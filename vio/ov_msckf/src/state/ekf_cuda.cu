#include "ekf_cuda.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ov_msckf {
namespace {

__global__ void k_add_diag(float *S, int m, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) S[(size_t)i * m + i] += v;
}
// fp64 -> fp32 conversion on the device, so the host never walks the matrix.
__global__ void k_d2f(const double *src, float *dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}

struct Ctx {
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solv = nullptr;
  cudaStream_t stream = nullptr;
  double *d_in = nullptr;              // staging for the fp64 uploads
  float *d_Pg = nullptr, *d_Psm = nullptr, *d_H = nullptr, *d_res = nullptr;
  float *d_M = nullptr, *d_S = nullptr, *d_X = nullptr, *d_D = nullptr, *d_T1 = nullptr, *d_dx = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  int lwork = 0;
  size_t cap_in = 0, cap_Pg = 0, cap_Psm = 0, cap_H = 0, cap_M = 0, cap_S = 0, cap_X = 0, cap_D = 0, cap_T1 = 0, cap_res = 0;
  bool ok = false;

  template <typename T> bool grow(T **p, size_t *cap, size_t need) {
    if (*cap >= need) return true;
    if (*p) cudaFree(*p);
    if (cudaMalloc(p, need * sizeof(T)) != cudaSuccess) { *p = nullptr; *cap = 0; return false; }
    *cap = need;
    return true;
  }
  bool init() {
    if (ok) return true;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return false;
    if (cusolverDnCreate(&solv) != CUSOLVER_STATUS_SUCCESS) return false;
    cublasSetStream(blas, stream);
    cusolverDnSetStream(solv, stream);
    if (cudaMalloc(&d_info, sizeof(int)) != cudaSuccess) return false;
    ok = true;
    return true;
  }
};

thread_local Ctx t_ctx;

} // namespace

bool ekf_cuda_enabled() {
  static const bool en = [] {
    const char *e = std::getenv("OV_GPU_EKF");
    if (!e || *e != '1') return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
      std::fprintf(stderr, "[ekf_cuda]: OV_GPU_EKF=1 but no CUDA device; using CPU\n");
      return false;
    }
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    std::fprintf(stderr, "[ekf_cuda]: fp32 GPU EKFUpdate enabled\n");
    return true;
  }();
  return en;
}

bool ekf_cuda_update(const double *Pg, int N, int Hc,
                     const double *Psm, const double *H, int m,
                     const double *res, double sigma2,
                     float *D_out, double *dx_out) {
  Ctx &c = t_ctx;
  if (!c.init()) return false;

  const size_t nPg = (size_t)N * Hc, nPsm = (size_t)Hc * Hc, nH = (size_t)m * Hc;
  const size_t nM = (size_t)N * m, nS = (size_t)m * m, nX = (size_t)m * N, nD = (size_t)N * N;
  const size_t nStage = nPg > nPsm ? (nPg > nH ? nPg : nH) : (nPsm > nH ? nPsm : nH);
  if (!c.grow(&c.d_in, &c.cap_in, nStage) || !c.grow(&c.d_Pg, &c.cap_Pg, nPg) ||
      !c.grow(&c.d_Psm, &c.cap_Psm, nPsm) || !c.grow(&c.d_H, &c.cap_H, nH) ||
      !c.grow(&c.d_M, &c.cap_M, nM) || !c.grow(&c.d_S, &c.cap_S, nS) ||
      !c.grow(&c.d_X, &c.cap_X, nX) || !c.grow(&c.d_D, &c.cap_D, nD) ||
      !c.grow(&c.d_T1, &c.cap_T1, (size_t)Hc * m) || !c.grow(&c.d_res, &c.cap_res, (size_t)m + N))
    return false;
  c.d_dx = c.d_res + m;

  auto up = [&](const double *src, size_t n, float *dst) {
    if (cudaMemcpyAsync(c.d_in, src, n * sizeof(double), cudaMemcpyHostToDevice, c.stream) != cudaSuccess) return false;
    k_d2f<<<(unsigned)((n + 255) / 256), 256, 0, c.stream>>>(c.d_in, dst, n);
    return true;
  };
  if (!up(Pg, nPg, c.d_Pg) || !up(Psm, nPsm, c.d_Psm) || !up(H, nH, c.d_H) || !up(res, (size_t)m, c.d_res))
    return false;

  const float one = 1.0f, zero = 0.0f, neg = -1.0f;
  // M = Pg * H^T            (N x Hc)(Hc x m)
  if (cublasSgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_T, N, m, Hc, &one, c.d_Pg, N, c.d_H, m, &zero, c.d_M, N) != CUBLAS_STATUS_SUCCESS) return false;
  // T1 = Psm * H^T          (Hc x Hc)(Hc x m)
  if (cublasSgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_T, Hc, m, Hc, &one, c.d_Psm, Hc, c.d_H, m, &zero, c.d_T1, Hc) != CUBLAS_STATUS_SUCCESS) return false;
  // S = H * T1 + s2*I       (m x Hc)(Hc x m)
  if (cublasSgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, m, m, Hc, &one, c.d_H, m, c.d_T1, Hc, &zero, c.d_S, m) != CUBLAS_STATUS_SUCCESS) return false;
  k_add_diag<<<(unsigned)((m + 255) / 256), 256, 0, c.stream>>>(c.d_S, m, (float)sigma2);
  // X = M^T then solve S X = X   ->  X = S^-1 M^T   (m x N)
  if (cublasSgeam(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, N, &one, c.d_M, N, &zero, c.d_X, m, c.d_X, m) != CUBLAS_STATUS_SUCCESS) return false;
  int lw = 0;
  if (cusolverDnSpotrf_bufferSize(c.solv, CUBLAS_FILL_MODE_UPPER, m, c.d_S, m, &lw) != CUSOLVER_STATUS_SUCCESS) return false;
  if (lw > c.lwork) { if (c.d_work) cudaFree(c.d_work); if (cudaMalloc(&c.d_work, (size_t)lw * sizeof(float)) != cudaSuccess) return false; c.lwork = lw; }
  if (cusolverDnSpotrf(c.solv, CUBLAS_FILL_MODE_UPPER, m, c.d_S, m, c.d_work, lw, c.d_info) != CUSOLVER_STATUS_SUCCESS) return false;
  if (cusolverDnSpotrs(c.solv, CUBLAS_FILL_MODE_UPPER, m, N, c.d_S, m, c.d_X, m, c.d_info) != CUSOLVER_STATUS_SUCCESS) return false;
  // D = X^T * M^T           (N x m)(m x N)
  if (cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_T, N, N, m, &one, c.d_X, m, c.d_M, N, &zero, c.d_D, N) != CUBLAS_STATUS_SUCCESS) return false;
  // dx = X^T * res
  if (cublasSgemv(c.blas, CUBLAS_OP_T, m, N, &one, c.d_X, m, c.d_res, 1, &zero, c.d_dx, 1) != CUBLAS_STATUS_SUCCESS) return false;

  std::vector<float> hdx(N);
  if (cudaMemcpyAsync(D_out, c.d_D, nD * sizeof(float), cudaMemcpyDeviceToHost, c.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(hdx.data(), c.d_dx, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, c.stream) != cudaSuccess) return false;
  int info = 0;
  if (cudaMemcpyAsync(&info, c.d_info, sizeof(int), cudaMemcpyDeviceToHost, c.stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(c.stream) != cudaSuccess) return false;
  if (info != 0) return false;   // Cholesky failed -> let the CPU path handle it
  for (int i = 0; i < N; i++) dx_out[i] = (double)hdx[i];
  (void)neg;
  return true;
}

} // namespace ov_msckf

// ======================= double-single (compensated fp32) GEMM ==========================
// The plain-fp32 path above was REJECTED on fleet ATE (2x drift): eps32-level error in the
// correction, amplified through S^-1 conditioning, accumulates in P over a flight. This
// path is different in kind: inputs are split V = hi + lo (hi = fp32(V), lo = fp32(V-hi)),
// hi*hi products are made EXACT with fmaf (2ProdFMA) and accumulated in a double-single
// (hi,lo) pair (2Sum), and the hi*lo cross terms ride in the lo word. Measured accuracy is
// ~1e-12 relative -- the same class as fp64 GEMM accumulation at these inner dims -- at
// fp32 throughput. Used for the big N-scale products only; S/chol/solve stay in fp64 Eigen.
namespace ov_msckf {
namespace {

#define DS_TS 32
__global__ void k_split_ds(const double *s, float *hi, float *lo, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float h = (float)s[i]; hi[i] = h; lo[i] = (float)(s[i] - (double)h); }
}

__global__ void k_ds_nt(const float *Ah, const float *Al, const float *Bh, const float *Bl,
                        double *C, int M, int N, int K, int lda, int ldb, int ldc, int sym) {
  if (sym && (blockIdx.y * DS_TS + DS_TS - 1) < blockIdx.x * DS_TS)
    return; // strictly-below-diagonal block of an upper-triangular request
  __shared__ float sAh[DS_TS][DS_TS + 1], sAl[DS_TS][DS_TS + 1], sBh[DS_TS][DS_TS + 1], sBl[DS_TS][DS_TS + 1];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row0 = blockIdx.x * DS_TS, col0 = blockIdx.y * DS_TS;
  float ch[2][2] = {{0, 0}, {0, 0}}, cl[2][2] = {{0, 0}, {0, 0}};
  for (int k0 = 0; k0 < K; k0 += DS_TS) {
    for (int ij = 0; ij < 4; ij++) {
      int i = ij & 1, j = ij >> 1;
      int ar = row0 + ty + 16 * i, ak = k0 + tx + 16 * j;
      int br = col0 + ty + 16 * i;
      bool av = (ar < M && ak < K), bv = (br < N && ak < K);
      sAh[ty + 16 * i][tx + 16 * j] = av ? Ah[ar + (size_t)ak * lda] : 0.f;
      sAl[ty + 16 * i][tx + 16 * j] = av ? Al[ar + (size_t)ak * lda] : 0.f;
      sBh[ty + 16 * i][tx + 16 * j] = bv ? Bh[br + (size_t)ak * ldb] : 0.f;
      sBl[ty + 16 * i][tx + 16 * j] = bv ? Bl[br + (size_t)ak * ldb] : 0.f;
    }
    __syncthreads();
    const int kend = min(DS_TS, K - k0);
#pragma unroll 8
    for (int k = 0; k < kend; k++) {
      float ahs[2] = {sAh[ty][k], sAh[ty + 16][k]}, als[2] = {sAl[ty][k], sAl[ty + 16][k]};
      float bhs[2] = {sBh[tx][k], sBh[tx + 16][k]}, bls[2] = {sBl[tx][k], sBl[tx + 16][k]};
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
          float p = ahs[i] * bhs[j];
          float e = fmaf(ahs[i], bhs[j], -p);
          cl[i][j] = fmaf(ahs[i], bls[j], cl[i][j]);
          cl[i][j] = fmaf(als[i], bhs[j], cl[i][j]);
          cl[i][j] += e;
          float t = ch[i][j] + p;
          float bb = t - ch[i][j];
          float err = (ch[i][j] - (t - bb)) + (p - bb);
          ch[i][j] = t;
          cl[i][j] += err;
        }
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      int r = row0 + ty + 16 * i, cc = col0 + tx + 16 * j;
      if (r < M && cc < N && (!sym || cc >= r))
        C[r + (size_t)cc * ldc] = (double)ch[i][j] + (double)cl[i][j];
    }
}

struct DsCtx {
  cudaStream_t stream = nullptr;
  double *dA64 = nullptr, *dB64 = nullptr, *dC = nullptr;
  float *dAh = nullptr, *dAl = nullptr, *dBh = nullptr, *dBl = nullptr;
  size_t capA = 0, capB = 0, capC = 0;
  bool ok = false;
  template <typename T> bool grow(T **p, size_t *cap, size_t need, size_t elt) {
    if (*cap >= need) return true;
    if (*p) cudaFree(*p);
    if (cudaMalloc((void **)p, need * elt) != cudaSuccess) { *p = nullptr; *cap = 0; return false; }
    *cap = need;
    return true;
  }
  bool init() {
    if (ok) return true;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    ok = true;
    return true;
  }
};
thread_local DsCtx t_ds;

} // namespace

bool ekf_ds_available() {
  static const bool en = [] {
    const char *e = std::getenv("OV_DS_EKF");
    if (!e || *e != '1') return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
      std::fprintf(stderr, "[ekf_cuda]: OV_DS_EKF=1 but no CUDA device; using CPU\n");
      return false;
    }
    std::fprintf(stderr, "[ekf_cuda]: double-single GPU GEMM enabled for EKF big products\n");
    return true;
  }();
  return en;
}

bool ekf_ds_gemm_nt(const double *A, int M, int K, const double *B, int N, double *C, bool sym) {
  if (sym && (M != N || A != B)) return false;
  DsCtx &c = t_ds;
  if (!c.init()) return false;
  const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
  const bool same = (A == B);
  if (!c.grow(&c.dA64, &c.capA, nA, 8) || !c.grow(&c.dC, &c.capC, nC, 8)) return false;
  if (!same && !c.grow(&c.dB64, &c.capB, nB, 8)) return false;
  // split buffers ride behind the fp64 staging: grow to hold hi+lo each
  static thread_local size_t capAh = 0, capBh = 0;
  if (capAh < nA) {
    if (c.dAh) cudaFree(c.dAh);
    if (c.dAl) cudaFree(c.dAl);
    if (cudaMalloc((void **)&c.dAh, nA * 4) != cudaSuccess) { c.dAh = nullptr; capAh = 0; return false; }
    if (cudaMalloc((void **)&c.dAl, nA * 4) != cudaSuccess) { c.dAl = nullptr; capAh = 0; return false; }
    capAh = nA;
  }
  if (!same && capBh < nB) {
    if (c.dBh) cudaFree(c.dBh);
    if (c.dBl) cudaFree(c.dBl);
    if (cudaMalloc((void **)&c.dBh, nB * 4) != cudaSuccess) { c.dBh = nullptr; capBh = 0; return false; }
    if (cudaMalloc((void **)&c.dBl, nB * 4) != cudaSuccess) { c.dBl = nullptr; capBh = 0; return false; }
    capBh = nB;
  }
  if (cudaMemcpyAsync(c.dA64, A, nA * 8, cudaMemcpyHostToDevice, c.stream) != cudaSuccess) return false;
  k_split_ds<<<(unsigned)((nA + 255) / 256), 256, 0, c.stream>>>(c.dA64, c.dAh, c.dAl, nA);
  const float *bh = c.dAh, *bl = c.dAl;
  if (!same) {
    if (cudaMemcpyAsync(c.dB64, B, nB * 8, cudaMemcpyHostToDevice, c.stream) != cudaSuccess) return false;
    k_split_ds<<<(unsigned)((nB + 255) / 256), 256, 0, c.stream>>>(c.dB64, c.dBh, c.dBl, nB);
    bh = c.dBh; bl = c.dBl;
  }
  dim3 th(16, 16), bl2((M + DS_TS - 1) / DS_TS, (N + DS_TS - 1) / DS_TS);
  k_ds_nt<<<bl2, th, 0, c.stream>>>(c.dAh, c.dAl, bh, bl, c.dC, M, N, K, M, N, M, sym ? 1 : 0);
  if (cudaMemcpyAsync(C, c.dC, nC * 8, cudaMemcpyDeviceToHost, c.stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(c.stream) != cudaSuccess) return false;
  return true;
}

} // namespace ov_msckf

// ==================== fp32 compression QR + compensated SYRK =========================
namespace ov_msckf {
namespace {

struct QrCtx {
  cusolverDnHandle_t solv = nullptr;
  cudaStream_t stream = nullptr;
  double *dW64 = nullptr;   // staging r x (c+1)
  float *dW = nullptr, *dTau = nullptr, *dwork = nullptr;
  double *dR = nullptr;     // fp64 result staging (c x (c+1))
  int *dinfo = nullptr;
  size_t capW = 0, capTau = 0, capR = 0;
  int lwork = 0;
  bool ok = false;
  template <typename T> bool grow(T **p, size_t *cap, size_t need, size_t elt) {
    if (*cap >= need) return true;
    if (*p) cudaFree(*p);
    if (cudaMalloc((void **)p, need * elt) != cudaSuccess) { *p = nullptr; *cap = 0; return false; }
    *cap = need;
    return true;
  }
  bool init() {
    if (ok) return true;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    if (cusolverDnCreate(&solv) != CUSOLVER_STATUS_SUCCESS) return false;
    cusolverDnSetStream(solv, stream);
    if (cudaMalloc((void **)&dinfo, sizeof(int)) != cudaSuccess) return false;
    ok = true;
    return true;
  }
};
thread_local QrCtx t_qr;

__global__ void k_f2d_2d(const float *src, int lds, double *dst, int ldd, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) dst[i + (size_t)j * ldd] = (double)src[i + (size_t)j * lds];
}
// zero strictly-below-diagonal of the top c x c (R must be clean upper triangular)
__global__ void k_tri_clean(double *R, int ldd, int c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < c && j < c && i > j) R[i + (size_t)j * ldd] = 0.0;
}

} // namespace

bool ekf_gpu_qr_enabled() {
  static const bool en = [] {
    const char *e = std::getenv("OV_GPU_QR");
    if (!e || *e != '1') return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return false;
    std::fprintf(stderr, "[ekf_cuda]: fp32 GPU compression QR enabled\n");
    return true;
  }();
  return en;
}

bool ekf_gpu_qr(const double *H, int r, int c, const double *res, double *R_out, double *res_out) {
  QrCtx &q = t_qr;
  if (!q.init()) return false;
  const int ca = c + 1;                       // augmented [H | res]
  const size_t nW = (size_t)r * ca, nR = (size_t)c * ca;
  if (!q.grow(&q.dW64, &q.capW, nW, 8) || !q.grow(&q.dTau, &q.capTau, (size_t)ca, 4) ||
      !q.grow(&q.dR, &q.capR, nR, 8))
    return false;
  // reuse dW64's tail? keep separate fp32 buffer sized nW
  static thread_local size_t capWf = 0;
  if (capWf < nW) {
    if (q.dW) cudaFree(q.dW);
    if (cudaMalloc((void **)&q.dW, nW * 4) != cudaSuccess) { q.dW = nullptr; capWf = 0; return false; }
    capWf = nW;
  }
  // upload H then res into the augmented fp64 staging, convert to fp32
  if (cudaMemcpyAsync(q.dW64, H, (size_t)r * c * 8, cudaMemcpyHostToDevice, q.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(q.dW64 + (size_t)r * c, res, (size_t)r * 8, cudaMemcpyHostToDevice, q.stream) != cudaSuccess) return false;
  k_d2f<<<(unsigned)((nW + 255) / 256), 256, 0, q.stream>>>(q.dW64, q.dW, nW);
  int lw = 0;
  if (cusolverDnSgeqrf_bufferSize(q.solv, r, ca, q.dW, r, &lw) != CUSOLVER_STATUS_SUCCESS) return false;
  if (lw > q.lwork) {
    if (q.dwork) cudaFree(q.dwork);
    if (cudaMalloc((void **)&q.dwork, (size_t)lw * 4) != cudaSuccess) { q.dwork = nullptr; q.lwork = 0; return false; }
    q.lwork = lw;
  }
  if (cusolverDnSgeqrf(q.solv, r, ca, q.dW, r, q.dTau, q.dwork, q.lwork, q.dinfo) != CUSOLVER_STATUS_SUCCESS) return false;
  // top c rows of the factored augmented matrix -> fp64, clean sub-diagonal of the R block
  dim3 th(16, 16), bl((unsigned)((c + 15) / 16), (unsigned)((ca + 15) / 16));
  k_f2d_2d<<<bl, th, 0, q.stream>>>(q.dW, r, q.dR, c, c, ca);
  dim3 bl2((unsigned)((c + 15) / 16), (unsigned)((c + 15) / 16));
  k_tri_clean<<<bl2, th, 0, q.stream>>>(q.dR, c, c);
  int info = 0;
  if (cudaMemcpyAsync(R_out, q.dR, (size_t)c * c * 8, cudaMemcpyDeviceToHost, q.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(res_out, q.dR + (size_t)c * c, (size_t)c * 8, cudaMemcpyDeviceToHost, q.stream) != cudaSuccess) return false;
  if (cudaMemcpyAsync(&info, q.dinfo, sizeof(int), cudaMemcpyDeviceToHost, q.stream) != cudaSuccess) return false;
  if (cudaStreamSynchronize(q.stream) != cudaSuccess) return false;
  return info == 0;
}

namespace {
struct SyCtx {
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  double *dV64 = nullptr, *dD64 = nullptr;
  float *dHi = nullptr, *dLo = nullptr, *dD1 = nullptr, *dD2 = nullptr;
  size_t capV = 0, capD = 0;
  bool ok = false;
  bool init() {
    if (ok) return true;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return false;
    cublasSetStream(blas, stream);
    ok = true;
    return true;
  }
};
thread_local SyCtx t_sy;
__global__ void k_sum2_f2d(const float *D1, const float *D2, double *D, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) D[i] = (double)D1[i] + (double)D2[i];
}
} // namespace

bool ekf_gpu_syrk_enabled() {
  static const bool en = [] {
    const char *e = std::getenv("OV_GPU_SYRK");
    if (!e || *e != '1') return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return false;
    std::fprintf(stderr, "[ekf_cuda]: compensated fp32 GPU SYRK enabled\n");
    return true;
  }();
  return en;
}

bool ekf_gpu_syrk(const double *V, int N, int m, double *D_out) {
  SyCtx &s = t_sy;
  if (!s.init()) return false;
  const size_t nV = (size_t)N * m, nD = (size_t)N * N;
  static thread_local size_t capVf = 0, capDf = 0;
  if (s.capV < nV) {
    if (s.dV64) cudaFree(s.dV64);
    if (cudaMalloc((void **)&s.dV64, nV * 8) != cudaSuccess) { s.dV64 = nullptr; s.capV = 0; return false; }
    s.capV = nV;
  }
  if (capVf < nV) {
    if (s.dHi) cudaFree(s.dHi);
    if (s.dLo) cudaFree(s.dLo);
    if (cudaMalloc((void **)&s.dHi, nV * 4) != cudaSuccess) { s.dHi = nullptr; capVf = 0; return false; }
    if (cudaMalloc((void **)&s.dLo, nV * 4) != cudaSuccess) { s.dLo = nullptr; capVf = 0; return false; }
    capVf = nV;
  }
  if (s.capD < nD) {
    if (s.dD64) cudaFree(s.dD64);
    if (cudaMalloc((void **)&s.dD64, nD * 8) != cudaSuccess) { s.dD64 = nullptr; s.capD = 0; return false; }
    s.capD = nD;
  }
  if (capDf < nD) {
    if (s.dD1) cudaFree(s.dD1);
    if (s.dD2) cudaFree(s.dD2);
    if (cudaMalloc((void **)&s.dD1, nD * 4) != cudaSuccess) { s.dD1 = nullptr; capDf = 0; return false; }
    if (cudaMalloc((void **)&s.dD2, nD * 4) != cudaSuccess) { s.dD2 = nullptr; capDf = 0; return false; }
    capDf = nD;
  }
  if (cudaMemcpyAsync(s.dV64, V, nV * 8, cudaMemcpyHostToDevice, s.stream) != cudaSuccess) return false;
  k_split_ds<<<(unsigned)((nV + 255) / 256), 256, 0, s.stream>>>(s.dV64, s.dHi, s.dLo, nV);
  const float one = 1.f, zero = 0.f;
  if (cublasSsyrk(s.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, N, m, &one, s.dHi, N, &zero, s.dD1, N) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasSsyr2k(s.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, N, m, &one, s.dHi, N, s.dLo, N, &zero, s.dD2, N) != CUBLAS_STATUS_SUCCESS) return false;
  k_sum2_f2d<<<(unsigned)((nD + 255) / 256), 256, 0, s.stream>>>(s.dD1, s.dD2, s.dD64, nD);
  if (cudaMemcpyAsync(D_out, s.dD64, nD * 8, cudaMemcpyDeviceToHost, s.stream) != cudaSuccess) return false;
  return cudaStreamSynchronize(s.stream) == cudaSuccess;
}

} // namespace ov_msckf
