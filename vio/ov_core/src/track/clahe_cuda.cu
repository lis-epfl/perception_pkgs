// CUDA CLAHE for TrackKLT. Measured 6.14 ms/image on the Orin's CPU (cv::CLAHE) vs ~0.41 ms
// for this kernel; with the host<->device copies of a 1 MB image it lands near 0.5 ms.
//
// Buffers are thread_local because TrackKLT runs feed_new_camera under parallel_for_ once
// num_opencv_threads > 0, and a shared scratch buffer would race.
#include "clahe_cuda.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace ov_core {
namespace {

constexpr int TX = 8, TY = 8;

__global__ void k_hist(const unsigned char *img, size_t pitch, int w, int h, int *hists) {
  __shared__ int sh[256];
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) sh[i] = 0;
  __syncthreads();
  const int tx = blockIdx.x, ty = blockIdx.y;
  const int x0 = tx * w / TX, x1 = (tx + 1) * w / TX;
  const int y0 = ty * h / TY, y1 = (ty + 1) * h / TY;
  for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y)
    for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x)
      atomicAdd(&sh[img[(size_t)y * pitch + x]], 1);
  __syncthreads();
  int *out = hists + (ty * TX + tx) * 256;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) out[i] = sh[i];
}

// Mirrors OpenCV's CLAHE: clip, redistribute the excess uniformly, then CDF -> LUT.
__global__ void k_lut(const int *hists, int w, int h, float clip, unsigned char *luts) {
  __shared__ int sh[256];
  const int tile = blockIdx.x;
  const int *hin = hists + tile * 256;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = hin[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    const int tw = w / TX, th = h / TY;
    int limit = (int)(clip * (float)(tw * th) / 256.0f);
    if (limit < 1) limit = 1;
    long excess = 0;
    for (int i = 0; i < 256; i++)
      if (sh[i] > limit) { excess += sh[i] - limit; sh[i] = limit; }
    const int inc = (int)(excess / 256);
    int residual = (int)(excess - (long)inc * 256);
    for (int i = 0; i < 256; i++) sh[i] += inc;
    // OpenCV spreads the residual across the histogram with a stride; adding it to the lowest
    // bins instead skews the CDF and was the dominant error on low-texture cameras
    // (cam D: 234k differing pixels -> 3k once matched).
    if (residual != 0) {
      int step = 256 / residual; if (step < 1) step = 1;
      for (int i = 0; i < 256 && residual > 0; i += step, residual--) sh[i]++;
    }
    long total = 0;
    for (int i = 0; i < 256; i++) total += sh[i];
    const float sc = total > 0 ? 255.0f / (float)total : 0.0f;
    long cdf = 0;
    unsigned char *lo = luts + tile * 256;
    for (int i = 0; i < 256; i++) {
      cdf += sh[i];
      const float v = cdf * sc;
      lo[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v + 0.5f));
    }
  }
}

__global__ void k_apply(const unsigned char *src, size_t sp, unsigned char *dst, size_t dp,
                        int w, int h, const unsigned char *luts) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float tw = (float)w / TX, th = (float)h / TY;
  // OpenCV's tile grid origin is x/tw - 0.5, NOT (x+0.5)/tw - 0.5. The half-pixel shift moved
  // every interpolation weight and accounted for ~5x of the mismatch against cv::CLAHE.
  const float fx = (float)x / tw - 0.5f, fy = (float)y / th - 0.5f;
  const int ix = (int)floorf(fx), iy = (int)floorf(fy);
  const float ax = fx - ix, ay = fy - iy;
  const int x0 = min(max(ix, 0), TX - 1), x1 = min(max(ix + 1, 0), TX - 1);
  const int y0 = min(max(iy, 0), TY - 1), y1 = min(max(iy + 1, 0), TY - 1);
  const unsigned char v = src[(size_t)y * sp + x];
  const float v00 = luts[(y0 * TX + x0) * 256 + v], v01 = luts[(y0 * TX + x1) * 256 + v];
  const float v10 = luts[(y1 * TX + x0) * 256 + v], v11 = luts[(y1 * TX + x1) * 256 + v];
  const float top = v00 + ax * (v01 - v00), bot = v10 + ax * (v11 - v10);
  const float o = top + ay * (bot - top);
  dst[(size_t)y * dp + x] = (unsigned char)(o < 0 ? 0 : (o > 255 ? 255 : o + 0.5f));
}

__global__ void k_hist_b(const unsigned char *const *imgs, const size_t *pitches, int w, int h, int *hists) {
  __shared__ int sh[256];
  const int img_i = blockIdx.z;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) sh[i] = 0;
  __syncthreads();
  const int tx = blockIdx.x, ty = blockIdx.y;
  const int x0 = tx * w / TX, x1 = (tx + 1) * w / TX;
  const int y0 = ty * h / TY, y1 = (ty + 1) * h / TY;
  const unsigned char *img = imgs[img_i];
  const size_t pitch = pitches[img_i];
  for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y)
    for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x)
      atomicAdd(&sh[img[(size_t)y * pitch + x]], 1);
  __syncthreads();
  int *out = hists + ((size_t)img_i * TX * TY + (ty * TX + tx)) * 256;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) out[i] = sh[i];
}

__global__ void k_apply_b(const unsigned char *const *srcs, const size_t *sps,
                          unsigned char *const *dsts, const size_t *dps,
                          int w, int h, const unsigned char *luts) {
  const int img_i = blockIdx.z;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float tw = (float)w / TX, th = (float)h / TY;
  // OpenCV's tile grid origin is x/tw - 0.5, NOT (x+0.5)/tw - 0.5. The half-pixel shift moved
  // every interpolation weight and accounted for ~5x of the mismatch against cv::CLAHE.
  const float fx = (float)x / tw - 0.5f, fy = (float)y / th - 0.5f;
  const int ix = (int)floorf(fx), iy = (int)floorf(fy);
  const float ax = fx - ix, ay = fy - iy;
  const int x0 = min(max(ix, 0), TX - 1), x1 = min(max(ix + 1, 0), TX - 1);
  const int y0 = min(max(iy, 0), TY - 1), y1 = min(max(iy + 1, 0), TY - 1);
  const unsigned char *lut = luts + (size_t)img_i * TX * TY * 256;
  const unsigned char v = srcs[img_i][(size_t)y * sps[img_i] + x];
  const float v00 = lut[(y0 * TX + x0) * 256 + v], v01 = lut[(y0 * TX + x1) * 256 + v];
  const float v10 = lut[(y1 * TX + x0) * 256 + v], v11 = lut[(y1 * TX + x1) * 256 + v];
  const float top = v00 + ax * (v01 - v00), bot = v10 + ax * (v11 - v10);
  const float o = top + ay * (bot - top);
  dsts[img_i][(size_t)y * dps[img_i] + x] = (unsigned char)(o < 0 ? 0 : (o > 255 ? 255 : o + 0.5f));
}

struct Ctx {
  unsigned char *d_src = nullptr, *d_dst = nullptr, *d_lut = nullptr;
  int *d_hist = nullptr;
  size_t p_src = 0, p_dst = 0;
  int w = 0, h = 0;
  cudaStream_t stream = nullptr;
  bool ok = false;

  bool ensure(int W, int H) {
    if (ok && W == w && H == h) return true;
    release();
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    if (cudaMallocPitch(&d_src, &p_src, W, H) != cudaSuccess) return false;
    if (cudaMallocPitch(&d_dst, &p_dst, W, H) != cudaSuccess) return false;
    if (cudaMalloc(&d_hist, TX * TY * 256 * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&d_lut, TX * TY * 256) != cudaSuccess) return false;
    w = W; h = H; ok = true;
    return true;
  }
  void release() {
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    if (d_hist) cudaFree(d_hist);
    if (d_lut) cudaFree(d_lut);
    if (stream) cudaStreamDestroy(stream);
    d_src = d_dst = d_lut = nullptr; d_hist = nullptr; stream = nullptr; ok = false;
  }
  ~Ctx() { release(); }
};

thread_local Ctx t_ctx;

constexpr int MAXB = 8;
struct BCtx {
  unsigned char *d_img[MAXB] = {nullptr}, *d_out[MAXB] = {nullptr};
  size_t p_img[MAXB] = {0}, p_out[MAXB] = {0};
  const unsigned char **dev_src = nullptr; unsigned char **dev_dst = nullptr;
  size_t *dev_sp = nullptr, *dev_dp = nullptr;
  int *d_hist = nullptr; unsigned char *d_lut = nullptr;
  unsigned char *h_in[MAXB] = {nullptr}, *h_out[MAXB] = {nullptr};   // pinned staging
  cudaStream_t stream = nullptr;
  int w = 0, h = 0, n = 0; bool ok = false;

  bool ensure(int N, int W, int H) {
    if (ok && N == n && W == w && H == h) return true;
    release();
    if (N > MAXB) return false;
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    for (int i = 0; i < N; i++) {
      if (cudaMallocPitch(&d_img[i], &p_img[i], W, H) != cudaSuccess) return false;
      if (cudaMallocPitch(&d_out[i], &p_out[i], W, H) != cudaSuccess) return false;
    }
    if (cudaMalloc(&dev_src, N * sizeof(void *)) != cudaSuccess) return false;
    if (cudaMalloc(&dev_dst, N * sizeof(void *)) != cudaSuccess) return false;
    if (cudaMalloc(&dev_sp, N * sizeof(size_t)) != cudaSuccess) return false;
    if (cudaMalloc(&dev_dp, N * sizeof(size_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_hist, (size_t)N * TX * TY * 256 * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&d_lut, (size_t)N * TX * TY * 256) != cudaSuccess) return false;
    // Pinned host buffers: pageable memcpy2D goes through a driver staging copy and costs
    // real CPU. Pinned memory lets the DMA engine do the work.
    for (int i = 0; i < N; i++) {
      if (cudaHostAlloc(&h_in[i], (size_t)W * H, cudaHostAllocDefault) != cudaSuccess) return false;
      if (cudaHostAlloc(&h_out[i], (size_t)W * H, cudaHostAllocDefault) != cudaSuccess) return false;
    }
    cudaMemcpy(dev_src, d_img, N * sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dst, d_out, N * sizeof(void *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sp, p_img, N * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dp, p_out, N * sizeof(size_t), cudaMemcpyHostToDevice);
    n = N; w = W; h = H; ok = true;
    return true;
  }
  void release() {
    for (int i = 0; i < MAXB; i++) {
      if (d_img[i]) cudaFree(d_img[i]); if (d_out[i]) cudaFree(d_out[i]);
      if (h_in[i]) cudaFreeHost(h_in[i]); if (h_out[i]) cudaFreeHost(h_out[i]);
      d_img[i]=nullptr; d_out[i]=nullptr; h_in[i]=nullptr; h_out[i]=nullptr;
    }
    if (dev_src) cudaFree(dev_src); if (dev_dst) cudaFree(dev_dst);
    if (dev_sp) cudaFree(dev_sp); if (dev_dp) cudaFree(dev_dp);
    if (d_hist) cudaFree(d_hist); if (d_lut) cudaFree(d_lut);
    if (stream) cudaStreamDestroy(stream);
    dev_src=nullptr; dev_dst=nullptr; dev_sp=nullptr; dev_dp=nullptr;
    d_hist=nullptr; d_lut=nullptr; stream=nullptr; ok=false;
  }
  ~BCtx() { release(); }
};
thread_local BCtx t_bctx;

} // namespace

bool clahe_cuda_enabled() {
  static const bool en = [] {
    const char *e = std::getenv("OV_GPU_CLAHE");
    if (!e || *e != '1') return false;
    // MUST be set before the CUDA context exists. The default (cudaDeviceScheduleSpin/Auto)
    // busy-waits inside cudaStreamSynchronize, which is catastrophic when the CPU is shared
    // with other autonomy modules: measured sys time went 23.8 s -> 113.7 s and wall 138 s ->
    // 263 s under a 50% background load. Blocking sync yields the core instead.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
      std::fprintf(stderr, "[clahe_cuda]: OV_GPU_CLAHE=1 but no CUDA device; using CPU\n");
      return false;
    }
    std::fprintf(stderr, "[clahe_cuda]: GPU CLAHE enabled\n");
    return true;
  }();
  return en;
}

bool clahe_cuda(const unsigned char *src, int w, int h, size_t src_stride,
                unsigned char *dst, size_t dst_stride, float clip) {
  Ctx &c = t_ctx;
  if (!c.ensure(w, h)) return false;
  if (cudaMemcpy2DAsync(c.d_src, c.p_src, src, src_stride, w, h, cudaMemcpyHostToDevice, c.stream) != cudaSuccess)
    return false;
  cudaMemsetAsync(c.d_hist, 0, TX * TY * 256 * sizeof(int), c.stream);
  k_hist<<<dim3(TX, TY), dim3(16, 16), 0, c.stream>>>(c.d_src, c.p_src, w, h, c.d_hist);
  k_lut<<<TX * TY, 256, 0, c.stream>>>(c.d_hist, w, h, clip, c.d_lut);
  k_apply<<<dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), 0, c.stream>>>(
      c.d_src, c.p_src, c.d_dst, c.p_dst, w, h, c.d_lut);
  if (cudaMemcpy2DAsync(dst, dst_stride, c.d_dst, c.p_dst, w, h, cudaMemcpyDeviceToHost, c.stream) != cudaSuccess)
    return false;
  return cudaStreamSynchronize(c.stream) == cudaSuccess;
}

bool clahe_cuda_batch(const unsigned char *const *srcs, const size_t *src_strides,
                      unsigned char *const *dsts, const size_t *dst_strides,
                      int n, int w, int h, float clip) {
  BCtx &c = t_bctx;
  if (!c.ensure(n, w, h)) return false;
  for (int i = 0; i < n; i++) {
    for (int r = 0; r < h; r++) memcpy(c.h_in[i] + (size_t)r * w, srcs[i] + (size_t)r * src_strides[i], w);
    if (cudaMemcpy2DAsync(c.d_img[i], c.p_img[i], c.h_in[i], w, w, h,
                          cudaMemcpyHostToDevice, c.stream) != cudaSuccess)
      return false;
  }
  cudaMemsetAsync(c.d_hist, 0, (size_t)n * TX * TY * 256 * sizeof(int), c.stream);
  k_hist_b<<<dim3(TX, TY, n), dim3(16, 16), 0, c.stream>>>(c.dev_src, c.dev_sp, w, h, c.d_hist);
  k_lut<<<TX * TY * n, 256, 0, c.stream>>>(c.d_hist, w, h, clip, c.d_lut);
  k_apply_b<<<dim3((w + 31) / 32, (h + 7) / 8, n), dim3(32, 8), 0, c.stream>>>(
      c.dev_src, c.dev_sp, c.dev_dst, c.dev_dp, w, h, c.d_lut);
  for (int i = 0; i < n; i++)
    if (cudaMemcpy2DAsync(c.h_out[i], w, c.d_out[i], c.p_out[i], w, h,
                          cudaMemcpyDeviceToHost, c.stream) != cudaSuccess)
      return false;
  if (cudaStreamSynchronize(c.stream) != cudaSuccess) return false;
  for (int i = 0; i < n; i++)
    for (int r = 0; r < h; r++) memcpy(dsts[i] + (size_t)r * dst_strides[i], c.h_out[i] + (size_t)r * w, w);
  return true;
}

} // namespace ov_core
