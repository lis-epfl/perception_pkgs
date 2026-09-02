#pragma once
// fp32 cuBLAS/cuSOLVER back end for StateHelper::EKFUpdate's dense linear algebra.
//
// Deliberately does NOT hold the covariance in fp32. P stays fp64 on the host; only the
// CORRECTION terms are computed in single precision:
//     M  = Pg * H^T                (N x m)
//     S  = H * Psm * H^T + s2*I    (m x m)
//     X  = S^-1 * M^T              (m x N)   [Cholesky solve, not an explicit inverse]
//     D  = K * M^T = X^T * M^T     (N x N)   <- returned, subtracted from P in fp64
//     dx = K * res = X^T * res     (N)
// so single-precision error enters the increment, not the accumulated covariance.
//
// Orin note: fp64 on this GPU runs at ~1/32 rate and measured 0.92-1.11x vs Eigen, i.e.
// worthless; fp32 measured 14-45x. That is why this is fp32 at all.
#include <cstddef>

namespace ov_msckf {

/// True if OV_GPU_EKF=1 and a CUDA device with cuBLAS/cuSOLVER is usable.
bool ekf_cuda_enabled();

/// All inputs column-major (Eigen default). Returns false on any failure so the caller can
/// fall back to the CPU path. D_out is N*N column-major; dx_out is N.
bool ekf_cuda_update(const double *Pg, int N, int Hc,
                     const double *Psm, const double *H, int m,
                     const double *res, double sigma2,
                     float *D_out, double *dx_out);

} // namespace ov_msckf

namespace ov_msckf {

/// True if OV_DS_EKF=1 and a CUDA device is usable. Independent of ekf_cuda_enabled().
bool ekf_ds_available();

/// C[MxN] = A[MxK] * B[NxK]^T in double-single compensated fp32 (fp64 in/out, ~1e-12
/// relative accuracy -- fp64-class for this filter, unlike plain fp32). Column-major.
/// If sym (requires B==A pointer-wise and M==N), only the upper triangle of C is written.
/// Returns false on any CUDA failure so the caller can run the Eigen path instead.
bool ekf_ds_gemm_nt(const double *A, int M, int K, const double *B, int N, double *C, bool sym);

} // namespace ov_msckf

namespace ov_msckf {

/// True if OV_GPU_QR=1 and CUDA usable. fp32 compression QR for big stacked systems.
bool ekf_gpu_qr_enabled();

/// In: H (r x c fp64 col-major), res (r). Out (on success): H's top c x c holds R (upper),
/// res's top c holds Q^T res. Caller resizes. fp32 internally (~1e-4 relative on R) --
/// the compressed system is a re-representation of the measurement; linearization error
/// dwarfs this, but it is NOT exact: fleet-gate before shipping. Returns false -> CPU path.
bool ekf_gpu_qr(const double *H, int r, int c, const double *res, double *R_out, double *res_out);

/// True if OV_GPU_SYRK=1 and CUDA usable.
bool ekf_gpu_syrk_enabled();

/// D = V * V^T (N x N, fp64 out, upper triangle valid) via compensated fp32 cuBLAS:
/// V split hi+lo, D = syrk(hi) + syr2k(hi,lo), summed in fp64 on device (~1e-6 relative --
/// forward-stable, no S^-1 amplification, but NOT exact: fleet-gate). false -> CPU path.
bool ekf_gpu_syrk(const double *V, int N, int m, double *D_out);

} // namespace ov_msckf
