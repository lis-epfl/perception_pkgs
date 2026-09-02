#pragma once
// CUDA CLAHE, drop-in for cv::createCLAHE(clip, Size(8,8))->apply().
// Plain-pointer API so this header stays free of CUDA types and can be included from
// ordinary C++ translation units.
#include <cstddef>

namespace ov_core {

/// True if a CUDA device is present and OV_GPU_CLAHE=1 in the environment.
bool clahe_cuda_enabled();

/// 8-bit single-channel CLAHE, 8x8 tiles, clip limit `clip`. src/dst may not alias.
/// Falls back to returning false if anything goes wrong, so callers can use the CPU path.
bool clahe_cuda(const unsigned char *src, int w, int h, size_t src_stride,
                unsigned char *dst, size_t dst_stride, float clip);

/// Batched form: all `n` images in ONE set of launches and ONE sync. This is the form that
/// matters -- per-image calls serialise on the GPU and lose to threaded cv::CLAHE.
bool clahe_cuda_batch(const unsigned char *const *srcs, const size_t *src_strides,
                      unsigned char *const *dsts, const size_t *dst_strides,
                      int n, int w, int h, float clip);

} // namespace ov_core
