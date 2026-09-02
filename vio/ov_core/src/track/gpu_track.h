#pragma once
// GPU-resident tracking front end for TrackKLT.
//
// Split of responsibilities, chosen to preserve OpenVINS semantics exactly:
//   GPU  pixels  -- CLAHE, pyramid, FAST+NMS+per-cell quota, pyramidal KLT
//   CPU  points  -- pruning, the min_px_dist occupancy grid, greedy dedup, id assignment,
//                   undistortion and RANSAC
// The CPU side is O(#features) (a few hundred), so it costs nothing, and keeping it there means
// feature *selection order* is unchanged -- which matters because MSCKF picks features by id.
//
// Images are uploaded once per frame and never come back; only keypoints are downloaded.
// Mask convention matches TrackKLT: a pixel with value > 127 is REJECTED.
#include <cstddef>
#include <vector>

namespace ov_core {

/// True if OV_GPU_TRACK=1 and a CUDA device is usable.
bool gpu_track_enabled();

/// CLAHE + pyramid for this image; becomes the camera's "current" resident frame.
// `ts` is the frame-set timestamp.  Under OV_NVJPG=1 it is the key into the device-side
// staging map: a HIT means the pixels are ALREADY on the device (decoded by NVJPG straight
// into a dmabuf the CUDA context has registered), so the host->device upload below is
// skipped entirely.  -1 (the default) means "no staging, upload as usual".
bool gpu_prepare(std::size_t cam_id, const unsigned char *img, int w, int h, std::size_t stride,
                 const unsigned char *mask, std::size_t mask_stride, bool do_clahe, float clahe_clip, double ts = -1.0);

/// Detect on the camera's PREVIOUS resident frame (or the current one when on_current, i.e. the
/// very first frame of a track). Honours the base mask, excludes a square of half-width
/// min_px_dist around each occ point, and only fills cells whose per_cell_want > 0.
/// Candidates come back sorted by descending FAST score so the caller's greedy dedup keeps the
/// strongest, matching Grider_GRID's response ordering.
bool gpu_detect(std::size_t cam_id, bool on_current,
                const float *occ_x, const float *occ_y, int n_occ, int min_px_dist,
                const int *per_cell_want, int grid_x, int grid_y, int fast_threshold,
                std::vector<float> &out_x, std::vector<float> &out_y, bool submit_only = false);

/// Completes a gpu_detect(..., submit_only=true): sync, download, sort by descending score.
bool gpu_detect_complete(std::size_t cam_id, std::vector<float> &out_x, std::vector<float> &out_y);

/// Pyramidal KLT from the previous resident pyramid to the current one.
/// status[i] = 1 when the point was tracked successfully.
/// start_level: highest pyramid level to begin coarse-to-fine at (-1 = full pyramid).
bool gpu_klt(std::size_t cam_id, const std::vector<float> &prev_x, const std::vector<float> &prev_y,
             std::vector<float> &cur_x, std::vector<float> &cur_y, std::vector<unsigned char> &status,
             int start_level = -1);

/// Current frame becomes the previous one (pointer swaps, no copies).
void gpu_commit(std::size_t cam_id);

/// Forget the previous frame for this camera (tracking reset).
void gpu_reset(std::size_t cam_id);

} // namespace ov_core
