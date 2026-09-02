#ifndef OV_CORE_NVJPG_DECODE_H
#define OV_CORE_NVJPG_DECODE_H
// ---------------------------------------------------------------------------------------
// OV_NVJPG=1 -- NVJPG hardware JPEG decode, zero-copy into CUDA.  Default OFF.
//
// Producer side (the decode-pool threads, aux cores):  ov_nvjpg_stage()
//   NVJPG decodeToFd -> dmabuf fd -> NvBufSurfaceFromFd -> NvBufSurfaceMapEglImage
//   -> cudaGraphicsEGLRegisterImage (CACHED PER FD) -> device pointer to the Y plane.
//   Returns a PLACEHOLDER host pointer; the caller wraps it in a header-only cv::Mat so the
//   existing std::shared_future<std::map<int,cv::Mat>> contract is untouched.
//
// Consumer side (gpu_prepare, filter cores):  ov_nvjpg_take() / ov_nvjpg_mark_consumed()
//   A HIT means "the pixels for (cam,ts) are already on the device" -- gpu_prepare then
//   SKIPS its cudaMemcpy2DAsync H2D entirely and reads the surface directly.
//
// LIFETIME INVARIANT (the correctness crux): a ring slot may not be re-issued to the
// decoder until either (a) the CUDA event recorded after the last kernel that reads its
// surface has fired, or (b) the consumer watermark has passed its timestamp, which proves
// the frame was dropped and will never be consumed.  Violating it silently hands frame N
// the pixels of frame N+R.  ov_nvjpg_stage BLOCKS rather than violate it, and counts it.
// ---------------------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <string>

namespace ov_core {

/// R18_RING_FIX -- tell the ring how deep the reader-ahead can get BEFORE the first
/// ov_nvjpg_enabled() call.  The ring is then sized from that number rather than from a
/// constant that happened to equal it.  See ov_nvjpg_stage() for the derivation.
void ov_nvjpg_set_lookahead(int max_frames_ahead);

/// OV_NVJPG=1 and the hardware path bound.  Cheap after the first call.
bool ov_nvjpg_enabled();

/// Explicit startup refusal (called once from the runner with the resolved config).
void ov_nvjpg_refuse_if(bool cond, const char *why);

/// Startup self-test (OV_NVJPG_SELFTEST=1): P1..P4 of the plan.  Returns false on failure.
bool ov_nvjpg_selftest(const unsigned char *jpg, size_t n);

/// PRODUCER.  Decodes into a ring slot, returns the placeholder host pointer plus the
/// geometry to build the header-only cv::Mat with.
/// RETURNS nullptr IF NO RING SLOT COULD BE ACQUIRED WITHIN THE ACQUIRE BUDGET, and the
/// caller MUST then decode on the CPU.  That bounded-give-up is what makes the decode-pool
/// worker unblockable and therefore what breaks the three-way deadlock; see the long
/// comment on the acquire loop.  nullptr is also returned on decode / EGL failure, which
/// before R18 silently dropped the camera from the frame-set.
const unsigned char *ov_nvjpg_stage(int cam_id, double ts, const unsigned char *jpg, size_t n,
                                    int *w, int *h, size_t *step);

/// True iff p is the shared placeholder page, i.e. "this Mat carries no pixels".
bool ov_nvjpg_is_placeholder(const unsigned char *p);

/// CONSUMER.  Claims the staged surface for (cam_id, ts).  slot is an opaque handle for
/// ov_nvjpg_mark_consumed / ov_nvjpg_readback.
bool ov_nvjpg_take(int cam_id, double ts, const void **d_y, size_t *pitch, int *w, int *h, int *slot);

/// CONSUMER.  Record the retirement event on the stream that read the surface.
void ov_nvjpg_mark_consumed(int slot, void *cuda_stream);

/// The consumer has finished with (or discarded) every frame-set at or before ts.
void ov_nvjpg_watermark(double ts);

/// OV_NVJPG_VERIFY=1 support: pull the Y plane back to host (dst is w*h, tight).
bool ov_nvjpg_readback(int slot, unsigned char *dst);

/// Same, addressed by (cam,ts) while the slot is still STAGED (verify gate only).
bool ov_nvjpg_readback_ts(int cam_id, double ts, unsigned char *dst, int *w, int *h);
bool ov_nvjpg_readback_cpu_ts(int cam_id, double ts, unsigned char *dst, int *w, int *h);

/// "[nvjpg]: on ring=3 fds_cached=12 ring_blocks=0 ..." for the run log.
std::string ov_nvjpg_stats();

} // namespace ov_core
#endif
