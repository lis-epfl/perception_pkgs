#include <chrono>
// GPU-resident tracking: CLAHE -> pyramid -> FAST(+NMS+per-cell quota) -> KLT.
// Images are uploaded once and never downloaded; only keypoints cross the bus.
#include "gpu_track.h"
#include "nvjpg_decode.h"
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Array.h>
#include <vpi/Stream.h>
#include <vpi/CUDAInterop.h>
#include <vpi/Status.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace ov_core {
namespace {

constexpr int TX = 8, TY = 8;
// OpenCV maxLevel=5 -> 6 pyramid levels (0..5). VPI takes a COUNT, so 6.
// Force a sync at the end of each stage so per-stage timing is HONEST. Without it gpu_prepare
// is fully async and its measured cost is submission only, while the real work lands on
// whichever later call happens to sync -- which made prepare look 2.4x faster than the CPU and
// KLT look 3x slower than it is.
inline bool custom_klt() {
  static const bool e = [] {
    const char *v = std::getenv("OV_GPU_CUSTOM_KLT");
    return !v || *v != '0';               // DEFAULT ON once validated
  }();
  return e;
}
// OV_DET_NOVPI=1: gpu_detect only needs d_px/d_py for the occupancy paint. ensure_pts also
// destroys+recreates three VPI arrays that are DEAD STATE under OV_GPU_CUSTOM_KLT (gpu_klt's
// custom branch uses d_p0/d_p1/d_lkst and jumps past the whole VPI block), and
// vpiArrayCreate/Destroy are device-synchronizing allocator calls that stall the other three
// cameras' streams too. Bit-exact to skip them: k_paint only reads d_px/d_py[0,n_occ).
inline bool det_novpi() {
  static const bool e = [] { const char *v = std::getenv("OV_DET_NOVPI"); return v && *v == '1'; }();
  return e;
}
// OV_DETECT_SPLIT=1: run the detect kernels on their OWN cuda stream so a submit-ahead
// detection cannot be waited on by the next gpu_prepare (whose pageable cudaMemcpy2DAsync
// syncs the stream before the copy is initiated). Detect reads d_prev/d_mprev, prepare writes
// d_raw/d_cur/d_mcur/d_hist/d_lut -- disjoint, so two streams are safe.
inline bool det_split() {
  static const bool e = [] {
    // OV_DETERMINISTIC / OV_DET_STREAM1: keep the submit-ahead, drop the second stream.
    // Two streams share d_prev / d_cur / d_mprev / d_mcur with NOTHING ordering them, and
    // gpu_commit swaps those pointers on the host, outside both streams. One stream makes
    // every access totally ordered; it changes no arithmetic and no scheduling decision that
    // the estimator can observe -- only which queue the kernels sit in.
    // OV_DETERM_SPLIT=1 (bench only): keep the second stream ON under OV_DETERMINISTIC, so the
    // oracle can be used to BISECT the stream race instead of merely avoiding it. With the
    // round-17 cross-stream ordering in place the two must produce BIT-IDENTICAL trajectories;
    // that equality is the proof that the ordering is complete.
    const char *d = std::getenv("OV_DETERMINISTIC");
    const char *ks = std::getenv("OV_DETERM_SPLIT");
    if (d && *d == '1' && !(ks && *ks == '1')) return false;
    const char *s = std::getenv("OV_DET_STREAM1");
    if (s && *s == '1') return false;
    const char *v = std::getenv("OV_DETECT_SPLIT"); return v && *v == '1';
  }();
  return e;
}
inline bool sync_each() {
  static const bool e = [] {
    const char *v = std::getenv("OV_GPU_SYNC_EACH");
    return v && *v == '1';
  }();
  return e;
}
inline bool wrap_stream() {
  static const bool w = [] {
    const char *e = std::getenv("OV_GPU_WRAP_STREAM");
    return e && *e == '1';
  }();
  return w;
}
inline int pyr_levels() {
  static const int L = [] {
    const char *e = std::getenv("OV_GPU_PYR_LEVELS");
    return e ? atoi(e) : 6;
  }();
  return L;
}
constexpr int MAXCELL = 256;

// ======================= OV_DETERMINISTIC (default OFF) =======================
// Bit-reproducibility gate. The recorded root cause of this pipeline's run-to-run scatter is
// that GPU candidate emission uses atomicAdd, so arrival order -- and, when a buffer fills,
// the retained SET -- is decided by warp scheduling. Nothing under this gate changes what is
// COMPUTED for a given pixel; it changes only which of several equally-scored candidates wins
// a tie and in what order they are handed to the caller.
inline bool det_sync() {
  // ROUND 17: **DEFAULT ON**.  Cross-stream ordering between the detection stream (cs_det) and
  // the image stream (cs).  Until round 17 this was implied only by OV_DETERMINISTIC, i.e. the
  // SHIPPED configuration ran OV_DETECT_SPLIT=1 with NOTHING ordering the two streams:
  //   * gpu_detect(submit_only) queues k_maskcopy/k_cand/k_score2/k_nms/k_select on cs_det;
  //     they READ d_prev and d_mprev.
  //   * gpu_commit swaps d_prev<->d_cur and d_mprev<->d_mcur as a pure HOST pointer exchange
  //     that participates in neither stream.
  //   * a later gpu_prepare then WRITES that same physical buffer on cs (k_apply/k_copy into
  //     d_cur, the mask k_copy into d_mcur, k_pyrdown into cp_cur levels).
  // => write-after-read across two unsynchronised streams on the same allocation.  The window
  // is NOT one frame: perform_detection_monocular_gpu returns EARLY (num_featsneeded, or the
  // OV_DETECT_EVERY / burst gate) BEFORE it reaches the harvest, so a submitted detection can
  // stay un-drained for up to OV_DETECT_EVERY frames while prepare keeps overwriting.
  // OV_DET_SYNC=0 restores the old (racy) behaviour, for pricing it only.
  static const bool s = [] {
    const char *e = std::getenv("OV_DET_SYNC");
    if (e && *e == '0') return false;
    return true;
  }();
  return s;
}
// OV_KCAND_FULL=1: size k_cand's candidate buffer at w*h, which makes overflow impossible by
// construction (a pixel can be a candidate at most once).  The 131072 cap SILENTLY DROPS
// everything past it in warp-arrival order; measured to fire on 12-39% of detections with
// kcand_max up to 242,749.  This CHANGES RESULTS on the shipped path (it stops dropping
// candidates), so it is a separate gate with its own accuracy gate.  Implied by
// OV_DETERMINISTIC; OV_KCAND_FULL=0 forces the old cap even under the oracle, which is how the
// accuracy of the fix is measured.
inline bool kcand_full() {
  static const bool k = [] {
    const char *e = std::getenv("OV_KCAND_FULL");
    if (e) return *e == '1';
    const char *d = std::getenv("OV_DETERMINISTIC");
    return d && *d == '1';
  }();
  return k;
}
// R18_KSEL -- SPLIT THE TWO det_mode()-GATED k_select BEHAVIOURS APART.  They are not the
// same kind of change and must not keep sharing one switch:
//
//  OV_KSEL_ORDER  (canonicalisation, FREE, no candidate changes identity)
//     The harvest stable_sorts by score alone, and a stable sort preserves input order among
//     ties -- so k_select's atomicAdd warp-arrival order survives into the caller's greedy
//     min_px_dist dedup and picks a DIFFERENT feature set run to run.  Adding (y, x) to the
//     comparator makes the ordering a function of the candidate SET alone.  The scores being
//     compared are bit-identical either way.  This is the last known result-affecting
//     nondeterminism source on the paced vehicle path.
//
//  OV_KSEL_CAP_FULL  (BEHAVIOUR CHANGE, needs an accuracy gate -- default OFF)
//     det_mode() also widens the PER-CELL candidate capacity from cap+8 to cap+64.  That is
//     not a tie-break: it changes WHICH candidates survive, and it silently drops the excess
//     in warp-arrival order when it binds.  Round 17 measured it binding 1.7-19.0 times per
//     run on the SHIP path and exactly 0 times under the oracle -- i.e. the oracle has been
//     scoring a wider cell than the vehicle ever runs, a second way (after OV_PREFETCH) in
//     which the oracle is not the vehicle's data path.  Gated separately, default OFF,
//     because unlike the tie-break it can move the answer.
inline bool ksel_order() {
  static const bool d = [] {
    const char *e = std::getenv("OV_KSEL_ORDER");
    return e && *e == '1';
  }();
  return d;
}
inline bool ksel_cap_full() {
  static const bool d = [] {
    const char *e = std::getenv("OV_KSEL_CAP_FULL");
    return e && *e == '1';
  }();
  return d;
}
inline bool det_mode() {
  static const bool d = [] { const char *e = std::getenv("OV_DETERMINISTIC"); return e && *e == '1'; }();
  return d;
}
// Census that makes the two capacity claims EVIDENCE rather than argument. Both counters are
// read from the run's own stderr; if either is non-zero the corresponding buffer bound, and a
// bound buffer is a silent, scheduler-dependent drop.
std::atomic<int>  g_cand_cap_seen{0}; // the cap actually allocated (read back, never argued)
std::atomic<long> g_kcand_over{0};    // detections where k_cand's candidate cap was hit
std::atomic<long> g_kcand_max{0};     // largest candidate count seen
std::atomic<long> g_ksel_over{0};     // cells where k_select's per-cell emission cap was hit
std::atomic<long> g_ksel_calls{0};
// ROUND 17 ordering census: how many cross-stream waits were actually issued.  Both are pure
// evidence -- a run with OV_DETECT_SPLIT=1 and detwait=0 would mean the guard never armed.
std::atomic<long> g_detwait{0};      // cs waited for cs_det (WAR: prepare after detect)
std::atomic<long> g_imgwait{0};      // cs_det waited for cs  (RAW: detect after prepare)
std::atomic<long> g_detdrain{0};     // cs_det drained on reset/teardown
struct DetDetermDump {
  ~DetDetermDump() {
    if (!g_ksel_calls.load()) return;
    std::fprintf(stderr,
                 "[detdeterm]: det_mode=%d ksel_order=%d ksel_cap_full=%d harvests=%ld "
                 "kcand_over=%ld kcand_max=%ld ksel_cell_over=%ld\n",
                 (int)det_mode(), (int)(det_mode() || ksel_order()),
                 (int)(det_mode() || ksel_cap_full()), g_ksel_calls.load(), g_kcand_over.load(),
                 g_kcand_max.load(), g_ksel_over.load());
    std::fprintf(stderr, "[detorder]: split=%d sync=%d kcand_full=%d detwait=%ld imgwait=%ld drain=%ld cand_cap=%d\n",
                 (int)det_split(), (int)det_sync(), (int)kcand_full(), g_detwait.load(), g_imgwait.load(),
                 g_detdrain.load(), g_cand_cap_seen.load());
  }
} g_detderm_dump;

// ===================== CLAHE (matches cv::createCLAHE(clip, 8x8)) =====================
__global__ void k_hist(const unsigned char *img, size_t pitch, int w, int h, int *hists) {
  __shared__ int sh[256];
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) sh[i] = 0;
  __syncthreads();
  const int tx = blockIdx.x, ty = blockIdx.y;
  const int x0 = tx * w / TX, x1 = (tx + 1) * w / TX, y0 = ty * h / TY, y1 = (ty + 1) * h / TY;
  for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y)
    for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x)
      atomicAdd(&sh[img[(size_t)y * pitch + x]], 1);
  __syncthreads();
  int *o = hists + (ty * TX + tx) * 256;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) o[i] = sh[i];
}
__global__ void k_lut(const int *hists, int w, int h, float clip, unsigned char *luts) {
  __shared__ int sh[256];
  const int tile = blockIdx.x;
  const int *hin = hists + tile * 256;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = hin[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    const int tw = w / TX, th = h / TY;
    int limit = (int)(clip * (float)(tw * th) / 256.0f); if (limit < 1) limit = 1;
    long ex = 0;
    for (int i = 0; i < 256; i++) if (sh[i] > limit) { ex += sh[i] - limit; sh[i] = limit; }
    const int inc = (int)(ex / 256);
    int residual = (int)(ex - (long)inc * 256);
    for (int i = 0; i < 256; i++) sh[i] += inc;
    if (residual != 0) {                       // OpenCV spreads the residual with a stride
      int step = 256 / residual; if (step < 1) step = 1;
      for (int i = 0; i < 256 && residual > 0; i += step, residual--) sh[i]++;
    }
    long tot = 0; for (int i = 0; i < 256; i++) tot += sh[i];
    const float sc = tot > 0 ? 255.0f / (float)tot : 0.0f;
    long cdf = 0; unsigned char *lo = luts + tile * 256;
    for (int i = 0; i < 256; i++) { cdf += sh[i]; float v = cdf * sc;
      lo[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v + 0.5f)); }
  }
}
__global__ void k_apply(const unsigned char *src, size_t sp, unsigned char *dst, size_t dp,
                        int w, int h, const unsigned char *luts) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const float tw = (float)w / TX, th = (float)h / TY;
  const float fx = (float)x / tw - 0.5f, fy = (float)y / th - 0.5f;   // OpenCV grid origin
  const int ix = (int)floorf(fx), iy = (int)floorf(fy);
  const float ax = fx - ix, ay = fy - iy;
  const int x0 = min(max(ix, 0), TX - 1), x1 = min(max(ix + 1, 0), TX - 1);
  const int y0 = min(max(iy, 0), TY - 1), y1 = min(max(iy + 1, 0), TY - 1);
  const unsigned char v = src[(size_t)y * sp + x];
  const float v00 = luts[(y0 * TX + x0) * 256 + v], v01 = luts[(y0 * TX + x1) * 256 + v];
  const float v10 = luts[(y1 * TX + x0) * 256 + v], v11 = luts[(y1 * TX + x1) * 256 + v];
  const float t = v00 + ax * (v01 - v00), b = v10 + ax * (v11 - v10), o = t + ay * (b - t);
  dst[(size_t)y * dp + x] = (unsigned char)(o < 0 ? 0 : (o > 255 ? 255 : o + 0.5f));
}
__global__ void k_sum(const unsigned char *p, size_t pitch, int w, int h, unsigned long long *out) {
  __shared__ unsigned int acc;
  if (threadIdx.x == 0 && threadIdx.y == 0) acc = 0;
  __syncthreads();
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) atomicAdd(&acc, (unsigned int)p[(size_t)y * pitch + x]);
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(out, (unsigned long long)acc);
}
__global__ void k_copy(const unsigned char *src, size_t sp, unsigned char *dst, size_t dp, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) dst[(size_t)y * dp + x] = src[(size_t)y * sp + x];
}

// ===================== detection mask =====================
// Base mask normalised to 0/255, then exclusion squares painted around existing features --
// the device-side equivalent of TrackKLT's cv::rectangle onto mask0_updated.
__global__ void k_maskcopy(const unsigned char *src, size_t sp, unsigned char *dst, size_t dp,
                           int w, int h, int have) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) dst[(size_t)y * dp + x] = (have && src[(size_t)y * sp + x] > 127) ? 255 : 0;
}
__global__ void k_paint(unsigned char *m, size_t mp, int w, int h,
                        const float *px, const float *py, int n, int r) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const int cx = (int)px[i], cy = (int)py[i];
  if (cx - r < 0 || cx + r >= w || cy - r < 0 || cy + r >= h) return;   // TrackKLT skips edge cases
  for (int dy = -r + (int)threadIdx.y; dy <= r; dy += blockDim.y)
    for (int dx = -r + (int)threadIdx.x; dx <= r; dx += blockDim.x)
      m[(size_t)(cy + dy) * mp + (cx + dx)] = 255;
}

// ===================== FAST-9 + NMS + per-cell quota =====================
__constant__ int c_ox[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3,-3,-3,-2,-1};
__constant__ int c_oy[16] = {-3,-3,-2,-1, 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3};
#define SB_X 32
#define SB_Y 16
__global__ void k_score(const unsigned char *img, size_t pitch, int w, int h, int thr,
                        const unsigned char *mask, size_t mpitch, unsigned char *score, size_t spitch) {
  __shared__ unsigned char tile[SB_Y + 6][SB_X + 6];
  const int bx = blockIdx.x * SB_X, by = blockIdx.y * SB_Y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x, nthr = blockDim.x * blockDim.y;
  for (int i = tid; i < (SB_X + 6) * (SB_Y + 6); i += nthr) {
    int ly = i / (SB_X + 6), lx = i - ly * (SB_X + 6);
    int gx = min(max(bx + lx - 3, 0), w - 1), gy = min(max(by + ly - 3, 0), h - 1);
    tile[ly][lx] = img[(size_t)gy * pitch + gx];
  }
  __syncthreads();
  const int x = bx + threadIdx.x, y = by + threadIdx.y;
  if (x >= w || y >= h) return;
  if (x < 3 || y < 3 || x >= w - 3 || y >= h - 3) { score[(size_t)y * spitch + x] = 0; return; }
  if (mask[(size_t)y * mpitch + x] > 127) { score[(size_t)y * spitch + x] = 0; return; }
  const int lx = threadIdx.x + 3, ly = threadIdx.y + 3;
  const int p = tile[ly][lx];
  // >=4 of the 8 EVEN circle points -- a valid necessary condition for a 9-arc.
  // (">=3 of 4 cardinals" is the FAST-12 test and wrongly drops FAST-9 corners.)
  const int hi = p + thr, lo = p - thr;
  int nb = 0, nd = 0;
#pragma unroll
  for (int i = 0; i < 16; i += 2) { int v = tile[ly + c_oy[i]][lx + c_ox[i]]; nb += (v > hi); nd += (v < lo); }
  if (nb < 4 && nd < 4) { score[(size_t)y * spitch + x] = 0; return; }
  int c[16];
#pragma unroll
  for (int i = 0; i < 16; i++) c[i] = tile[ly + c_oy[i]][lx + c_ox[i]];
  int best = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    int mb = 255, md = 255;
#pragma unroll
    for (int j = 0; j < 9; j++) { int v = c[(i + j) & 15]; int a = v - p, d = p - v;
      if (a < mb) mb = a; if (d < md) md = d; }
    int s = mb > md ? mb : md;
    if (s > best) best = s;
  }
  score[(size_t)y * spitch + x] = (best > thr) ? (unsigned char)min(best, 255) : 0;
}
__global__ void k_nms(const unsigned char *score, size_t sp, int w, int h, unsigned char *out, size_t op) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int s = score[(size_t)y * sp + x];
  if (s == 0 || x < 4 || y < 4 || x >= w - 4 || y >= h - 4) { out[(size_t)y * op + x] = 0; return; }
  unsigned char keep = s;
#pragma unroll
  for (int dy = -1; dy <= 1; dy++)
#pragma unroll
    for (int dx = -1; dx <= 1; dx++)
      if ((dx || dy) && score[(size_t)(y + dy) * sp + (x + dx)] > s) keep = 0;
  out[(size_t)y * op + x] = keep;
}
// One block per grid cell. Histogram the NMS scores, walk down to the cutoff that admits
// want[cell], then emit. Cost and output stay bounded regardless of scene texture.
__global__ void k_select(const unsigned char *nms, size_t pitch, int w, int h,
                         int gxn, int gyn, const int *want, int cap,
                         float *ox, float *oy, unsigned char *osc, int *counts) {
  __shared__ int hist[256];
  __shared__ int cutoff;
  const int cx = blockIdx.x, cy = blockIdx.y, cell = cy * gxn + cx;
  if (threadIdx.x == 0 && threadIdx.y == 0) counts[cell] = 0;
  __syncthreads();
  const int quota = want[cell];
  if (quota <= 0) return;
  const int x0 = cx * w / gxn, x1 = (cx + 1) * w / gxn;
  const int y0 = cy * h / gyn, y1 = (cy + 1) * h / gyn;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 256; i += blockDim.x * blockDim.y) hist[i] = 0;
  __syncthreads();
  for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y)
    for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x) {
      int s = nms[(size_t)y * pitch + x];
      if (s) atomicAdd(&hist[s], 1);
    }
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int acc = 0, cut = 1;
    for (int s = 255; s >= 1; s--) { acc += hist[s]; cut = s; if (acc >= quota) break; }
    cutoff = cut;
  }
  __syncthreads();
  const int cut = cutoff;
  for (int y = y0 + threadIdx.y; y < y1; y += blockDim.y)
    for (int x = x0 + threadIdx.x; x < x1; x += blockDim.x) {
      int s = nms[(size_t)y * pitch + x];
      if (s >= cut) {
        int k = atomicAdd(&counts[cell], 1);
        if (k < cap) {
          ox[(size_t)cell * cap + k] = (float)x;
          oy[(size_t)cell * cap + k] = (float)y;
          osc[(size_t)cell * cap + k] = (unsigned char)s;
        }
      }
    }
}


// ===================== custom pyramid + pyramidal LK (no VPI) =====================
// VPI's KLT forces its point arrays to be recreated every call (~7 ms/frame-set, proven
// mandatory for correctness). Plain CUDA buffers have no such constraint, so a hand-written
// pyramidal LK removes that cost entirely -- and the VPI pyramid/stream interop with it.
constexpr int LK_MAXLVL = 8;
struct CuPyr {
  unsigned char *d[LK_MAXLVL] = {nullptr};
  size_t p[LK_MAXLVL] = {0};
  int w[LK_MAXLVL] = {0}, h[LK_MAXLVL] = {0};
};

// 5-tap [1 4 6 4 1] separable Gaussian + decimate, reflect-101 borders: matches cv::pyrDown,
// which is what cv::buildOpticalFlowPyramid uses.
__global__ void k_pyrdown(const unsigned char *src, size_t sp, int sw, int sh,
                          unsigned char *dst, size_t dp, int dw, int dh) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dw || y >= dh) return;
  const int k5[5] = {1, 4, 6, 4, 1};
  const int cx = 2 * x, cy = 2 * y;
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 5; j++) {
    int yy = cy + j - 2;
    yy = yy < 0 ? -yy : (yy >= sh ? 2 * sh - 2 - yy : yy);
    float row = 0.f;
#pragma unroll
    for (int i = 0; i < 5; i++) {
      int xx = cx + i - 2;
      xx = xx < 0 ? -xx : (xx >= sw ? 2 * sw - 2 - xx : xx);
      row += k5[i] * (float)src[(size_t)yy * sp + xx];
    }
    acc += k5[j] * row;
  }
  dst[(size_t)y * dp + x] = (unsigned char)(acc * (1.f / 256.f) + 0.5f);
}

__device__ inline float lk_sample(const unsigned char *img, size_t pitch, int w, int h, float x, float y) {
  x = fminf(fmaxf(x, 0.f), (float)w - 1.001f);
  y = fminf(fmaxf(y, 0.f), (float)h - 1.001f);
  const int ix = (int)x, iy = (int)y;
  const float fx = x - ix, fy = y - iy;
  const float a = __ldg(&img[(size_t)iy * pitch + ix]), b = __ldg(&img[(size_t)iy * pitch + ix + 1]);
  const float c = __ldg(&img[(size_t)(iy + 1) * pitch + ix]), d = __ldg(&img[(size_t)(iy + 1) * pitch + ix + 1]);
  return (a + fx * (b - a)) + fy * ((c + fx * (d - c)) - (a + fx * (b - a)));
}

// One block (16x16) per feature. 15x15 window, iterate coarse->fine.
// OV_LK128=1 switches to 16x8 blocks (two window rows per thread): half the warps per block
// doubles resident blocks per SM, which hides the scattered-u8-load latency better.
#define LK_HW 7

__global__ void k_lk128(CuPyr prev, CuPyr cur, int nlev, int lvl0,
                        const float2 *p0, float2 *p1, unsigned char *status,
                        int n, int iters, float eps) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const int tx = threadIdx.x, ty = threadIdx.y;      // 16 x 8
  const int tid = ty * 16 + tx;
  const bool actx = tx < 2 * LK_HW + 1;
  const bool act0 = actx;                            // rows 0..7
  const bool act1 = actx && (ty + 8) < 2 * LK_HW + 1; // rows 8..14
  const float ox = (float)(tx - LK_HW);
  const float oy0 = (float)(ty - LK_HW), oy1 = (float)(ty + 8 - LK_HW);
  __shared__ float3 redG[4];
  __shared__ float2 redB[4];
  __shared__ float sGinv[4];
  __shared__ float2 sNext;
  __shared__ int sOK;

  if (tid == 0) { sOK = 1; sNext = make_float2(p0[i].x / (float)(1 << lvl0), p0[i].y / (float)(1 << lvl0)); }
  __syncthreads();

  for (int L = lvl0; L >= 0; L--) {
    const float scale = 1.f / (float)(1 << L);
    const float2 pP = make_float2(p0[i].x * scale, p0[i].y * scale);
    const int W = prev.w[L], H = prev.h[L];
    float Ipix0 = 0.f, Ix0 = 0.f, Iy0 = 0.f, Ipix1 = 0.f, Ix1 = 0.f, Iy1 = 0.f;
    if (act0 && sOK) {
      const float sx = pP.x + ox, sy = pP.y + oy0;
      Ipix0 = lk_sample(prev.d[L], prev.p[L], W, H, sx, sy);
      Ix0 = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx + 1.f, sy) -
                    lk_sample(prev.d[L], prev.p[L], W, H, sx - 1.f, sy));
      Iy0 = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx, sy + 1.f) -
                    lk_sample(prev.d[L], prev.p[L], W, H, sx, sy - 1.f));
    }
    if (act1 && sOK) {
      const float sx = pP.x + ox, sy = pP.y + oy1;
      Ipix1 = lk_sample(prev.d[L], prev.p[L], W, H, sx, sy);
      Ix1 = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx + 1.f, sy) -
                    lk_sample(prev.d[L], prev.p[L], W, H, sx - 1.f, sy));
      Iy1 = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx, sy + 1.f) -
                    lk_sample(prev.d[L], prev.p[L], W, H, sx, sy - 1.f));
    }
    float gxx = Ix0 * Ix0 + Ix1 * Ix1, gxy = Ix0 * Iy0 + Ix1 * Iy1, gyy = Iy0 * Iy0 + Iy1 * Iy1;
    for (int o = 16; o > 0; o >>= 1) {
      gxx += __shfl_down_sync(0xffffffffu, gxx, o);
      gxy += __shfl_down_sync(0xffffffffu, gxy, o);
      gyy += __shfl_down_sync(0xffffffffu, gyy, o);
    }
    if ((tid & 31) == 0) redG[tid >> 5] = make_float3(gxx, gxy, gyy);
    __syncthreads();
    if (tid < 4) {
      float3 v = redG[tid];
      for (int o = 2; o > 0; o >>= 1) {
        v.x += __shfl_down_sync(0xfu, v.x, o);
        v.y += __shfl_down_sync(0xfu, v.y, o);
        v.z += __shfl_down_sync(0xfu, v.z, o);
      }
      if (tid == 0) redG[0] = v;
    }
    __syncthreads();
    if (tid == 0) {
      const float det = redG[0].x * redG[0].z - redG[0].y * redG[0].y;
      if (det < 1e-4f || !sOK) {
        sOK = 0;
      } else {
        const float inv = 1.f / det;
        sGinv[0] = redG[0].z * inv;  sGinv[1] = -redG[0].y * inv;
        sGinv[2] = -redG[0].y * inv; sGinv[3] = redG[0].x * inv;
      }
    }
    __syncthreads();
    if (!sOK) continue;

    for (int it = 0; it < iters; it++) {
      float bx = 0.f, by = 0.f;
      if (act0) {
        const float J = lk_sample(cur.d[L], cur.p[L], W, H, sNext.x + ox, sNext.y + oy0);
        const float r = Ipix0 - J;
        bx += r * Ix0; by += r * Iy0;
      }
      if (act1) {
        const float J = lk_sample(cur.d[L], cur.p[L], W, H, sNext.x + ox, sNext.y + oy1);
        const float r = Ipix1 - J;
        bx += r * Ix1; by += r * Iy1;
      }
      for (int o = 16; o > 0; o >>= 1) {
        bx += __shfl_down_sync(0xffffffffu, bx, o);
        by += __shfl_down_sync(0xffffffffu, by, o);
      }
      if ((tid & 31) == 0) redB[tid >> 5] = make_float2(bx, by);
      __syncthreads();
      if (tid < 4) {
        float2 v = redB[tid];
        for (int o = 2; o > 0; o >>= 1) {
          v.x += __shfl_down_sync(0xfu, v.x, o);
          v.y += __shfl_down_sync(0xfu, v.y, o);
        }
        if (tid == 0) redB[0] = v;
      }
      __syncthreads();
      __shared__ float2 sDelta;
      if (tid == 0) {
        sDelta.x = sGinv[0] * redB[0].x + sGinv[1] * redB[0].y;
        sDelta.y = sGinv[2] * redB[0].x + sGinv[3] * redB[0].y;
        sNext.x += sDelta.x; sNext.y += sDelta.y;
      }
      __syncthreads();
      if (sDelta.x * sDelta.x + sDelta.y * sDelta.y < eps * eps) break;
    }
    if (tid == 0 && L > 0) { sNext.x *= 2.f; sNext.y *= 2.f; }
    __syncthreads();
  }
  if (tid == 0) {
    const int W0 = cur.w[0], H0 = cur.h[0];
    const bool in = sNext.x >= 0.f && sNext.y >= 0.f && sNext.x < (float)W0 && sNext.y < (float)H0;
    p1[i] = sNext;
    status[i] = (sOK && in) ? 1 : 0;
  }
}
__global__ void k_lk(CuPyr prev, CuPyr cur, int nlev, int lvl0,
                     const float2 *p0, float2 *p1, unsigned char *status,
                     int n, int iters, float eps) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const bool act = (tx < 2 * LK_HW + 1) && (ty < 2 * LK_HW + 1);
  const float ox = (float)(tx - LK_HW), oy = (float)(ty - LK_HW);
  __shared__ float3 redG[256];
  __shared__ float2 redB[256];
  __shared__ float sGinv[4];
  __shared__ float2 sNext;
  __shared__ int sOK;
  const int tid = ty * 16 + tx;

  if (tid == 0) { sOK = 1; sNext = make_float2(p0[i].x / (float)(1 << lvl0), p0[i].y / (float)(1 << lvl0)); }
  __syncthreads();

  for (int L = lvl0; L >= 0; L--) {
    const float scale = 1.f / (float)(1 << L);
    const float2 pP = make_float2(p0[i].x * scale, p0[i].y * scale);
    const int W = prev.w[L], H = prev.h[L];
    // window samples + gradients on the PREVIOUS image (fixed per level)
    float Ipix = 0.f, Ix = 0.f, Iy = 0.f;
    if (act && sOK) {
      const float sx = pP.x + ox, sy = pP.y + oy;
      Ipix = lk_sample(prev.d[L], prev.p[L], W, H, sx, sy);
      Ix = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx + 1.f, sy) -
                   lk_sample(prev.d[L], prev.p[L], W, H, sx - 1.f, sy));
      Iy = 0.5f * (lk_sample(prev.d[L], prev.p[L], W, H, sx, sy + 1.f) -
                   lk_sample(prev.d[L], prev.p[L], W, H, sx, sy - 1.f));
    }
    // warp-shuffle reduction: 8 warp partials, then warp 0 folds them (no tree of syncthreads)
    float gxx = act ? Ix * Ix : 0.f, gxy = act ? Ix * Iy : 0.f, gyy = act ? Iy * Iy : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
      gxx += __shfl_down_sync(0xffffffffu, gxx, o);
      gxy += __shfl_down_sync(0xffffffffu, gxy, o);
      gyy += __shfl_down_sync(0xffffffffu, gyy, o);
    }
    if ((tid & 31) == 0) redG[tid >> 5] = make_float3(gxx, gxy, gyy);
    __syncthreads();
    if (tid < 8) {
      float3 v = redG[tid];
      for (int o = 4; o > 0; o >>= 1) {
        v.x += __shfl_down_sync(0xffu, v.x, o);
        v.y += __shfl_down_sync(0xffu, v.y, o);
        v.z += __shfl_down_sync(0xffu, v.z, o);
      }
      if (tid == 0) redG[0] = v;
    }
    __syncthreads();
    if (tid == 0) {
      const float det = redG[0].x * redG[0].z - redG[0].y * redG[0].y;
      if (det < 1e-4f || !sOK) {
        sOK = 0;
      } else {
        const float inv = 1.f / det;
        sGinv[0] = redG[0].z * inv;  sGinv[1] = -redG[0].y * inv;
        sGinv[2] = -redG[0].y * inv; sGinv[3] = redG[0].x * inv;
      }
    }
    __syncthreads();
    if (!sOK) continue;

    for (int it = 0; it < iters; it++) {
      float bx = 0.f, by = 0.f;
      if (act) {
        const float J = lk_sample(cur.d[L], cur.p[L], W, H, sNext.x + ox, sNext.y + oy);
        const float r = Ipix - J;
        bx = r * Ix; by = r * Iy;
      }
      for (int o = 16; o > 0; o >>= 1) {
        bx += __shfl_down_sync(0xffffffffu, bx, o);
        by += __shfl_down_sync(0xffffffffu, by, o);
      }
      if ((tid & 31) == 0) redB[tid >> 5] = make_float2(bx, by);
      __syncthreads();
      if (tid < 8) {
        float2 v = redB[tid];
        for (int o = 4; o > 0; o >>= 1) {
          v.x += __shfl_down_sync(0xffu, v.x, o);
          v.y += __shfl_down_sync(0xffu, v.y, o);
        }
        if (tid == 0) redB[0] = v;
      }
      __syncthreads();
      __shared__ float2 sDelta;
      if (tid == 0) {
        sDelta.x = sGinv[0] * redB[0].x + sGinv[1] * redB[0].y;
        sDelta.y = sGinv[2] * redB[0].x + sGinv[3] * redB[0].y;
        sNext.x += sDelta.x; sNext.y += sDelta.y;
      }
      __syncthreads();
      if (sDelta.x * sDelta.x + sDelta.y * sDelta.y < eps * eps) break;
    }
    if (tid == 0 && L > 0) { sNext.x *= 2.f; sNext.y *= 2.f; }
    __syncthreads();
  }
  if (tid == 0) {
    const int W0 = cur.w[0], H0 = cur.h[0];
    const bool in = sNext.x >= 0.f && sNext.y >= 0.f && sNext.x < (float)W0 && sNext.y < (float)H0;
    p1[i] = sNext;
    status[i] = (sOK && in) ? 1 : 0;
  }
}


// Two-phase FAST. k_score ran the 144-op scoring loop for every warp containing one candidate
// (measured 1.11 ms/image, 9x off the bandwidth floor). Phase A does only the cheap ring test
// and compacts candidates; phase B scores candidates alone (~3-5% of pixels).
__global__ void k_cand(const unsigned char *img, size_t pitch, int w, int h, int thr,
                       const unsigned char *mask, size_t mpitch,
                       unsigned char *score, size_t spitch, int2 *cand, int *cnt, int cap) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  score[(size_t)y * spitch + x] = 0;
  if (x < 3 || y < 3 || x >= w - 3 || y >= h - 3) return;
  if (mask[(size_t)y * mpitch + x] > 127) return;
  const int p = img[(size_t)y * pitch + x];
  const int hi = p + thr, lo = p - thr;
  int nb = 0, nd = 0;
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const int v = img[(size_t)(y + c_oy[i]) * pitch + (x + c_ox[i])];
    nb += (v > hi); nd += (v < lo);
  }
  if (nb < 4 && nd < 4) return;
  const int k = atomicAdd(cnt, 1);
  if (k < cap) cand[k] = make_int2(x, y);
}
__global__ void k_score2(const unsigned char *img, size_t pitch, int w, int h, int thr,
                         const int2 *cand, const int *cnt, int cap,
                         unsigned char *score, size_t spitch) {
  const int total = min(*cnt, cap);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    const int x = cand[idx].x, y = cand[idx].y;
    const int p = img[(size_t)y * pitch + x];
    int c[16];
#pragma unroll
    for (int i = 0; i < 16; i++) c[i] = img[(size_t)(y + c_oy[i]) * pitch + (x + c_ox[i])];
    int best = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      int mb = 255, md = 255;
#pragma unroll
      for (int j = 0; j < 9; j++) { const int v = c[(i + j) & 15]; const int a = v - p, d = p - v;
        if (a < mb) mb = a; if (d < md) md = d; }
      const int sc = mb > md ? mb : md;
      if (sc > best) best = sc;
    }
    if (best > thr) score[(size_t)y * spitch + x] = (unsigned char)min(best, 255);
  }
}

#define VOK(x) do { if ((x) != VPI_SUCCESS) return false; } while (0)
#define COK(x) do { if ((x) != cudaSuccess) return false; } while (0)

struct Cam {
  unsigned char *d_raw = nullptr;
  unsigned char *d_cur = nullptr, *d_prev = nullptr;      // equalised images
  unsigned char *d_mcur = nullptr, *d_mprev = nullptr;    // masks
  unsigned char *d_mdet = nullptr;                        // detection mask (base + exclusions)
  unsigned char *d_sc = nullptr, *d_nm = nullptr;
  size_t p_raw = 0, p_cur = 0, p_prev = 0, p_mcur = 0, p_mprev = 0, p_mdet = 0, p_sc = 0, p_nm = 0;
  int *d_hist = nullptr; unsigned char *d_lut = nullptr;
  float *d_ox = nullptr, *d_oy = nullptr; unsigned char *d_osc = nullptr;
  int *d_cnt = nullptr, *d_want = nullptr;
  int cand_cap = 0;
  cudaEvent_t ev_det = nullptr;   // detection-submitted marker on cs_det (WAR: cs waits on it)
  cudaEvent_t ev_img = nullptr;   // image-stream marker on cs        (RAW: cs_det waits on it)
  // Set on the thread that submits, cleared on the thread that issues the wait. Those are not
  // always the same thread (submit-ahead runs on the feed thread, the inline detect runs on the
  // camera's parallel_for_ worker), so they are atomics rather than plain bools.
  std::atomic<bool> det_ev_pending{false};
  std::atomic<bool> img_ev_pending{false};
  float *d_px = nullptr, *d_py = nullptr;
  int cap_cell = 0;
  int det_cap = 0, det_ncell = 0;
  uint64_t mask_ck = 0;
  VPIStream stream = nullptr;
  cudaStream_t cs = nullptr;
  cudaStream_t cs_det = nullptr;          // OV_DETECT_SPLIT: detect kernels off the KLT stream
  VPIImage img_cur = nullptr, img_prev = nullptr;
  VPIPyramid pyr_cur = nullptr, pyr_prev = nullptr;
  VPIArray a_prev = nullptr, a_cur = nullptr, a_st = nullptr;
  VPIPayload lk = nullptr;
  CuPyr cp_prev, cp_cur;                  // custom pyramids (level 0 aliases d_prev/d_cur)
  float2 *d_p0 = nullptr, *d_p1 = nullptr; unsigned char *d_lkst = nullptr;
  float2 *h_p0 = nullptr, *h_p1 = nullptr; unsigned char *h_lkst = nullptr;   // pinned staging
  int lk_cap = 0;
  int2 *d_cand = nullptr; int *d_ccnt = nullptr;
  int npts_cap = 0, d_pts_cap = 0, w = 0, h = 0;
  struct ArrSet { VPIArray prev, cur, st; };
  std::map<int, ArrSet> arr_cache;
  long n_prepare = 0, n_commit = 0, n_klt = 0;
  bool has_prev = false, has_mask = false, has_mask_prev = false, ok = false;
};

std::mutex g_mtx;
std::map<size_t, Cam> g_cam;

// One wrapper per image buffer, created ONCE. The previous version destroyed and recreated
// the wrapper on every commit and ignored the result -- per-frame VPI object churn, and a
// silent failure there leaves img_cur dangling so the pyramid is built from garbage (observed:
// cams 1-3 tracked 0 features while cam 0 was fine). Wrapper and buffer are now swapped as a
// pair, so they can never drift apart.
bool wrap_one(Cam &c, unsigned char *buf, size_t pitch, VPIImage *out) {
  VPIImageData d; memset(&d, 0, sizeof(d));
  d.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  d.buffer.pitch.format = VPI_IMAGE_FORMAT_U8;
  d.buffer.pitch.numPlanes = 1;
  d.buffer.pitch.planes[0].width = c.w;
  d.buffer.pitch.planes[0].height = c.h;
  d.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
  d.buffer.pitch.planes[0].pitchBytes = (int32_t)pitch;
  d.buffer.pitch.planes[0].data = buf;
  VOK(vpiImageCreateWrapper(&d, NULL, VPI_BACKEND_CUDA, out));
  return true;
}

// VPI's PyrLK keys off the array CAPACITY, not the size field: with capacity 16384 and only
// n~220 valid points it tracked 57-68%, while arrays sized n+8 tracked 99.3% on the very same
// images, points, pyramids and payload (component-wise in-process A/B). So the arrays must be
// (re)created to match n. Pyramids and payload can persist -- only the arrays matter.
// VPI needs capacity ~= the number of points submitted, so the arrays cannot simply be
// oversized. Padding to a fixed capacity with duplicate points was tried and VPI then reports
// EVERY point lost. Instead cache one array-triple per distinct n: n revisits the same few
// hundred values, so after a second or two nothing is allocated any more.
// VPI arrays must be created FRESH for each submit in this version -- reusing them silently
// degrades tracking, and it is reuse, not capacity slack, that does it:
//   reused @ fixed capacity 16384 ....... 57% tracked
//   reused @ cache keyed on size/16 ..... 41%
//   created fresh, capacity n+8 ......... 96-98%   <-- required
// This costs ~7 ms/frame-set (12 create/destroy) and is why gpu_klt is 10 ms rather than the
// 2 ms the standalone benchmark shows. Worth it: without it the ATE does not converge.
// Grow-only device buffers for the detection occupancy paint. No VPI objects touched.
bool ensure_occ(Cam &c, int n) {
  const int want = n + 8;
  if (want <= c.d_pts_cap) return true;
  if (c.d_px) { cudaFree(c.d_px); cudaFree(c.d_py); }
  COK(cudaMalloc(&c.d_px, (size_t)(want + 64) * sizeof(float)));
  COK(cudaMalloc(&c.d_py, (size_t)(want + 64) * sizeof(float)));
  c.d_pts_cap = want + 64;
  return true;
}

bool ensure_pts(Cam &c, int n) {
  const int want = n + 8;
  if (c.npts_cap == want && c.a_prev) {
    // same size as last call -- still recreate, see note above
  }
  if (c.a_prev) { vpiArrayDestroy(c.a_prev); vpiArrayDestroy(c.a_cur); vpiArrayDestroy(c.a_st); }
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c.a_prev));
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c.a_cur));
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_U8, 0, &c.a_st));
  if (want > c.d_pts_cap) {
    if (c.d_px) { cudaFree(c.d_px); cudaFree(c.d_py); }
    COK(cudaMalloc(&c.d_px, (size_t)(want + 64) * sizeof(float)));
    COK(cudaMalloc(&c.d_py, (size_t)(want + 64) * sizeof(float)));
    c.d_pts_cap = want + 64;
  }
  c.npts_cap = want;
  return true;
}

bool unused_ensure_pts(Cam &c, int n) {
  const int want = n + 8;
  if (c.npts_cap == want) return true;
  if (c.a_prev) { vpiArrayDestroy(c.a_prev); vpiArrayDestroy(c.a_cur); vpiArrayDestroy(c.a_st); }
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c.a_prev));
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c.a_cur));
  VOK(vpiArrayCreate(want, VPI_ARRAY_TYPE_U8, 0, &c.a_st));
  if (c.d_px) { cudaFree(c.d_px); cudaFree(c.d_py); }
  COK(cudaMalloc(&c.d_px, (size_t)want * sizeof(float)));
  COK(cudaMalloc(&c.d_py, (size_t)want * sizeof(float)));
  c.npts_cap = want;
  return true;
}

bool ensure(Cam &c, int w, int h) {
  if (c.ok && c.w == w && c.h == h) return true;
  c.w = w; c.h = h;
  COK(cudaStreamCreate(&c.cs));
  COK(cudaStreamCreate(&c.cs_det));
  // Wrapping our CUDA stream as the VPI stream removes the host sync between our kernels and
  // VPI's, but measured SLOWER end-to-end (GPU/CPU wall ratio 1.26 -> 1.40) -- VPI's own stream
  // appears better optimised than an externally supplied one. Off by default.
  if (wrap_stream()) {
    VOK(vpiStreamCreateWrapperCUDA(c.cs, 0, &c.stream));
  } else {
    VOK(vpiStreamCreate(0, &c.stream));
  }
  COK(cudaMallocPitch(&c.d_raw, &c.p_raw, w, h));
  COK(cudaMallocPitch(&c.d_cur, &c.p_cur, w, h));
  COK(cudaMallocPitch(&c.d_prev, &c.p_prev, w, h));
  COK(cudaMallocPitch(&c.d_mcur, &c.p_mcur, w, h));
  COK(cudaMallocPitch(&c.d_mprev, &c.p_mprev, w, h));
  COK(cudaMallocPitch(&c.d_mdet, &c.p_mdet, w, h));
  COK(cudaMallocPitch(&c.d_sc, &c.p_sc, w, h));
  COK(cudaMallocPitch(&c.d_nm, &c.p_nm, w, h));
  COK(cudaMalloc(&c.d_hist, TX * TY * 256 * sizeof(int)));
  COK(cudaMalloc(&c.d_lut, TX * TY * 256));
  COK(cudaMalloc(&c.d_want, MAXCELL * sizeof(int)));
  COK(cudaMalloc(&c.d_cnt, MAXCELL * sizeof(int)));
  if (!wrap_one(c, c.d_cur, c.p_cur, &c.img_cur)) return false;
  if (!wrap_one(c, c.d_prev, c.p_prev, &c.img_prev)) return false;
  VOK(vpiPyramidCreate(w, h, VPI_IMAGE_FORMAT_U8, pyr_levels(), 0.5f, 0, &c.pyr_cur));
  VOK(vpiPyramidCreate(w, h, VPI_IMAGE_FORMAT_U8, pyr_levels(), 0.5f, 0, &c.pyr_prev));
  VOK(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, w, h, VPI_IMAGE_FORMAT_U8, pyr_levels(), 0.5f, &c.lk));
  if (!ensure_pts(c, 1024)) return false;
  // custom pyramid levels (level 0 aliases the CLAHE output buffers)
  {
    int lw = w, lh = h;
    c.cp_cur.d[0] = c.d_cur;  c.cp_cur.p[0] = c.p_cur;  c.cp_cur.w[0] = w; c.cp_cur.h[0] = h;
    c.cp_prev.d[0] = c.d_prev; c.cp_prev.p[0] = c.p_prev; c.cp_prev.w[0] = w; c.cp_prev.h[0] = h;
    for (int L = 1; L < pyr_levels(); L++) {
      lw = (lw + 1) / 2; lh = (lh + 1) / 2;
      COK(cudaMallocPitch(&c.cp_cur.d[L], &c.cp_cur.p[L], lw, lh));
      COK(cudaMallocPitch(&c.cp_prev.d[L], &c.cp_prev.p[L], lw, lh));
      c.cp_cur.w[L] = lw; c.cp_cur.h[L] = lh;
      c.cp_prev.w[L] = lw; c.cp_prev.h[L] = lh;
    }
    const int cap = 8192;
    COK(cudaMalloc(&c.d_p0, cap * sizeof(float2)));
    COK(cudaMalloc(&c.d_p1, cap * sizeof(float2)));
    COK(cudaMalloc(&c.d_lkst, cap));
    COK(cudaHostAlloc(&c.h_p0, cap * sizeof(float2), cudaHostAllocDefault));
    COK(cudaHostAlloc(&c.h_p1, cap * sizeof(float2), cudaHostAllocDefault));
    COK(cudaHostAlloc(&c.h_lkst, cap, cudaHostAllocDefault));
    c.lk_cap = cap;
    // OV_DETERMINISTIC: k_cand appends candidates with atomicAdd and SILENTLY DROPS everything
    // past its cap, so once a frame overflows, WHICH pixels keep their FAST score -- and hence
    // the NMS map, the per-cell histogram, the cutoff and the candidate COUNT -- becomes a
    // function of warp scheduling. Step 0 measured exactly this: 2 of 16 back-to-back
    // detections on IDENTICAL inputs returned a DIFFERENT NUMBER of candidates, which k_select
    // alone cannot produce. One int2 per pixel makes overflow impossible by construction (a
    // pixel can be a candidate at most once). The kernel arithmetic is untouched.
    COK(cudaEventCreateWithFlags(&c.ev_det, cudaEventDisableTiming));
    COK(cudaEventCreateWithFlags(&c.ev_img, cudaEventDisableTiming));
    c.cand_cap = kcand_full() ? (w * h) : 131072;
    g_cand_cap_seen.store(c.cand_cap, std::memory_order_relaxed);
    COK(cudaMalloc(&c.d_cand, (size_t)c.cand_cap * sizeof(int2)));
    COK(cudaMalloc(&c.d_ccnt, sizeof(int)));
  }
  c.has_prev = false; c.ok = true;
  return true;
}

} // namespace

bool gpu_track_enabled() {
  static const bool en = [] {
    const char *e = std::getenv("OV_GPU_TRACK");
    // '1' = full GPU tracking; '2' = GPU KLT only (CPU CLAHE/pyramid/detect) for bisection
    if (!e || (*e != '1' && *e != '2')) return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
      std::fprintf(stderr, "[gpu_track]: OV_GPU_TRACK=1 but no CUDA device; using CPU\n");
      return false;
    }
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    std::fprintf(stderr, "[gpu_track]: GPU-resident tracking enabled BUILD_MARKER_7788\n");
    return true;
  }();
  return en;
}

void gpu_reset(std::size_t cam_id) {
  std::lock_guard<std::mutex> g(g_mtx);
  auto it = g_cam.find(cam_id);
  if (it == g_cam.end()) return;
  // A reset invalidates d_prev.  If a detection is still queued against it on cs_det, the
  // stream-ordered guards above are no longer enough -- the caller is about to declare the
  // buffer meaningless -- so drain it here.  Cheap: resets are rare (track loss / re-init).
  if (it->second.cs_det && it->second.det_ev_pending.exchange(false)) {
    cudaStreamSynchronize(it->second.cs_det);
    g_detdrain.fetch_add(1, std::memory_order_relaxed);
  }
  it->second.has_prev = false;
}

bool gpu_prepare(std::size_t cam_id, const unsigned char *img, int w, int h, std::size_t stride,
                 const unsigned char *mask, std::size_t mask_stride, bool do_clahe, float clahe_clip,
                 double ts) {
  Cam *cp;
  { std::lock_guard<std::mutex> g(g_mtx); cp = &g_cam[cam_id]; }
  Cam &c = *cp;
  if (!ensure(c, w, h)) return false;
  // WAR guard: everything below writes c.d_raw / c.d_cur / the masks on c.cs, and gpu_commit
  // may have swapped those pointers under a detection still in flight on c.cs_det. Order the
  // two streams instead of hoping. Stream-ordered wait: the host does not block.
  // Stream-ordered wait: the HOST does not block, so the submit-ahead that OV_DETECT_SPLIT
  // exists to buy is preserved.  cs is in-order, so ONE wait also covers every later prepare
  // until the next detection re-arms the event -- which is why the flag is cleared here.
  if (det_sync() && c.ev_det && c.det_ev_pending.exchange(false)) {
    cudaStreamWaitEvent(c.cs, c.ev_det, 0);
    g_detwait.fetch_add(1, std::memory_order_relaxed);
  }
  // ---- OV_NVJPG: is this frame's Y plane already on the device? -----------------------
  // A hit DELETES the pageable cudaMemcpy2DAsync H2D below (a synchronous CPU staging copy
  // of 1.024 MB/cam that runs on the filter cores) and points the CLAHE/copy kernels at the
  // NVJPG surface directly.  c.d_raw is simply unused on this path.
  const unsigned char *src = c.d_raw;
  std::size_t src_pitch = c.p_raw;
  int nv_slot = -1;
  if (ov_nvjpg_enabled() && ov_nvjpg_is_placeholder(img)) {
    const void *d_y = nullptr;
    std::size_t p = 0;
    int nw = 0, nh = 0;
    if (!ov_nvjpg_take((int)cam_id, ts, &d_y, &p, &nw, &nh, &nv_slot)) {
      // The placeholder carries NO pixels.  Falling through to the H2D would upload a blank
      // page and silently destroy the frame, so this is fatal by construction.
      std::fprintf(stderr,
                   "[nvjpg]: FATAL staging MISS for cam %zu ts %.6f -- placeholder Mat with no "
                   "device surface (ring lifetime violated).\n",
                   cam_id, ts);
      return false;
    }
    if (nw != w || nh != h) {
      // R18_RING_FIX: ov_nvjpg_take() has ALREADY moved this slot to state 2 with no
      // retirement event.  Returning here without marking it left the slot permanently
      // unreclaimable and parked the next producer for that camera forever -- one of the
      // three ways the ring deadlock could be entered.  Retire it on the stream we would
      // have used, so the slot comes back even on this fatal path.
      ov_nvjpg_mark_consumed(nv_slot, (void *)c.cs);
      std::fprintf(stderr, "[nvjpg]: FATAL geometry %dx%d != %dx%d\n", nw, nh, w, h);
      return false;
    }
    src = (const unsigned char *)d_y;
    src_pitch = p;
  } else {
    COK(cudaMemcpy2DAsync(c.d_raw, c.p_raw, img, stride, w, h, cudaMemcpyHostToDevice, c.cs));
  }
  dim3 b(32, 8), gr((w + 31) / 32, (h + 7) / 8);
  if (do_clahe) {
    COK(cudaMemsetAsync(c.d_hist, 0, TX * TY * 256 * sizeof(int), c.cs));
    k_hist<<<dim3(TX, TY), dim3(16, 16), 0, c.cs>>>(src, src_pitch, w, h, c.d_hist);
    k_lut<<<TX * TY, 256, 0, c.cs>>>(c.d_hist, w, h, clahe_clip, c.d_lut);
    k_apply<<<gr, b, 0, c.cs>>>(src, src_pitch, c.d_cur, c.p_cur, w, h, c.d_lut);
  } else {
    k_copy<<<gr, b, 0, c.cs>>>(src, src_pitch, c.d_cur, c.p_cur, w, h);
  }
  // Last kernel that reads the surface has been queued: the ring slot is retired when this
  // event fires.  MUST be recorded before any later stream work is queued.
  if (nv_slot >= 0) ov_nvjpg_mark_consumed(nv_slot, (void *)c.cs);
  c.has_mask = (mask != nullptr);
  if (mask) {
    // Masks are near-static (regenerated only when intrinsics drift). Skip the 1 MB upload
    // when a 256-byte sample checksum is unchanged; commit's d_mprev copy keeps both sides.
    uint64_t ck = 1469598103934665603ULL;
    for (int i = 0; i < 256; i++) {
      ck ^= mask[((size_t)(i * 131) % ((size_t)h * mask_stride))];
      ck *= 1099511628211ULL;
    }
    if (ck != c.mask_ck) {
      COK(cudaMemcpy2DAsync(c.d_mcur, c.p_mcur, mask, mask_stride, w, h, cudaMemcpyHostToDevice, c.cs));
      c.mask_ck = ck;
    } else {
      // content unchanged but d_mcur may hold the swapped prev buffer -> device-side copy
      dim3 mb(32, 8), mg((w + 31) / 32, (h + 7) / 8);
      k_copy<<<mg, mb, 0, c.cs>>>(c.d_mprev, c.p_mprev, c.d_mcur, c.p_mcur, w, h);
    }
  }
  // If the VPI stream wraps c.cs the ordering is automatic; otherwise sync c.cs so VPI sees a
  // fully written d_cur. Either way there is NO vpiStreamSync afterwards: the pyramid goes to
  // c.stream and every later VPI op for this camera uses that same stream, so VPI orders them.
  if (custom_klt()) {
    // levels 1..N-1 from the CLAHE output, all on c.cs -- no VPI, no cross-stream handoff
    c.cp_cur.d[0] = c.d_cur; c.cp_cur.p[0] = c.p_cur;
    for (int L = 1; L < pyr_levels(); L++) {
      dim3 pb(16, 16), pg((c.cp_cur.w[L] + 15) / 16, (c.cp_cur.h[L] + 15) / 16);
      k_pyrdown<<<pg, pb, 0, c.cs>>>(c.cp_cur.d[L - 1], c.cp_cur.p[L - 1], c.cp_cur.w[L - 1], c.cp_cur.h[L - 1],
                                     c.cp_cur.d[L], c.cp_cur.p[L], c.cp_cur.w[L], c.cp_cur.h[L]);
    }
    if (sync_each()) COK(cudaStreamSynchronize(c.cs));
  } else {
    if (!wrap_stream()) COK(cudaStreamSynchronize(c.cs));
    VOK(vpiSubmitGaussianPyramidGenerator(c.stream, VPI_BACKEND_CUDA, c.img_cur, c.pyr_cur, VPI_BORDER_CLAMP));
    if (sync_each()) VOK(vpiStreamSync(c.stream));
  }
  // RAW marker: everything that writes d_cur / d_mcur / the pyramid for THIS frame is now
  // queued on cs.  After the next gpu_commit those very buffers become d_prev / d_mprev, i.e.
  // exactly what the detection stream reads.  The custom-KLT path happens to host-sync cs
  // inside gpu_klt before any detection is submitted, so this edge is redundant TODAY -- it is
  // recorded anyway so the ordering does not silently depend on an unrelated function keeping
  // a synchronous tail.
  if (det_sync() && det_split() && c.ev_img) {
    cudaEventRecord(c.ev_img, c.cs);
    c.img_ev_pending.store(true, std::memory_order_release);
  }
  c.n_prepare++;
  return true;
}

// ---- OV_DET_VERIFY bookkeeping (see gpu_detect) ----
std::atomic<long> g_dv_calls{0}, g_dv_ordered_ok{0}, g_dv_set_ok{0}, g_dv_size_ok{0};
std::atomic<long> g_dv_pts{0}, g_dv_pts_ordered_diff{0};
struct DetVerifyDump {
  ~DetVerifyDump() {
    long n = g_dv_calls.load();
    if (!n) return;
    std::fprintf(stderr,
                 "[detverify]: calls=%ld  size_equal=%ld  ORDERED_IDENTICAL=%ld  SET_IDENTICAL=%ld"
                 "  pts=%ld ordered_pos_diffs=%ld\n",
                 n, g_dv_size_ok.load(), g_dv_ordered_ok.load(), g_dv_set_ok.load(),
                 g_dv_pts.load(), g_dv_pts_ordered_diff.load());
  }
} g_dv_dump;

void det_verify_cmp(const std::vector<float> &ax, const std::vector<float> &ay,
                    const std::vector<float> &bx, const std::vector<float> &by) {
  g_dv_calls.fetch_add(1, std::memory_order_relaxed);
  g_dv_pts.fetch_add((long)ax.size(), std::memory_order_relaxed);
  if (ax.size() != bx.size()) return;
  g_dv_size_ok.fetch_add(1, std::memory_order_relaxed);
  long pos_diff = 0;
  for (size_t i = 0; i < ax.size(); i++)
    if (ax[i] != bx[i] || ay[i] != by[i]) pos_diff++;      // exact float compare, no tolerance
  g_dv_pts_ordered_diff.fetch_add(pos_diff, std::memory_order_relaxed);
  if (pos_diff == 0) g_dv_ordered_ok.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::pair<float, float>> A(ax.size()), B(bx.size());
  for (size_t i = 0; i < ax.size(); i++) { A[i] = {ax[i], ay[i]}; B[i] = {bx[i], by[i]}; }
  std::sort(A.begin(), A.end());
  std::sort(B.begin(), B.end());
  if (A == B) g_dv_set_ok.fetch_add(1, std::memory_order_relaxed);
}

bool gpu_detect(std::size_t cam_id, bool on_current,
                const float *occ_x, const float *occ_y, int n_occ, int min_px_dist,
                const int *per_cell_want, int grid_x, int grid_y, int fast_threshold,
                std::vector<float> &out_x, std::vector<float> &out_y, bool submit_only) {
  Cam *cp = nullptr;
  { std::lock_guard<std::mutex> g(g_mtx);
    auto it = g_cam.find(cam_id);
    if (it != g_cam.end() && it->second.ok) cp = &it->second; }
  if (!cp) return false;
  Cam &c = *cp;
  cudaStream_t ds = (det_split() && c.cs_det) ? c.cs_det : c.cs;
  const int w = c.w, h = c.h, ncell = grid_x * grid_y;
  if (ncell > MAXCELL) return false;
  if (!on_current && !c.has_prev) return false;

  const unsigned char *src = on_current ? c.d_cur : c.d_prev;
  const size_t sp = on_current ? c.p_cur : c.p_prev;
  const unsigned char *msk = on_current ? c.d_mcur : c.d_mprev;
  const size_t mp = on_current ? c.p_mcur : c.p_mprev;
  const int have_mask = on_current ? (c.has_mask ? 1 : 0) : (c.has_mask_prev ? 1 : 0);

  out_x.clear(); out_y.clear();
  int cap = 0;
  for (int i = 0; i < ncell; i++) cap = std::max(cap, per_cell_want[i]);
  if (cap <= 0) return true;
  // headroom: several pixels can share the cutoff score. k_select emits every pixel at or above
  // the cutoff and keeps the first `cap` ARRIVALS, so if a cell ever exceeds `cap` the surviving
  // SET is scheduler-decided. Step 0 measured this never binding (SET_IDENTICAL == size_equal on
  // 16/16 calls); det mode widens the margin and the [detdeterm] census proves it stays unbound.
  cap += (det_mode() || ksel_cap_full()) ? 64 : 8;  // R18_KSEL: see the gate comment
  if (cap > c.cap_cell) {
    if (c.d_ox) { cudaFree(c.d_ox); cudaFree(c.d_oy); cudaFree(c.d_osc); }
    COK(cudaMalloc(&c.d_ox, (size_t)MAXCELL * cap * sizeof(float)));
    COK(cudaMalloc(&c.d_oy, (size_t)MAXCELL * cap * sizeof(float)));
    COK(cudaMalloc(&c.d_osc, (size_t)MAXCELL * cap));
    c.cap_cell = cap;
  }
  COK(cudaMemcpy2DAsync(c.d_want, ncell * sizeof(int), per_cell_want, ncell * sizeof(int),
                        ncell * sizeof(int), 1, cudaMemcpyHostToDevice, ds));

  dim3 b(32, 8), gr((w + 31) / 32, (h + 7) / 8);
  static const bool fast2 = [] {
    const char *e = std::getenv("OV_GPU_FAST2");
    return !e || *e != '0';               // two-phase FAST, default ON
  }();
  // The entire device-side detection, parameterised ONLY on which occupancy-buffer sizing
  // helper runs: use_vpi=true is the historical ensure_pts (3x vpiArrayDestroy + 3x
  // vpiArrayCreate every call), use_vpi=false is OV_DET_NOVPI's grow-only ensure_occ. Every
  // kernel, argument and launch configuration below is identical in both cases -- which is
  // what OV_DET_VERIFY re-runs back to back to check numerically.
  auto submit_all = [&](bool use_vpi) -> bool {
    // RAW: do not read d_prev / d_mprev on cs_det until the image stream has finished writing
    // the buffers that became them.  Stream-ordered; the host does not block.
    if (det_sync() && ds != c.cs && c.ev_img && c.img_ev_pending.exchange(false)) {
      cudaStreamWaitEvent(ds, c.ev_img, 0);
      g_imgwait.fetch_add(1, std::memory_order_relaxed);
    }
    k_maskcopy<<<gr, b, 0, ds>>>(msk, mp, c.d_mdet, c.p_mdet, w, h, have_mask);
    if (n_occ > 0 && min_px_dist > 0) {
      if (!(use_vpi ? ensure_pts(c, n_occ) : ensure_occ(c, n_occ))) return false;
      COK(cudaMemcpyAsync(c.d_px, occ_x, (size_t)n_occ * sizeof(float), cudaMemcpyHostToDevice, ds));
      COK(cudaMemcpyAsync(c.d_py, occ_y, (size_t)n_occ * sizeof(float), cudaMemcpyHostToDevice, ds));
      k_paint<<<n_occ, dim3(16, 16), 0, ds>>>(c.d_mdet, c.p_mdet, w, h, c.d_px, c.d_py, n_occ, min_px_dist);
    }
    if (fast2) {
      COK(cudaMemsetAsync(c.d_ccnt, 0, sizeof(int), ds));
      k_cand<<<gr, b, 0, ds>>>(src, sp, w, h, fast_threshold, c.d_mdet, c.p_mdet,
                                 c.d_sc, c.p_sc, c.d_cand, c.d_ccnt, c.cand_cap);
      k_score2<<<128, 256, 0, ds>>>(src, sp, w, h, fast_threshold, c.d_cand, c.d_ccnt, c.cand_cap,
                                      c.d_sc, c.p_sc);
    } else {
      k_score<<<dim3((w + SB_X - 1) / SB_X, (h + SB_Y - 1) / SB_Y), dim3(SB_X, SB_Y), 0, ds>>>(
          src, sp, w, h, fast_threshold, c.d_mdet, c.p_mdet, c.d_sc, c.p_sc);
    }
    k_nms<<<gr, b, 0, ds>>>(c.d_sc, c.p_sc, w, h, c.d_nm, c.p_nm);
    k_select<<<dim3(grid_x, grid_y), dim3(32, 8), 0, ds>>>(c.d_nm, c.p_nm, w, h, grid_x, grid_y,
                                                    c.d_want, cap, c.d_ox, c.d_oy, c.d_osc, c.d_cnt);
    c.det_cap = cap; c.det_ncell = ncell;
    // Mark the point on cs_det past which d_prev/d_cur are no longer being read by detection.
    if (det_sync() && ds != c.cs && c.ev_det) {
      cudaEventRecord(c.ev_det, ds);
      c.det_ev_pending.store(true, std::memory_order_release);
    }
    return true;
  };

  // OV_DET_VERIFY=1 : run the detection TWICE on identical inputs, first with the VPI churn
  //                   (ensure_pts) then without it (ensure_occ), and compare the two candidate
  //                   lists the caller would consume.
  // OV_DET_VERIFY=2 : control -- both halves use ensure_pts, so whatever mismatch it reports is
  //                   the pipeline's OWN k_select atomicAdd emission-order nondeterminism.
  // The run behaves like the gate-OFF path either way (half A's output is what is returned).
  static const int det_verify = [] { const char *e = std::getenv("OV_DET_VERIFY"); return e ? atoi(e) : 0; }();
  if (det_verify > 0 && !submit_only) {
    std::vector<float> ax, ay, bx, by;
    if (!submit_all(true)) return false;
    if (!gpu_detect_complete(cam_id, ax, ay)) return false;
    if (!submit_all(det_verify == 2)) return false;
    if (!gpu_detect_complete(cam_id, bx, by)) return false;
    det_verify_cmp(ax, ay, bx, by);
    out_x = ax; out_y = ay;
    return true;
  }

  if (!submit_all(!(det_novpi() && custom_klt()))) return false;
  if (submit_only) return true;          // one submitter thread queues all cams; complete later
  return gpu_detect_complete(cam_id, out_x, out_y);
}

bool gpu_detect_complete(std::size_t cam_id, std::vector<float> &out_x, std::vector<float> &out_y) {
  Cam *cp = nullptr;
  { std::lock_guard<std::mutex> g(g_mtx);
    auto it = g_cam.find(cam_id);
    if (it != g_cam.end() && it->second.ok) cp = &it->second; }
  if (!cp) return false;
  Cam &c = *cp;
  cudaStream_t ds = (det_split() && c.cs_det) ? c.cs_det : c.cs;
  const int cap = c.det_cap, ncell = c.det_ncell;
  out_x.clear(); out_y.clear();
  if (cap <= 0 || ncell <= 0) return true;

  std::vector<int> cnt(ncell);
  std::vector<float> hx((size_t)ncell * cap), hy((size_t)ncell * cap);
  std::vector<unsigned char> hs((size_t)ncell * cap);
  COK(cudaMemcpyAsync(cnt.data(), c.d_cnt, ncell * sizeof(int), cudaMemcpyDeviceToHost, ds));
  COK(cudaMemcpyAsync(hx.data(), c.d_ox, hx.size() * sizeof(float), cudaMemcpyDeviceToHost, ds));
  COK(cudaMemcpyAsync(hy.data(), c.d_oy, hy.size() * sizeof(float), cudaMemcpyDeviceToHost, ds));
  COK(cudaMemcpyAsync(hs.data(), c.d_osc, hs.size(), cudaMemcpyDeviceToHost, ds));
  int ccnt_h = 0;   // 4 bytes riding an existing async batch: the capacity census, see above
  COK(cudaMemcpyAsync(&ccnt_h, c.d_ccnt, sizeof(int), cudaMemcpyDeviceToHost, ds));
  COK(cudaStreamSynchronize(ds));
  g_ksel_calls.fetch_add(1, std::memory_order_relaxed);
  if (c.cand_cap > 0) {
    if (ccnt_h > (long)g_kcand_max.load(std::memory_order_relaxed))
      g_kcand_max.store(ccnt_h, std::memory_order_relaxed);
    if (ccnt_h > c.cand_cap) g_kcand_over.fetch_add(1, std::memory_order_relaxed);
  }
  for (int cell = 0; cell < ncell; cell++)
    if (cnt[cell] > cap) g_ksel_over.fetch_add(1, std::memory_order_relaxed);

  // Sort by descending score so the caller's greedy dedup keeps the strongest candidates,
  // mirroring Grider_GRID's response ordering.
  struct Cand { float x, y; unsigned char s; };
  std::vector<Cand> all;
  all.reserve(256);
  for (int cell = 0; cell < ncell; cell++) {
    int n = std::min(cnt[cell], cap);
    for (int i = 0; i < n; i++)
      all.push_back({hx[(size_t)cell * cap + i], hy[(size_t)cell * cap + i], hs[(size_t)cell * cap + i]});
  }
  // A stable_sort on a uint8 key PRESERVES input order among ties, i.e. it faithfully
  // propagates k_select's atomic arrival order into the caller, whose greedy min_px_dist dedup
  // then keeps a different feature set. Ordering equal-score candidates by raster position
  // instead is a total order: the sequence becomes a function of the SET alone. The scores
  // being compared are bit-identical either way -- this is a tie-break, not an approximation.
  // R18_KSEL: det_mode() still implies the total order, so every round-16/17 oracle
  // fingerprint is reproduced bit-for-bit; OV_KSEL_ORDER=1 is what makes it reachable on a
  // PACED run without also disabling pacing, prefetch and the second stream.
  if (det_mode() || ksel_order())
    std::stable_sort(all.begin(), all.end(), [](const Cand &a, const Cand &b) {
      if (a.s != b.s) return a.s > b.s;
      if (a.y != b.y) return a.y < b.y;
      return a.x < b.x;
    });
  else
    std::stable_sort(all.begin(), all.end(), [](const Cand &a, const Cand &b) { return a.s > b.s; });
  out_x.reserve(all.size()); out_y.reserve(all.size());
  for (auto &q : all) { out_x.push_back(q.x); out_y.push_back(q.y); }
  return true;
}

bool gpu_klt(std::size_t cam_id, const std::vector<float> &prev_x, const std::vector<float> &prev_y,
             std::vector<float> &cur_x, std::vector<float> &cur_y, std::vector<unsigned char> &status,
             int start_level) {
  Cam *cp = nullptr;
  { std::lock_guard<std::mutex> g(g_mtx);
    auto it = g_cam.find(cam_id);
    if (it != g_cam.end() && it->second.ok) cp = &it->second; }
  if (!cp) return false;
  Cam &c = *cp;
  const int n = (int)prev_x.size();
  cur_x.assign(prev_x.begin(), prev_x.end());
  cur_y.assign(prev_y.begin(), prev_y.end());
  status.assign(n, 0);
  if (n == 0) return true;
  if (!c.has_prev) return false;
  c.n_klt++;
  if (custom_klt()) {
    if (n > c.lk_cap) { std::fprintf(stderr, "[gpu_track]: lk n=%d > cap\n", n); return false; }
    for (int i = 0; i < n; i++) { c.h_p0[i].x = prev_x[i]; c.h_p0[i].y = prev_y[i]; }
    { cudaError_t e = cudaMemcpyAsync(c.d_p0, c.h_p0, n * sizeof(float2), cudaMemcpyHostToDevice, c.cs);
      if (e != cudaSuccess) { std::fprintf(stderr, "[gpu_track]: lk upload: %s\n", cudaGetErrorString(e)); return false; } }
    static const int lk_iters = [] {
      const char *e = std::getenv("OV_GPU_KLT_ITERS");
      return e ? atoi(e) : 30;
    }();
    int lvl0 = (start_level < 0) ? pyr_levels() - 1 : start_level;
    if (start_level == -2 && n >= 8) {
      // Pilot pass: full-pyramid KLT on a sparse subset of THIS frame pair measures the actual
      // current flow (rotation AND translation, no one-frame lag), then the main launch starts
      // only as high in the pyramid as that flow requires. Pilot buffers live in the tail of
      // the (8192-point) LK arrays; the main upload above only touches [0, n).
      const int m = n < 48 ? n : 48;
      const int off = c.lk_cap - 64;
      const int stride = n / m;
      for (int i = 0; i < m; i++) c.h_p0[off + i] = c.h_p0[i * stride];
      COK(cudaMemcpyAsync(c.d_p0 + off, c.h_p0 + off, m * sizeof(float2), cudaMemcpyHostToDevice, c.cs));
      k_lk<<<m, dim3(16, 16), 0, c.cs>>>(c.cp_prev, c.cp_cur, pyr_levels(), pyr_levels() - 1,
                                         c.d_p0 + off, c.d_p1 + off, c.d_lkst + off, m, lk_iters, 0.01f);
      COK(cudaMemcpyAsync(c.h_p1 + off, c.d_p1 + off, m * sizeof(float2), cudaMemcpyDeviceToHost, c.cs));
      COK(cudaMemcpyAsync(c.h_lkst + off, c.d_lkst + off, m, cudaMemcpyDeviceToHost, c.cs));
      COK(cudaStreamSynchronize(c.cs));
      int ok = 0;
      float mx = 0.f;
      for (int i = 0; i < m; i++)
        if (c.h_lkst[off + i]) {
          ok++;
          const float dx = fabsf(c.h_p1[off + i].x - c.h_p0[off + i].x);
          const float dy = fabsf(c.h_p1[off + i].y - c.h_p0[off + i].y);
          mx = fmaxf(mx, fmaxf(dx, dy));
        }
      if (ok < m / 2) {
        lvl0 = pyr_levels() - 1; // scene too hard to trust the pilot -- full pyramid
      } else {
        lvl0 = 0;
        while (lvl0 < pyr_levels() - 1 && (float)(7 << lvl0) < 1.5f * mx)
          lvl0++;
      }
    }
    if (lvl0 < 0) lvl0 = 0;
    if (lvl0 > pyr_levels() - 1) lvl0 = pyr_levels() - 1;
    static const bool lk_ev = [] { const char *e = std::getenv("OV_GPU_LK_EVENTS"); return e && *e == '1'; }();
    static std::mutex ev_mtx;
    static double ev_kern_ms = 0, ev_wall_ms = 0;
    static long ev_n = 0;
    static struct EvDump {
      ~EvDump() {
        if (ev_n)
          std::fprintf(stderr, "[lk_events]: n=%ld kernel avg %.3f ms, submit->sync wall avg %.3f ms\n",
                       ev_n, ev_kern_ms / ev_n, ev_wall_ms / ev_n);
      }
    } ev_dump;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    std::chrono::steady_clock::time_point wall0;
    if (lk_ev) {
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      cudaEventRecord(ev0, c.cs);
      wall0 = std::chrono::steady_clock::now();
    }
    static const bool lk128 = [] { const char *e = std::getenv("OV_LK128"); return e && *e == '1'; }();
    if (lk128)
      k_lk128<<<n, dim3(16, 8), 0, c.cs>>>(c.cp_prev, c.cp_cur, pyr_levels(), lvl0,
                                           c.d_p0, c.d_p1, c.d_lkst, n, lk_iters, 0.01f);
    else
      k_lk<<<n, dim3(16, 16), 0, c.cs>>>(c.cp_prev, c.cp_cur, pyr_levels(), lvl0,
                                         c.d_p0, c.d_p1, c.d_lkst, n, lk_iters, 0.01f);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { std::fprintf(stderr, "[gpu_track]: k_lk LAUNCH: %s\n", cudaGetErrorString(e)); return false; } }
    if (lk_ev) cudaEventRecord(ev1, c.cs);
    COK(cudaMemcpyAsync(c.h_p1, c.d_p1, n * sizeof(float2), cudaMemcpyDeviceToHost, c.cs));
    COK(cudaMemcpyAsync(c.h_lkst, c.d_lkst, n, cudaMemcpyDeviceToHost, c.cs));
    { cudaError_t e = cudaStreamSynchronize(c.cs);
      if (e != cudaSuccess) { std::fprintf(stderr, "[gpu_track]: lk sync: %s\n", cudaGetErrorString(e)); return false; } }
    if (lk_ev) {
      float km = 0.f;
      cudaEventElapsedTime(&km, ev0, ev1);
      double wm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
      { std::lock_guard<std::mutex> lk(ev_mtx); ev_kern_ms += km; ev_wall_ms += wm; ev_n++; }
      cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    for (int i = 0; i < n; i++) {
      cur_x[i] = c.h_p1[i].x; cur_y[i] = c.h_p1[i].y;
      status[i] = c.h_lkst[i];
      if (!(cur_x[i] >= 0 && cur_y[i] >= 0 && cur_x[i] < c.w && cur_y[i] < c.h)) status[i] = 0;
    }
    if (std::getenv("OV_GPU_LK_DBG"))
      std::fprintf(stderr, "[lkdbg]: cam %zu n=%d p0[0]=(%.1f,%.1f) p1[0]=(%.2f,%.2f) st[0]=%d\n",
                   cam_id, n, prev_x[0], prev_y[0], c.h_p1[0].x, c.h_p1[0].y, (int)c.h_lkst[0]);
    goto lk_done;
  }
  { // ---- VPI path (scoped so the custom path's goto may bypass it) ----
  if (c.n_prepare != c.n_commit + 1)
    std::fprintf(stderr, "[gpu_track]: cam %zu PAIRING BROKEN prepares=%ld commits=%ld klts=%ld"
                         " (prev is %ld frames stale)\n",
                 cam_id, c.n_prepare, c.n_commit, c.n_klt, c.n_prepare - c.n_commit - 1);
  if (!ensure_pts(c, n)) return false;
  {
    VPIArrayData ad;
    VOK(vpiArrayLockData(c.a_prev, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &ad));
    auto *k = (VPIKeypointF32 *)ad.buffer.aos.data;
    for (int i = 0; i < n; i++) { k[i].x = prev_x[i]; k[i].y = prev_y[i]; }
    *ad.buffer.aos.sizePointer = n;
    VOK(vpiArrayUnlock(c.a_prev));
  }
  // NOTE: do NOT pre-seed a_cur. useInitialFlow defaults to 0, so VPI treats curPts as pure
  // output and runs its full coarse-to-fine search. Writing the previous positions into it
  // made VPI return only tiny refinements around them (mean flow 1.23 px where the true
  // motion is tens of px), which collapsed the estimated path to 2.0 m against 80.6 m actual.
  // Hypothesis: vpiCreateOpticalFlowPyrLK binds to the pyramid handles it first sees, so the
  // per-frame swap in gpu_commit makes the payload operate on the wrong pair. (VPI already
  // proved it caches handles once this session -- recreating the point arrays mid-run silently
  // stopped it writing curPts.) Recreate the payload per call to test.
  static const bool fresh_payload = [] {
    const char *e = std::getenv("OV_GPU_FRESH_PAYLOAD");
    return e && *e == '1';
  }();
  VPIPayload lk_local = c.lk;
  if (fresh_payload) {
    VOK(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, c.w, c.h, VPI_IMAGE_FORMAT_U8,
                                  pyr_levels(), 0.5f, &lk_local));
  }
  VPIOpticalFlowPyrLKParams p;
  VOK(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &p));
  p.windowDimension = 15;
  // Match OpenCV's TermCriteria(COUNT|EPS, 30, 0.01) that OpenVINS passes to
  // calcOpticalFlowPyrLK. VPI defaults to numIterations=6 / epsilon=0 -- 5x fewer iterations
  // means less-converged positions, which shows up as a lower per-frame survival rate
  // (90% vs 93%) and compounds into ~25% shorter tracks.
  p.numIterations = 30;
  p.epsilon = 0.01f;
  if (const char *it = std::getenv("OV_GPU_KLT_ITERS")) p.numIterations = atoi(it);
  if (const char *ep = std::getenv("OV_GPU_KLT_EPS")) p.epsilon = (float)atof(ep);
  VOK(vpiSubmitOpticalFlowPyrLK(c.stream, VPI_BACKEND_CUDA, lk_local, c.pyr_prev, c.pyr_cur,
                                c.a_prev, c.a_cur, c.a_st, &p));
  VOK(vpiStreamSync(c.stream));
  if (fresh_payload) vpiPayloadDestroy(lk_local);
  {
    VPIArrayData ad, sd;
    VOK(vpiArrayLockData(c.a_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &ad));
    VOK(vpiArrayLockData(c.a_st, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &sd));
    auto *k = (VPIKeypointF32 *)ad.buffer.aos.data;
    auto *st = (unsigned char *)sd.buffer.aos.data;
    for (int i = 0; i < n; i++) {
      cur_x[i] = k[i].x; cur_y[i] = k[i].y;
      // VPI's status is much STRICTER than OpenCV's: it flags high-LK-error points, whereas
      // cv::calcOpticalFlowPyrLK only flags catastrophic failures (measured: the CPU path
      // reports 196/196 tracked essentially every frame). OpenVINS relies on RANSAC for real
      // outlier rejection, so honouring VPI's flag threw away good correspondences -- on 77
      // frames it marked ALL points lost while findFundamentalMat accepted 37 of 38 of the very
      // same positions, emptying good_left and resetting the track.
      // Default: accept any in-bounds position and let RANSAC decide. OV_GPU_KLT_STRICT=1
      // restores VPI's own flag.
      static const bool permissive = [] {
        const char *e = std::getenv("OV_GPU_KLT_PERMISSIVE");
        return e && *e == '1';
      }();
      status[i] = permissive ? 1 : (st[i] ? 0 : 1);
      if (!(cur_x[i] >= 0 && cur_y[i] >= 0 && cur_x[i] < c.w && cur_y[i] < c.h)) status[i] = 0;
    }
    VOK(vpiArrayUnlock(c.a_st));
    VOK(vpiArrayUnlock(c.a_cur));
  }
  {
    int okc = 0;
    for (int i = 0; i < n; i++) okc += status[i];
    if (okc == 0 && n > 0) {
      VPIArrayData sd;
      int32_t stsize = -1;
      unsigned char raw[6] = {0};
      if (vpiArrayLockData(c.a_st, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &sd) == VPI_SUCCESS) {
        stsize = *sd.buffer.aos.sizePointer;
        auto *p = (unsigned char *)sd.buffer.aos.data;
        for (int i = 0; i < 6 && i < n; i++) raw[i] = p[i];
        vpiArrayUnlock(c.a_st);
      }
      int32_t psize = -1, csize = -1;
      VPIArrayData ad;
      if (vpiArrayLockData(c.a_prev, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &ad) == VPI_SUCCESS) {
        psize = *ad.buffer.aos.sizePointer; vpiArrayUnlock(c.a_prev); }
      if (vpiArrayLockData(c.a_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &ad) == VPI_SUCCESS) {
        csize = *ad.buffer.aos.sizePointer; vpiArrayUnlock(c.a_cur); }
      std::fprintf(stderr, "[gpu_track]: ALL-LOST cam %zu n=%d cap=%d | sizes prev=%d cur=%d st=%d"
                           " | raw st = %u %u %u %u %u %u\n",
                   cam_id, n, c.npts_cap, psize, csize, stsize,
                   raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]);
    }
  }
  } // ---- end VPI path ----
lk_done:;
  // Dump the inputs of a call that actually FAILED in situ, so the harness can replay the
  // hard case rather than an easy one.
  if (std::getenv("OV_GPU_DUMPBAD")) {
    static bool done = false;
    int okc = 0;
    for (int i = 0; i < n; i++) okc += status[i];
    if (!done && n > 150 && okc < n / 5) {
      done = true;
      std::vector<unsigned char> hp((size_t)c.w * c.h), hc((size_t)c.w * c.h);
      cudaMemcpy2D(hp.data(), c.w, c.d_prev, c.p_prev, c.w, c.h, cudaMemcpyDeviceToHost);
      cudaMemcpy2D(hc.data(), c.w, c.d_cur, c.p_cur, c.w, c.h, cudaMemcpyDeviceToHost);
      FILE *f1 = fopen("/home/lis/bad_prev.gray", "wb"); fwrite(hp.data(), 1, hp.size(), f1); fclose(f1);
      FILE *f2 = fopen("/home/lis/bad_cur.gray", "wb");  fwrite(hc.data(), 1, hc.size(), f2); fclose(f2);
      FILE *f3 = fopen("/home/lis/bad_pts.txt", "w");
      for (int i = 0; i < n; i++) fprintf(f3, "%.6f %.6f\n", prev_x[i], prev_y[i]);
      fclose(f3);
      std::fprintf(stderr, "[gpu_track]: DUMPED BAD call cam %zu n=%d in-situ tracked=%d\n", cam_id, n, okc);
    }
  }
  // (the in-process A/B diagnostic that identified the VPI array-capacity root cause
  //  lived here; removed once the finding was recorded)
  if (const char *fv = std::getenv("OV_GPU_FLOW_DEBUG")) {
    if (*fv == '1' && n > 0) {
      double sum = 0, mx = 0; int cnt = 0;
      for (int i = 0; i < n; i++) {
        if (!status[i]) continue;
        double d = std::sqrt((cur_x[i]-prev_x[i])*(cur_x[i]-prev_x[i]) + (cur_y[i]-prev_y[i])*(cur_y[i]-prev_y[i]));
        sum += d; if (d > mx) mx = d; cnt++;
      }
      std::fprintf(stderr, "[flow]: cam %zu n=%d tracked=%d mean|flow|=%.4f max=%.3f\n",
                   cam_id, n, cnt, cnt ? sum/cnt : 0.0, mx);
    }
  }
  if (const char *v = std::getenv("OV_GPU_TRACK_DEBUG")) {
    if (*v == '1') {
      int okc = 0;
      for (int i = 0; i < n; i++) okc += status[i];
      unsigned long long *dsum, hs_prev = 0, hs_cur = 0, hs_raw = 0, hs_msk = 0;
      if (cudaMalloc(&dsum, sizeof(unsigned long long)) == cudaSuccess) {
        dim3 bb(32, 8), gg((c.w + 31) / 32, (c.h + 7) / 8);
        cudaMemset(dsum, 0, sizeof(unsigned long long));
        k_sum<<<gg, bb>>>(c.d_prev, c.p_prev, c.w, c.h, dsum);
        cudaDeviceSynchronize();
        cudaMemcpy(&hs_prev, dsum, sizeof(hs_prev), cudaMemcpyDeviceToHost);
        cudaMemset(dsum, 0, sizeof(unsigned long long));
        k_sum<<<gg, bb>>>(c.d_cur, c.p_cur, c.w, c.h, dsum);
        cudaDeviceSynchronize();
        cudaMemcpy(&hs_cur, dsum, sizeof(hs_cur), cudaMemcpyDeviceToHost);
        cudaMemset(dsum, 0, sizeof(unsigned long long));
        k_sum<<<gg, bb>>>(c.d_raw, c.p_raw, c.w, c.h, dsum);
        cudaDeviceSynchronize();
        cudaMemcpy(&hs_raw, dsum, sizeof(hs_raw), cudaMemcpyDeviceToHost);
        cudaMemset(dsum, 0, sizeof(unsigned long long));
        k_sum<<<gg, bb>>>(c.d_mcur, c.p_mcur, c.w, c.h, dsum);
        cudaDeviceSynchronize();
        cudaMemcpy(&hs_msk, dsum, sizeof(hs_msk), cudaMemcpyDeviceToHost);
        cudaFree(dsum);
      }
      std::fprintf(stderr, "[gpu_track]: cam %zu klt %d/%d | prev=%llu cur=%llu raw=%llu mask=%llu%s%s\n",
                   cam_id, okc, n, hs_prev, hs_cur, hs_raw, hs_msk,
                   (hs_cur == hs_msk ? "  <<< cur==MASK" : ""),
                   (hs_cur == hs_raw ? "  <<< cur==RAW(no CLAHE)" : ""));
    }
  }
  return true;
}

void gpu_commit(std::size_t cam_id) {
  Cam *cp = nullptr;
  { std::lock_guard<std::mutex> g(g_mtx);
    auto it = g_cam.find(cam_id);
    if (it != g_cam.end() && it->second.ok) cp = &it->second; }
  if (!cp) return;
  Cam &c = *cp;
  // Swapping the pyramid/image handles every frame gave 46% KLT survival in situ while a
  // replay of the very same images and points gave 86%. Images, points, pyramid rebuild and a
  // fresh payload were each ruled out individually, so the fault is in the swapped state
  // itself. Copy-and-rebuild instead: identical to the sequence the replay performs, at the
  // cost of one device-to-device copy plus one pyramid build per frame.
  if (custom_klt()) {
    std::swap(c.d_prev, c.d_cur);   std::swap(c.p_prev, c.p_cur);
    std::swap(c.d_mprev, c.d_mcur); std::swap(c.p_mprev, c.p_mcur);
    std::swap(c.cp_prev, c.cp_cur);
    c.cp_cur.d[0] = c.d_cur; c.cp_cur.p[0] = c.p_cur;
    c.cp_prev.d[0] = c.d_prev; c.cp_prev.p[0] = c.p_prev;
    c.has_mask_prev = c.has_mask;
    c.has_prev = true;
    c.n_commit++;
    return;
  }
  static const bool noswap = [] {
    const char *e = std::getenv("OV_GPU_NOSWAP");
    return !e || *e != '0';                       // default ON
  }();
  if (noswap) {
    dim3 b(32, 8), gr((c.w + 31) / 32, (c.h + 7) / 8);
    k_copy<<<gr, b, 0, c.cs>>>(c.d_cur, c.p_cur, c.d_prev, c.p_prev, c.w, c.h);
    k_copy<<<gr, b, 0, c.cs>>>(c.d_mcur, c.p_mcur, c.d_mprev, c.p_mprev, c.w, c.h);
    if (!wrap_stream()) cudaStreamSynchronize(c.cs);
    vpiSubmitGaussianPyramidGenerator(c.stream, VPI_BACKEND_CUDA, c.img_prev, c.pyr_prev, VPI_BORDER_CLAMP);
  } else {
    std::swap(c.pyr_prev, c.pyr_cur);
    std::swap(c.d_prev, c.d_cur);   std::swap(c.p_prev, c.p_cur);
    std::swap(c.img_prev, c.img_cur);            // wrapper travels with its buffer
    std::swap(c.d_mprev, c.d_mcur); std::swap(c.p_mprev, c.p_mcur);
  }
  c.has_mask_prev = c.has_mask;
  c.has_prev = true;
  c.n_commit++;
}

} // namespace ov_core
