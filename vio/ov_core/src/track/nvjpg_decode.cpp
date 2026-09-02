#include "nvjpg_decode.h"

#include <cuda_runtime.h>
#include <cuda_egl_interop.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <atomic>

#include "nvbufsurface.h"

namespace ov_core {
namespace {

// ---- the isolated shim's C ABI -------------------------------------------------------
typedef int (*fn_abi)(void);
typedef void *(*fn_open)(int);
typedef int (*fn_decode)(void *, int, unsigned char *, unsigned long, int *, unsigned *, unsigned *,
                         unsigned *);
typedef const char *(*fn_prov)(void);
typedef void (*fn_close)(void *);
#define OVNV_ABI_COOKIE 0x4e564a01
#define V4L2_PIX_FMT_YUV420M_FOURCC 0x32314d59 /* 'YM12' */

struct Slot {
  int fd = -1;
  // OUR OWN device buffer.  NvJPEGDecoder::decodeToFd hands back a BORROWED view of a
  // process-wide recycled surface pool: jpeg_finish_decompress (NvJpegDecoder.cpp:140) puts
  // the buffer back BEFORE `fd = cinfo.fd` is read at :147, so the next decode -- on ANY
  // instance, from ANY thread -- may already own it.  MEASURED: 12 decoder instances yield
  // 13 distinct fds, and a CPU map of the surface disagrees with libjpeg exactly as much as
  // the CUDA view does (250/5300 bit-exact both ways), i.e. the surface really does hold
  // another frame.  So the surface is copied out, device-to-device, under the decode lock.
  void *d_own = nullptr;
  size_t p_own = 0;
  const void *d_y = nullptr;
  size_t pitch = 0;
  int w = 0, h = 0;
  double ts = -1e18;
  cudaEvent_t ev = nullptr;
  bool ev_recorded = false;
  // 0 free | 1 staged (decoded, not yet claimed) | 2 claimed (consumer owns it)
  // 3 RESERVED (R18): acquired by a producer whose decode is still in flight.  Before R18
  //   the slot sat at 0 across the whole decode window and was kept private only by the
  //   next[] cursor having moved past it; once the acquire SEARCHES the ring (it must, see
  //   ov_nvjpg_stage) a second producer on the same camera could hand out the same slot.
  int state = 0;
};

struct State {
  bool on = false;
  bool inited = false;
  bool refused = false;
  std::string refuse_why;
  void *dso = nullptr;
  void *h = nullptr; // shim handle
  fn_decode f_decode = nullptr;
  fn_close f_close = nullptr;
  fn_prov f_prov = nullptr;
  int ncam = 4, ring = 3;
  int lookahead = 3;        // R18_RING_FIX: reader-ahead depth, set by the runner
  double acq_ms = 250.0;    // R18: acquire budget before falling back to a CPU decode
  std::vector<std::vector<Slot>> slot; // [cam][ring]
  std::vector<int> next;               // [cam]
  std::map<std::pair<int, double>, int> stage; // (cam,ts) -> global slot id
  struct Reg { cudaGraphicsResource_t res; NvBufSurface *surf; };
  std::map<unsigned long long, Reg> egl;   // PHYSICAL BUFFER dataPtr -> registration
  double watermark = -1e18;
  std::map<int, std::set<unsigned long long>> fd_data;
  std::set<unsigned long long> n_surf_ids;
  std::mutex decode_m;      // the borrowed-surface window: decode -> copy-out is atomic
  cudaStream_t cpy = nullptr;
  unsigned char *page = nullptr; // ONE shared placeholder page
  size_t page_bytes = 0;
  int pw = 0, ph = 0;
  // counters
  std::atomic<long> n_stage{0}, n_take{0}, n_miss{0}, n_block{0}, n_abandon{0}, n_fail{0};
  std::atomic<long> n_acqfail{0};   // R18: acquire gave up -> caller decoded on the CPU
  double block_ms = 0.0;
  double max_block_ms = 0.0;
  std::mutex m;
};

State &S() {
  static State s;
  return s;
}

double now_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// Bind libov_nvjpg.so.  RTLD_LOCAL keeps libnvjpeg's 116 jpeg_* out of the global scope so
// libopencv_imgcodecs keeps binding to libjpeg.so.8; RTLD_DEEPBIND is MANDATORY, not
// belt-and-braces, because glibc searches the GLOBAL scope first for an RTLD_LOCAL object's
// own lookups and libjpeg.so.8 is already global via imgcodecs -- without it our own
// NvJpegDecoder.cpp would bind to STOCK libjpeg and corrupt cinfo in the other direction.
bool bind_dso(State &s) {
  const char *cand[] = {"libov_nvjpg.so", nullptr};
  for (int i = 0; cand[i]; i++) {
    s.dso = dlopen(cand[i], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (s.dso) break;
  }
  if (!s.dso) {
    std::fprintf(stderr, "[nvjpg]: dlopen(libov_nvjpg.so) FAILED: %s\n", dlerror());
    return false;
  }
  fn_abi f_abi = (fn_abi)dlsym(s.dso, "ovnv_abi");
  fn_open f_open = (fn_open)dlsym(s.dso, "ovnv_open");
  s.f_decode = (fn_decode)dlsym(s.dso, "ovnv_decode");
  s.f_prov = (fn_prov)dlsym(s.dso, "ovnv_jpeg_provider");
  s.f_close = (fn_close)dlsym(s.dso, "ovnv_close");
  if (!f_abi || !f_open || !s.f_decode || !s.f_prov || !s.f_close) {
    std::fprintf(stderr, "[nvjpg]: shim symbols missing\n");
    return false;
  }
  if (f_abi() != OVNV_ABI_COOKIE) {
    std::fprintf(stderr, "[nvjpg]: shim ABI cookie mismatch (stale libov_nvjpg.so?)\n");
    return false;
  }
  // P1: the ESTIMATOR's own libjpeg must still be stock libjpeg after the dlopen.
  void *ph = dlsym(RTLD_DEFAULT, "jpeg_read_header");
  Dl_info di;
  const char *who = (ph && dladdr(ph, &di) && di.dli_fname) ? di.dli_fname : "<none>";
  if (!std::strstr(who, "libjpeg.so.8")) {
    std::fprintf(stderr,
                 "[nvjpg]: FATAL P1 -- global jpeg_read_header resolves to '%s', not libjpeg.so.8. "
                 "cv::imdecode would take libnvjpeg's incompatible jpeg_decompress_struct layout. "
                 "Aborting BEFORE any decode.\n",
                 who);
    return false;
  }
  std::fprintf(stderr, "[nvjpg]: P1 ok global jpeg_read_header <- %s\n", who);
  std::fprintf(stderr, "[nvjpg]: P2    shim jpeg provider      <- %s\n", s.f_prov());
  s.h = f_open(s.ncam);
  if (!s.h) {
    std::fprintf(stderr, "[nvjpg]: ovnv_open(%d) FAILED\n", s.ncam);
    return false;
  }
  return true;
}

int envi(const char *k, int d) {
  const char *e = std::getenv(k);
  return (e && *e) ? std::atoi(e) : d;
}

// BENCH ONLY, DEFAULT OFF, NEVER SHIP.  Restores the pre-R18 slot acquire so the deadlock
// can be provoked on demand and the fix measured against a real reference arm.
bool legacy_acq() {
  static const bool v = [] {
    const char *e = std::getenv("OV_NVJPG_LEGACY_ACQ");
    return e && *e == '1';
  }();
  return v;
}

} // namespace

void ov_nvjpg_set_lookahead(int max_frames_ahead) {
  State &s = S();
  if (max_frames_ahead > 0) s.lookahead = max_frames_ahead;
}

namespace {

void init_once() {
  State &s = S();
  if (s.inited) return;
  s.inited = true;
  const char *e = std::getenv("OV_NVJPG");
  if (!e || *e != '1') return;
  if (s.refused) {
    std::fprintf(stderr, "[nvjpg]: REFUSED -- %s\n", s.refuse_why.c_str());
    std::exit(EXIT_FAILURE);
  }
  s.ncam = envi("OV_NVJPG_NCAM", 4);
  // R18_RING_FIX -- SIZE THE RING FROM THE LOOK-AHEAD, DO NOT GUESS IT.
  // Peak simultaneous slot demand for ONE camera, counted off the producer/consumer code:
  //   1  the frame the consumer is inside feed_frame() for: taken (state 2) and not retired
  //      until its CLAHE/copy kernel completes, which is asynchronous.
  //   L  the frames sitting in raq -- exactly MAX_FRAMES_AHEAD of them (state 1).
  //   1  the frame whose 4 decode tasks the producer POSTED and then blocked in push():
  //      the tasks are posted BEFORE the MAX_FRAMES_AHEAD wait, so one frame's worth of
  //      staging always runs ahead of the raq accounting.
  //   1  the frame being staged right now (state 3).
  // => L + 3.  With L = 3 that is 6, and the shipped default was 3, i.e. HALF of the
  // structural requirement.  ring == MAX_FRAMES_AHEAD was never a safe pairing.
  const int need = s.lookahead + 3;
  s.ring = envi("OV_NVJPG_RING", need);
  if (s.ring < need && !envi("OV_NVJPG_RING_FORCE", 0)) {
    std::fprintf(stderr, "[nvjpg]: OV_NVJPG_RING=%d is below the structural minimum %d "
                         "(lookahead %d + 3); raising it. Set OV_NVJPG_RING_FORCE=1 to "
                         "reproduce the pre-R18 depth for bisection.\n",
                 s.ring, need, s.lookahead);
    s.ring = need;
  }
  if (s.ring < 2) s.ring = 2;
  if (s.ring > 16) s.ring = 16;
  { const char *a = std::getenv("OV_NVJPG_ACQ_MS");
    if (a && *a) s.acq_ms = std::atof(a);
    if (s.acq_ms < 1.0) s.acq_ms = 1.0; }
  // EGL display, once.
  EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, nullptr, nullptr)) {
    std::fprintf(stderr, "[nvjpg]: FATAL no EGL display -- zero-copy unavailable\n");
    std::exit(EXIT_FAILURE);
  }
  if (!bind_dso(s)) std::exit(EXIT_FAILURE);
  s.slot.assign(s.ncam, std::vector<Slot>(s.ring));
  s.next.assign(s.ncam, 0);
  for (int c = 0; c < s.ncam; c++)
    for (int r = 0; r < s.ring; r++)
      if (cudaEventCreateWithFlags(&s.slot[c][r].ev, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "[nvjpg]: FATAL cudaEventCreate\n");
        std::exit(EXIT_FAILURE);
      }
  if (cudaStreamCreateWithFlags(&s.cpy, cudaStreamNonBlocking) != cudaSuccess) {
    std::fprintf(stderr, "[nvjpg]: FATAL cudaStreamCreate\n");
    std::exit(EXIT_FAILURE);
  }
  s.on = true;
  std::fprintf(stderr,
               "[nvjpg]: ENABLED ncam=%d ring=%d lookahead=%d acq_ms=%.0f (%d decoder "
               "instances, copy-out ON)\n",
               s.ncam, s.ring, s.lookahead, s.acq_ms, s.ncam);
}

// Register (or look up) the EGLImage -> CUDA device pointer for the buffer the decoder just
// wrote.  KEYED BY PHYSICAL BUFFER (surfaceList[0].dataPtr), NEVER BY fd.
//
// MEASURED, and it is the whole correctness story of this lever: libnvjpeg hands decodeToFd a
// BORROWED buffer out of a process-wide pool, and the same integer fd names a DIFFERENT
// physical buffer over time -- fd 67 named 4 of them, 5 fds ever appeared but 16 distinct
// dataPtrs did.  Caching the registration by fd (what /home/lis/nvjpg_zero.cu does, and what
// the plan specified) therefore reads a STALE buffer roughly 3 times in 4: exactly the
// uniform 332/1325-per-camera bit-exact rate measured before this change.  The prior
// zero-copy benchmark never compared pixels, so it never saw this.
bool egl_lookup(State &s, int fd, const void **d_y, size_t *pitch) {
  NvBufSurface *surf = nullptr;
  if (NvBufSurfaceFromFd(fd, (void **)&surf) != 0 || !surf) {
    std::fprintf(stderr, "[nvjpg]: NvBufSurfaceFromFd(%d) failed\n", fd);
    return false;
  }
  const unsigned long long key = (unsigned long long)(uintptr_t)surf->surfaceList[0].dataPtr;
  auto it = s.egl.find(key);
  if (it == s.egl.end()) {
    if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
      std::fprintf(stderr, "[nvjpg]: NvBufSurfaceMapEglImage failed\n");
      return false;
    }
    cudaGraphicsResource_t res = nullptr;
    cudaError_t ce = cudaGraphicsEGLRegisterImage(
        &res, (EGLImageKHR)surf->surfaceList[0].mappedAddr.eglImage,
        cudaGraphicsRegisterFlagsReadOnly);
    if (ce != cudaSuccess) {
      std::fprintf(stderr, "[nvjpg]: cudaGraphicsEGLRegisterImage: %s\n", cudaGetErrorString(ce));
      return false;
    }
    it = s.egl.insert({key, State::Reg{res, surf}}).first;
  }
  cudaEglFrame ef;
  if (cudaGraphicsResourceGetMappedEglFrame(&ef, it->second.res, 0, 0) != cudaSuccess) return false;
  *d_y = ef.frame.pPitch[0].ptr;
  *pitch = ef.frame.pPitch[0].pitch;
  return *d_y != nullptr;
}

} // namespace

bool ov_nvjpg_enabled() {
  init_once();
  return S().on;
}

void ov_nvjpg_refuse_if(bool cond, const char *why) {
  State &s = S();
  if (!cond) return;
  s.refused = true;
  if (!s.refuse_why.empty()) s.refuse_why += " AND ";
  s.refuse_why += why;
}

bool ov_nvjpg_is_placeholder(const unsigned char *p) {
  State &s = S();
  return s.on && s.page && p >= s.page && p < s.page + s.page_bytes;
}

const unsigned char *ov_nvjpg_stage(int cam_id, double ts, const unsigned char *jpg, size_t n,
                                    int *w, int *h, size_t *step) {
  State &s = S();
  if (!s.on || cam_id < 0 || cam_id >= s.ncam) return nullptr;

  // ---- claim a ring slot, honouring the lifetime invariant --------------------------
  // R18_RING_FIX -- THIS LOOP USED TO BE THE OTHER HALF OF A THREE-WAY DEADLOCK.
  //
  // THE CYCLE (captured with gdb, /var/tmp/vioperf/hang.bt):
  //   consumer  run_serial_msckf.cpp:1025  in fr.fut.get(), waiting for frame N's decode
  //   4x pool   nvjpg_decode.cpp (here)    waiting for a ring slot only the consumer frees
  //   producer  run_serial_msckf.cpp:1161  waiting on MAX_FRAMES_AHEAD for the consumer
  // OvDecodePool has exactly ncam=4 workers and every frame posts exactly 4 tasks, so ALL
  // FOUR workers can be parked in here at once -- and in the captured hang they were, three
  // on frame N and one straggler on frame N-1 cam 1, which is precisely the task the
  // consumer was blocked on.  Nothing could then make progress.
  //
  // THREE INDEPENDENT DEFECTS FED IT, AND ALL THREE ARE FIXED HERE:
  //  (a) THE SLOT WAS CHOSEN BEFORE IT WAS CHECKED.  The old code did r = next[cam]++ and
  //      then waited on THAT slot even when another slot in the same ring was free.  The
  //      round-robin cursor is only correct if slots retire in issue order, and they do
  //      not: four workers stage four cameras concurrently and enter this function out of
  //      frame order, so "the next slot" is not "the oldest slot".  Now we SEARCH.
  //  (b) THE WAIT WAS UNBOUNDED.  An unbounded wait on a pool worker is what let the cycle
  //      close.  It is now bounded by s.acq_ms.
  //  (c) state==2 WITH NO RECORDED EVENT HAD NO ESCAPE AT ALL -- the old loop's only exits
  //      were state 0, state 2 with an event, and state 1 under the watermark.  A slot that
  //      was taken but never marked (the geometry-mismatch return in gpu_track.cu did
  //      exactly that) parked a worker forever.  The bounded budget now covers it, and the
  //      gpu_track.cu leak is fixed at its source as well.
  //
  // ON EXPIRY WE RETURN nullptr AND THE CALLER DECODES ON THE CPU.  That is the documented
  // fallback, and it is what makes the cycle STRUCTURALLY impossible rather than merely
  // improbable: no pool worker can be parked indefinitely, so the frame the consumer is
  // waiting on always completes, so the consumer always advances, so slots always retire.
  // The ring depth (see init_once) is what stops the fallback ever being reached in
  // practice; the fallback is what stops a hang if that reasoning is ever wrong.
  int r = -1;
  if (legacy_acq()) {
    // ---- PRE-R18 ACQUIRE, VERBATIM.  THIS IS THE CODE THAT DEADLOCKS. -----------------
    // Commits to next[cam] before checking it, and then waits on that one slot forever.
    std::unique_lock<std::mutex> lk(s.m);
    r = s.next[cam_id];
    s.next[cam_id] = (r + 1) % s.ring;
    Slot *sl = &s.slot[cam_id][r];
    double t_block = -1.0;
    for (;;) {
      if (sl->state == 0) break;
      if (sl->state == 2 && sl->ev_recorded) {
        cudaEvent_t ev = sl->ev;
        if (cudaEventQuery(ev) == cudaSuccess) { sl->state = 0; sl->ev_recorded = false; break; }
        if (t_block < 0) t_block = now_ms();
        lk.unlock();
        cudaEventSynchronize(ev);
        lk.lock();
        sl->state = 0;
        sl->ev_recorded = false;
        break;
      }
      if (sl->state == 1 && s.watermark >= sl->ts) {
        s.stage.erase({cam_id, sl->ts});
        sl->state = 0;
        s.n_abandon++;
        break;
      }
      if (t_block < 0) t_block = now_ms();
      lk.unlock();
      struct timespec req { 0, 200000 };
      nanosleep(&req, nullptr);
      lk.lock();
    }
    if (t_block >= 0) {
      const double bl = now_ms() - t_block;
      s.n_block++;
      s.block_ms += bl;
      if (bl > s.max_block_ms) s.max_block_ms = bl;
    }
  } else {
    std::unique_lock<std::mutex> lk(s.m);
    const double t0 = now_ms();
    double t_block = -1.0;
    for (;;) {
      // Start at the cursor purely for fairness/locality; correctness comes from the scan.
      for (int i = 0; i < s.ring; i++) {
        const int cand = (s.next[cam_id] + i) % s.ring;
        Slot *sl = &s.slot[cam_id][cand];
        if (sl->state == 0) { r = cand; break; }
        if (sl->state == 1 && s.watermark >= sl->ts) {
          // the consumer moved past this frame-set without ever using it (throttled/filtered)
          s.stage.erase({cam_id, sl->ts});
          sl->state = 0;
          s.n_abandon++;
          r = cand;
          break;
        }
        if (sl->state == 2 && sl->ev_recorded && cudaEventQuery(sl->ev) == cudaSuccess) {
          sl->state = 0;
          sl->ev_recorded = false;
          r = cand;
          break;
        }
        // state 3 (another producer's in-flight decode) is never stealable.
      }
      if (r >= 0) break;
      if (t_block < 0) t_block = now_ms();
      if (now_ms() - t0 >= s.acq_ms) break;   // budget spent -> CPU fallback
      lk.unlock();
      struct timespec req { 0, 200000 };
      nanosleep(&req, nullptr);
      lk.lock();
    }
    if (t_block >= 0) {
      const double bl = now_ms() - t_block;
      s.n_block++;
      s.block_ms += bl;
      if (bl > s.max_block_ms) s.max_block_ms = bl;
    }
    if (r < 0) {
      // No slot in s.acq_ms.  Do NOT wait longer -- that is the deadlock.  Tell the caller
      // to decode this image on the CPU instead; the frame is still delivered, in full.
      s.n_acqfail++;
      return nullptr;
    }
    s.slot[cam_id][r].state = 3;             // RESERVED across the unlocked decode window
    s.next[cam_id] = (r + 1) % s.ring;
  }

  // ---- hardware decode + copy-out, ATOMIC ---------------------------------------------
  // The engine is serial (~1.5 ms/image) so serialising the four cameras costs nothing in
  // throughput, and it is what makes the borrowed surface safe: no other decode can recycle
  // the buffer between decodeToFd returning and our D2D copy completing.
  int fd = -1;
  unsigned dw = 0, dh = 0, pf = 0;
  const void *d_y = nullptr;
  size_t pitch = 0;
  {
    std::lock_guard<std::mutex> dk(s.decode_m);
    if (s.f_decode(s.h, cam_id, (unsigned char *)jpg, (unsigned long)n, &fd, &dw, &dh, &pf) != 0) {
      s.n_fail++;
      std::lock_guard<std::mutex> lk(s.m);
      s.slot[cam_id][r].state = 0;
      return nullptr;
    }
    if (pf != V4L2_PIX_FMT_YUV420M_FOURCC) {
      std::fprintf(stderr, "[nvjpg]: FATAL unexpected pixfmt 0x%x (want YM12)\n", pf);
      std::exit(EXIT_FAILURE);
    }
    {
      std::lock_guard<std::mutex> lk(s.m);
      if (!egl_lookup(s, fd, &d_y, &pitch)) {
        s.n_fail++;
        s.slot[cam_id][r].state = 0;
        return nullptr;
      }
      Slot &sl0 = s.slot[cam_id][r];
      if (!sl0.d_own) {
        if (cudaMallocPitch(&sl0.d_own, &sl0.p_own, (size_t)dw, (size_t)dh) != cudaSuccess) {
          std::fprintf(stderr, "[nvjpg]: FATAL cudaMallocPitch\n");
          std::exit(EXIT_FAILURE);
        }
      }
      cudaError_t ce = cudaMemcpy2DAsync(sl0.d_own, sl0.p_own, d_y, pitch, (size_t)dw, (size_t)dh,
                                         cudaMemcpyDeviceToDevice, s.cpy);
      if (ce == cudaSuccess) ce = cudaStreamSynchronize(s.cpy);
      if (ce != cudaSuccess) {
        std::fprintf(stderr, "[nvjpg]: FATAL D2D copy-out: %s\n", cudaGetErrorString(ce));
        std::exit(EXIT_FAILURE);
      }
      d_y = sl0.d_own;
      pitch = sl0.p_own;
    }
  }

  std::lock_guard<std::mutex> lk(s.m);
  if (!s.page) {
    s.pw = (int)dw;
    s.ph = (int)dh;
    s.page_bytes = (size_t)dw * dh;
    s.page = (unsigned char *)std::calloc(1, s.page_bytes);
  }
  Slot &sl = s.slot[cam_id][r];
  sl.fd = fd;
  sl.d_y = d_y;
  sl.pitch = pitch;
  sl.w = (int)dw;
  sl.h = (int)dh;
  sl.ts = ts;
  sl.state = 1;
  s.stage[{cam_id, ts}] = cam_id * s.ring + r;
  s.n_stage++;
  if (w) *w = (int)dw;
  if (h) *h = (int)dh;
  if (step) *step = (size_t)dw;
  return s.page;
}

bool ov_nvjpg_take(int cam_id, double ts, const void **d_y, size_t *pitch, int *w, int *h, int *slot) {
  State &s = S();
  if (!s.on) return false;
  std::lock_guard<std::mutex> lk(s.m);
  auto it = s.stage.find({cam_id, ts});
  if (it == s.stage.end()) {
    s.n_miss++;
    return false;
  }
  int gid = it->second;
  Slot &sl = s.slot[gid / s.ring][gid % s.ring];
  *d_y = sl.d_y;
  *pitch = sl.pitch;
  *w = sl.w;
  *h = sl.h;
  *slot = gid;
  sl.state = 2;
  sl.ev_recorded = false;
  s.stage.erase(it);
  s.n_take++;
  return true;
}

void ov_nvjpg_mark_consumed(int slot, void *cuda_stream) {
  State &s = S();
  if (!s.on || slot < 0) return;
  std::lock_guard<std::mutex> lk(s.m);
  Slot &sl = s.slot[slot / s.ring][slot % s.ring];
  cudaEventRecord(sl.ev, (cudaStream_t)cuda_stream);
  sl.ev_recorded = true;
}

void ov_nvjpg_watermark(double ts) {
  State &s = S();
  if (!s.on) return;
  std::lock_guard<std::mutex> lk(s.m);
  if (ts > s.watermark) s.watermark = ts;
}

bool ov_nvjpg_readback(int slot, unsigned char *dst) {
  State &s = S();
  if (!s.on || slot < 0) return false;
  Slot sl;
  {
    std::lock_guard<std::mutex> lk(s.m);
    sl = s.slot[slot / s.ring][slot % s.ring];
  }
  return cudaMemcpy2D(dst, (size_t)sl.w, sl.d_y, sl.pitch, (size_t)sl.w, (size_t)sl.h,
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

// DIAGNOSTIC: read the SAME surface through NvBufSurfaceMap (CPU mapping) instead of CUDA.
// If this matches libjpeg and the CUDA read does not, the surface is right and the CUDA view
// is stale; if both differ, the decoder put another frame in it.
bool ov_nvjpg_readback_cpu_ts(int cam_id, double ts, unsigned char *dst, int *w, int *h) {
  State &s = S();
  if (!s.on) return false;
  int fd = -1, ww = 0, hh = 0;
  NvBufSurface *surf = nullptr;
  {
    std::lock_guard<std::mutex> lk(s.m);
    auto it = s.stage.find({cam_id, ts});
    if (it == s.stage.end()) return false;
    Slot &sl = s.slot[it->second / s.ring][it->second % s.ring];
    fd = sl.fd; ww = sl.w; hh = sl.h;
    auto e = s.egl.find(fd);
    if (e == s.egl.end()) return false;
    surf = e->second.surf;
  }
  if (w) *w = ww;
  if (h) *h = hh;
  if (!dst || !surf) return false;
  return false; // copy-out path: the borrowed surface is no longer the source of truth
  if (NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_READ) != 0) return false;
  NvBufSurfaceSyncForCpu(surf, 0, 0);
  const unsigned char *src = (const unsigned char *)surf->surfaceList[0].mappedAddr.addr[0];
  unsigned int sp = surf->surfaceList[0].planeParams.pitch[0];
  for (int y = 0; y < hh; y++) std::memcpy(dst + (size_t)y * ww, src + (size_t)y * sp, (size_t)ww);
  NvBufSurfaceUnMap(surf, 0, 0);
  return true;
}

bool ov_nvjpg_readback_ts(int cam_id, double ts, unsigned char *dst, int *w, int *h) {
  State &s = S();
  if (!s.on) return false;
  Slot sl;
  {
    std::lock_guard<std::mutex> lk(s.m);
    auto it = s.stage.find({cam_id, ts});
    if (it == s.stage.end()) return false;
    sl = s.slot[it->second / s.ring][it->second % s.ring];
  }
  if (w) *w = sl.w;
  if (h) *h = sl.h;
  return cudaMemcpy2D(dst, (size_t)sl.w, sl.d_y, sl.pitch, (size_t)sl.w, (size_t)sl.h,
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool ov_nvjpg_selftest(const unsigned char *jpg, size_t n) {
  State &s = S();
  if (!s.on) return false;
  // P3: 1000x cv::imdecode AFTER the dlopen -- the exact thing the record says segfaults.
  //     Done by the CALLER (this TU must not include OpenCV).  Here: P4.
  int okc = 0;
  for (int i = 0; i < 1000; i++) {
    int w = 0, h = 0;
    size_t st = 0;
    const unsigned char *p = ov_nvjpg_stage(0, -1e17 + i, jpg, n, &w, &h, &st);
    if (p && w == 1280 && h == 800) okc++;
    // release immediately: nothing consumes these
    std::lock_guard<std::mutex> lk(s.m);
    for (int r = 0; r < s.ring; r++)
      if (s.slot[0][r].state == 1) { s.stage.erase({0, s.slot[0][r].ts}); s.slot[0][r].state = 0; }
  }
  std::fprintf(stderr, "[nvjpg]: P4 ovnv_decode 1000x -> %d/1000 fd>=0 pixfmt=YM12 1280x800\n", okc);
  {
    std::lock_guard<std::mutex> lk(s.m);
    std::fprintf(stderr, "[nvjpg]: P4 distinct dmabuf fds cached = %zu (expect ring=%d for cam0)\n",
                 s.egl.size(), s.ring);
  }
  s.n_stage = 0;
  s.n_block = 0;
  s.n_abandon = 0;
  s.block_ms = 0;
  return okc == 1000;
}

std::string ov_nvjpg_stats() {
  State &s = S();
  if (!s.on) return "[nvjpg]: off";
  std::lock_guard<std::mutex> lk(s.m);
  char b[512];
  std::snprintf(b, sizeof(b),
                "[nvjpg]: on ring=%d lookahead=%d acq_ms=%.0f bufs_registered=%zu staged=%ld "
                "taken=%ld miss=%ld ring_blocks=%ld block_ms=%.2f max_block_ms=%.2f "
                "cpu_fallback=%ld abandoned=%ld decode_fail=%ld distinct_bufs=%zu",
                s.ring, s.lookahead, s.acq_ms, s.egl.size(), (long)s.n_stage, (long)s.n_take,
                (long)s.n_miss, (long)s.n_block, s.block_ms, s.max_block_ms,
                (long)s.n_acqfail, (long)s.n_abandon, (long)s.n_fail, s.n_surf_ids.size());
  return std::string(b);
}

} // namespace ov_core
