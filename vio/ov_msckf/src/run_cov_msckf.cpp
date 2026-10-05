/*
 * Deterministic OFFLINE OpenVINS runner.
 *
 * Reads a rosbag2 recording directly and feeds IMU + camera measurements into
 * VioManager *synchronously, in timestamp order, in the main thread*. There is
 * no rclcpp executor, no real-time bag playback, and no async update thread.
 * Given a fixed OV_RNG_SEED this produces bit-identical results across runs,
 * eliminating the message-timing / frame-dropping variance of the live
 * subscribe path (`ros2 bag play | run_subscribe_msckf`).
 *
 * The recording is opened through BagSource (utils/bag_source.h), which merges
 * one or more files into a single timestamp-ordered stream and auto-detects
 * sqlite3 vs mcap. Its defaults are the study layout -- a single sqlite3 bag with
 * /imu0 and /cam<N> as sensor_msgs/Imu and sensor_msgs/Image -- so an unmodified
 * config behaves exactly as it always has. Topics and message types are config
 * keys; see bag_source.h.
 *
 * Usage:
 *   ros2 run ov_msckf run_serial_msckf <config.yaml> <bag[,bag2...]> <out_tum> \
 *        --ros-args -p use_stereo:=true -p max_cameras:=4
 *
 * Output: TUM trajectory file (t x y z qx qy qz qw), one pose per camera frame,
 * matching exactly what /ov_msckf/odomimu would publish (fast_state_propagate to
 * the frame timestamp, gated by initialized() && (t - init_time) >= 1).
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>

#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "core/VioManagerOptions.h" // pulls in ov_core::YamlParser + Printer
#include "utils/bag_source.h"       // merged multi-file reader + deferred decode
#include "utils/dataset_reader.h"
#include <future>
#include <deque>
#include <condition_variable>
#include <thread>
#include <malloc.h>
#include <execinfo.h>
#include <csignal>
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <cstring>
#include <omp.h>
#include <Eigen/Core>   // pulls in ov_core::CameraData / ImuData
#include "state/State.h"
#include "state/Propagator.h"
#include "track/nvjpg_decode.h"

namespace ov_msckf {
// Defined in VioManager.cpp: per-stage wall clock inside feed_measurement_camera.
extern std::map<std::string, double> g_vio_stage_secs;
extern double g_dbg_track_ms, g_dbg_upd_ms;
extern int g_dbg_upd_n;
extern std::map<std::string, long> g_vio_stage_calls;
} // namespace ov_msckf

// OV_TRK_DUMP=<path>: per-frame tracking attribution, written to its OWN file so the
// OV_LAT_DUMP 7-/9-column layouts stay byte-identical (r8sum.py keys off len(cols)>=9 and
// r5qsim.py off p[:6]; appending there would silently corrupt both).
namespace ov_core {
extern std::atomic<double> g_trk_det_ms, g_trk_klt_ms, g_trk_rsc_ms, g_trk_und_ms, g_trk_db_ms;
extern std::atomic<long> g_trk_npts, g_trk_ndet, g_trk_npts_new, g_trk_ncam;
} // namespace ov_core

using namespace ov_msckf;

// ---------------------------------------------------------------------------
// CPU partitioning / thread affinity / RT priority, and a dedicated JPEG-decode
// pool. Three INDEPENDENT env gates, all default OFF:
//   OV_DECODE_POOL=1      persistent decode threads, off the shared OpenCV TBB arena
//   OV_AFFINITY="0-3:4-7" "<critical cpu list>:<auxiliary cpu list>"
//   OV_RT_PRIO=<n>        SCHED_RR priority n on the critical (inheriting) threads
// Not one line of estimator arithmetic is touched: only WHICH core runs what, and
// which thread calls cv::imdecode. OV_DECODE_VERIFY=1 checks that in-binary.
// ---------------------------------------------------------------------------
static cpu_set_t g_cpu_crit, g_cpu_aux;
static bool g_aff_on = false;
// OV_UPD_CORES / OV_UPD_THREADS: widen the OpenMP team that runs the FILTER UPDATE beyond the
// critical cluster.  Default UNSET == today's behaviour byte-for-byte (the pool is never created,
// no thread's affinity or nice value changes).  libgomp caches its worker pool per team-MASTER and
// the workers keep whatever affinity they had at creation time, so the mask has to be handed to
// them once, from the thread that will later run the update -- see the warm-up in main().
static cpu_set_t g_cpu_upd;
static bool g_upd_wide = false;
static int g_upd_threads = 8;
static int g_upd_nice = 5; // OV_UPD_NICE: nice value of the team members that sit past the critical set
static bool g_rt_on = false;
// OV_DEC_PRIO: decode-window scheduling. 0/unset = today's behavior exactly.
//   1 = decode-pool threads are NOT niced down (they are the only latency-critical
//       work in the process while the consumer blocks in fr.fut.get()).
//   2 = 1, plus the producer is left on the FULL cpu mask but niced DOWN to +5, so its
//       ~3 ms-cadence IMU deserializes can use the idle critical cores during the decode
//       window instead of preempting a decoder, while never outweighing tracking.
// Thread placement / CFS weight only: no record content, no record ORDER, no estimator
// input is touched. Proven in-binary with OV_DECODE_VERIFY=1.
static int g_dec_prio = 0;

static bool ov_parse_cpulist(const char *s, cpu_set_t &m) { // "0-3" / "0,2,4" / "5"
  CPU_ZERO(&m);
  int a = -1, b = -1;
  bool any = false;
  while (*s) {
    char *e;
    long v = strtol(s, &e, 10);
    if (e == s) break;
    s = e;
    if (a < 0) a = b = (int)v;
    else b = (int)v;
    if (*s == '-') { s++; continue; }
    for (int c = std::min(a, b); c <= std::max(a, b); c++) {
      if (c >= 0 && c < CPU_SETSIZE) { CPU_SET(c, &m); any = true; }
    }
    a = b = -1;
    if (*s == ',') s++;
  }
  return any;
}

// Call ONCE as the first statement of main(): glibc's pthread_attr_t defaults to
// PTHREAD_INHERIT_SCHED and children inherit the affinity mask, so every thread the
// process later creates (TBB workers, GOMP teams, CUDA driver threads) inherits the
// critical mask/policy for free. Only threads we explicitly demote leave the set.
static void ov_sched_init() {
  const char *e = std::getenv("OV_AFFINITY");
  if (e && *e) {
    std::string s(e);
    size_t c = s.find(':');
    if (c != std::string::npos && ov_parse_cpulist(s.substr(0, c).c_str(), g_cpu_crit) &&
        ov_parse_cpulist(s.substr(c + 1).c_str(), g_cpu_aux)) {
      g_aff_on = (sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_crit) == 0);
      std::fprintf(stderr, "[sched]: critical=%s aux=%s ok=%d\n", s.substr(0, c).c_str(),
                   s.substr(c + 1).c_str(), (int)g_aff_on);
    } else {
      std::fprintf(stderr, "[sched]: OV_AFFINITY='%s' unparsable, ignored\n", e);
    }
  }
  // Wide-update mask.  Only meaningful when OV_AFFINITY is on -- with no affinity the process is
  // already free to use every core and there is nothing to widen.
  if (g_aff_on) {
    CPU_ZERO(&g_cpu_upd);
    g_cpu_upd = g_cpu_crit;
    if (const char *u = std::getenv("OV_UPD_CORES")) {
      cpu_set_t m;
      if (ov_parse_cpulist(u, m)) {
        g_cpu_upd = m;
        g_upd_wide = true;
      } else {
        std::fprintf(stderr, "[sched]: OV_UPD_CORES='%s' unparsable, ignored\n", u);
      }
    }
    if (const char *t = std::getenv("OV_UPD_THREADS")) {
      int v = atoi(t);
      if (v > 0) g_upd_threads = v;
    }
    if (const char *n = std::getenv("OV_UPD_NICE")) g_upd_nice = atoi(n);
  }
  if (const char *p = std::getenv("OV_RT_PRIO")) {
    int pr = atoi(p);
    if (pr > 0) {
      struct sched_param sp {};
      sp.sched_priority = pr;
      int rc = sched_setscheduler(0, SCHED_RR, &sp);
      if (rc == 0) mlockall(MCL_CURRENT | MCL_FUTURE);
      g_rt_on = (rc == 0);
      std::fprintf(stderr, "[sched]: SCHED_RR prio=%d rc=%d\n", pr, rc);
    }
  }
}

// Demote an already-created helper thread to the auxiliary cluster + SCHED_OTHER.
static void ov_sched_aux(pthread_t t) {
  if (!g_aff_on && !g_rt_on) return; // all gates off: leave the thread exactly as it was
  if (g_aff_on) pthread_setaffinity_np(t, sizeof(cpu_set_t), &g_cpu_aux);
  struct sched_param sp {};
  sp.sched_priority = 0;
  pthread_setschedparam(t, SCHED_OTHER, &sp);
}

// Same, from inside the thread itself (also sheds an inherited SCHED_RR policy).
static void ov_sched_aux_self(bool nice_down = true) {
  if (g_aff_on) sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_aux);
  struct sched_param sp {};
  sp.sched_priority = 0;
  pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
  if (nice_down) setpriority(PRIO_PROCESS, 0, 5); // PRIO_PROCESS + 0 == calling THREAD on Linux
}


// ---- OV_NVJPG bookkeeping -------------------------------------------------------------
static bool g_nvjpg = false;          // hardware decode active
// R18_RING_FIX: ONE definition of the reader-ahead depth.  The NVJPG ring is sized from
// this number at init (ov_nvjpg_set_lookahead), so the two can no longer drift apart --
// the shipped pairing ring(3) == MAX_FRAMES_AHEAD(3) is exactly how the deadlock became
// reachable, and it was reachable because the two constants were written independently.
static const int OV_MAX_FRAMES_AHEAD = 3;
// R18_READER -- WHY OV_PREFETCH IS NOT BEHAVIOUR-NEUTRAL.  Census only, no behaviour change.
//
// The two readers do NOT deliver the same frame-sets, and the difference is structural:
//   OV_PREFETCH=1 (THE VEHICLE): the producer assembles camera payloads in its own `stash`
//     and pushes a frame into `raq` ONLY when payloads.size() == ncam.  `pending` therefore
//     only ever holds COMPLETE 4-camera frame-sets.
//   OV_PREFETCH=0 (the round-16 oracle): camera records go straight into `pending` as they
//     are read, and flush_ready() feeds `pending.begin()` as soon as `latest_imu_t` passes
//     its stamp -- WITHOUT CHECKING THAT ALL FOUR CAMERAS HAVE ARRIVED.  One IMU record
//     interleaved between the 2nd and 3rd camera payload of a frame is enough to feed a
//     PARTIAL frame-set, and with OV_GROUP_CAMS=1 the tracker then simply tracks whichever
//     cameras were present.
// That is a different measurement stream, not a different schedule, which is why the
// trajectories differ deterministically.  These counters MEASURE it instead of arguing it.
static long g_rd_frames_fed = 0;      // frame-sets handed to feed_frame()
static long g_rd_partial_fed = 0;     // ... of which had < ncam camera payloads
static long g_rd_partial_imgs = 0;    // camera images missing across those frame-sets
static bool g_nvjpg_verify = false;   // OV_NVJPG_VERIFY=1 pixel A/B against cv::imdecode
static long g_nv_imgs = 0, g_nv_bitexact = 0, g_nv_maxdiff = 0;
static double g_nv_absdiff_sum = 0.0; static long g_nv_pix = 0;
static long g_nv_d1 = 0, g_nv_d2 = 0, g_nv_d3 = 0;      // #pixels |d|>=1,>=2,>=3
static long g_nv_gmin = 255, g_nv_gmax = 0, g_nv_hmin = 255, g_nv_hmax = 0;
static double g_nv_gmean = 0, g_nv_hmean = 0;
static long g_nv_fast_inter = 0, g_nv_fast_union = 0, g_nv_fast_frames = 0;
static long g_nv_cpumap_bitexact = 0, g_nv_cpumap_maxdiff = 0, g_nv_cpumap_n = 0;
static long g_nv_cam_be[8] = {0}, g_nv_cam_n[8] = {0};
static std::mutex g_nv_vmtx;
// ---- OV_DEC_CPU=1 (S0b): true CPU-seconds burned inside CamPayload::decode() ----------
// The [timing] "jpeg decode" row is the harvest WAIT, not CPU. This is CPU.
static bool g_dec_cpu = false;
static std::atomic<long long> g_dec_cpu_ns{0};
static std::atomic<long> g_dec_cpu_calls{0};
static inline long long thread_cpu_ns() {
  struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

// ---------------------------------------------------------------------------------------
// OV_NVJPG_VERIFY=1 -- PIXEL CHARACTERISATION, in ONE process, against the exact reference
// the gate-OFF path uses (cv::imdecode IMREAD_GRAYSCALE).  Structurally a clone of the
// OV_DECODE_VERIFY block.  The IDCT difference only reaches ATE through feature SELECTION,
// so the FAST corner-set Jaccard is reported alongside the raw pixel deltas -- that is the
// number that predicts risk, and a good pixel result does NOT clear the ATE gate.
// ---------------------------------------------------------------------------------------
static void nvjpg_verify_one(int cam_id, const CamPayload &pl) {
  int w = 0, h = 0;
  std::vector<unsigned char> gy;
  {
    int ww = 0, hh = 0;
    if (!ov_core::ov_nvjpg_readback_ts(cam_id, pl.ts, nullptr, &ww, &hh)) {
      // first call just learns the geometry; do the real read below
    }
    w = ww; h = hh;
    if (w <= 0 || h <= 0) return;
    gy.resize((size_t)w * h);
    if (!ov_core::ov_nvjpg_readback_ts(cam_id, pl.ts, gy.data(), &ww, &hh)) return;
  }
  cv::Mat ref = cv::imdecode(pl.data, cv::IMREAD_GRAYSCALE);
  if (ref.empty() || ref.cols != w || ref.rows != h) return;
  // DIAGNOSTIC: the same surface, read through the CPU mapping instead of CUDA.
  {
    std::vector<unsigned char> cy((size_t)w * h);
    int ww = 0, hh = 0;
    if (ov_core::ov_nvjpg_readback_cpu_ts(cam_id, pl.ts, cy.data(), &ww, &hh)) {
      long cmd = 0;
      for (int r = 0; r < h; r++) {
        const unsigned char *pb = ref.ptr<unsigned char>(r);
        for (int x = 0; x < w; x++) {
          long d = std::labs((long)cy[(size_t)r * w + x] - (long)pb[x]);
          if (d > cmd) cmd = d;
        }
      }
      std::lock_guard<std::mutex> lk(g_nv_vmtx);
      g_nv_cpumap_n++;
      if (cmd == 0) g_nv_cpumap_bitexact++;
      if (cmd > g_nv_cpumap_maxdiff) g_nv_cpumap_maxdiff = cmd;
    }
  }
  cv::Mat gpu((int)h, (int)w, CV_8UC1, gy.data());
  long md = 0, d1 = 0, d2 = 0, d3 = 0;
  double asum = 0, gsum = 0, hsum = 0;
  long gmin = 255, gmax = 0, hmin = 255, hmax = 0;
  for (int r = 0; r < h; r++) {
    const unsigned char *pa = gpu.ptr<unsigned char>(r), *pb = ref.ptr<unsigned char>(r);
    for (int x = 0; x < w; x++) {
      long d = std::labs((long)pa[x] - (long)pb[x]);
      if (d > md) md = d;
      if (d >= 1) d1++;
      if (d >= 2) d2++;
      if (d >= 3) d3++;
      asum += (double)d;
      gsum += pa[x]; hsum += pb[x];
      if (pa[x] < gmin) gmin = pa[x];
      if (pa[x] > gmax) gmax = pa[x];
      if (pb[x] < hmin) hmin = pb[x];
      if (pb[x] > hmax) hmax = pb[x];
    }
  }
  // FAST corner-set Jaccard: the selection-level metric.
  std::vector<cv::KeyPoint> ka, kb;
  cv::FAST(gpu, ka, 20, true);
  cv::FAST(ref, kb, 20, true);
  std::set<std::pair<int, int>> A, B;
  for (auto &k : ka) A.insert({(int)k.pt.x, (int)k.pt.y});
  for (auto &k : kb) B.insert({(int)k.pt.x, (int)k.pt.y});
  long inter = 0;
  for (auto &e : A) if (B.count(e)) inter++;
  long uni = (long)A.size() + (long)B.size() - inter;
  std::lock_guard<std::mutex> lk(g_nv_vmtx);
  g_nv_imgs++;
  g_nv_pix += (long)w * h;
  if (md == 0) g_nv_bitexact++;
  if (cam_id >= 0 && cam_id < 8) { g_nv_cam_n[cam_id]++; if (md == 0) g_nv_cam_be[cam_id]++; }
  if (md > g_nv_maxdiff) g_nv_maxdiff = md;
  g_nv_d1 += d1; g_nv_d2 += d2; g_nv_d3 += d3;
  g_nv_absdiff_sum += asum;
  g_nv_gmean += gsum / ((double)w * h);
  g_nv_hmean += hsum / ((double)w * h);
  if (gmin < g_nv_gmin) g_nv_gmin = gmin;
  if (gmax > g_nv_gmax) g_nv_gmax = gmax;
  if (hmin < g_nv_hmin) g_nv_hmin = hmin;
  if (hmax > g_nv_hmax) g_nv_hmax = hmax;
  g_nv_fast_inter += inter; g_nv_fast_union += uni; g_nv_fast_frames++;
}

// One persistent thread per camera, pinned one-per-aux-core. Jobs are opaque
// std::function<void()>; the pool knows nothing about images.
struct OvDecodePool {
  std::vector<std::thread> th;
  std::deque<std::function<void()>> q;
  std::mutex m;
  std::condition_variable cv_;
  bool stop = false;
  explicit OvDecodePool(int n) {
    for (int i = 0; i < n; i++)
      th.emplace_back([this, i] {
        ov_sched_aux_self(g_dec_prio < 1); // OV_DEC_PRIO>=1: stay at nice 0
        if (g_aff_on) {
          cpu_set_t one;
          CPU_ZERO(&one);
          int k = 0;
          for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &g_cpu_aux) && k++ == i) { CPU_SET(c, &one); break; }
          if (CPU_COUNT(&one)) pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
        }
        for (;;) {
          std::function<void()> j;
          {
            std::unique_lock<std::mutex> lk(m);
            cv_.wait(lk, [&] { return stop || !q.empty(); });
            if (stop && q.empty()) return;
            j = std::move(q.front());
            q.pop_front();
          }
          j();
        }
      });
  }
  void post(std::function<void()> j) {
    { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(j)); }
    cv_.notify_one();
  }
  ~OvDecodePool() {
    { std::lock_guard<std::mutex> lk(m); stop = true; }
    cv_.notify_all();
    for (auto &t : th) t.join();
  }
};
static std::unique_ptr<OvDecodePool> g_decpool;
static bool g_decode_verify = false;
// ---------------------------------------------------------------------------------------
// OV_ASYNC_EMIT (round 6): DECOUPLED POSE EMISSION.
// The pose is emitted from the PRODUCER thread the instant a camera frame-set is complete, by
// IMU-propagating the newest COMPLETED filter update (an immutable ov_msckf::PoseSnap) forward
// to that frame's stamp.  The estimator is never joined on the pose path.
//
// THIS CHANGES WHAT THE EMITTED POSE IS, and both quantities are dumped so neither can be
// quoted as the other:
//   <out>.tum        joined stream, filter stamps  -- x_hat(T) exactly as today
//   <out>.async.tum  decoupled stream, per-frame stamps -- IMU-propagate(x_hat(t_k) -> t_N)
// Extra per-frame columns land in OV_LAT_DUMP: emit_lat_ms (arrival -> decoupled emission) and
// corr_age_ms (t_N - t_k, the age of the newest correction inside the emitted pose).
//
// The emitter owns its OWN copy of the IMU stream (fed on the producer thread, read on the
// producer thread, never shared) so the filter's IMU path is byte-untouched and there is no
// lock on the emit path other than the snapshot handoff.
// OV_PIPELINE (round 20): the EKF update of frame N runs on a worker while frame N+1 is
// tracked.  The bench-side consequence is the pose-output path: nothing may read the live
// state right after feed_measurement_camera returns.  Mirrors VioManager::pipeline_on().
static bool g_pipeline = false;
static bool g_async_emit = false;
static bool g_async_dual = false;
static bool g_async_verify = false;
static long g_av_n = 0, g_av_ok = 0;
static double g_av_max = -1.0;
static long g_ae_emitted = 0, g_ae_nosnap = 0, g_ae_noimu = 0;
static long g_dv_frames = 0, g_dv_bitexact = 0, g_dv_imgs = 0, g_dv_maxdiff = 0, g_dv_mismatch = 0;


// Coarse wall-clock accounting for the whole run. OpenVINS' own
// record_timing_information covers only the frames where an EKF update fires
// (update_min_dt gates it to 7.5 Hz while KLT tracks every frame), so it explains
// well under half of a pass. These timers close the gap: every second between
// process start and exit lands in exactly one bucket.
namespace {
struct Stopwatch {
  std::map<std::string, double> total;
  std::map<std::string, long> count;
  void add(const std::string &k, double secs) { total[k] += secs; count[k] += 1; }
  void report(double wall) const {
    double acc = 0;
    for (auto const &kv : total) acc += kv.second;
    PRINT_INFO(GREEN "\n[timing]: %-26s %8s %10s %9s\n" RESET, "stage", "sec", "calls", "share");
    for (auto const &kv : total)
      PRINT_INFO(GREEN "[timing]: %-26s %8.2f %10ld %8.1f%%\n" RESET, kv.first.c_str(), kv.second,
                 count.at(kv.first), 100.0 * kv.second / wall);
    PRINT_INFO(GREEN "[timing]: %-26s %8.2f %10s %8.1f%%\n" RESET, "-- accounted --", acc, "", 100.0 * acc / wall);
    PRINT_INFO(GREEN "[timing]: %-26s %8.2f %10s %8.1f%%\n" RESET, "-- unaccounted --", wall - acc, "",
               100.0 * (wall - acc) / wall);
    PRINT_INFO(GREEN "[timing]: %-26s %8.2f\n" RESET, "WALL", wall);
  }
};
Stopwatch SW;
inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Scoped {
  const char *k; double t0;
  explicit Scoped(const char *key) : k(key), t0(now_s()) {}
  ~Scoped() { SW.add(k, now_s() - t0); }
};
} // namespace


// ---- ROUND 21 INSTRUMENT: dump the marginal 6x6 covariance of the IMU pose ---------------
// OV_COV_OUT=<file> -> one line per emitted pose:  t  P(0,0) .. P(5,5)  (row-major, 36 values)
// StateHelper ordering for _imu->pose() is [theta_GtoI (3) ; p_IinG (3)].
// Read-only w.r.t. the filter; fired from exactly the same points as the .tum line.
static FILE *g_covf = nullptr;
static bool  g_covf_tried = false;
static inline FILE *ov_covf() {
  if (!g_covf_tried) {
    g_covf_tried = true;
    const char *p = std::getenv("OV_COV_OUT");
    if (p && *p) {
      g_covf = std::fopen(p, "w");
      if (g_covf) {
        std::fprintf(g_covf, "# t_cam P[6x6] row-major, order [th_x th_y th_z px py pz]\n");
        std::atexit([]() { if (g_covf) { std::fflush(g_covf); std::fclose(g_covf); g_covf = nullptr; } });
      }
    }
  }
  return g_covf;
}

int main(int argc, char **argv) {
  // Any SIGSEGV/SIGABRT prints a raw backtrace to stderr before dying, so rare startup crashes
  // leave evidence even when nothing was attached and stdout buffering ate the log tail.
  // backtrace() is not strictly async-signal-safe, but on a crashing process it is the best
  // available and has nothing left to corrupt.
  struct CrashTrace {
    static void handler(int sig) {
      void *frames[48];
      int n = backtrace(frames, 48);
      char head[64];
      int m = snprintf(head, sizeof(head), "\n[crash]: signal %d, %d frames:\n", sig, n);
      ssize_t wr = write(2, head, m);
      backtrace_symbols_fd(frames, n, 2);
      (void)wr;
      signal(sig, SIG_DFL);
      raise(sig); // re-raise so the core file is still produced
    }
  };
  signal(SIGSEGV, CrashTrace::handler);
  signal(SIGABRT, CrashTrace::handler);
  signal(SIGBUS, CrashTrace::handler);

  // FIRST real statement of main(): everything created later inherits this.
  ov_sched_init();
  {
    // MUST be read before OvDecodePool spawns its workers -- they consult it in their prologue.
    const char *dpr = std::getenv("OV_DEC_PRIO");
    g_dec_prio = dpr ? atoi(dpr) : 0;
    if (g_dec_prio) std::fprintf(stderr, "[sched]: OV_DEC_PRIO=%d\n", g_dec_prio);
    const char *dp = std::getenv("OV_DECODE_POOL");
    if (dp && *dp == '1') {
      g_decpool = std::unique_ptr<OvDecodePool>(new OvDecodePool(4));
      std::fprintf(stderr, "[sched]: decode pool ON (4 threads)\n");
    }
    const char *dv = std::getenv("OV_DECODE_VERIFY");
    g_decode_verify = (dv && *dv == '1');
    const char *dc = std::getenv("OV_DEC_CPU");
    g_dec_cpu = (dc && *dc == '1');
    const char *nvv = std::getenv("OV_NVJPG_VERIFY");
    g_nvjpg_verify = (nvv && *nvv == '1');
  }
  {
    const char *e = std::getenv("OV_ASYNC_EMIT");
    g_async_emit = (e && *e == '1');
    const char *d = std::getenv("OV_ASYNC_EMIT_DUAL");
    g_async_dual = (d && *d == '1');
    const char *v = std::getenv("OV_ASYNC_VERIFY");
    g_async_verify = (v && *v == '1');
  }

  // Large Eigen temporaries (covariance grows past 128KB) otherwise go through mmap/munmap on
  // every alloc -- page-fault zeroing each time. Keep them on the heap for reuse.
  mallopt(M_MMAP_THRESHOLD, 512 * 1024 * 1024);
  mallopt(M_TRIM_THRESHOLD, 512 * 1024 * 1024);
  // Eigen splits even ~100x300 GEMMs across OMP threads; at our sizes the fork/join can cost
  // more than the math. OV_EIGEN_NT=1 pins Eigen's internal GEMM threading.
  if (const char *ent = std::getenv("OV_EIGEN_NT"))
    Eigen::setNbThreads(atoi(ent));
  // OV_UPD_CORES: create libgomp's worker pool HERE, on the thread that will drive the consumer
  // loop (main calls sys->feed_measurement_camera, which runs tracking then the update), and hand
  // every member the wide mask.  Widening the master afterwards does nothing: the workers are
  // cached in the master's thread pool with the affinity they were born with.  Members past the
  // critical set are niced DOWN so they can never outrank the nice+5 aux-cluster JPEG decoders,
  // whose latency is directly additive to the joined pose latency.
  if (g_upd_wide) {
    const int nt = g_upd_threads, ncrit = CPU_COUNT(&g_cpu_crit);
    int nfail = 0;
    volatile double sink = 0.0;
#pragma omp parallel num_threads(nt) reduction(+ : sink)
    {
      if (sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_upd) != 0) {
#pragma omp atomic
        nfail++;
      }
      if (omp_get_thread_num() >= ncrit && g_upd_nice > 0)
        setpriority(PRIO_PROCESS, 0, g_upd_nice); // PRIO_PROCESS + 0 == calling THREAD on Linux
      sink += omp_get_thread_num();
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_crit); // master back on the critical set
    std::fprintf(stderr, "[sched]: upd pool nt=%d got=%d wide=%s nice=%d fail=%d sink=%.0f\n", nt,
                 omp_get_max_threads(), std::getenv("OV_UPD_CORES"), g_upd_nice, nfail, (double)sink);
  }
  const double _wall0 = now_s();

  // Deterministic OpenCV RNG (findFundamentalMat RANSAC in TrackKLT). OV_RNG_SEED overrides.
  {
    const char *seed_env = std::getenv("OV_RNG_SEED");
    uint64_t seed = 42;
    if (seed_env && *seed_env) {
      try { seed = std::stoull(seed_env); } catch (...) {}
    }
    cv::theRNG().state = seed;
  }

  // Strip ROS args, keep positional: [exe, config, bag, out]
  auto pos = rclcpp::init_and_remove_ros_arguments(argc, argv);
  if (pos.size() < 4) {
    PRINT_ERROR(RED "usage: run_serial_msckf <config.yaml> <bag_dir> <out_tum> [--ros-args -p use_stereo:=.. -p max_cameras:=..]\n" RESET);
    return EXIT_FAILURE;
  }
  std::string config_path = pos[1];
  std::string bag_path = pos[2];
  std::string out_path = pos[3];

  // Node exists only so YamlParser can pick up -p use_stereo / -p max_cameras overrides
  // (identical to the launch path: "overriding node use_stereo with value from ROS!").
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("run_serial_msckf", options);

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node(node);

  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);

  // ---------------- OV_NVJPG startup: refusals, bind, self-test ------------------------
  // Every one of these four sites READS HOST PIXELS.  Under OV_NVJPG the host Mat is a
  // header over a blank shared page, so they would silently corrupt the run rather than
  // fail.  Refuse loudly at startup with a named reason instead.
  if (std::getenv("OV_NVJPG") && *std::getenv("OV_NVJPG") == '1') {
    const char *gt = std::getenv("OV_GPU_TRACK");
    ov_core::ov_nvjpg_refuse_if(params.downsample_cameras,
                                "downsample_cameras=true (VioManager pyrDown reads host pixels)");
    ov_core::ov_nvjpg_refuse_if(params.histogram_method == ov_core::TrackBase::HistogramMethod::HISTOGRAM,
                                "histogram_method=HISTOGRAM (cv::equalizeHist reads host pixels)");
    ov_core::ov_nvjpg_refuse_if(!gt || *gt != '1', "OV_GPU_TRACK!=1 (no device consumer)");
    ov_core::ov_nvjpg_refuse_if(params.use_stereo, "use_stereo=true (host stereo matching)");
    // R18_RING_FIX: MUST precede ov_nvjpg_enabled(), which is what runs init_once().
    ov_core::ov_nvjpg_set_lookahead(OV_MAX_FRAMES_AHEAD);
    g_nvjpg = ov_core::ov_nvjpg_enabled();
    if (g_nvjpg && std::getenv("OV_NVJPG_SELFTEST") && *std::getenv("OV_NVJPG_SELFTEST") == '1') {
      // P3/P4: prove IN-BINARY that cv::imdecode still works after the dlopen (the exact
      // thing the record says segfaults) and that the hardware path really bound.
      std::vector<unsigned char> jp;
      { std::ifstream f("/home/lis/jpgs/camA_00.jpg", std::ios::binary);
        jp.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }
      if (jp.empty()) { std::fprintf(stderr, "[nvjpg]: SELFTEST needs /home/lis/jpgs/camA_00.jpg\n"); return EXIT_FAILURE; }
      unsigned long long ck = 0; int nz = 0;
      for (int i = 0; i < 1000; i++) {
        cv::Mat m = cv::imdecode(jp, cv::IMREAD_GRAYSCALE);
        if (m.empty()) { std::fprintf(stderr, "[nvjpg]: FATAL P3 imdecode returned empty at i=%d\n", i); return EXIT_FAILURE; }
        unsigned long long c2 = 0;
        for (int r = 0; r < m.rows; r += 37) { const unsigned char *q = m.ptr<unsigned char>(r);
          for (int x = 0; x < m.cols; x += 11) c2 = c2 * 1099511628211ULL + q[x]; }
        if (i == 0) ck = c2; else if (c2 != ck) nz++;
      }
      std::fprintf(stderr, "[nvjpg]: P3 cv::imdecode 1000x AFTER dlopen -> no crash, checksum stable (%d drifts)\n", nz);
      if (!ov_core::ov_nvjpg_selftest(jp.data(), jp.size())) {
        std::fprintf(stderr, "[nvjpg]: FATAL P4 failed\n"); return EXIT_FAILURE;
      }
    }
  }
  params.use_multi_threading_subs = false; // serial / synchronous
  auto sys = std::make_shared<VioManager>(params);
  // Read the gate back from the MANAGER, not the environment: VioManager refuses the pipeline
  // (loudly, with a named reason) on any configuration it cannot make race-free, and the bench
  // must follow that decision rather than the env var.
  g_pipeline = sys->pipeline_on();

  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    return EXIT_FAILURE;
  }

  const int ncam = params.state_options.num_cameras;
  const bool use_stereo = params.use_stereo;
  const bool use_mask = params.use_mask;
  PRINT_INFO(GREEN "[serial]: ncam=%d use_stereo=%d use_mask=%d bag=%s\n" RESET, ncam, (int)use_stereo, (int)use_mask, bag_path.c_str());

  // Per-cam startup masks (auto fisheye disk masks live in params.masks). Cache
  // once; VioManager::maybe_refresh_fisheye_masks() overwrites message.masks in
  // place each frame as intrinsics drift, so the startup mask is just the seed.
  const std::map<size_t, cv::Mat> startup_masks = sys->get_params().masks;
  auto get_mask = [&](int cam_id, int rows, int cols) -> cv::Mat {
    if (use_mask) {
      auto it = startup_masks.find((size_t)cam_id);
      if (it != startup_masks.end() && !it->second.empty())
        return it->second;
    }
    return cv::Mat::zeros(rows, cols, CV_8UC1);
  };

  // ---- bag reader ----
  // The recording may be split across several files (swarm-nxt records IMU and
  // cameras separately), so bag_path accepts a comma-separated list of URIs.
  BagSourceOptions bopt;
  {
    size_t start = 0;
    while (start <= bag_path.size()) {
      size_t comma = bag_path.find(',', start);
      std::string uri = bag_path.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      if (!uri.empty())
        bopt.uris.push_back(uri);
      if (comma == std::string::npos)
        break;
      start = comma + 1;
    }
  }
  // Every key below defaults to the study layout, so an unmodified config reads a
  // single sqlite3 bag of /imu0 + /cam<N> exactly as this runner always has.
  parser->parse_config("imu_topic", bopt.imu_topic, false);
  parser->parse_config("imu_msg_type", bopt.imu_msg_type, false);
  parser->parse_config("cam_msg_type", bopt.cam_msg_type, false);
  for (int i = 0; i < ncam; i++) {
    std::string topic;
    parser->parse_config("cam_topic" + std::to_string(i), topic, false);
    if (!topic.empty())
      bopt.cam_topics[topic] = i;
  }
  // Optional trim. Calibration needs a static start plus ~10-15 s of motion, so a long
  // recording costs passes it does not need: runtime is ~(bag length) x (passes) x
  // 1-4x real time, and on an Orin that is ~4x per pass.
  double t_start = 0.0, t_end = -1.0;
  parser->parse_config("bag_t_start", t_start, false);
  parser->parse_config("bag_t_end", t_end, false);
  bopt.t_start = t_start;
  if (t_end > 0.0)
    bopt.t_end = t_end;
  BagSource source(bopt);

  // A fully-assembled camera frame. Payloads are still encoded here on purpose --
  // see the decode note in feed_frame.
  struct Frame {
    double ts;
    std::map<int, CamPayload> payloads; // cam_id -> undecoded image
    // decode prefetch (OV_PREFETCH=1): launched the moment the frame-set is complete in
    // `pending`, i.e. while the estimator is still busy with the PREVIOUS frame -- the same
    // overlap a live deployment gets from its driver thread. Payload references stay valid:
    // map nodes are stable and the entry is only erased after feed_frame consumed the future.
    std::shared_future<std::map<int, cv::Mat>> fut;
    bool has_fut = false;
    // OV_ASYNC_EMIT: filled by the PRODUCER at emission time, read by the consumer purely so
    // the per-frame latency dump carries both metrics on the same row.
    double emit_lat_ms = -1.0;  // arrival -> decoupled pose written
    double corr_age_ms = -1.0;  // frame stamp MINUS the stamp of the newest completed update
    long emit_seq = -1;         // snapshot publication counter used for this frame
    double arrive_wall = -1.0; // wall time the frame-set became complete ("arrived")
    // (stamped arrive_wall - paced release instant); lat_true == lat + pace_off, so BOTH the
    // old and the new latency definition are recoverable from ONE run's dump. 0 when not paced.
    double pace_off = 0.0;
  };
  std::map<double, Frame> pending; // keyed by stamp, sorted ascending
  double latest_imu_t = -1;

  // OV_REALTIME=1: release records at their sensor-clock rate against the wall clock, so the
  // run behaves like a live system (frames "arrive"; nothing can be touched early). Latency is
  // then measured per fed frame: arrival of the complete frame-set -> pose out.
  static const bool rt_pace = [] {
    // OV_DETERMINISTIC forces the paced reader OFF, and that is a deliberate concession, not an
    // oversight. Real-time pacing is the ONE nondeterminism source that is structural: the
    // OV_SPREAD_ADAPT sub-update budget reads g_frames_queued (a wall-clock backlog), OV_PRESEED
    // skips on g_frame_wait_ms, and the preseed readiness test is a wait_for(0s) poll. All three
    // exist to REACT to how fast the machine ran, so determinism and real-time reactivity are in
    // direct conflict there. Every one of those sites is already gated on rt_pace, so turning
    // pacing off collapses all of them with no further code change. The scoring mode gives up
    // the pacing; the estimator arithmetic and every other gate are untouched.
    const char *d = std::getenv("OV_DETERMINISTIC");
    if (d && *d == '1') return false;
    const char *e = std::getenv("OV_REALTIME");
    return e && *e == '1';
  }();
  double rt_t0_bag = -1.0, rt_t0_wall = -1.0;
  auto pace = [&](double tbag) {
    if (!rt_pace) return;
    if (rt_t0_bag < 0) { rt_t0_bag = tbag; rt_t0_wall = now_s(); return; }
    const double target = rt_t0_wall + (tbag - rt_t0_bag);
    const double now = now_s();
    if (target > now + 0.0002)
      std::this_thread::sleep_for(std::chrono::duration<double>(target - now));
  };
  std::vector<double> lat_s; // per-fed-frame latency, seconds
  lat_s.reserve(8192);
  // ---- MEASUREMENT-DEFINITION gates (they change the METRIC, never the system) ----
  // OV_LAT_TRUE=1 moves the latency ORIGIN from "4th camera payload deserialized" to the
  // sensor-paced RELEASE instant. It makes every reported number WORSE; that is the honest
  // sensor->pose latency. It must be on for BOTH arms of a comparison or for neither.
  static const bool g_lat_true = [] { const char *e = std::getenv("OV_LAT_TRUE"); return e && *e == '1'; }();
  // OV_PACE_PROBE=1 records (stamped arrive_wall - paced release instant) per frame-set, i.e.
  // the camera-record deserialize bias, WITHOUT changing which one is used for latency.
  static const bool g_pace_probe = [] { const char *e = std::getenv("OV_PACE_PROBE"); return e && *e == '1'; }();
  std::vector<double> pace_off_s; // producer-thread only; read after producer.join()
  pace_off_s.reserve(8192);

  std::vector<std::string> lines;
  lines.reserve(6000);
  // OV_ASYNC_EMIT: decoupled stream (producer thread only until producer.join()).
  std::vector<std::string> async_lines;
  std::vector<std::string> async_diag;
  if (g_async_emit) { async_lines.reserve(16384); if (g_async_dual) async_diag.reserve(16384); }
  double last_logged_ts = -1; // dedupe: with OV_UPDATE_MIN_DT decimation, skipped
                              // frames leave state->_timestamp unchanged

  // Harvest the online-calibration state at the end of the run (the tool reads this).
  auto write_calib_to = [&](FILE *cf) {
    auto st = sys->get_state();
    if (!cf) return;
    Eigen::Vector3d ba = st->_imu->bias_a(), bg = st->_imu->bias_g();
    double toff = st->_calib_dt_CAMtoIMU->value()(0);
    std::fprintf(cf, "{\"toff\":%.9f,\"ba\":[%.9f,%.9f,%.9f],\"bg\":[%.9f,%.9f,%.9f],\"cams\":{",
                 toff, ba(0), ba(1), ba(2), bg(0), bg(1), bg(2));
    bool first = true;
    for (auto const &kv : st->_cam_intrinsics) {
      int cid = kv.first;
      Eigen::VectorXd intr = kv.second->value();
      Eigen::Matrix3d R_CtoI = st->_calib_IMUtoCAM.at(cid)->Rot().transpose();
      Eigen::Vector3d p_CinI = -R_CtoI * st->_calib_IMUtoCAM.at(cid)->pos();
      if (!first) std::fprintf(cf, ",");
      first = false;
      std::fprintf(cf, "\"%d\":{\"intr\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
        "\"R_CtoI\":[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f],\"p_CinI\":[%.9f,%.9f,%.9f]}",
        cid, intr(0),intr(1),intr(2),intr(3),intr(4),intr(5),intr(6),intr(7),
        R_CtoI(0,0),R_CtoI(0,1),R_CtoI(0,2),R_CtoI(1,0),R_CtoI(1,1),R_CtoI(1,2),R_CtoI(2,0),R_CtoI(2,1),R_CtoI(2,2),
        p_CinI(0),p_CinI(1),p_CinI(2));
    }
    std::fprintf(cf, "}");
    {
      // IMU intrinsics state dump (kalibr model: Dw/Da lower-tri vecs, R_GYROtoIMU
      // estimated). Values equal the chain seed when calib_imu_intrinsics is off.
      Eigen::VectorXd dw = st->_calib_imu_dw->value();
      Eigen::VectorXd da = st->_calib_imu_da->value();
      Eigen::VectorXd tg = st->_calib_imu_tg->value();
      Eigen::VectorXd qg = st->_calib_imu_GYROtoIMU->value();
      Eigen::VectorXd qa = st->_calib_imu_ACCtoIMU->value();
      std::fprintf(cf,
        ",\"imu\":{\"dw\":[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f],"
        "\"da\":[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f],"
        "\"tg\":[%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f],"
        "\"q_GYROtoIMU\":[%.9f,%.9f,%.9f,%.9f],\"q_ACCtoIMU\":[%.9f,%.9f,%.9f,%.9f]}",
        dw(0),dw(1),dw(2),dw(3),dw(4),dw(5), da(0),da(1),da(2),da(3),da(4),da(5),
        tg(0),tg(1),tg(2),tg(3),tg(4),tg(5),tg(6),tg(7),tg(8),
        qg(0),qg(1),qg(2),qg(3), qa(0),qa(1),qa(2),qa(3));
    }
    std::fprintf(cf, "}");   // no newline: the series wraps this in an outer object
  };

  auto write_calib = [&](const std::string &path) {
    FILE *cf = std::fopen(path.c_str(), "w");
    if (!cf) return;
    write_calib_to(cf);
    std::fprintf(cf, "\n");
    std::fclose(cf);
  };

  // Periodic snapshots of the online-calibration state, one JSON object per line.
  //
  // The end-of-run harvest is a single sample, so the only convergence evidence the
  // tool has is agreement BETWEEN passes -- and that cannot tell "converged" from
  // "never moved". A run whose calibration stayed pinned at its seed reports a
  // suspiciously LOW self-consistency residual while the trajectory diverges; this
  // fleet produced exactly that (resid 0.0187 against a 14 km trajectory error).
  //
  // With a series, settling is measurable inside ONE pass: if the last few snapshots
  // oscillate around a stable mean, the calibration has converged, and a second pass
  // exists only to confirm what the series already shows.
  FILE *series = nullptr;
  double series_dt = 0.0, series_next = -1;
  parser->parse_config("calib_series_dt", series_dt, false);
  if (series_dt > 0.0)
    series = std::fopen((out_path + ".calib_series.jsonl").c_str(), "w");

  // ---------------- OV_PIPELINE pose sink ------------------------------------------------
  // Byte-for-byte the same logging block as the in-loop one, but fired by VioManager at the
  // points where the update worker is provably idle (post-drain / post-ZUPT / post-init)
  // instead of after feed_measurement_camera returns, where the worker is mid-update and
  // state->_imu is a torn read.  Same value, same state->_timestamp: the drain that fires it
  // is the exact instant the serial build would have finished the same sub-updates.
  auto log_pose_quiescent = [&]() {
    if (!sys->initialized())
      return;
    auto state = sys->get_state();
    if ((state->_timestamp - sys->initialized_time()) < 1.0)
      return;
    if (state->_timestamp == last_logged_ts)
      return;
    last_logged_ts = state->_timestamp;
    Eigen::Vector4d q = state->_imu->quat();
    Eigen::Vector3d p = state->_imu->pos();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f",
                  state->_timestamp, p(0), p(1), p(2), q(0), q(1), q(2), q(3));
    lines.emplace_back(buf);
    if (FILE *cf_ = ov_covf()) {
      std::vector<std::shared_ptr<ov_type::Type>> hord_;
      hord_.push_back(state->_imu->pose());
      Eigen::MatrixXd Pm_ = ov_msckf::StateHelper::get_marginal_covariance(state, hord_);
      std::fprintf(cf_, "%.9f", state->_timestamp);
      for (int r_ = 0; r_ < 6; r_++)
        for (int c_ = 0; c_ < 6; c_++) std::fprintf(cf_, " %.9g", Pm_(r_, c_));
      std::fprintf(cf_, "\n");
    }
    if (series != nullptr) {
      const double ts = state->_timestamp;
      if (series_next < 0)
        series_next = ts;
      if (ts >= series_next) {
        std::fprintf(series, "{\"t\":%.6f,\"calib\":", ts);
        write_calib_to(series);
        std::fprintf(series, "}\n");
        series_next = ts + series_dt;
      }
    }
  };
  if (g_pipeline) {
    sys->set_pose_sink(log_pose_quiescent);
    // OV_PUB_WHEN_READY: VioManager also fires the sink from the pipeline worker, at the end of each
    // update batch.  Same rows in the same order (the dedup on state->_timestamp is in the sink),
    // so <out>.tum is byte-identical with the gate on or off.
    std::fprintf(stderr, "[pipe]: bench pose output = %s (live-state read disabled)\n",
                 sys->pub_when_ready() ? "pipeline worker, when each update finishes (OV_PUB_WHEN_READY)" : "post-drain sink");
  }

  // track_frequency throttle (replicates ROS2Visualizer::callback_stereo lines
  // 554-558): per lead-camera, skip a frame whose timestamp is < last_tracked +
  // 1/track_frequency. With 30Hz cams and track_frequency=14.5 this tracks every
  // ~3rd frame -> ~100ms parallax baseline (the dominant accuracy lever per the
  // tuning history). track_frequency<=0 disables throttling (track every frame).
  std::map<int, double> cam_last_track;
  const double track_dt = (params.track_frequency > 1e-6) ? 1.0 / params.track_frequency : 0.0;

  // Feed one assembled frame's measurements, then log the pose once.
  //
  // Ordering here is load-bearing for speed: the throttle runs BEFORE any decode.
  // With track_frequency 29.0 against 30 Hz cameras roughly every other frame is
  // discarded, and decoding first would spend a full JPEG decode per camera on
  // frames that are then dropped. Only the groups that survive get decoded.
  // OV_DETERMINISTIC folds OV_PREFETCH OFF (round-25 follow-up to round 24, which had to pass
  // OV_PREFETCH=0 alongside the gate). Reader-ahead runs a PRODUCER thread that fills `pending`
  // and `raq` concurrently with the consumer, and the OV_PRESEED hook picks its frame out of
  // those same containers -- a producer race. A scoring mode that can be configured HALF-ON is
  // a trap, so the gate now owns every lever it needs. Proven behaviour-neutral in-binary: with
  // OV_PREFETCH=0 passed explicitly the trajectories are bit-identical either way.
  static const bool prefetch = [] {
    const char *d = std::getenv("OV_DETERMINISTIC");
    // OV_DETERM_PRESEED keeps the preseed hook alive under the oracle -- and the hook's frame
    // source is Frame::fut, which ONLY the reader-ahead producer ever sets. Without this the
    // arm would build an empty CameraData every time and prove nothing at all. So the preseed
    // bench gate implies the reader-ahead, and that implication is in the binary, not the
    // recipe. (Round 16 folded OV_PREFETCH into OV_DETERMINISTIC to kill exactly this class of
    // half-configurable trap; this is the same fold applied one level down.)
    const char *kp = std::getenv("OV_DETERM_PRESEED");
    const bool keep = (kp && (*kp == '1' || *kp == '2')) ||
                      [] { const char *k = std::getenv("OV_DETERM_PREFETCH"); return k && *k == '1'; }();
    if (d && *d == '1' && !keep) return false;
    const char *e = std::getenv("OV_PREFETCH"); return e && *e == '1'; }();
  // The decoupled emitter lives on the reader-ahead PRODUCER thread; without OV_PREFETCH there
  // is no such thread and emitting from the consumer would just re-serialise it behind the
  // estimator -- i.e. it would measure nothing. Refuse loudly rather than report a fake number.
  if (g_async_emit && !prefetch) {
    std::fprintf(stderr, "[asyncemit]: OV_ASYNC_EMIT requires OV_PREFETCH=1 (producer thread). DISABLED.\n");
    g_async_emit = false;
  }
  static double g_frame_wait_ms = 0.0;
  // ROUND 17 preseed census. Read from the run's own stderr; an arm whose preseed=0 is an arm
  // that did not test preseed.
  static long g_pre_calls = 0, g_pre_fired = 0, g_pre_noframe = 0, g_pre_notready = 0, g_pre_skip = 0;
  static struct PreseedDump {
    ~PreseedDump() {
      if (!g_pre_calls) return;
      std::fprintf(stderr, "[preseed]: calls=%ld fired=%ld no_frame=%ld decode_not_ready=%ld backlog_skip=%ld\n",
                   g_pre_calls, g_pre_fired, g_pre_noframe, g_pre_notready, g_pre_skip);
    }
  } g_preseed_dump;
  static std::atomic<int> g_fq_max{0}; // high-water mark of the arrival counter (verification)

  auto feed_frame_impl = [&](const Frame &fr, const std::vector<std::vector<int>> &groups_in,
                             std::map<int, cv::Mat> &imgs) {
    bool fed_any = false;

    const int rows = imgs.begin()->second.rows;
    const int cols = imgs.begin()->second.cols;
    // 3. Feed.
    for (auto const &g : groups_in) {
      bool complete = true;
      for (int c : g)
        if (!imgs.count(c)) complete = false;
      if (!complete) continue;
      ov_core::CameraData msg;
      msg.timestamp = fr.ts;
      for (int c : g) {
        msg.sensor_ids.push_back(c);
        msg.images.push_back(imgs.at(c));
        { Scoped _s("  build mask"); msg.masks.push_back(get_mask(c, rows, cols)); }
      }
      { Scoped _s("  feed_measurement_camera"); sys->feed_measurement_camera(msg); }
      fed_any = true;
    }
    // Log the filtered IMU state directly at the update time. (We do NOT use
    // fast_state_propagate(state, fr.ts): with a negative cam-imu timeoffset the
    // state update time is AHEAD of fr.ts, so propagating to fr.ts is backward
    // and fails. NOTE (2026-06-12, corrected): state->_timestamp is the CAMERA-
    // clock frame stamp (verified == bag /cam0 stamps); the physical/IMU-clock
    // instant of the pose is state->_timestamp + calib_dt (toff ~ -39 ms, in the
    // .calib.json). Downstream eval must shift stamps by toff before comparing
    // to GT — see FINAL/TIMESTAMP_CLOCK_RESULTS.md. The live node's poseimu
    // publisher already applies this (+t_ItoC in ROS2Visualizer::publish_state).)
    // OV_ASYNC_VERIFY=1: run the shipped fast_state_propagate AND the snapshot twin in this
    // one process on the same live state and the same IMU buffer, and record the max abs
    // difference.  This is the exactness proof for the propagation the emitter does; it says
    // nothing about the SEMANTIC change (which state is propagated) -- that is the pose-delta
    // measurement between <out>.tum and <out>.async.tum.
    if (g_async_verify && !g_pipeline && fed_any && sys->initialized()) {
      double d = -1.0;
      const int ok = sys->get_propagator()->fast_propagate_selfcheck(sys->get_state(), fr.ts, d);
      g_av_n++;
      if (ok) {
        g_av_ok++;
        if (d > g_av_max) g_av_max = d;
      }
    }
    // OV_PIPELINE: the update for THIS frame's tick is still running on the worker right now,
    // so `state` is being mutated and must not be read here (torn _imu / _Cov).  The pose is
    // logged instead from log_pose_quiescent(), which VioManager fires at the points where the
    // worker is provably idle -- same value, same state->_timestamp, one frame later in WALL
    // time only.  See sys->set_pose_sink() below.
    if (!g_pipeline && fed_any && sys->initialized() && (sys->get_state()->_timestamp - sys->initialized_time()) >= 1.0 &&
        sys->get_state()->_timestamp != last_logged_ts) {
      auto state = sys->get_state();
      last_logged_ts = state->_timestamp;
      Eigen::Vector4d q = state->_imu->quat(); // q_GtoI, JPL xyzw (matches odomimu)
      Eigen::Vector3d p = state->_imu->pos();  // p_IinG
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f",
                    state->_timestamp, p(0), p(1), p(2), q(0), q(1), q(2), q(3));
      lines.emplace_back(buf);
    if (FILE *cf_ = ov_covf()) {
      std::vector<std::shared_ptr<ov_type::Type>> hord_;
      hord_.push_back(state->_imu->pose());
      Eigen::MatrixXd Pm_ = ov_msckf::StateHelper::get_marginal_covariance(state, hord_);
      std::fprintf(cf_, "%.9f", state->_timestamp);
      for (int r_ = 0; r_ < 6; r_++)
        for (int c_ = 0; c_ < 6; c_++) std::fprintf(cf_, " %.9g", Pm_(r_, c_));
      std::fprintf(cf_, "\n");
    }
      if (series != nullptr) {
        const double ts = state->_timestamp;
        if (series_next < 0)
          series_next = ts;              // first snapshot as soon as the filter is up
        if (ts >= series_next) {
          std::fprintf(series, "{\"t\":%.6f,\"calib\":", ts);
          write_calib_to(series);
          std::fprintf(series, "}\n");   // closes the {"t":..,"calib":..} wrapper
          series_next = ts + series_dt;
        }
      }
    }
  };

  auto feed_frame = [&](const Frame &fr) {
    // OV_NVJPG lifetime: publish the consumer watermark on SCOPE EXIT, so it covers BOTH the
    // consumed path and every early return (all-groups-throttled, empty images). A ring slot
    // staged for a timestamp <= the watermark provably will never be claimed and is retired.
    struct NvMark {
      double ts;
      ~NvMark() { if (g_nvjpg) ov_core::ov_nvjpg_watermark(ts); }
    } _nvmark{fr.ts};
    // OV_SPREAD_ADAPT look-ahead counter: this frame is leaving the arrival queue and entering
    // the estimator. Decrement FIRST -- feed_frame can return early (all groups throttled), and
    // the counter must not leak. arrive_wall > 0 <=> the frame-set was complete <=> it was
    // counted at arrival. After this point the counter holds strictly LATER frame-sets.
    if (rt_pace && fr.arrive_wall > 0) {
      const int _prev = ov_core::g_frames_queued.fetch_sub(1, std::memory_order_relaxed);
      if (_prev <= 0) // invariant: every decrement is paired with an earlier arrival increment
        std::fprintf(stderr, "[fqbug]: counter underflow, prev=%d ts=%.6f\n", _prev, fr.ts);
    }
    auto throttled = [&](int lead) {
      auto it = cam_last_track.find(lead);
      if (it != cam_last_track.end() && fr.ts < it->second + track_dt) return true;
      cam_last_track[lead] = fr.ts;
      return false;
    };

    // 1. Which camera groups survive the throttle?
    std::vector<std::vector<int>> groups;
    if (use_stereo && ncam % 2 == 0) {
      for (int p = 0; p < ncam / 2; p++) {
        int l = 2 * p, r = 2 * p + 1;
        if (!fr.payloads.count(l) || !fr.payloads.count(r)) continue;
        if (throttled(l)) continue;
        groups.push_back({l, r});
      }
    } else if (std::getenv("OV_GROUP_CAMS") && *std::getenv("OV_GROUP_CAMS") == '1') {
      // One multi-image message per frame-set: TrackKLT then runs its parallel_for_ across
      // cameras, so tracking (and per-camera CPU work: RANSAC, undistort) overlaps instead
      // of running 4x sequentially. Throttle on the lead camera (hardware-synced).
      std::vector<int> all;
      for (auto const &kv : fr.payloads) all.push_back(kv.first);
      if (!all.empty() && !throttled(all.front())) groups.push_back(all);
    } else {
      for (auto const &kv : fr.payloads) {
        int cid = kv.first;
        if (throttled(cid)) continue;
        groups.push_back({cid});
      }
    }
    if (groups.empty()) {
      if (fr.has_fut) fr.fut.wait();     // mispredicted keep: let the decode finish before erase
      return;
    }

    // 2. Decode only what those groups need, concurrently. Each decode is an
    //    independent pure function of its own payload, so this does not disturb
    //    the bit-identical-across-runs property the serial runner exists for.
    std::vector<int> need;
    for (auto const &g : groups)
      for (int c : g)
        if (std::find(need.begin(), need.end(), c) == need.end())
          need.push_back(c);
    // Convoy guard: how long this frame sat in the queue. The preseed hook skips GPU
    // prefill when we are draining a backlog -- during back-to-back feeds it collides with
    // the next frame's KLT and inflates tracking 2x (measured: every 25-36 ms track spike
    // followed a <7 ms update).
    g_frame_wait_ms = (fr.arrive_wall > 0) ? 1000.0 * (now_s() - fr.arrive_wall) : 0.0;
    // Harvest the stash-time decode if one was launched; otherwise decode here.
    std::map<int, cv::Mat> imgs;
    if (fr.has_fut) {
      const double _tdec = now_s();
      imgs = fr.fut.get();               // usually already done (ran during the previous frame)
      SW.add("  jpeg decode", now_s() - _tdec);
    } else {
      std::vector<cv::Mat> decoded(need.size());
      const double _tdec = now_s();
      cv::parallel_for_(cv::Range(0, (int)need.size()), [&](const cv::Range &rng) {
        for (int i = rng.start; i < rng.end; i++)
          decoded[i] = fr.payloads.at(need[i]).decode();
      });
      SW.add("  jpeg decode", now_s() - _tdec);
      for (size_t i = 0; i < need.size(); i++)
        if (!decoded[i].empty()) imgs[need[i]] = decoded[i];
    }
    for (auto it = imgs.begin(); it != imgs.end();) {
      if (it->second.empty() || !std::count(need.begin(), need.end(), it->first)) it = imgs.erase(it);
      else ++it;
    }
    if (!imgs.empty()) {
      const double _tp = now_s();
      ov_core::g_rt_behind_ms = fr.arrive_wall > 0 ? 1000.0 * (now_s() - fr.arrive_wall) : 0.0;
      feed_frame_impl(fr, groups, imgs);
      const double proc = now_s() - _tp;
      if (fr.arrive_wall > 0) {
        lat_s.push_back(now_s() - fr.arrive_wall);
        static FILE *lf = [] {
          const char *p = std::getenv("OV_LAT_DUMP");
          return p ? std::fopen(p, "w") : (FILE *)nullptr;
        }();
        if (lf && !g_async_emit)
          std::fprintf(lf, "%.6f %.2f %.2f %.2f %.2f %d %.3f\n", fr.ts, 1000.0 * lat_s.back(), 1000.0 * proc,
                       g_dbg_track_ms, g_dbg_upd_ms, g_dbg_upd_n, 1000.0 * fr.pace_off);
        // OV_ASYNC_EMIT: two EXTRA columns so the joined latency and the decoupled latency sit
        // on the same row and neither can be mistaken for the other.
        //   col8 emit_lat_ms  = arrival -> DECOUPLED pose written (producer thread)
        //   col9 corr_age_ms  = frame stamp - stamp of the newest COMPLETED update in that pose
        else if (lf)
          std::fprintf(lf, "%.6f %.2f %.2f %.2f %.2f %d %.3f %.3f %.3f\n", fr.ts, 1000.0 * lat_s.back(), 1000.0 * proc,
                       g_dbg_track_ms, g_dbg_upd_ms, g_dbg_upd_n, 1000.0 * fr.pace_off, fr.emit_lat_ms, fr.corr_age_ms);
        // Separate stream: ts lat proc trk_total det klt und rsc db npts ndet npts_new ncam
        static FILE *tf = [] {
          const char *p = std::getenv("OV_TRK_DUMP");
          return (p && *p && *p != '0' && *p != '1') ? std::fopen(p, "w") : (FILE *)nullptr;
        }();
        if (tf)
          std::fprintf(tf, "%.6f %.2f %.2f %.2f %.3f %.3f %.3f %.3f %.3f %ld %ld %ld %ld\n", fr.ts,
                       1000.0 * lat_s.back(), 1000.0 * proc, g_dbg_track_ms,
                       ov_core::g_trk_det_ms.load(std::memory_order_relaxed),
                       ov_core::g_trk_klt_ms.load(std::memory_order_relaxed),
                       ov_core::g_trk_und_ms.load(std::memory_order_relaxed),
                       ov_core::g_trk_rsc_ms.load(std::memory_order_relaxed),
                       ov_core::g_trk_db_ms.load(std::memory_order_relaxed),
                       ov_core::g_trk_npts.load(std::memory_order_relaxed),
                       ov_core::g_trk_ndet.load(std::memory_order_relaxed),
                       ov_core::g_trk_npts_new.load(std::memory_order_relaxed),
                       ov_core::g_trk_ncam.load(std::memory_order_relaxed));
      }
    }
  };

  // Flush any pending frame whose stamp the IMU has now passed (IMU available
  // through the image time => propagation/update is well-posed).
  auto flush_ready = [&]() {
    while (!pending.empty() && pending.begin()->first <= latest_imu_t) {
      // R18_READER census.  Counting only -- the frame is fed exactly as before, so every
      // round-16/17 fingerprint is preserved on both readers.
      const int _np = (int)pending.begin()->second.payloads.size();
      g_rd_frames_fed++;
      if (_np < ncam) { g_rd_partial_fed++; g_rd_partial_imgs += (ncam - _np); }
      feed_frame(pending.begin()->second);
      pending.erase(pending.begin());
    }
  };

  SW.add("setup+config+masks", now_s() - _wall0);

  size_t n_imu = 0, n_img = 0, n_frames = 0;
  if (!prefetch) {
    for (;;) {
      double _t = now_s();
      BagRecord rec = source.next();
      SW.add("bag read+deserialize", now_s() - _t);
      if (rec.kind == BagRecord::NONE) break;
      pace(rec.kind == BagRecord::IMU ? rec.imu.timestamp : rec.cam.ts);
      if (rec.kind == BagRecord::IMU) {
        { Scoped _s("feed imu"); sys->feed_measurement_imu(rec.imu); }
        latest_imu_t = rec.imu.timestamp;
        n_imu++;
        { Scoped _s("flush->feed_frame"); flush_ready(); }
      } else {
        const int cid = rec.cam.cam_id;
        if (cid < 0 || cid >= ncam) continue; // ignore cams beyond max_cameras
        const double ts = rec.cam.ts;
        pending[ts].ts = ts;
        pending[ts].payloads[cid] = std::move(rec.cam);
        n_img++;
        if ((int)pending[ts].payloads.size() == ncam) {
          n_frames++;
          pending[ts].arrive_wall = now_s();
          if (rt_pace) {
            const int _p = ov_core::g_frames_queued.fetch_add(1, std::memory_order_relaxed) + 1;
            if (_p > g_fq_max) g_fq_max = _p;
          }
        }
      }
    }
  } else {
    // Reader-ahead (OV_PREFETCH=1): a producer thread owns bag reading, frame assembly and the
    // decode launch, so frame N's JPEGs decode WHILE the estimator is still inside frame N-1.
    // The consumer below applies records in exactly the original order, so estimates are
    // unchanged; only where the decode milliseconds are spent moves.
    struct RAItem {
      int kind = 0; // 0 imu, 1 frame
      ov_core::ImuData imu;
      Frame frame;
    };
    std::deque<RAItem> raq;
    std::mutex ram;
    std::condition_variable ra_pop, ra_push;
    bool ra_done = false;
    int frames_ahead = 0;
    const int MAX_FRAMES_AHEAD = OV_MAX_FRAMES_AHEAD;  // R18: sized against the NVJPG ring

    std::thread producer([&]() {
      // setpriority(PRIO_PROCESS, 0, ...) targets the CALLING thread on Linux, so the
      // producer's own nice level can only be set from in here.
      if (g_dec_prio >= 2) setpriority(PRIO_PROCESS, 0, 5);
      std::map<double, Frame> stash;
      double pred_last = -1e18;
      // OV_ASYNC_EMIT: the emitter's PRIVATE IMU history. Written and read only here, on this
      // one thread, so it needs no lock and -- crucially -- the filter's own IMU path
      // (sys->feed_measurement_imu on the consumer) is left byte-identical.  Feeding the
      // emitter here is what stops it being IMU-starved during exactly the convoys it exists
      // to bridge: the consumer stops advancing the filter's IMU buffer while it is blocked
      // inside tracking/update, this buffer keeps advancing at sensor pace.
      std::vector<ov_core::ImuData> emit_imu;
      size_t emit_since_trim = 0;
      double first_snap_t = -1.0;
      if (g_async_emit) emit_imu.reserve(8192);
      auto push = [&](RAItem &&it) {
        std::unique_lock<std::mutex> lk(ram);
        if (it.kind == 1) {
          ra_push.wait(lk, [&] { return frames_ahead < MAX_FRAMES_AHEAD; });
          frames_ahead++;
        }
        raq.push_back(std::move(it));
        ra_pop.notify_one();
      };
      for (;;) {
        BagRecord rec = source.next();
        if (rec.kind == BagRecord::NONE) break;
        pace(rec.kind == BagRecord::IMU ? rec.imu.timestamp : rec.cam.ts);
        if (rec.kind == BagRecord::IMU) {
          if (g_async_emit) {
            emit_imu.push_back(rec.imu);
            // bounded history: the emitter only ever integrates over [snap.t, frame stamp],
            // measured at p99 ~0.10 s and max ~0.24 s, so 3 s is ~12x headroom.
            if (++emit_since_trim >= 256) {
              emit_since_trim = 0;
              const double cut = emit_imu.back().timestamp - 3.0;
              size_t k = 0;
              while (k < emit_imu.size() && emit_imu[k].timestamp < cut) k++;
              if (k > 0) emit_imu.erase(emit_imu.begin(), emit_imu.begin() + k);
            }
          }
          RAItem it; it.kind = 0; it.imu = rec.imu;
          push(std::move(it));
          continue;
        }
        const int cid = rec.cam.cam_id;
        if (cid < 0 || cid >= ncam) continue;
        const double ts = rec.cam.ts;
        stash[ts].ts = ts;
        stash[ts].payloads[cid] = std::move(rec.cam);
        auto sit = stash.find(ts);
        if ((int)sit->second.payloads.size() == ncam) {
          Frame fr = std::move(sit->second);
          stash.erase(sit);
          const double _aw_deser = now_s();                    // 4th camera payload deserialized
          const double _aw_rel = (rt_pace && rt_t0_bag >= 0)   // sensor-paced release instant
                                     ? (rt_t0_wall + (ts - rt_t0_bag))
                                     : _aw_deser;
          fr.arrive_wall = g_lat_true ? _aw_rel : _aw_deser;
          fr.pace_off = (rt_pace && rt_t0_bag >= 0) ? (_aw_deser - _aw_rel) : 0.0;
          if (g_pace_probe && rt_pace && rt_t0_bag >= 0) pace_off_s.push_back(fr.pace_off);
          // OV_SPREAD_ADAPT look-ahead counter: this frame-set is now COMPLETE and waiting.
          if (rt_pace) {
            const int _p = ov_core::g_frames_queued.fetch_add(1, std::memory_order_relaxed) + 1;
            if (_p > g_fq_max) g_fq_max = _p;
          }
          // ---- OV_ASYNC_EMIT: emit the pose NOW, before anything that can block. ----
          // Placed ahead of the decode launch and ahead of the bounded push() so a consumer
          // convoy (which can back this thread up against MAX_FRAMES_AHEAD) cannot delay the
          // emission of the frame that has already arrived.
          if (g_async_emit) {
            auto snap = ov_msckf::ov_pose_snap_load();
            if (!snap || snap->t <= 0) {
              g_ae_nosnap++;
            } else {
              if (first_snap_t < 0) first_snap_t = snap->t;
              // mirror the joined stream's "wait 1 s after init" guard, computed from data this
              // thread owns (never from sys->initialized_time(), which the consumer writes).
              if (ts - first_snap_t >= 1.0) {
                Eigen::Matrix<double, 13, 1> sp;
                if (ov_msckf::Propagator::fast_propagate_snap(*snap, emit_imu, ts, sp)) {
                  const double t_emit = now_s();
                  fr.emit_lat_ms = 1000.0 * (t_emit - fr.arrive_wall);
                  fr.corr_age_ms = 1000.0 * (ts - snap->t);
                  fr.emit_seq = snap->seq;
                  g_ae_emitted++;
                  char abuf[256];
                  // same TUM convention as the joined stream: t x y z qx qy qz qw,
                  // q = q_GtoI (JPL xyzw), p = p_IinG, stamp = CAMERA clock.
                  std::snprintf(abuf, sizeof(abuf), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f", ts, sp(4), sp(5), sp(6), sp(0),
                                sp(1), sp(2), sp(3));
                  async_lines.emplace_back(abuf);
                  if (g_async_dual) {
                    char dbuf[192];
                    std::snprintf(dbuf, sizeof(dbuf), "%.9f %.3f %.3f %ld %.6f", ts, fr.emit_lat_ms, fr.corr_age_ms, snap->seq,
                                  snap->t);
                    async_diag.emplace_back(dbuf);
                  }
                } else {
                  g_ae_noimu++;
                }
              }
            }
          }
          // mirror of feed_frame's throttle: a mispredicted keep wastes one decode, a
          // mispredicted drop falls back to the synchronous decode -- both safe.
          if (!(ts < pred_last + track_dt)) {
            pred_last = ts;
            auto blobs = std::make_shared<std::map<int, CamPayload>>(std::move(fr.payloads));
            fr.payloads.clear();
            for (auto const &kv : *blobs) { // keep ids visible for the grouping logic
              fr.payloads[kv.first].cam_id = kv.first;
              fr.payloads[kv.first].ts = kv.second.ts;
            }
            if (g_decpool) {
              // OV_DECODE_POOL=1: decode on persistent aux-cluster threads instead of a
              // fresh std::async thread whose cv::parallel_for_ re-enters the ONE process-wide
              // OpenCV TBB arena that also serves the 4-camera tracking parallel_for_. Same
              // std::shared_future type, same cam_id-keyed std::map, so every consumer
              // (harvest, mispredicted-keep wait, preseed non-blocking probe) is unchanged.
              auto ids = std::make_shared<std::vector<int>>();
              for (auto const &kv : *blobs) ids->push_back(kv.first);
              auto pr = std::make_shared<std::promise<std::map<int, cv::Mat>>>();
              auto out = std::make_shared<std::map<int, cv::Mat>>();
              auto left = std::make_shared<std::atomic<int>>((int)ids->size());
              auto mtx = std::make_shared<std::mutex>();
              fr.fut = pr->get_future().share();
              if (ids->empty()) {
                pr->set_value(std::map<int, cv::Mat>());
              } else {
                for (size_t i = 0; i < ids->size(); i++)
                  g_decpool->post([blobs, ids, out, left, mtx, pr, i] {
                    cv::Mat img;
                    const long long _c0 = g_dec_cpu ? thread_cpu_ns() : 0;
                    try {
                      const CamPayload &pl = blobs->at((*ids)[i]);
                      if (g_nvjpg && pl.compressed && !pl.data.empty()) {
                        // OV_NVJPG: the JPEG goes to the NVJPG fixed-function engine and the
                        // Y plane lands in a dmabuf already registered with this CUDA context.
                        // What travels through the std::shared_future is a HEADER-ONLY cv::Mat
                        // over one shared placeholder page -- no pixels, no 1 MB allocation --
                        // so every downstream consumer (harvest, mispredicted-keep wait,
                        // preseed probe, size checks) is byte-for-byte unchanged in behaviour.
                        int w = 0, h = 0; size_t st = 0;
                        const unsigned char *page = ov_core::ov_nvjpg_stage(
                            (*ids)[i], pl.ts, pl.data.data(), pl.data.size(), &w, &h, &st);
                        if (page) {
                          img = cv::Mat(h, w, CV_8UC1, const_cast<unsigned char *>(page), st);
                        } else {
                          // R18_RING_FIX -- THE DOCUMENTED FALLBACK.  ov_nvjpg_stage gives up
                          // rather than park this pool worker forever (that park is what
                          // closed the decode deadlock), and it also returns null on decode /
                          // EGL failure.  Decode on the CPU instead: pl.decode() is the exact
                          // OV_NVJPG=0 path, it yields REAL pixels, and gpu_prepare then takes
                          // its ordinary H2D branch because the Mat is not the placeholder.
                          // BEFORE R18 this branch left img empty and the camera was silently
                          // dropped from the frame-set -- a wrong answer, not a slow one.
                          img = pl.decode();
                        }
                        if (g_nvjpg_verify && page) nvjpg_verify_one((*ids)[i], pl);
                      } else {
                        img = pl.decode(); // pure fn of its own payload
                      }
                    } catch (...) {
                      img = cv::Mat();
                    }
                    if (g_dec_cpu) {
                      g_dec_cpu_ns.fetch_add(thread_cpu_ns() - _c0, std::memory_order_relaxed);
                      g_dec_cpu_calls.fetch_add(1, std::memory_order_relaxed);
                    }
                    {
                      std::lock_guard<std::mutex> lk(*mtx);
                      if (!img.empty()) (*out)[(*ids)[i]] = img;
                    }
                    if (--*left == 0) pr->set_value(std::move(*out));
                  });
              }
              if (g_decode_verify) {
                // IN-BINARY PROOF: recompute the gate-OFF result (the exact std::async body,
                // run here synchronously) and compare byte-for-byte with what the pool produced.
                std::map<int, cv::Mat> ref;
                std::vector<int> rid;
                for (auto const &kv : *blobs) rid.push_back(kv.first);
                std::vector<cv::Mat> dec(rid.size());
                cv::parallel_for_(cv::Range(0, (int)rid.size()), [&](const cv::Range &rng) {
                  for (int i = rng.start; i < rng.end; i++)
                    dec[i] = blobs->at(rid[i]).decode();
                });
                for (size_t i = 0; i < rid.size(); i++)
                  if (!dec[i].empty()) ref[rid[i]] = dec[i];
                const std::map<int, cv::Mat> &got = fr.fut.get();
                g_dv_frames++;
                bool same_keys = (ref.size() == got.size());
                for (auto const &kv : ref) if (!got.count(kv.first)) same_keys = false;
                if (!same_keys) g_dv_mismatch++;
                for (auto const &kv : ref) {
                  auto g = got.find(kv.first);
                  if (g == got.end()) continue;
                  g_dv_imgs++;
                  const cv::Mat &A = kv.second, &B = g->second;
                  if (A.rows != B.rows || A.cols != B.cols || A.type() != B.type()) {
                    g_dv_mismatch++;
                    continue;
                  }
                  long md = 0;
                  for (int r = 0; r < A.rows; r++) {
                    const unsigned char *pa = A.ptr<unsigned char>(r), *pb = B.ptr<unsigned char>(r);
                    if (std::memcmp(pa, pb, (size_t)A.cols * A.elemSize()) == 0) continue;
                    for (int c = 0; c < A.cols * (int)A.elemSize(); c++) {
                      long d = std::labs((long)pa[c] - (long)pb[c]);
                      if (d > md) md = d;
                    }
                  }
                  if (md == 0) g_dv_bitexact++;
                  if (md > g_dv_maxdiff) g_dv_maxdiff = md;
                }
              }
            } else {
              fr.fut = std::async(std::launch::async, [blobs]() {
                std::map<int, cv::Mat> out;
                std::vector<int> ids;
                for (auto const &kv : *blobs) ids.push_back(kv.first);
                std::vector<cv::Mat> dec(ids.size());
                const long long _c0 = g_dec_cpu ? thread_cpu_ns() : 0;
                cv::parallel_for_(cv::Range(0, (int)ids.size()), [&](const cv::Range &rng) {
                  for (int i = rng.start; i < rng.end; i++)
                    dec[i] = blobs->at(ids[i]).decode();
                });
                if (g_dec_cpu) {
                  g_dec_cpu_ns.fetch_add(thread_cpu_ns() - _c0, std::memory_order_relaxed);
                  g_dec_cpu_calls.fetch_add((long)ids.size(), std::memory_order_relaxed);
                }
                for (size_t i = 0; i < ids.size(); i++)
                  if (!dec[i].empty()) out[ids[i]] = dec[i];
                return out;
              }).share();
            }
            fr.has_fut = true;
          }
          RAItem it; it.kind = 1; it.frame = std::move(fr);
          push(std::move(it));
        }
      }
      // EOF: hand over incomplete frame-sets untouched (synchronous decode path)
      for (auto &kv : stash) {
        RAItem it; it.kind = 1; it.frame = std::move(kv.second);
        push(std::move(it));
      }
      { std::lock_guard<std::mutex> lk(ram); ra_done = true; }
      ra_pop.notify_one();
    });

    // The reader-ahead thread does bag read + deserialize (24.4 s of a 60 s run) and must
    // not sit on the critical cluster. No-op unless OV_AFFINITY is set.
    if (g_dec_prio >= 2) {
      // OV_DEC_PRIO=2: do NOT confine the producer to the aux cluster (where it would preempt
      // a nice-0 decoder). Give it the full mask and nice it DOWN to +5 instead: during the
      // decode window cores 0-3 are idle, and outside it CFS weight keeps it behind tracking.
      // Its nice+5 is applied from inside the thread itself (setpriority is per-thread).
      cpu_set_t all;
      CPU_ZERO(&all);
      for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &g_cpu_crit) || CPU_ISSET(c, &g_cpu_aux)) CPU_SET(c, &all);
      if (g_aff_on) pthread_setaffinity_np(producer.native_handle(), sizeof(all), &all);
      struct sched_param sp {};
      sp.sched_priority = 0;
      pthread_setschedparam(producer.native_handle(), SCHED_OTHER, &sp);
    } else {
      ov_sched_aux(producer.native_handle());
    }

    // Preseed hook: after tracking of frame N, hand frame N+1's already-decoded images to the
    // tracker so its GPU prepare fills the device during frame N's EKF update. OV_PRESEED=1.
    static const bool preseed_on = [] {
      // OV_DETERMINISTIC: OFF. The hook's frame source (`pending` / `raq`) is filled by the
      // reader thread, so which frame it finds -- and whether that frame is complete -- is a
      // producer race. Preseed only moves gpu_prepare earlier in wall-clock time; it is
      // idempotent per (camera, timestamp) via TrackKLT::preseeded, so removing it costs
      // throughput and nothing else. Proven in-binary: see the R24NP neutrality check.
      // OV_DETERM_PRESEED (bench only): keep the hook ON under the oracle so its result-
      // neutrality can be PROVEN rather than assumed.  =1 blocks for the decode (preseed always
      // happens); =2 keeps the SHIPPED non-blocking probe, so WHETHER a given frame is
      // preseeded still varies with wall-clock -- if the trajectory is bit-identical across
      // draws under =2, the hook is result-neutral under timing variation, which is the claim.
      const char *d = std::getenv("OV_DETERMINISTIC");
      const char *kp = std::getenv("OV_DETERM_PRESEED");
      if (d && *d == '1' && !(kp && (*kp == '1' || *kp == '2'))) return false;
      const char *e = std::getenv("OV_PRESEED");
      return e && *e == '1';
    }();
    if (preseed_on) {
      sys->set_post_track_hook([&](double cur_ts) {
        static const double skip_wait = [] { const char *e = std::getenv("OV_PRESEED_SKIP_MS"); return e ? atof(e) : 8.0; }();
        static const int det_pre = [] { const char *e = std::getenv("OV_DETERM_PRESEED"); return e ? atoi(e) : 0; }();
        static const bool det_on = [] { const char *e = std::getenv("OV_DETERMINISTIC"); return e && *e == '1'; }();
        const bool block_for_decode = det_on && det_pre != 2;
        g_pre_calls++;
        if (!det_on && g_frame_wait_ms > skip_wait) {
          g_pre_skip++;
          return; // draining a backlog: keep the GPU clear for the next frame's KLT
        }
        // ---- ROUND 17: SELECT the frame under the handoff discipline, THEN build. ----
        // The old version ran `build` (which can block on a std::shared_future and allocate the
        // four masks) INSIDE the `ram` critical section on the raq path, and touched Frame
        // objects directly.  Selection now copies only the two things a Frame handoff owns --
        // the timestamp and the shared_future -- while the lock is held, and every allocation,
        // every wait and every get() happens after the lock is dropped.  A shared_future is
        // exactly the right object to copy out: it is the producer/consumer handoff for the
        // decoded images, and copying it is what makes the read-after-select safe no matter
        // what the producer does to the deque afterwards.
        double pre_ts = -1.0;
        std::shared_future<std::map<int, cv::Mat>> pre_fut;
        bool have = false, from_pending = false;
        {
          // `pending` is owned by the CONSUMER thread, which is the thread running this hook
          // (post_track_hook is invoked from inside VioManager::track_image_and_update on the
          // caller's thread).  The reader thread never touches `pending`: it fills its own
          // `stash`, hands complete frames over through `raq` under `ram`, and the consumer
          // moves them into `pending`.  So this lookup needs no lock -- but it DOES need the
          // frame to be complete, which for a `pending` entry is guaranteed by construction:
          // the producer only pushes a frame once all ncam payloads have arrived.
          auto it2 = pending.upper_bound(cur_ts); // current frame is still in `pending` here
          if (it2 != pending.end()) {
            from_pending = true;                  // the old code committed to `pending` HERE:
            if (it2->second.has_fut) {            // a frame with no decode future ends the hook,
              pre_ts = it2->second.ts;            // it does NOT fall through to raq. Preserved
              pre_fut = it2->second.fut;          // exactly, so "preseed is result-neutral" keeps
              have = true;                        // meaning the same thing it did before.
            }
          }
        }
        if (!have && !from_pending) {
          std::lock_guard<std::mutex> lk(ram);
          for (auto &qi : raq)
            if (qi.kind == 1 && qi.frame.ts > cur_ts) {
              if (qi.frame.has_fut) { pre_ts = qi.frame.ts; pre_fut = qi.frame.fut; have = true; }
              break;
            }
        }
        if (!have) { g_pre_noframe++; return; }
        if (!block_for_decode && pre_fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
          g_pre_notready++;
          return;
        }
        ov_core::CameraData pre;
        {
          auto imgs = pre_fut.get();     // shared_future: the consumer's own get() is unaffected
          pre.timestamp = pre_ts;
          for (auto &kv : imgs) {
            pre.sensor_ids.push_back(kv.first);
            pre.images.push_back(kv.second);
            pre.masks.push_back(get_mask(kv.first, kv.second.rows, kv.second.cols));
          }
        }
        if (!pre.sensor_ids.empty()) { g_pre_fired++; sys->preseed_next_frame(pre); }
      });
    }
    for (;;) {
      RAItem it;
      {
        std::unique_lock<std::mutex> lk(ram);
        double _t = now_s();
        ra_pop.wait(lk, [&] { return !raq.empty() || ra_done; });
        SW.add("bag read+deserialize", now_s() - _t); // consumer-side wait on the reader
        if (raq.empty() && ra_done) break;
        it = std::move(raq.front());
        raq.pop_front();
        if (it.kind == 1) { frames_ahead--; ra_push.notify_one(); }
      }
      if (it.kind == 0) {
        { Scoped _s("feed imu"); sys->feed_measurement_imu(it.imu); }
        latest_imu_t = it.imu.timestamp;
        n_imu++;
        { Scoped _s("flush->feed_frame"); flush_ready(); }
      } else {
        n_img += it.frame.payloads.size();
        n_frames++;
        pending[it.frame.ts] = std::move(it.frame);
      }
    }
    producer.join();
  }
  // EOF: flush whatever remains (IMU stream ended).
  // (pipeline drain happens after flush_ready below)
  latest_imu_t = std::numeric_limits<double>::infinity();
  { Scoped _s("flush->feed_frame"); flush_ready(); }
  // OV_PIPELINE: the LAST tick's update is still in flight.  Drain it and take its pose
  // before anything below reads the state (final .tum row, .calib.json harvest).
  sys->pipeline_flush();
  if (g_pace_probe && !pace_off_s.empty()) {
    std::vector<double> v = pace_off_s;
    std::sort(v.begin(), v.end());
    auto q = [&](double pr) { return 1000.0 * v[(size_t)(pr * (v.size() - 1))]; };
    double sum = 0;
    for (double x : v) sum += x;
    std::fprintf(stderr,
                 "[paceoff]: n=%zu mean=%.3f p10=%.3f p50=%.3f p90=%.3f p99=%.3f min=%.3f max=%.3f ms"
                 " (stamped arrive_wall MINUS paced release instant)\n",
                 v.size(), 1000.0 * sum / v.size(), q(0.10), q(0.50), q(0.90), q(0.99), q(0.0), q(1.0));
  }
  if (g_lat_true) std::fprintf(stderr, "[lattrue]: latency origin = PACED RELEASE INSTANT (not deserialize-complete)\n");
  if (rt_pace)
    std::fprintf(stderr, "[fqcnt]: final=%d maxseen=%d\n",
                 ov_core::g_frames_queued.load(std::memory_order_relaxed), g_fq_max.load(std::memory_order_relaxed));
  if (g_decode_verify)
    std::fprintf(stderr,
                 "[decverify]: frames=%ld images=%ld BITEXACT=%ld mismatch=%ld maxabsdiff=%ld\n",
                 g_dv_frames, g_dv_imgs, g_dv_bitexact, g_dv_mismatch, g_dv_maxdiff);
  if (g_dec_cpu) {
    const double cs = 1e-9 * (double)g_dec_cpu_ns.load();
    std::fprintf(stderr, "[deccpu]: decode CPU-seconds=%.3f calls=%ld  per-call=%.3f ms\n", cs,
                 (long)g_dec_cpu_calls.load(),
                 g_dec_cpu_calls.load() ? 1e3 * cs / (double)g_dec_cpu_calls.load() : 0.0);
  }
  std::fprintf(stderr,
               "[reader]: mode=%s frames_fed=%ld partial_fed=%ld partial_imgs=%ld ncam=%d\n",
               prefetch ? "ahead" : "serial", g_rd_frames_fed, g_rd_partial_fed,
               g_rd_partial_imgs, ncam);
  if (g_nvjpg) std::fprintf(stderr, "%s\n", ov_core::ov_nvjpg_stats().c_str());
  if (g_nvjpg_verify && g_nv_imgs) {
    std::fprintf(stderr,
                 "[nvverify]: images=%ld BITEXACT=%ld maxabsdiff=%ld mean|d|=%.5f "
                 "pct|d|>=1=%.4f%% >=2=%.4f%% >=3=%.4f%%\n",
                 g_nv_imgs, g_nv_bitexact, g_nv_maxdiff, g_nv_absdiff_sum / (double)g_nv_pix,
                 100.0 * g_nv_d1 / (double)g_nv_pix, 100.0 * g_nv_d2 / (double)g_nv_pix,
                 100.0 * g_nv_d3 / (double)g_nv_pix);
    std::fprintf(stderr, "[nvverify]: GPU min/max/mean %ld/%ld/%.3f | CPU min/max/mean %ld/%ld/%.3f\n",
                 g_nv_gmin, g_nv_gmax, g_nv_gmean / (double)g_nv_imgs, g_nv_hmin, g_nv_hmax,
                 g_nv_hmean / (double)g_nv_imgs);
    std::fprintf(stderr, "[nvverify]: CPU-MAP of the SAME surface vs imdecode: n=%ld BITEXACT=%ld maxabsdiff=%ld\n",
                 g_nv_cpumap_n, g_nv_cpumap_bitexact, g_nv_cpumap_maxdiff);
    std::fprintf(stderr, "[nvverify]: per-cam BITEXACT %ld/%ld %ld/%ld %ld/%ld %ld/%ld\n",
                 g_nv_cam_be[0], g_nv_cam_n[0], g_nv_cam_be[1], g_nv_cam_n[1],
                 g_nv_cam_be[2], g_nv_cam_n[2], g_nv_cam_be[3], g_nv_cam_n[3]);
    std::fprintf(stderr, "[nvverify]: FAST corner-set Jaccard = %.4f over %ld images (inter=%ld union=%ld)\n",
                 g_nv_fast_union ? (double)g_nv_fast_inter / (double)g_nv_fast_union : 0.0,
                 g_nv_fast_frames, g_nv_fast_inter, g_nv_fast_union);
  }
  if (!lat_s.empty()) {
    std::vector<double> v = lat_s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto q = [&](double p) { return 1000.0 * v[(size_t)(p * (v.size() - 1))]; };
    int over = 0;
    for (double x : v) over += (x > 1.0 / 29.0);
    PRINT_INFO("[latency]: n=%zu mean=%.1f p50=%.1f p90=%.1f p99=%.1f max=%.1f ms, over-budget(34.5ms)=%d (%.2f%%)%s\n",
               v.size(), 1000.0 * sum / v.size(), q(0.50), q(0.90), q(0.99), q(1.0), over,
               100.0 * over / v.size(), rt_pace ? " [REALTIME-PACED]" : " [offline, no pacing]");
  }

  if (g_async_verify)
    std::fprintf(stderr, "[asyncverify]: n=%ld ok=%ld MAXABSDIFF=%.3e (fast_propagate_snap vs fast_state_propagate, mean 13-vector)\n",
                 g_av_n, g_av_ok, g_av_max);
  if (g_async_emit) {
    std::fprintf(stderr, "[asyncemit]: emitted=%ld no_snapshot=%ld imu_short=%ld\n", g_ae_emitted, g_ae_nosnap, g_ae_noimu);
    FILE *af = std::fopen((out_path + ".async.tum").c_str(), "w");
    if (af) {
      for (auto const &l : async_lines) std::fprintf(af, "%s\n", l.c_str());
      std::fclose(af);
    }
    if (g_async_dual) {
      FILE *df = std::fopen((out_path + ".async.csv").c_str(), "w");
      if (df) {
        std::fprintf(df, "# ts emit_lat_ms corr_age_ms snap_seq snap_t\n");
        for (auto const &l : async_diag) std::fprintf(df, "%s\n", l.c_str());
        std::fclose(df);
      }
    }
  }

  // Write TUM
  FILE *f = std::fopen(out_path.c_str(), "w");
  if (!f) { PRINT_ERROR(RED "[serial]: cannot open out %s\n" RESET, out_path.c_str()); return EXIT_FAILURE; }
  for (auto const &l : lines) std::fprintf(f, "%s\n", l.c_str());
  std::fclose(f);

  // Dump OV's CONVERGED calibration: per-cam intrinsics + body_P_cam extrinsic (R_CtoI, p_CinI)
  // + cam-imu timeoffset + biases. (Same writer used for the mid-run snapshots above.)
  { Scoped _s("write outputs"); write_calib(out_path + ".calib.json"); }
  if (series != nullptr)
    std::fclose(series);

  {
    // VioManager's internal buckets, so the 80%+ spent inside feed_measurement_camera
    // is broken down rather than reported as one number.
    using ov_msckf::g_vio_stage_secs;
    using ov_msckf::g_vio_stage_calls;
    PRINT_INFO(GREEN "\n[vio-stage]: %-30s %8s %10s\n" RESET, "stage", "sec", "calls");
    for (auto const &kv : g_vio_stage_secs)
      PRINT_INFO(GREEN "[vio-stage]: %-30s %8.2f %10ld\n" RESET, kv.first.c_str(), kv.second,
                 g_vio_stage_calls[kv.first]);
  }
  SW.report(now_s() - _wall0);
  PRINT_INFO(GREEN "[serial]: done. imu=%zu img=%zu frames=%zu poses=%zu -> %s\n" RESET,
             n_imu, n_img, n_frames, lines.size(), out_path.c_str());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
