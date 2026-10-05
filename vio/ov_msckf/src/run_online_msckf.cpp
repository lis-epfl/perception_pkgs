/*
 * run_online_msckf -- the ONLINE (live-subscriber) OpenVINS node for the swarm-nxt
 * 4-camera fisheye VIO, carrying the ingest optimisations that the offline bench
 * runner (run_serial_msckf) developed over rounds 4-19.
 *
 * WHAT THIS IS AND WHY IT EXISTS
 * -----------------------------
 * `run_subscribe_msckf` + ROS2Visualizer::setup_subscribers cannot consume the flight
 * cameras at all:
 *   * it subscribes sensor_msgs/msg/Imu, the vehicle publishes px4_msgs/msg/SensorCombined
 *     (no header, `timestamp` in microseconds, FRD body frame),
 *   * it subscribes sensor_msgs/msg/Image, the vehicle publishes CompressedImage (JPEG),
 *   * with 4 cameras + use_stereo it builds TWO stereo-paired CameraData per frame-set,
 *     not the ONE grouped 4-image message the hybrid update path is designed around,
 *   * and none of OV_NVJPG / OV_DECODE_POOL / OV_AFFINITY / OV_GROUP_CAMS is reachable
 *     from it (they are implemented inside run_serial_msckf.cpp).
 *
 * This node fixes all four. It is deliberately a SEPARATE binary: run_subscribe_msckf
 * and run_serial_msckf are untouched, so every number in the campaign record still
 * describes the binary it was measured on.
 *
 * INGEST, top to bottom
 * ---------------------
 *  1. QoS.  Camera drivers publish SENSOR_DATA (BEST_EFFORT).  A RELIABLE subscriber is
 *     QoS-INCOMPATIBLE with a BEST_EFFORT publisher and receives NOTHING, silently.  A
 *     BEST_EFFORT subscriber matches BOTH, so that is the default here; the history depth
 *     is deep rather than the stock 10 so a momentary consumer stall does not evict.
 *     Receipt is never assumed -- every topic is COUNTED and the census is printed.
 *
 *  2. IMU conversion.  ov_core::sensor_combined_to_imu (ov_core/utils/swarmnxt_msgs.h),
 *     the SAME function the offline runner uses through BagSource.  That is a correctness
 *     requirement, not tidiness: self-calibration fits `timeshift_cam_imu` against
 *     whatever field the calibration path called "the sensor time", and if the flight path
 *     disagrees the calibration is fitted for a pipeline that does not exist.  It also
 *     carries the FRD->FLU axis flip, whose omission diverges the trajectory while every
 *     gate stays green.
 *
 *  3. Stamp-keyed 4-camera synchroniser.  The four cameras are hardware-synced and share a
 *     stamp EXACTLY, so this keys on the integer nanosecond stamp -- no ApproximateTime, no
 *     slop window.  It emits ONE grouped ov_core::CameraData per frame-set.
 *     CLOSING RULE (zero added latency in the common case, and no timer needed):
 *       * a stamp with all ncam images closes IMMEDIATELY;
 *       * when a stamp S closes, every still-open stamp S' < S is closed as PARTIAL first,
 *         in ascending order.  This is exact, not heuristic: per-topic delivery is ordered,
 *         so if every camera has already delivered S > S', no camera can ever deliver S'.
 *       * a time-based escape (OV_ONLINE_HOLD_MS) closes a trailing stamp that never
 *         completes, so one dead camera cannot stall the filter.
 *     Partial frame-sets are FED IN STAMP ORDER, not discarded.  MEASURED DIFFERENCE FROM
 *     THE OFFLINE RUNNER, and it is the ONLY one: run_serial_msckf holds an incomplete
 *     stash entry until EOF and only then hands it to feed_frame, by which time
 *     cam_last_track has run past its stamp and the track_frequency throttle discards it.
 *     On nxt6_s2 that is 5 frame-sets (13 images) and on nxt3_s2 it is 2 (4 images); this
 *     node feeds them, so its fed set is a strict SUPERSET of the offline one.
 *
 *  4. Decode.  OV_NVJPG=1 stages the JPEG on the NVJPG fixed-function engine straight into
 *     a CUDA-registered dmabuf (ov_nvjpg_stage), on OV_DECODE_POOL threads pinned one per
 *     auxiliary core.  ALL the round-18/19 correctness fixes are inherited unchanged
 *     because they live in ov_core: the searching/RESERVED acquire (round 19 proved the
 *     PRE-R18 acquire returns a DIFFERENT TRAJECTORY EVERY RUN under load), the
 *     ring>=lookahead+3 sizing via ov_nvjpg_set_lookahead(), and the null-stage fallback to
 *     pl.decode() instead of silently dropping the camera from the frame-set.  The consumer
 *     watermark ov_nvjpg_watermark() is published on scope exit of the feed, covering every
 *     early return, exactly as in run_serial_msckf.
 *
 *  5. Threads.  Three roles, and the split is the whole point:
 *       ROS executor      -- deserialize + enqueue ONLY.  It must never block, because a
 *                            blocked executor is a DROPPED MESSAGE and a dropped message
 *                            looks like speed.
 *       decode pool       -- NVJPG staging / libjpeg, auxiliary cores.
 *       estimator thread  -- feed_measurement_imu / feed_measurement_camera / publish,
 *                            critical cores.
 *     OV_AFFINITY="<crit>:<aux>" partitions them.  DDS receive threads are created while
 *     the calling thread holds the FULL mask so they are never confined to the critical
 *     cluster (they inherit whatever mask exists at creation).
 *
 *  6. Ordering.  Records reach the estimator in the SAME order the offline runner applies
 *     them: IMU in stamp order, and a frame-set only once the IMU stream has passed its
 *     stamp (`latest_imu_t >= fr.ts`), which is the offline flush_ready() predicate.
 *
 * NOT PORTED, on purpose:
 *   OV_REALTIME  -- a bag-pacing simulation; meaningless when messages arrive for real.
 *   OV_PREFETCH / OV_PRESEED -- reader-ahead over a bag FILE.  There is no analogue when
 *     frames arrive from DDS: the "look ahead" here is exactly the arrival queue, and it is
 *     already what feeds the decode pool.  (OV_PRESEED was also measured inert on the paced
 *     path, firing 0-3 times per ~6285 calls.)
 *
 * STAMPS.  Poses are published by ROS2Visualizer::publish_state, which stamps
 * timestamp_inI = state->_timestamp + _calib_dt_CAMtoIMU.  That is correct and is NOT
 * re-stamped here.  Do not replace it with node->now().
 *
 * WHEN THE POSE LEAVES.  [online-latency] stops at the hand-off of the update to the pipeline
 * worker (p50 13.5 ms); it is NOT the pose latency.  Under OV_PIPELINE the pose is emitted by
 * VioManager's pose_sink, which by default fires only after the NEXT frame-set has been
 * tracked: measured 47 ms after the pose's own images arrived (80 ms when the sub-update queue
 * runs one set behind).  OV_PUB_WHEN_READY=1 makes the worker fire the same sink the moment
 * its update is done.  The [online-pose] census printed at exit is the number a consumer sees:
 * frame-set complete -> the pose carrying that stamp published.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <pthread.h>
#include <string>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <execinfo.h>
#include <fstream>
#include <functional>
#include <future>
#include <malloc.h>
#include <map>
#include <memory>
#include <mutex>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#ifdef OV_HAVE_PX4_MSGS
#include <px4_msgs/msg/sensor_combined.hpp>
#endif

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS2Visualizer.h"
#include "state/State.h"
#include "track/nvjpg_decode.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/swarmnxt_msgs.h"

using namespace ov_msckf;

// =====================================================================================
// CPU partitioning.  Same three env gates and the same parser as run_serial_msckf, kept
// literally identical so the ship string means the same thing in both binaries.
// =====================================================================================
static cpu_set_t g_cpu_crit, g_cpu_aux, g_cpu_all;
static bool g_aff_on = false;

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
    for (int c = std::min(a, b); c <= std::max(a, b); c++)
      if (c >= 0 && c < CPU_SETSIZE) { CPU_SET(c, &m); any = true; }
    a = b = -1;
    if (*s == ',') s++;
  }
  return any;
}

static void ov_sched_init() {
  CPU_ZERO(&g_cpu_all);
  for (int c = 0; c < (int)sysconf(_SC_NPROCESSORS_ONLN) && c < CPU_SETSIZE; c++) CPU_SET(c, &g_cpu_all);
  const char *e = std::getenv("OV_AFFINITY");
  if (!e || !*e) return;
  std::string s(e);
  size_t c = s.find(':');
  if (c != std::string::npos && ov_parse_cpulist(s.substr(0, c).c_str(), g_cpu_crit) &&
      ov_parse_cpulist(s.substr(c + 1).c_str(), g_cpu_aux)) {
    g_aff_on = (sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_crit) == 0);
    std::fprintf(stderr, "[sched]: critical=%s aux=%s ok=%d\n", s.substr(0, c).c_str(), s.substr(c + 1).c_str(),
                 (int)g_aff_on);
  } else {
    std::fprintf(stderr, "[sched]: OV_AFFINITY='%s' unparsable, ignored\n", e);
  }
}
/// Widen the CALLING thread to every core.  Used around DDS/participant creation so the
/// rmw receive threads -- which inherit the creator's mask -- are never confined to the
/// critical cluster where they would contend with tracking and drop messages.
static void ov_sched_wide_self() { if (g_aff_on) sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_all); }
static void ov_sched_crit_self() { if (g_aff_on) sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_crit); }
static void ov_sched_aux_self(bool nice_down) {
  if (g_aff_on) sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_aux);
  struct sched_param sp {};
  sp.sched_priority = 0;
  pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
  if (nice_down) setpriority(PRIO_PROCESS, 0, 5);
}

// One persistent thread per camera, pinned one-per-aux-core.  Verbatim from
// run_serial_msckf.cpp so the decode window behaves identically.
struct OvDecodePool {
  std::vector<std::thread> th;
  std::deque<std::function<void()>> q;
  std::mutex m;
  std::condition_variable cv_;
  bool stop = false;
  explicit OvDecodePool(int n) {
    for (int i = 0; i < n; i++)
      th.emplace_back([this, i] {
        ov_sched_aux_self(false); // decode latency is directly additive to pose latency: nice 0
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
static bool g_nvjpg = false;

// One camera image still in its on-the-wire form.  Same contract as
// ov_msckf::CamPayload (utils/bag_source.h) -- redeclared rather than included so this
// node does not drag in rosbag2.
struct LivePayload {
  int cam_id = -1;
  double ts = -1.0;
  bool compressed = false;
  std::vector<uint8_t> data;
  uint32_t width = 0, height = 0, step = 0;
  cv::Mat decode() const {
    if (compressed) return cv::imdecode(data, cv::IMREAD_GRAYSCALE);
    if (width == 0 || height == 0 || data.empty()) return cv::Mat();
    const size_t stride = (step != 0) ? step : width;
    if (data.size() < stride * height) return cv::Mat();
    cv::Mat wrapped((int)height, (int)width, CV_8UC1, const_cast<uint8_t *>(data.data()), stride);
    return wrapped.clone();
  }
};

struct FrameSet {
  double ts = -1.0;
  int64_t ts_ns = 0;
  std::map<int, LivePayload> payloads;
  std::shared_future<std::map<int, cv::Mat>> fut;
  bool has_fut = false;
  double arrive_wall = -1.0; // wall clock at CLOSE (all cams in, or provably never will be)
  int n_images = 0;
};

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- census (printed at exit; this is the frame-accounting gate) ----------------------
static std::atomic<long> C_imu_recv{0}, C_imu_bad{0};
static long C_cam_recv[8] = {0};
static std::atomic<long> C_sets_closed{0}, C_sets_complete{0}, C_sets_partial{0};
static std::atomic<long> C_imgs_closed{0}, C_late_drop{0}, C_hold_close{0}, C_drain_close{0};
static std::atomic<long> C_sets_fed{0}, C_imgs_fed{0}, C_sets_throttled{0}, C_decode_empty{0};
static std::atomic<long> C_decode_prefetched{0}, C_decode_sync{0}, C_poses{0};
static std::atomic<int> C_qdepth_max{0}, C_imuq_max{0};

int main(int argc, char **argv) {
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
      raise(sig);
    }
  };
  signal(SIGSEGV, CrashTrace::handler);
  signal(SIGABRT, CrashTrace::handler);
  signal(SIGBUS, CrashTrace::handler);

  // ---- ONE SANE DEFAULT SET -------------------------------------------------------
  // setenv(..., overwrite=0) so an explicit environment ALWAYS wins and the resolved
  // value is visible to ov_core, which reads these same names.  The ship string sets all
  // three to exactly these values, so passing it changes nothing.
  setenv("OV_NVJPG", "1", 0);
  setenv("OV_DECODE_POOL", "1", 0);
  setenv("OV_AFFINITY", "0-3:4-7", 0);
  setenv("OV_GROUP_CAMS", "1", 0);

  ov_sched_init(); // first real statement after the defaults: children inherit the mask
  {
    const char *dp = std::getenv("OV_DECODE_POOL");
    if (dp && *dp == '1') {
      g_decpool = std::unique_ptr<OvDecodePool>(new OvDecodePool(4));
      std::fprintf(stderr, "[sched]: decode pool ON (4 threads)\n");
    }
  }
  mallopt(M_MMAP_THRESHOLD, 512 * 1024 * 1024);
  mallopt(M_TRIM_THRESHOLD, 512 * 1024 * 1024);
  if (const char *ent = std::getenv("OV_EIGEN_NT")) Eigen::setNbThreads(atoi(ent));
  {
    const char *seed_env = std::getenv("OV_RNG_SEED");
    uint64_t seed = 42;
    if (seed_env && *seed_env) { try { seed = std::stoull(seed_env); } catch (...) {} }
    cv::theRNG().state = seed;
  }

  auto pos = rclcpp::init_and_remove_ros_arguments(argc, argv);
  if (pos.size() < 2) {
    PRINT_ERROR(RED "usage: run_online_msckf <config.yaml> [out_tum] [--ros-args -p max_cameras:=4 ...]\n" RESET);
    return EXIT_FAILURE;
  }
  const std::string config_path = pos[1];
  std::string out_path = (pos.size() >= 3) ? pos[2] : "";
  if (out_path.empty()) { const char *o = std::getenv("OV_ONLINE_TUM"); if (o && *o) out_path = o; }

  // DDS participant + its receive threads are born here -- give them every core.
  ov_sched_wide_self();
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("run_online_msckf", options);
  ov_sched_crit_self();

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node(node);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);

  // ---- OV_NVJPG startup: the same four refusals, and the SAME ring sizing call --------
  const int lookahead = [] { const char *e = std::getenv("OV_ONLINE_LOOKAHEAD"); int v = e ? atoi(e) : 3; return v > 0 ? v : 3; }();
  if (std::getenv("OV_NVJPG") && *std::getenv("OV_NVJPG") == '1') {
    const char *gt = std::getenv("OV_GPU_TRACK");
    ov_core::ov_nvjpg_refuse_if(params.downsample_cameras, "downsample_cameras=true (VioManager pyrDown reads host pixels)");
    ov_core::ov_nvjpg_refuse_if(params.histogram_method == ov_core::TrackBase::HistogramMethod::HISTOGRAM,
                                "histogram_method=HISTOGRAM (cv::equalizeHist reads host pixels)");
    ov_core::ov_nvjpg_refuse_if(!gt || *gt != '1', "OV_GPU_TRACK!=1 (no device consumer)");
    ov_core::ov_nvjpg_refuse_if(params.use_stereo, "use_stereo=true (host stereo matching)");
    // R18_RING_FIX: MUST precede ov_nvjpg_enabled(), which is what runs init_once().
    // Ring depth = lookahead + 3; the online lookahead is the decode-prefetch budget below.
    ov_core::ov_nvjpg_set_lookahead(lookahead);
    g_nvjpg = ov_core::ov_nvjpg_enabled();
  }

  params.use_multi_threading_subs = false; // this node owns its own threading
  params.use_multi_threading_pubs = false; // no background trackhist thread
  auto sys = std::make_shared<VioManager>(params);
  auto viz = std::make_shared<ROS2Visualizer>(node, sys); // publishers ONLY -- no setup_subscribers

  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    return EXIT_FAILURE;
  }

  const int ncam = params.state_options.num_cameras;
  const bool use_stereo = params.use_stereo;
  const bool use_mask = params.use_mask;
  const double track_dt = (params.track_frequency > 1e-6) ? 1.0 / params.track_frequency : 0.0;
  const std::map<size_t, cv::Mat> startup_masks = sys->get_params().masks;
  auto get_mask = [&](int cam_id, int rows, int cols) -> cv::Mat {
    if (use_mask) {
      auto it = startup_masks.find((size_t)cam_id);
      if (it != startup_masks.end() && !it->second.empty()) return it->second;
    }
    return cv::Mat::zeros(rows, cols, CV_8UC1);
  };

  // ---- topics, from the SAME config keys the offline runner reads --------------------
  std::string imu_topic = "/imu0", imu_msg_type = "sensor_msgs/msg/Imu", cam_msg_type = "sensor_msgs/msg/Image";
  parser->parse_config("imu_topic", imu_topic, false);
  parser->parse_config("imu_msg_type", imu_msg_type, false);
  parser->parse_config("cam_msg_type", cam_msg_type, false);
  std::vector<std::string> cam_topics((size_t)ncam);
  for (int i = 0; i < ncam; i++) {
    std::string t;
    parser->parse_config("cam_topic" + std::to_string(i), t, false);
    if (t.empty()) t = "/cam" + std::to_string(i) + "/image_raw";
    cam_topics[(size_t)i] = t;
  }

  // ---- QoS ---------------------------------------------------------------------------
  // A RELIABLE subscriber does not match a BEST_EFFORT publisher: it receives NOTHING, and
  // nothing anywhere reports an error.  BEST_EFFORT matches both, so it is the default.
  const bool qos_reliable = [] { const char *e = std::getenv("OV_ONLINE_QOS"); return e && std::strcmp(e, "reliable") == 0; }();
  const int cam_depth = [] { const char *e = std::getenv("OV_ONLINE_CAM_DEPTH"); return e ? atoi(e) : 120; }();
  const int imu_depth = [] { const char *e = std::getenv("OV_ONLINE_IMU_DEPTH"); return e ? atoi(e) : 4000; }();
  auto mkqos = [&](int depth) {
    rclcpp::QoS q(rclcpp::KeepLast((size_t)depth));
    if (qos_reliable) q.reliable(); else q.best_effort();
    q.durability_volatile();
    return q;
  };
  const double hold_s = [] { const char *e = std::getenv("OV_ONLINE_HOLD_MS"); return (e ? atof(e) : 250.0) * 1e-3; }();

  PRINT_INFO(GREEN "[online]: ncam=%d use_stereo=%d use_mask=%d group_cams=%d track_dt=%.4f\n" RESET, ncam,
             (int)use_stereo, (int)use_mask, 1, track_dt);
  PRINT_INFO(GREEN "[online]: imu %s (%s)\n" RESET, imu_topic.c_str(), imu_msg_type.c_str());
  for (int i = 0; i < ncam; i++)
    PRINT_INFO(GREEN "[online]: cam%d %s (%s)\n" RESET, i, cam_topics[(size_t)i].c_str(), cam_msg_type.c_str());
  std::fprintf(stderr, "[online]: qos=%s cam_depth=%d imu_depth=%d lookahead=%d hold_ms=%.0f nvjpg=%d pool=%d\n",
               qos_reliable ? "RELIABLE" : "BEST_EFFORT", cam_depth, imu_depth, lookahead, 1e3 * hold_s,
               (int)g_nvjpg, (int)(g_decpool != nullptr));

  // =====================================================================================
  // Assembler + queues.
  // =====================================================================================
  std::mutex q_mtx;
  std::condition_variable q_cv;
  std::deque<ov_core::ImuData> imuq;         // stamp-ordered by construction
  std::deque<FrameSet> readyq;               // CLOSED frame-sets, stamp-ordered
  std::atomic<bool> stop_flag{false};

  struct Partial {
    double ts = -1.0;
    std::map<int, LivePayload> payloads;
    double first_arrive = 0.0;
  };
  std::mutex a_mtx;
  std::map<int64_t, Partial> open;           // stamp_ns -> partially assembled set
  int64_t last_closed_ns = INT64_MIN;
  std::atomic<int> inflight_decodes{0};      // closed-but-unfed sets holding a decode future
  double pred_last = -1e18;                  // throttle prediction, mirrors the offline producer

  // Launch the decode for a closed frame-set, unless we are already `lookahead` deep --
  // in which case the payloads travel undecoded and the estimator decodes synchronously
  // on harvest.  Never blocks: this runs on the DDS executor thread.
  auto launch_decode = [&](FrameSet &fr) {
    const bool throttle_pred = (track_dt > 0.0 && fr.ts < pred_last + track_dt);
    if (!throttle_pred) pred_last = fr.ts;
    if (throttle_pred) return;                              // would be dropped anyway
    if (inflight_decodes.load() >= lookahead) { C_decode_sync++; return; }
    if (!g_decpool) { C_decode_sync++; return; }
    auto blobs = std::make_shared<std::map<int, LivePayload>>(std::move(fr.payloads));
    fr.payloads.clear();
    auto ids = std::make_shared<std::vector<int>>();
    for (auto const &kv : *blobs) {
      ids->push_back(kv.first);
      fr.payloads[kv.first].cam_id = kv.first;              // keep ids visible to the grouping logic
      fr.payloads[kv.first].ts = kv.second.ts;
    }
    auto pr = std::make_shared<std::promise<std::map<int, cv::Mat>>>();
    auto out = std::make_shared<std::map<int, cv::Mat>>();
    auto left = std::make_shared<std::atomic<int>>((int)ids->size());
    auto mtx = std::make_shared<std::mutex>();
    fr.fut = pr->get_future().share();
    fr.has_fut = true;
    inflight_decodes++;
    C_decode_prefetched++;
    if (ids->empty()) { pr->set_value(std::map<int, cv::Mat>()); return; }
    for (size_t i = 0; i < ids->size(); i++)
      g_decpool->post([blobs, ids, out, left, mtx, pr, i] {
        cv::Mat img;
        try {
          const LivePayload &pl = blobs->at((*ids)[i]);
          if (g_nvjpg && pl.compressed && !pl.data.empty()) {
            int w = 0, h = 0; size_t st = 0;
            const unsigned char *page =
                ov_core::ov_nvjpg_stage((*ids)[i], pl.ts, pl.data.data(), pl.data.size(), &w, &h, &st);
            if (page) img = cv::Mat(h, w, CV_8UC1, const_cast<unsigned char *>(page), st);
            else img = pl.decode(); // R18_RING_FIX documented fallback: real pixels, never a dropped camera
          } else {
            img = pl.decode();
          }
        } catch (...) { img = cv::Mat(); }
        { std::lock_guard<std::mutex> lk(*mtx); if (!img.empty()) (*out)[(*ids)[i]] = img; }
        if (--*left == 0) pr->set_value(std::move(*out));
      });
  };

  // Close `it` (called with a_mtx held) and hand it to the estimator.
  auto close_entry = [&](std::map<int64_t, Partial>::iterator it) {
    FrameSet fr;
    fr.ts = it->second.ts;
    fr.ts_ns = it->first;
    fr.payloads = std::move(it->second.payloads);
    fr.n_images = (int)fr.payloads.size();
    fr.arrive_wall = now_s();
    last_closed_ns = it->first;
    open.erase(it);
    C_sets_closed++;
    C_imgs_closed += fr.n_images;
    if (fr.n_images == ncam) C_sets_complete++;
    else {
      C_sets_partial++;
      std::fprintf(stderr, "[assemble]: PARTIAL set ts=%.9f images=%d/%d\n", fr.ts, fr.n_images, ncam);
    }
    launch_decode(fr);
    {
      std::lock_guard<std::mutex> lk(q_mtx);
      readyq.push_back(std::move(fr));
      if ((int)readyq.size() > C_qdepth_max.load()) C_qdepth_max = (int)readyq.size();
    }
    q_cv.notify_one();
  };

  // Close every open stamp strictly older than `key_ns`, ascending.  Exact, see header.
  auto close_older_than = [&](int64_t key_ns) {
    while (!open.empty() && open.begin()->first < key_ns) close_entry(open.begin());
  };

  auto on_camera = [&](int cam_id, int64_t ts_ns, double ts, LivePayload &&pl) {
    std::lock_guard<std::mutex> lk(a_mtx);
    C_cam_recv[cam_id]++;
    if (ts_ns <= last_closed_ns) { // arrived after its own frame-set was already handed on
      C_late_drop++;
      std::fprintf(stderr, "[assemble]: LATE cam%d ts=%.9f (last_closed=%.9f) DROPPED\n", cam_id, ts,
                   1e-9 * (double)last_closed_ns);
      return;
    }
    auto &p = open[ts_ns];
    p.ts = ts;
    if (p.first_arrive == 0.0) p.first_arrive = now_s();
    p.payloads[cam_id] = std::move(pl);
    if ((int)p.payloads.size() == ncam) {
      close_older_than(ts_ns);                 // ascending order is preserved
      close_entry(open.find(ts_ns));
    } else {
      // dead-camera escape only: a trailing stamp nothing will ever complete.
      const double t = now_s();
      while (!open.empty() && (t - open.begin()->second.first_arrive) > hold_s) {
        C_hold_close++;
        close_entry(open.begin());
      }
    }
  };

  // ---- subscriptions ------------------------------------------------------------------
  ov_sched_wide_self();
  rclcpp::SubscriptionBase::SharedPtr sub_imu;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_cam;
  auto push_imu = [&](const ov_core::ImuData &m) {
    { std::lock_guard<std::mutex> lk(q_mtx);
      imuq.push_back(m);
      if ((int)imuq.size() > C_imuq_max.load()) C_imuq_max = (int)imuq.size(); }
    q_cv.notify_one();
    C_imu_recv++;
  };
  if (imu_msg_type == "px4_msgs/msg/SensorCombined") {
#ifdef OV_HAVE_PX4_MSGS
    sub_imu = node->create_subscription<px4_msgs::msg::SensorCombined>(
        imu_topic, mkqos(imu_depth), [&](const px4_msgs::msg::SensorCombined::SharedPtr msg) {
          ov_core::ImuData m;
          ov_core::sensor_combined_to_imu(*msg, m); // shared with the offline path ON PURPOSE
          if (!(m.timestamp > 0)) { C_imu_bad++; return; }
          push_imu(m);
        });
#else
    PRINT_ERROR(RED "[online]: imu_msg_type=px4_msgs/msg/SensorCombined but px4_msgs was NOT found at build time\n" RESET);
    return EXIT_FAILURE;
#endif
  } else {
    sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, mkqos(imu_depth), [&](const sensor_msgs::msg::Imu::SharedPtr msg) {
          ov_core::ImuData m;
          m.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
          m.wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
          m.am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
          push_imu(m);
        });
  }
  for (int i = 0; i < ncam; i++) {
    if (cam_msg_type == "sensor_msgs/msg/CompressedImage") {
      subs_cam.push_back(node->create_subscription<sensor_msgs::msg::CompressedImage>(
          cam_topics[(size_t)i], mkqos(cam_depth), [&, i](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            LivePayload pl;
            pl.cam_id = i;
            pl.compressed = true;
            pl.ts = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
            pl.data = std::move(msg->data);
            const int64_t ns = (int64_t)msg->header.stamp.sec * 1000000000LL + (int64_t)msg->header.stamp.nanosec;
            on_camera(i, ns, pl.ts, std::move(pl));
          }));
    } else {
      subs_cam.push_back(node->create_subscription<sensor_msgs::msg::Image>(
          cam_topics[(size_t)i], mkqos(cam_depth), [&, i](const sensor_msgs::msg::Image::SharedPtr msg) {
            LivePayload pl;
            pl.cam_id = i;
            pl.compressed = false;
            pl.ts = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
            pl.width = msg->width; pl.height = msg->height; pl.step = msg->step;
            pl.data = std::move(msg->data);
            const int64_t ns = (int64_t)msg->header.stamp.sec * 1000000000LL + (int64_t)msg->header.stamp.nanosec;
            on_camera(i, ns, pl.ts, std::move(pl));
          }));
    }
  }
  ov_sched_crit_self();

  // Liveness probe.  QoS incompatibility is SILENT: a mismatched subscriber reports no
  // error, it just never fires.  This prints matched-publisher counts next to received
  // counts, so "subscribed" is never inferred from the absence of an error message.
  auto probe = node->create_wall_timer(std::chrono::seconds(5), [&, imu_topic, cam_topics, ncam] {
    std::string s;
    char b[256];
    std::snprintf(b, sizeof(b), "imu pubs=%zu recv=%ld |", sub_imu->get_publisher_count(), C_imu_recv.load());
    s += b;
    for (int i = 0; i < ncam; i++) {
      std::snprintf(b, sizeof(b), " cam%d pubs=%zu recv=%ld", i, subs_cam[(size_t)i]->get_publisher_count(),
                    C_cam_recv[i]);
      s += b;
    }
    std::snprintf(b, sizeof(b), " | sets=%ld fed=%ld poses=%ld", C_sets_closed.load(), C_sets_fed.load(),
                  C_poses.load());
    s += b;
    std::fprintf(stderr, "[online-live]: %s\n", s.c_str());
  });

  // =====================================================================================
  // Estimator thread.  Applies records in the SAME order the offline runner does.
  // =====================================================================================
  std::vector<std::string> lines;
  std::vector<double> lat_ms, proc_ms;
  double last_logged_ts = -1;
  std::map<int, double> cam_last_track;
  double latest_imu_t = -1;
  FILE *latf = [] { const char *p = std::getenv("OV_ONLINE_LAT"); return (p && *p) ? std::fopen(p, "w") : (FILE *)nullptr; }();
  // (frame-set stamp, steady clock) pairs: when each fed set was complete, and when the pose
  // carrying that stamp had been published.  Joined at exit into the [online-pose] census.
  // fed_at is written by the estimator thread only; emit_at by emit_pose only, which runs on the
  // estimator thread or -- OV_PUB_WHEN_READY -- on the pipeline worker, never on both at once.
  std::vector<std::pair<double, double>> fed_at, emit_at;
  fed_at.reserve(1 << 16);
  emit_at.reserve(1 << 16);

  // The ONE pose-output path.  poseimu is stamped by ROS2Visualizer::publish_state as
  // state->_timestamp + _calib_dt_CAMtoIMU (the ~-44 ms camera->IMU offset) -- it is NOT
  // re-stamped with now(), which would assert every pose ~44 ms in the future.
  const bool g_pipeline_on = sys->pipeline_on();
  auto emit_pose = [&]() {
    if (!sys->initialized()) return;
    auto state = sys->get_state();
    if ((state->_timestamp - sys->initialized_time()) < 1.0) return;
    if (state->_timestamp == last_logged_ts) return;
    last_logged_ts = state->_timestamp;
    if (rclcpp::ok()) { try { viz->visualize(); } catch (...) {} } // the drain runs after shutdown
    emit_at.emplace_back(state->_timestamp, now_s());
    Eigen::Vector4d q = state->_imu->quat();  // q_GtoI, JPL xyzw
    Eigen::Vector3d p = state->_imu->pos();   // p_IinG
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f", state->_timestamp, p(0), p(1), p(2),
                  q(0), q(1), q(2), q(3));
    lines.emplace_back(buf);
    C_poses++;
  };
  if (g_pipeline_on) {
    sys->set_pose_sink(emit_pose);
    std::fprintf(stderr, "[pipe]: ON -- pose output = %s (live-state read disabled)\n",
                 sys->pub_when_ready() ? "pipeline worker, when each update finishes (OV_PUB_WHEN_READY)" : "post-drain sink");
  } else {
    std::fprintf(stderr, "[pipe]: OFF -- pose output emitted inline after each fed frame-set\n");
  }

  int qdepth_at_feed = 0;
  auto feed_frame = [&](FrameSet &fr) {
    // Consumer watermark on SCOPE EXIT so it covers every early return: a ring slot staged
    // for ts <= watermark provably will never be claimed and is retired.
    struct NvMark { double ts; ~NvMark() { if (g_nvjpg) ov_core::ov_nvjpg_watermark(ts); } } _nvmark{fr.ts};
    const double t_feed0 = now_s();
    if (fr.has_fut) inflight_decodes--;

    auto throttled = [&](int lead) {
      if (track_dt <= 0.0) return false;
      auto it = cam_last_track.find(lead);
      if (it != cam_last_track.end() && fr.ts < it->second + track_dt) return true;
      cam_last_track[lead] = fr.ts;
      return false;
    };
    std::vector<std::vector<int>> groups;
    if (use_stereo && ncam % 2 == 0) {
      for (int p = 0; p < ncam / 2; p++) {
        int l = 2 * p, r = 2 * p + 1;
        if (!fr.payloads.count(l) || !fr.payloads.count(r)) continue;
        if (throttled(l)) continue;
        groups.push_back({l, r});
      }
    } else {
      // ONE grouped CameraData per frame-set (the OV_GROUP_CAMS shape the hybrid update
      // and TrackKLT's cross-camera parallel_for_ are designed around).
      std::vector<int> all;
      for (auto const &kv : fr.payloads) all.push_back(kv.first);
      if (!all.empty() && !throttled(all.front())) groups.push_back(all);
    }
    if (groups.empty()) {
      C_sets_throttled++;
      if (fr.has_fut) fr.fut.wait();
      return;
    }
    std::vector<int> need;
    for (auto const &g : groups)
      for (int c : g)
        if (std::find(need.begin(), need.end(), c) == need.end()) need.push_back(c);

    std::map<int, cv::Mat> imgs;
    if (fr.has_fut) {
      imgs = fr.fut.get();
    } else {
      // Synchronous decode (no prefetch was launched).  Goes through the same NVJPG stager
      // so the device-side contract is identical.
      std::vector<cv::Mat> decoded(need.size());
      cv::parallel_for_(cv::Range(0, (int)need.size()), [&](const cv::Range &rng) {
        for (int i = rng.start; i < rng.end; i++) {
          const LivePayload &pl = fr.payloads.at(need[i]);
          if (g_nvjpg && pl.compressed && !pl.data.empty()) {
            int w = 0, h = 0; size_t st = 0;
            const unsigned char *page = ov_core::ov_nvjpg_stage(need[i], pl.ts, pl.data.data(), pl.data.size(), &w, &h, &st);
            decoded[i] = page ? cv::Mat(h, w, CV_8UC1, const_cast<unsigned char *>(page), st) : pl.decode();
          } else {
            decoded[i] = pl.decode();
          }
        }
      });
      for (size_t i = 0; i < need.size(); i++)
        if (!decoded[i].empty()) imgs[need[i]] = decoded[i];
    }
    for (auto it = imgs.begin(); it != imgs.end();) {
      if (it->second.empty() || !std::count(need.begin(), need.end(), it->first)) it = imgs.erase(it);
      else ++it;
    }
    if (imgs.empty()) { C_decode_empty++; return; }

    const int rows = imgs.begin()->second.rows, cols = imgs.begin()->second.cols;
    bool fed_any = false;
    for (auto const &g : groups) {
      bool complete = true;
      for (int c : g) if (!imgs.count(c)) complete = false;
      if (!complete) continue;
      ov_core::CameraData msg;
      msg.timestamp = fr.ts;
      for (int c : g) {
        msg.sensor_ids.push_back(c);
        msg.images.push_back(imgs.at(c));
        msg.masks.push_back(get_mask(c, rows, cols));
      }
      sys->feed_measurement_camera(msg);
      C_imgs_fed += (long)g.size();
      fed_any = true;
    }
    if (!fed_any) return;
    C_sets_fed++;
    const double t_done = now_s();
    const double lat = 1000.0 * (t_done - fr.arrive_wall);
    const double proc = 1000.0 * (t_done - t_feed0);
    lat_ms.push_back(lat);
    proc_ms.push_back(proc);
    fed_at.emplace_back(fr.ts, fr.arrive_wall);
    // ts  arrival->fed ms  processing ms  ready-queue depth  images  1=prefetched decode
    //   col7 arrive_wall = the MONOTONIC (steady_clock, boot-epoch) reading at which this
    //   frame-set CLOSED, i.e. all its images were in hand.  It is NOT a wall/UTC epoch and it
    //   is NOT commensurable with fr.ts (the recording's camera stamp); only DIFFERENCES of
    //   col7 are meaningful, which is exactly what is needed.  Columns 1-6 are unchanged.
    //   Purpose: the end-to-end EKF2 age of a LIVE path is (arrival delay) + (compute) +
    //   (-toff), and the arrival delay -- DDS transport plus, in a bag-replay harness, the
    //   player's own pacing error -- is computable only as
    //       arrival_skew(i) = (col7_i - col7_0) - (ts_i - ts_0),
    //   which needs the arrival clock recorded beside the frame stamp.  Without it the census
    //   can only see the node's own compute and would flatter the age.
    if (latf)
      std::fprintf(latf, "%.9f %.3f %.3f %d %d %d %.9f\n", fr.ts, lat, proc, qdepth_at_feed, fr.n_images,
                   (int)fr.has_fut, fr.arrive_wall);

    // Publish + log the trajectory row.  Under OV_PIPELINE the update for THIS frame is
    // still running on the worker right now, so state->_imu / _Cov would be a TORN READ:
    // emit_pose is then driven by VioManager's pose_sink instead, which fires only where
    // the worker is provably idle.  Same value, same state->_timestamp.
    if (!g_pipeline_on) emit_pose();
  };

  std::deque<FrameSet> held; // closed frame-sets waiting for the IMU stream to pass their stamp
  std::thread est([&] {
    ov_sched_crit_self();
    for (;;) {
      std::deque<ov_core::ImuData> imu_batch;
      std::deque<FrameSet> frames;
      bool stopping = false;
      {
        std::unique_lock<std::mutex> lk(q_mtx);
        q_cv.wait_for(lk, std::chrono::milliseconds(20),
                      [&] { return stop_flag.load() || !imuq.empty() || !readyq.empty(); });
        imu_batch.swap(imuq);
        frames.swap(readyq);
        stopping = stop_flag.load();
      }
      // Apply IMU first, then flush every frame-set the IMU stream has passed -- the
      // offline flush_ready() predicate, unchanged.
      for (auto &f : frames) held.push_back(std::move(f));
      for (auto &m : imu_batch) {
        sys->feed_measurement_imu(m);
        latest_imu_t = m.timestamp;
      }
      while (!held.empty() && held.front().ts <= latest_imu_t) {
        qdepth_at_feed = (int)held.size();
        feed_frame(held.front());
        held.pop_front();
      }
      if (stopping) {
        // EOF drain, the offline equivalent of `latest_imu_t = inf; flush_ready();`:
        // the IMU stream has ended, so nothing is waiting on it any more.
        latest_imu_t = std::numeric_limits<double>::infinity();
        while (!held.empty()) { qdepth_at_feed = (int)held.size(); feed_frame(held.front()); held.pop_front(); }
        // OV_PIPELINE: the LAST tick's update is still in flight.  Drain it and take its
        // pose before anything below reads the state.
        if (g_pipeline_on) sys->pipeline_flush();
        break;
      }
    }
  });

  // Spin: executor on every core, niced down so it can never outrank tracking, but never
  // blocked either -- a blocked executor is a dropped message.
  std::thread spin([&] {
    if (g_aff_on) { sched_setaffinity(0, sizeof(cpu_set_t), &g_cpu_all); setpriority(PRIO_PROCESS, 0, 2); }
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
  });
  spin.join(); // returns on SIGINT / rclcpp shutdown

  // ---- drain -------------------------------------------------------------------------
  { // close everything still open in the assembler (EOF equivalent of the offline stash drain)
    std::lock_guard<std::mutex> lk(a_mtx);
    while (!open.empty()) { C_drain_close++; close_entry(open.begin()); }
  }
  stop_flag = true;
  q_cv.notify_all();
  est.join();
  if (latf) std::fclose(latf);

  if (!out_path.empty()) {
    std::ofstream of(out_path);
    for (auto const &l : lines) of << l << "\n";
    of.close();
  }

  // ---- CALIBRATION HARVEST.  Same JSON as run_serial_msckf's `<out>.calib.json` --------
  // The estimator thread has been joined and, under OV_PIPELINE, pipeline_flush() already ran
  // in its stopping branch, so the state is quiescent here -- this is not a torn read.
  // THE SCORER NEEDS "toff": pose stamps in the .tum are on the CAMERA clock and GT is on the
  // IMU clock; the physical instant is state->_timestamp + _calib_dt_CAMtoIMU.
  if (!out_path.empty()) {
    FILE *cf = std::fopen((out_path + ".calib.json").c_str(), "w");
    if (cf) {
      auto st = sys->get_state();
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
      std::fprintf(cf, "}\n");
      std::fclose(cf);
      std::fprintf(stderr, "[online]: calib toff=%.6f ms=%.3f written to %s.calib.json\n",
                   toff, 1000.0 * toff, out_path.c_str());
    }
  }

  // ---- CENSUS. This is the frame-accounting gate; read it, do not trust the exit code. --
  long cam_total = 0;
  std::string per_cam;
  for (int i = 0; i < ncam; i++) {
    cam_total += C_cam_recv[i];
    per_cam += (i ? "," : "") + std::to_string(C_cam_recv[i]);
  }
  std::fprintf(stderr,
               "[online]: imu_recv=%ld imu_bad=%ld cam_recv=[%s] cam_total=%ld\n"
               "[online]: sets_closed=%ld complete=%ld partial=%ld imgs_closed=%ld late_drop=%ld hold_close=%ld drain_close=%ld\n"
               "[online]: sets_fed=%ld imgs_fed=%ld throttled=%ld decode_empty=%ld prefetched=%ld sync_decode=%ld\n"
               "[online]: poses=%ld qdepth_max=%d imuq_max=%d\n",
               C_imu_recv.load(), C_imu_bad.load(), per_cam.c_str(), cam_total, C_sets_closed.load(),
               C_sets_complete.load(), C_sets_partial.load(), C_imgs_closed.load(), C_late_drop.load(),
               C_hold_close.load(), C_drain_close.load(), C_sets_fed.load(), C_imgs_fed.load(), C_sets_throttled.load(),
               C_decode_empty.load(), C_decode_prefetched.load(), C_decode_sync.load(), C_poses.load(),
               C_qdepth_max.load(), C_imuq_max.load());
  auto dump = [&](const char *tag, std::vector<double> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) { return v[(size_t)std::min(v.size() - 1, (size_t)(p * v.size()))]; };
    double sum = 0; for (double x : v) sum += x;
    std::fprintf(stderr, "%s: n=%zu mean=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f ms\n", tag, v.size(),
                 sum / v.size(), pct(0.50), pct(0.90), pct(0.99), v.back());
  };
  dump("[online-latency]", lat_ms);   // frame-set complete -> fed (queueing INCLUDED)
  dump("[online-proc]", proc_ms);     // time inside feed_frame (decode+track+update+publish)
  {
    // Frame-set complete -> the pose carrying that stamp published.  Split by how the pose came
    // about, because the two populations differ by design: a filter step (stamps update_min_dt
    // apart, the flight case) and a per-frame pose (zero-velocity update, on the ground).
    std::map<double, double> arrived;
    for (auto const &f : fed_at) arrived[f.first] = f.second;
    std::vector<double> age_step, age_frame;
    FILE *pf = [] { const char *p = std::getenv("OV_ONLINE_POSE_LAT"); return (p && *p) ? std::fopen(p, "w") : (FILE *)nullptr; }();
    const double step_gap = 0.75 * params.update_min_dt;
    double prev = -1.0;
    for (auto const &e : emit_at) {
      auto it = arrived.find(e.first);
      if (it != arrived.end() && it->second > 0) {
        const double age = 1000.0 * (e.second - it->second);
        const bool step = step_gap > 0 && prev > 0 && (e.first - prev) > step_gap;
        (step ? age_step : age_frame).push_back(age);
        if (pf) std::fprintf(pf, "%.9f %.3f %d %.9f\n", e.first, age, (int)step, e.second); // stamp  age ms  1=filter step  emit clock
      }
      prev = e.first;
    }
    if (pf) std::fclose(pf);
    dump("[online-pose]: filter steps", age_step);
    dump("[online-pose]: per-frame", age_frame);
  }
  if (g_nvjpg) std::fprintf(stderr, "%s\n", ov_core::ov_nvjpg_stats().c_str());
  std::fflush(stderr);

  g_decpool.reset();
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
