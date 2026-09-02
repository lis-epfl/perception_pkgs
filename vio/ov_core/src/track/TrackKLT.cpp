/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackKLT.h"

#include <cstdlib>
#include <cstring>

#include "Grider_FAST.h"
#include "Grider_GRID.h"
#include "cam/CamBase.h"
#include "cam/CamRadtan.h"
#include "cam/CamEqui.h"
#include "cam/CamDS.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "vprof.h"
#include "clahe_cuda.h"
#include "gpu_track.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <chrono>

namespace {
// mean track length = observations / features-created; the single number that says whether
// features are persisting across frames
struct TrkStats {
  std::atomic<long> created{0}, observations{0}, pruned{0}, kept_in{0};
  ~TrkStats() {
    long c = created.load(), o = observations.load();
    fprintf(stderr, "[trkstat]: features_created=%ld observations=%ld mean_track_len=%.2f"
                    " pruned_at_detect=%ld carried_in=%ld\n",
            c, o, c ? (double)o / c : 0.0, pruned.load(), kept_in.load());
  }
};
TrkStats g_trk_stats;
} // namespace


using namespace ov_core;

namespace ov_core {
double g_rt_behind_ms = 0.0;
std::atomic<int> g_frames_queued{0};
}

// Detection-cadence state (OV_DETECT_EVERY / OV_DETECT_ADAPT / OV_DETECT_AHEAD), file scope so
// both the per-frame gate and the end-of-feed submit-ahead can consult it.
static std::mutex g_det_mtx;
static std::map<size_t, int> g_det_ctr;
static std::map<size_t, double> g_det_ema;
// OV_DETECT_BURST_MIN state: frame number of the last ALLOWED burst per camera. Written ONLY by
// the consuming gate in perform_detection_monocular_gpu (single owner, exactly like g_det_ema);
// the submit-ahead paths read it. Guarded by g_det_mtx.
static std::map<size_t, long> g_det_last_burst;
// OV_DETECT_SPLIT: submit the detect kernels a frame ahead (end of feed_new_camera, i.e. AFTER
// every camera's gpu_commit, so d_prev IS the image pts_last lives in) on a separate cuda
// stream, and only harvest on the frame that consumes them.
static bool det_split_cfg() {
  static const bool v = [] { const char *e = std::getenv("OV_DETECT_SPLIT"); return e && *e == '1'; }();
  return v;
}
// OV_DETECT_PHASE_ABS: cadence keyed on an ABSOLUTE frame index instead of a per-camera counter
// that the adaptive-burst path resets. The 4 cameras are hardware-synced, so one burst resets
// all four to the same phase and they detect together from then on -- the p99 population.
static bool det_phase_abs() {
  static const bool v = [] { const char *e = std::getenv("OV_DETECT_PHASE_ABS"); return e && *e == '1'; }();
  return v;
}
static double det_adapt_frac() {
  static const double v = [] { const char *e = std::getenv("OV_DETECT_ADAPT"); return e ? atof(e) : 0.8; }();
  return v;
}
// OV_DETECT_BURST_MIN=K (0 = OFF, default): hysteresis on the OV_DETECT_ADAPT safety net.
// The 4 cameras are hardware-synced, so an aggressive yaw collapses KLT survival on all four
// at once, every camera bursts, and each burst does g_det_ctr[cam]=-phase, which re-phase-locks
// the cadence -- so they keep bursting together while the EMA decays (~9 frames at 0.95/0.05).
// Live N is pinned at the post-detect peak, and RANSAC is violently superlinear in N (MEASURED
// this round: 0.34 ms/cam at N~175 -> 19.7 ms/cam at N~375). K rate-limits REPEAT bursts on the
// same camera to at most one per K frames; the FIRST burst after a collapse is never suppressed
// and the ordinary cadence is never suppressed.
static int det_burst_min() {
  static const int v = [] { const char *e = std::getenv("OV_DETECT_BURST_MIN"); return e ? atoi(e) : 0; }();
  return v;
}
static std::atomic<long> g_frame_no{0};

// ---- OV_TRK_DUMP: per-frame tracking attribution (default OFF, pure instrumentation) ----
// Every accumulation is behind trk_dump_enabled(); with the gate unset nothing here is
// evaluated beyond one predicted-not-taken branch, and no estimator state is touched.
// NOTE: TrackBase's rT1..rT5 CANNOT be used for this. They are shared members of TrackBase
// (TrackBase.h:219) and feed_monocular runs under a 4-camera parallel_for_, so the four
// camera threads interleave their writes. All spans below use function-LOCAL steady_clock.
namespace ov_core {
std::atomic<double> g_trk_det_ms{0.0}, g_trk_klt_ms{0.0}, g_trk_rsc_ms{0.0};
std::atomic<double> g_trk_und_ms{0.0}, g_trk_db_ms{0.0};
std::atomic<long> g_trk_npts{0}, g_trk_ndet{0}, g_trk_npts_new{0}, g_trk_ncam{0};
} // namespace ov_core
bool ov_core_trk_dump_enabled() {
  static const bool v = [] { const char *e = std::getenv("OV_TRK_DUMP"); return e && *e && *e != '0'; }();
  return v;
}
static inline double trk_now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// std::atomic<double>::fetch_add is C++20; CAS loop keeps this valid under C++17.
static inline void trk_add(std::atomic<double> &a, double v) {
  double cur = a.load(std::memory_order_relaxed);
  while (!a.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {}
}
static std::atomic<long> g_det_hit{0}, g_det_stale{0}, g_det_inline{0}, g_det_burst{0};
static std::atomic<long> g_det_burst_sup{0};   // OV_DETECT_BURST_MIN: bursts rate-limited away
namespace {
struct DetSplitStats {
  ~DetSplitStats() {
    if (g_det_hit.load() || g_det_stale.load() || g_det_inline.load())
      fprintf(stderr, "[detsplit]: harvested=%ld stale_dropped=%ld inline=%ld bursts=%ld burst_suppressed=%ld\n",
              g_det_hit.load(), g_det_stale.load(), g_det_inline.load(), g_det_burst.load(),
              g_det_burst_sup.load());
  }
} g_detsplit_stats;
}

// ---- OV_RANSAC_TAIL instrumentation (default OFF; zero cost when the gate is off) ----
static std::atomic<long> g_rt_calls{0}, g_rt_exact{0}, g_rt_approx{0}, g_rt_mismatch{0};
static std::atomic<long> g_rt_degen{0};                       // probe returned no F -> forced exact
static std::atomic<long> g_rt_vcalls{0}, g_rt_vpts{0};        // approx-path verify (OV_RANSAC_TAIL_VERIFY=2)
static std::atomic<long> g_rt_vflip01{0}, g_rt_vflip10{0};    // ref=0->cand=1 (new inlier) / ref=1->cand=0 (new outlier)
static std::atomic<long> g_rt_vref1{0}, g_rt_vcand1{0};       // inlier counts under ref / candidate
// OV_RANSAC_HIST=1: gate-independent census of the RANSAC input size (diagnostic only).
static std::atomic<long> g_rh_calls{0}, g_rh_pts{0}, g_rh_b[6];
// ---- OV_RANSAC_OLDFIT instrumentation (default OFF; zero cost when the gate is off) ----
static std::atomic<long> g_of_calls{0}, g_of_approx{0}, g_of_degen{0}, g_of_passthru{0}, g_of_small{0};
static std::atomic<long> g_of_pts{0}, g_of_nfit{0}, g_of_b[6];   // n_fit histogram on the approx path
static std::atomic<long> g_of_vcalls{0}, g_of_vpts{0};
static std::atomic<long> g_of_vref1{0}, g_of_vcand1{0}, g_of_vflip01{0}, g_of_vflip10{0};
namespace {
struct RTailStats {
  ~RTailStats() {
    if (g_rt_calls.load())
      fprintf(stderr, "[rtail]: calls=%ld exact=%ld approx=%ld degen_exact=%ld mismatch=%ld\n",
              g_rt_calls.load(), g_rt_exact.load(), g_rt_approx.load(), g_rt_degen.load(), g_rt_mismatch.load());
    if (g_rt_vcalls.load())
      fprintf(stderr, "[rtailv]: approx_verified=%ld pts=%ld ref_in=%ld cand_in=%ld flip_0to1=%ld flip_1to0=%ld\n",
              g_rt_vcalls.load(), g_rt_vpts.load(), g_rt_vref1.load(), g_rt_vcand1.load(),
              g_rt_vflip01.load(), g_rt_vflip10.load());
    if (g_of_calls.load()) {
      fprintf(stderr, "[oldfit]: calls=%ld approx=%ld degen=%ld passthru=%ld small=%ld pts=%ld nfit=%ld\n",
              g_of_calls.load(), g_of_approx.load(), g_of_degen.load(), g_of_passthru.load(), g_of_small.load(),
              g_of_pts.load(), g_of_nfit.load());
      fprintf(stderr, "[oldfit] nfit_hist: <60=%ld 60-99=%ld 100-149=%ld 150-199=%ld 200-299=%ld >=300=%ld\n",
              g_of_b[0].load(), g_of_b[1].load(), g_of_b[2].load(), g_of_b[3].load(), g_of_b[4].load(), g_of_b[5].load());
    }
    if (g_of_vcalls.load())
      fprintf(stderr, "[oldfitv]: approx_verified=%ld pts=%ld ref_in=%ld cand_in=%ld flip_0to1=%ld flip_1to0=%ld\n",
              g_of_vcalls.load(), g_of_vpts.load(), g_of_vref1.load(), g_of_vcand1.load(),
              g_of_vflip01.load(), g_of_vflip10.load());
    if (g_rh_calls.load())
      fprintf(stderr, "[rhist]: calls=%ld meanpts=%.1f  <64=%ld 64-127=%ld 128-191=%ld 192-255=%ld 256-383=%ld >=384=%ld\n",
              g_rh_calls.load(), (double)g_rh_pts.load() / (double)g_rh_calls.load(),
              g_rh_b[0].load(), g_rh_b[1].load(), g_rh_b[2].load(), g_rh_b[3].load(), g_rh_b[4].load(), g_rh_b[5].load());
  }
} g_rtail_stats;
}
// OV_RANSAC_OLDFIT=1 (default OFF): fit the fundamental matrix on the OLD-TRACK PREFIX only,
// then classify ALL points exactly (Sampson) at the same threshold. See perform_matching.
static bool ransac_oldfit_on() {
  static const bool v = [] { const char *e = std::getenv("OV_RANSAC_OLDFIT"); return e && *e == '1'; }();
  return v;
}

// ================= OV_UNDIST_BATCH (default OFF) ==========================================
// Two bit-exact changes to point undistortion, behind ONE gate:
//  (1) perform_matching's 2N per-point CamBase::undistort_cv calls become 2 batched calls.
//  (2) the feature-DB write STOPS undistorting the surviving points a THIRD time and reuses
//      the pts1_n it already has. good_left[j] == pts_left_new[good_idx[j]] == pts1[good_idx[j]]
//      (perform_matching copies pts1 back into kpts1 before returning), so the normalized value
//      is exactly pts1_n[good_idx[j]]; _camw and _cam1 are the same calib_for(cam_id) object and
//      snapshot_calib() only runs on the main thread AFTER feed_new_camera returns, so the
//      calibration cannot move between the two sites.
// pts0_n is BATCHED but deliberately NOT CACHED across frames: the previous frame's normalized
// coords were computed against the PREVIOUS calib snapshot and the intrinsics are online-
// estimated states, so a cross-frame cache would perturb the RANSAC fit set -- the exact axis on
// which OV_RANSAC_TAIL / OV_RANSAC_SUB / OV_RANSAC_CONF all died.
static bool undist_batch_on() {
  static const bool v = [] { const char *e = std::getenv("OV_UNDIST_BATCH"); return e && *e == '1'; }();
  return v;
}
static bool undist_verify_on() {
  static const bool v = [] { const char *e = std::getenv("OV_UNDIST_BATCH_VERIFY"); return e && *e == '1'; }();
  return v;
}
namespace {
std::atomic<long> g_ub_n{0}, g_ub_bad{0};              // batched points checked / mismatching
std::atomic<long> g_ub_rn{0}, g_ub_rbad{0};            // DB-write reuses checked / mismatching
std::atomic<long> g_ub_maxdiff_bits{0};                // max |diff| as its IEEE-754 bit pattern
std::atomic<long> g_ub_calls{0}, g_ub_pts{0};          // batched calls / points (cost census)
std::atomic<int>  g_ub_banner{0};
inline void ub_maxdiff(float d) {
  if (!(d > 0.0f))
    return;
  long b;
  std::memcpy(&b, &d, 4); b &= 0xffffffffL;            // positive floats compare as ints
  long cur = g_ub_maxdiff_bits.load(std::memory_order_relaxed);
  while (b > cur && !g_ub_maxdiff_bits.compare_exchange_weak(cur, b, std::memory_order_relaxed)) {}
}
struct UndistStats {
  ~UndistStats() {
    if (g_ub_calls.load() == 0 && g_ub_n.load() == 0)
      return;
    long b = g_ub_maxdiff_bits.load(); float d = 0.0f; std::memcpy(&d, &b, 4);
    fprintf(stderr, "[undist]: batched_calls=%ld batched_pts=%ld | verify n=%ld bitexact=%ld"
                    " reuse_n=%ld reuse_bitexact=%ld maxabsdiff=%.3e\n",
            g_ub_calls.load(), g_ub_pts.load(),
            g_ub_n.load(), g_ub_n.load() - g_ub_bad.load(),
            g_ub_rn.load(), g_ub_rn.load() - g_ub_rbad.load(), (double)d);
  }
} g_undist_stats;
} // namespace
static int det_every_cfg() {
  static const int v = [] { const char *e = std::getenv("OV_DETECT_EVERY"); return e ? atoi(e) : 1; }();
  return v;
}

void TrackKLT::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // Preprocessing steps that we do not parallelize
  // NOTE: DO NOT PARALLELIZE THESE!
  // NOTE: These seem to be much slower if you parallelize them...
  rT1 = boost::posix_time::microsec_clock::local_time();
  size_t num_images = message.images.size();
  const bool gpu_mode = ov_core::gpu_track_enabled();
  const long frame_no = ++g_frame_no;   // one bump per frame-set (absolute detect phase)

  // Create every per-camera map entry HERE, serially, before any parallel section.
  // std::map / std::unordered_map operator[] INSERTS on first access; on the very first
  // frame-set the four camera threads all did that first insert concurrently, corrupting the
  // container (captured core: RB-tree insert_and_rebalance SIGSEGV under det_submitted[cam_id]
  // in perform_detection_monocular_gpu — the rare "startup segfault", ~2% of cold starts).
  // With all keys pre-created, parallel code only ever assigns through existing nodes: no
  // rebalance, no rehash, no structural mutation. emplace is a no-op once the key exists.
  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {
    size_t cid = message.sensor_ids.at(msg_id);
    det_submitted.emplace(cid, false);
    det_pending_gen.emplace(cid, -1L);
    pts_last.emplace(cid, std::vector<cv::KeyPoint>());
    ids_last.emplace(cid, std::vector<size_t>());
    img_last.emplace(cid, cv::Mat());
    img_pyramid_last.emplace(cid, std::vector<cv::Mat>());
    img_mask_last.emplace(cid, cv::Mat());
    img_curr.emplace(cid, cv::Mat());
    img_pyramid_curr.emplace(cid, std::vector<cv::Mat>());
  }
  static const bool gpu_klt_only_fnc = [] {
    const char *e = std::getenv("OV_GPU_TRACK");
    return e && *e == '2';
  }();

  // Batched GPU CLAHE for the WHOLE frame-set: one upload, one set of launches, one sync.
  // Per-image GPU calls serialise on the device and lose to threaded cv::CLAHE (measured
  // 8.26 s vs 6.24 s over a 60 s window at num_opencv_threads=8), so batching is the point.
  std::vector<cv::Mat> gpu_eq;
  if (!gpu_mode && histogram_method == HistogramMethod::CLAHE && ov_core::clahe_cuda_enabled() && num_images > 0) {
    bool all_ok = !message.images.at(0).empty();
    const int bw = all_ok ? message.images.at(0).cols : 0;
    const int bh = all_ok ? message.images.at(0).rows : 0;
    for (size_t i = 0; i < num_images && all_ok; i++) {
      const cv::Mat &m = message.images.at(i);
      if (m.type() != CV_8UC1 || m.cols != bw || m.rows != bh) all_ok = false;
    }
    if (all_ok) {
      VPROF("1.track/a_clahe_gpu_batch");
      gpu_eq.resize(num_images);
      std::vector<const unsigned char *> srcs(num_images);
      std::vector<unsigned char *> dsts(num_images);
      std::vector<size_t> sst(num_images), dst(num_images);
      for (size_t i = 0; i < num_images; i++) {
        gpu_eq[i].create(bh, bw, CV_8UC1);
        srcs[i] = message.images.at(i).data;
        sst[i] = message.images.at(i).step;
        dsts[i] = gpu_eq[i].data;
        dst[i] = gpu_eq[i].step;
      }
      if (!ov_core::clahe_cuda_batch(srcs.data(), sst.data(), dsts.data(), dst.data(),
                                     (int)num_images, bw, bh, 10.0f))
        gpu_eq.clear();   // fall back to CPU below
    }
  }

  for (size_t msg_id = 0; msg_id < num_images; msg_id++) {

    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize (skipped entirely in GPU mode -- done on the device below)
    cv::Mat img;
    if (gpu_mode) {
      img = message.images.at(msg_id);
    } else if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(msg_id), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      VPROF("1.track/a_clahe");
      if (!gpu_eq.empty()) {
        img = gpu_eq[msg_id];               // produced by the batched GPU pass above
      } else {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
        clahe->apply(message.images.at(msg_id), img);
        // OV_CLAHE_PERTURB=1: nudge ~0.3% of pixels by +-1 level -- the SAME magnitude by which
        // the CUDA CLAHE differs from cv::CLAHE. Calibrates how much ATE moves for a
        // perturbation of that size, so a GPU-vs-CPU ATE gap can be judged against it.
        static const bool perturb = [] {
          const char *e = std::getenv("OV_CLAHE_PERTURB");
          return e && *e == '1';
        }();
        if (perturb) {
          uint32_t st = 0x9E3779B9u ^ (uint32_t)cam_id;
          for (int r = 0; r < img.rows; r++) {
            uchar *row = img.ptr<uchar>(r);
            for (int cc = 0; cc < img.cols; cc++) {
              st ^= st << 13; st ^= st >> 17; st ^= st << 5;
              if ((st & 0x3FFu) == 0) {           // ~1/1024 of pixels
                int v = row[cc] + (((st >> 10) & 1u) ? 1 : -1);
                row[cc] = (uchar)(v < 0 ? 0 : (v > 255 ? 255 : v));
              }
            }
          }
        }
      }
    } else {
      img = message.images.at(msg_id);
    }

    // Extract image pyramid
    std::vector<cv::Mat> imgpyr;
    if (gpu_mode && gpu_klt_only_fnc) {
      // bisection: CPU CLAHE + CPU pyramid, but ALSO prepare the device so GPU KLT can run
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
      clahe->apply(message.images.at(msg_id), img);
      cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
      if (!ov_core::gpu_prepare(cam_id, message.images.at(msg_id).data, message.images.at(msg_id).cols,
                                message.images.at(msg_id).rows, message.images.at(msg_id).step,
                                message.masks.at(msg_id).empty() ? nullptr : message.masks.at(msg_id).data,
                                message.masks.at(msg_id).empty() ? 0 : message.masks.at(msg_id).step,
                                true, 10.0f)) {
        PRINT_ERROR(RED "[gpu_track]: gpu_prepare failed\n" RESET);
        std::exit(EXIT_FAILURE);
      }
    } else if (gpu_mode) {
      // CLAHE + pyramid happen on the device and STAY there. img_curr keeps the raw host
      // image (already resident, no download) purely so downstream size/None checks and any
      // visualisation still have something valid; img_pyramid_curr is intentionally empty
      // because nothing on the GPU path reads it.
      VPROF("1.track/b_pyramid");
      bool preseeded_hit = false;
      {
        std::lock_guard<std::mutex> lk(preseed_mtx);
        auto it = preseeded.find(cam_id);
        if (it != preseeded.end()) {
          if (it->second == message.timestamp) preseeded_hit = true;
          preseeded.erase(it); // stale entries (throttle-dropped frames) just fall through
        }
      }
      if (!preseeded_hit &&
          !ov_core::gpu_prepare(cam_id, message.images.at(msg_id).data, message.images.at(msg_id).cols,
                                message.images.at(msg_id).rows, message.images.at(msg_id).step,
                                message.masks.at(msg_id).empty() ? nullptr : message.masks.at(msg_id).data,
                                message.masks.at(msg_id).empty() ? 0 : message.masks.at(msg_id).step,
                                histogram_method == HistogramMethod::CLAHE, 10.0f, message.timestamp)) {
        PRINT_ERROR(RED "[gpu_track]: gpu_prepare failed for cam %zu\n" RESET, cam_id);
        std::exit(EXIT_FAILURE);
      }
      img = message.images.at(msg_id);
    } else {
      VPROF("1.track/b_pyramid");
      cv::buildOpticalFlowPyramid(img, imgpyr, win_size, pyr_levels);
    }

    // Save!
    img_curr[cam_id] = img;
    img_pyramid_curr[cam_id] = imgpyr;
  }

  // Single-submitter: with a grouped message all prepares are already queued; queue every
  // camera's detection too before spawning the parallel completion threads.
  // Measured: no gain (47.2 vs 45.6 ms/iter) — the parallel threads already submit within
  // microseconds, and the duplicated bookkeeping costs ~2 ms serial. Kept behind env.
  static const bool submit1 = [] { const char *e = std::getenv("OV_SUBMIT1"); return e && *e == '1'; }();
  if (submit1 && gpu_mode && num_images > 1) {
    for (size_t msg_id = 0; msg_id < num_images; msg_id++) {
      size_t cid = message.sensor_ids.at(msg_id);
      if (!pts_last[cid].empty()) {
        perform_detection_submit(cid, message.masks.at(msg_id), true);
        if (det_submitted[cid]) det_pending_gen[cid] = frame_no;   // consumed on THIS frame
      }
    }
  }

  // Either call our stereo or monocular version
  // If we are doing binocular tracking, then we should parallize our tracking
  if (num_images == 1) {
    feed_monocular(message, 0);
  } else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  } else if (num_images >= 2 && (num_images % 2 == 0) && use_stereo) {
    // Multi-stereo: process consecutive (2p,2p+1) pairs as independent stereo
    parallel_for_(cv::Range(0, (int)(num_images / 2)), LambdaBody([&](const cv::Range &range) {
                    for (int p = range.start; p < range.end; p++) {
                      feed_stereo(message, 2 * p, 2 * p + 1);
                    }
                  }));
  } else if (!use_stereo) {
    parallel_for_(cv::Range(0, (int)num_images), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++) {
                      feed_monocular(message, i);
                    }
                  }));
  } else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or N stereo pairs (even N)", num_images);
    std::exit(EXIT_FAILURE);
  }

  // Submit-ahead (OV_DETECT_AHEAD=1): a camera whose cadence fires NEXT frame detects on the
  // image that just became "previous" here, with the occupancy/mask state that is final now.
  // Submitting at end-of-feed lets the FAST/NMS/select kernels run while the CPU is inside the
  // EKF update, erasing the detection spike from the frame that completes it.
  static const bool det_ahead = [] { const char *e = std::getenv("OV_DETECT_AHEAD"); return e && *e == '1'; }();
  if ((det_ahead || det_split_cfg()) && gpu_mode && !gpu_klt_only_fnc && det_every_cfg() > 1) {
    for (size_t msg_id = 0; msg_id < num_images; msg_id++) {
      size_t cid = message.sensor_ids.at(msg_id);
      bool due;
      {
        static const bool stagger = [] { const char *e = std::getenv("OV_DETECT_STAGGER"); return e && *e == '1'; }();
        const int phase = stagger ? (int)((cid * (size_t)det_every_cfg()) / 4) : 0;
        std::lock_guard<std::mutex> lk(g_det_mtx);
        // Must mirror the consumption gate EXACTLY (see perform_detection_monocular_gpu):
        //   abs phase: (frame + phase) % every == 0, evaluated for the NEXT frame
        //   per-cam counter: (++ctr + phase) % every == 0
        due = det_phase_abs() ? (((frame_no + 1 + phase) % det_every_cfg()) == 0)
                              : (((g_det_ctr[cid] + 1 + phase) % det_every_cfg()) == 0);
      }
      // Cheap pre-filter for the burst case: pruning only ever REMOVES points, so a camera
      // whose raw live count is already comfortably above the burst threshold cannot burst.
      // 15% slack keeps it conservative; a miss just means that detection runs inline.
      bool maybe_burst = false;
      if (!due) {
        const double af = det_adapt_frac();
        std::lock_guard<std::mutex> lk(g_det_mtx);
        const double m = g_det_ema.count(cid) ? g_det_ema[cid] : 0.0;
        maybe_burst = (af > 0.0 && m > 0.0 && (double)pts_last[cid].size() < 1.15 * af * m);
        // Same OV_DETECT_BURST_MIN rate limit as the two gates below (read-only). Only ever
        // clears maybe_burst, never `due`, so cadence submissions are unaffected.
        const int bmin = det_burst_min();
        if (maybe_burst && bmin > 0) {
          const long fn_next = frame_no + 1;
          auto itb = g_det_last_burst.find(cid);
          if (itb != g_det_last_burst.end() && fn_next - itb->second < (long)bmin)
            maybe_burst = false;
        }
      }
      if ((due || maybe_burst) && !pts_last[cid].empty() && !det_submitted[cid]) {
        perform_detection_submit(cid, message.masks.at(msg_id), due);
        if (det_submitted[cid]) det_pending_gen[cid] = frame_no + 1;   // consumed NEXT frame
      }
    }
  }
}

void TrackKLT::preseed_gpu(const CameraData &message) {
  if (!ov_core::gpu_track_enabled()) return;
  for (size_t msg_id = 0; msg_id < message.sensor_ids.size(); msg_id++) {
    size_t cam_id = message.sensor_ids.at(msg_id);
    if (cam_id >= mtx_feeds.size()) continue;
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));
    {
      std::lock_guard<std::mutex> lk(preseed_mtx);
      auto it = preseeded.find(cam_id);
      if (it != preseeded.end() && it->second == message.timestamp) continue; // already queued
    }
    const cv::Mat &im = message.images.at(msg_id);
    const cv::Mat &mk = message.masks.at(msg_id);
    if (im.empty() || im.type() != CV_8UC1) continue;
    if (!ov_core::gpu_prepare(cam_id, im.data, im.cols, im.rows, im.step,
                              mk.empty() ? nullptr : mk.data, mk.empty() ? 0 : mk.step,
                              histogram_method == HistogramMethod::CLAHE, 10.0f, message.timestamp))
      continue;
    std::lock_guard<std::mutex> lk(preseed_mtx);
    preseeded[cam_id] = message.timestamp;
  }
}

void TrackKLT::feed_monocular(const CameraData &message, size_t msg_id) {

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Get our image objects for this image
  cv::Mat img = img_curr.at(cam_id);
  std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
  cv::Mat mask = message.masks.at(msg_id);
  const bool gpu_mode = ov_core::gpu_track_enabled();
  static const bool gpu_klt_only = [] {
    const char *e = std::getenv("OV_GPU_TRACK");
    return e && *e == '2';
  }();
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id].empty()) {
    // Detect new features
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    if (gpu_mode && !gpu_klt_only) {
      // no previous frame yet -> seed from the CURRENT resident image
      perform_detection_monocular_gpu(cam_id, true, img.cols, img.rows, mask, good_left, good_ids_left);
      ov_core::gpu_commit(cam_id);
    } else if (gpu_mode) {
      perform_detection_monocular(imgpyr, mask, good_left, good_ids_left, cam_id);
      ov_core::gpu_commit(cam_id);
    } else {
      perform_detection_monocular(imgpyr, mask, good_left, good_ids_left, cam_id);
    }
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id].size();
  auto pts_left_old = pts_last[cam_id];
  auto ids_left_old = ids_last[cam_id];
  // OV_RANSAC_OLDFIT boundary. NOTE: `pts_before_detect` is NOT the old/new boundary. The
  // detection call PRUNES pts_left_old first (edge margin, min_px_dist occupancy, grid bounds,
  // mask -- TrackKLT.cpp erase sites in perform_detection_monocular{,_gpu}) and only THEN appends
  // this frame's keypoints, so the pre-call size overcounts the survivors by the number pruned
  // and would feed fresh keypoints into the fit set -- precisely the contamination this lever
  // exists to remove. Ids are assigned from ++currid, so every new id is strictly greater than
  // every surviving old id and is appended at the end: the survivors are an exact contiguous
  // prefix, recoverable after the fact regardless of which detection path ran.
  const bool oldfit_on = ransac_oldfit_on();
  const size_t max_old_id =
      (oldfit_on && !ids_left_old.empty()) ? *std::max_element(ids_left_old.begin(), ids_left_old.end()) : 0;
  // OV_DETECT_AFTER=1: run detection AFTER matching so KLT never queues behind the FAST/NMS
  // kernels on the camera stream. New features then start tracking one frame later (they are
  // appended to pts_last at the end); the seeding path (empty pts) keeps the classic order.
  static const bool det_after = [] { const char *e = std::getenv("OV_DETECT_AFTER"); return e && *e == '1'; }();
  const bool defer_detect = det_after && gpu_mode && !gpu_klt_only && !pts_left_old.empty();
  const bool _tdmp = ov_core_trk_dump_enabled();
  const double _tk0 = _tdmp ? trk_now_s() : 0.0;
  if (!defer_detect) { VPROF("1.track/c_detect");
  if (gpu_mode && !gpu_klt_only)
    perform_detection_monocular_gpu(cam_id, false, img.cols, img.rows, img_mask_last[cam_id], pts_left_old, ids_left_old);
  else
    perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old, cam_id); }
  double _tk1 = 0.0;
  if (_tdmp) {
    _tk1 = trk_now_s();
    trk_add(ov_core::g_trk_det_ms, 1000.0 * (_tk1 - _tk0));
    // live count AFTER top-off == exactly what gpu_klt is asked to track this frame
    ov_core::g_trk_npts.fetch_add((long)pts_left_old.size(), std::memory_order_relaxed);
    ov_core::g_trk_ncam.fetch_add(1, std::memory_order_relaxed);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Count the surviving old tracks (contiguous prefix, see the note above). -1 = gate off, which
  // makes perform_matching take today's path verbatim.
  int n_old_fit = -1;
  if (oldfit_on && !ids_left_old.empty()) {
    size_t k = 0;
    while (k < ids_left_old.size() && ids_left_old[k] <= max_old_id)
      k++;
    n_old_fit = (int)k;
  }

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

  // OV_UNDIST_BATCH: filled by perform_matching with the normalized coords of pts_left_new.
  // Left EMPTY when the gate is off or on any early return -> the DB write below falls back to
  // today's per-point undistort_cv, so this is inert unless the gate is armed.
  const bool _ub = undist_batch_on();
  std::vector<cv::Point2f> pts1_norm;

  // Lets track temporally
  { VPROF("1.track/d_match");
  if (gpu_mode) {
    std::vector<float> px, py, cx, cy;
    std::vector<uchar> st;
    px.reserve(pts_left_old.size()); py.reserve(pts_left_old.size());
    for (auto &k : pts_left_old) { px.push_back(k.pt.x); py.push_back(k.pt.y); }
    // Flow-adaptive pyramid depth: a full-pyramid pilot pass inside gpu_klt measures the
    // CURRENT frame's flow and picks the start level (-2 = auto). Redo below is belt-and-braces.
    static const bool adapt_lvl = [] { const char *e = std::getenv("OV_KLT_ADAPT_LVL"); return e && *e == '1'; }();
    const int lvl0 = adapt_lvl ? -2 : -1;
    if (!ov_core::gpu_klt(cam_id, px, py, cx, cy, st, lvl0)) {
      PRINT_ERROR(RED "[gpu_track]: gpu_klt failed for cam %zu\n" RESET, cam_id);
      std::exit(EXIT_FAILURE);
    }
    if (adapt_lvl && !st.empty()) {
      size_t ok = 0;
      for (auto v : st) ok += (v != 0);
      if ((double)ok < 0.6 * (double)st.size()) {
        // burst: coarse levels were needed after all -- redo full pyramid, same inputs
        if (!ov_core::gpu_klt(cam_id, px, py, cx, cy, st, -1)) {
          PRINT_ERROR(RED "[gpu_track]: gpu_klt redo failed for cam %zu\n" RESET, cam_id);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    for (size_t i = 0; i < pts_left_new.size() && i < cx.size(); i++) {
      pts_left_new[i].pt.x = cx[i];
      pts_left_new[i].pt.y = cy[i];
    }
    // pre_status short-circuits the CPU KLT; undistortion and RANSAC still run below
    perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll, &st, n_old_fit,
                     _ub ? &pts1_norm : nullptr);
  } else {
    perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll, nullptr, n_old_fit,
                     _ub ? &pts1_norm : nullptr);
  } }
  assert(pts_left_new.size() == ids_left_old.size());
  double _tk2 = 0.0;
  if (_tdmp) { _tk2 = trk_now_s(); trk_add(ov_core::g_trk_klt_ms, 1000.0 * (_tk2 - _tk1)); }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty()) {
    if (gpu_mode)
      ov_core::gpu_commit(cam_id);
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id].clear();
    ids_last[cam_id].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;
  // OV_UNDIST_BATCH: source index of each survivor, i.e. good_left[j] == pts_left_new[good_idx[j]].
  std::vector<size_t> good_idx;
  good_left.reserve(pts_left_new.size());
  good_ids_left.reserve(pts_left_new.size());
  good_idx.reserve(pts_left_new.size());

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
        (int)pts_left_new.at(i).pt.y >= img.rows)
      continue;
    // Check if it is in the mask
    // NOTE: mask has max value of 255 (white) if it should be
    if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
      continue;
    // If it is a good track, and also tracked from left to right
    if (mask_ll[i]) {
      good_left.push_back(pts_left_new[i]);
      good_ids_left.push_back(ids_left_old[i]);
      good_idx.push_back(i);
    }
  }

  // Update our feature database, with theses new observations
  static const bool defer_writes = [] {
    const char *e = std::getenv("OV_ASYNC_UPDATE");
    return e && *e == '1';
  }();
  ov_core::CamBase *_camw = calib_for(cam_id);
  // OV_UNDIST_BATCH step 2: these M points were ALREADY undistorted as pts1_n inside
  // perform_matching a few hundred microseconds ago, with the SAME camera object (_camw and
  // perform_matching's _cam1 are both calib_for(cam_id)) and the SAME calibration snapshot
  // (snapshot_calib() runs on the main thread only AFTER feed_new_camera returns). Reuse is
  // bit-exact. The size guard makes every early-return path fall back to today's code.
  const bool _ub_reuse = _ub && pts1_norm.size() == pts_left_new.size();
  const bool _ub_vfy = _ub_reuse && undist_verify_on();
  auto _npt = [&](size_t i) -> cv::Point2f {
    if (!_ub_reuse)
      return _camw->undistort_cv(good_left.at(i).pt);
    const cv::Point2f v = pts1_norm[good_idx[i]];
    if (_ub_vfy) {
      const cv::Point2f r = _camw->undistort_cv(good_left.at(i).pt);
      g_ub_rn.fetch_add(1, std::memory_order_relaxed);
      if (std::memcmp(&r, &v, sizeof(cv::Point2f)) != 0) {
        g_ub_rbad.fetch_add(1, std::memory_order_relaxed);
        ub_maxdiff(std::fabs(r.x - v.x)); ub_maxdiff(std::fabs(r.y - v.y));
      }
    }
    return v;
  };
  if (defer_writes || defer_runtime.load(std::memory_order_relaxed)) {
    // The update worker may still be iterating Feature maps; buffer and commit after the join.
    std::lock_guard<std::mutex> lkp(pending_mtx);
    // OV_DETERMINISTIC: the four camera threads append to ONE vector under this mutex, so the
    // database insertion SEQUENCE -- and therefore libstdc++'s unordered_map bucket order, and
    // therefore feats_lost / feats_marg / feats_maxtracks -- varies with thread interleaving.
    // Per-camera buckets concatenated in sensor-id order at flush make it a fixed sequence at
    // zero cost; within a camera the natural track order is preserved exactly.
    std::vector<PendingObs> &dst =
        (det_ids_on() && cam_id < PEND_NCAM) ? pending_by_cam[cam_id] : pending_obs;
    for (size_t i = 0; i < good_left.size(); i++) {
      cv::Point2f npt_l = _npt(i);
      dst.push_back({good_ids_left.at(i), message.timestamp, cam_id,
                     good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y});
      g_trk_stats.observations++;
    }
  } else {
    VPROF("1.track/e_db_write");
    std::vector<FeatureDatabase::BulkObs> bulk;
    bulk.reserve(good_left.size());
    for (size_t i = 0; i < good_left.size(); i++) {
      cv::Point2f npt_l = _npt(i);
      bulk.push_back({good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y});
      g_trk_stats.observations++;
    }
    database->update_features_bulk(bulk);
  }


  // Move forward in time
  {
    VPROF("1.track/f_advance");
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_pyramid_last[cam_id] = imgpyr;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
  }
  if (defer_detect) {
    // deferred top-off: detect on what is now the PREVIOUS image (post-commit), appending to
    // pts_last so the new points are tracked from the next frame on. Same inputs the classic
    // order would use next frame.
    std::vector<cv::KeyPoint> pts_new = pts_last[cam_id];
    std::vector<size_t> ids_new = ids_last[cam_id];
    { VPROF("1.track/c_detect");
    perform_detection_monocular_gpu(cam_id, false, img.cols, img.rows, mask, pts_new, ids_new); }
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    pts_last[cam_id] = pts_new;
    ids_last[cam_id] = ids_new;
  }
  // Device-side equivalent: the current pyramid/image/mask become the previous ones. Must
  // happen on EVERY frame, not just the seeding one, or the previous pyramid never advances
  // and KLT would keep tracking against frame 0.
  if (gpu_mode)
    ov_core::gpu_commit(cam_id);
  if (_tdmp) trk_add(ov_core::g_trk_db_ms, 1000.0 * (trk_now_s() - _tk2));
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%zu detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Get our image objects for this image
  cv::Mat img_left = img_curr.at(cam_id_left);
  cv::Mat img_right = img_curr.at(cam_id_right);
  std::vector<cv::Mat> imgpyr_left = img_pyramid_curr.at(cam_id_left);
  std::vector<cv::Mat> imgpyr_right = img_pyramid_curr.at(cam_id_right);
  cv::Mat mask_left = message.masks.at(msg_id_left);
  cv::Mat mask_right = message.masks.at(msg_id_right);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // If we didn't have any successful tracks last time, just extract this time
  // This also handles, the tracking initalization on the first call to this extractor
  if (pts_last[cam_id_left].empty() && pts_last[cam_id_right].empty()) {
    // Track into the new image
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    perform_detection_stereo(imgpyr_left, imgpyr_right, mask_left, mask_right, cam_id_left, cam_id_right, good_left, good_right,
                             good_ids_left, good_ids_right);
    // Save the current image and pyramid
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    return;
  }

  // First we should make that the last images have enough features so we can do KLT
  // This will "top-off" our number of tracks so always have a constant number
  int pts_before_detect = (int)pts_last[cam_id_left].size();
  auto pts_left_old = pts_last[cam_id_left];
  auto pts_right_old = pts_last[cam_id_right];
  auto ids_left_old = ids_last[cam_id_left];
  auto ids_right_old = ids_last[cam_id_right];
  perform_detection_stereo(img_pyramid_last[cam_id_left], img_pyramid_last[cam_id_right], img_mask_last[cam_id_left],
                           img_mask_last[cam_id_right], cam_id_left, cam_id_right, pts_left_old, pts_right_old, ids_left_old,
                           ids_right_old);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Our return success masks, and predicted new features
  std::vector<uchar> mask_ll, mask_rr;
  std::vector<cv::KeyPoint> pts_left_new = pts_left_old;
  std::vector<cv::KeyPoint> pts_right_new = pts_right_old;

  // Lets track temporally
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
                  for (int i = range.start; i < range.end; i++) {
                    bool is_left = (i == 0);
                    perform_matching(img_pyramid_last[is_left ? cam_id_left : cam_id_right], is_left ? imgpyr_left : imgpyr_right,
                                     is_left ? pts_left_old : pts_right_old, is_left ? pts_left_new : pts_right_new,
                                     is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                                     is_left ? mask_ll : mask_rr);
                  }
                }));
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // left to right matching
  // TODO: we should probably still do this to reject outliers
  // TODO: maybe we should collect all tracks that are in both frames and make they pass this?
  // std::vector<uchar> mask_lr;
  // perform_matching(imgpyr_left, imgpyr_right, pts_left_new, pts_right_new, cam_id_left, cam_id_right, mask_lr);
  rT5 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // If any of our masks are empty, that means we didn't have enough to do ransac, so just return
  if (mask_ll.empty() && mask_rr.empty()) {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left].clear();
    pts_last[cam_id_right].clear();
    ids_last[cam_id_left].clear();
    ids_last[cam_id_right].clear();
    PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
    return;
  }

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;

  // Loop through all left points
  for (size_t i = 0; i < pts_left_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x > img_left.cols ||
        (int)pts_left_new.at(i).pt.y > img_left.rows)
      continue;
    // See if we have the same feature in the right
    bool found_right = false;
    size_t index_right = 0;
    for (size_t n = 0; n < ids_right_old.size(); n++) {
      if (ids_left_old.at(i) == ids_right_old.at(n)) {
        found_right = true;
        index_right = n;
        break;
      }
    }
    // If it is a good track, and also tracked from left to right
    // Else track it as a mono feature in just the left image
    if (mask_ll[i] && found_right && mask_rr[index_right]) {
      // Ensure we do not have any bad KLT tracks (i.e., points are negative)
      if (pts_right_new.at(index_right).pt.x < 0 || pts_right_new.at(index_right).pt.y < 0 ||
          (int)pts_right_new.at(index_right).pt.x >= img_right.cols || (int)pts_right_new.at(index_right).pt.y >= img_right.rows)
        continue;
      good_left.push_back(pts_left_new.at(i));
      good_right.push_back(pts_right_new.at(index_right));
      good_ids_left.push_back(ids_left_old.at(i));
      good_ids_right.push_back(ids_right_old.at(index_right));
      // PRINT_DEBUG("adding to stereo - %u , %u\n", ids_left_old.at(i), ids_right_old.at(index_right));
    } else if (mask_ll[i]) {
      good_left.push_back(pts_left_new.at(i));
      good_ids_left.push_back(ids_left_old.at(i));
      // PRINT_DEBUG("adding to left - %u \n",ids_left_old.at(i));
    }
  }

  // Loop through all right points
  for (size_t i = 0; i < pts_right_new.size(); i++) {
    // Ensure we do not have any bad KLT tracks (i.e., points are negative)
    if (pts_right_new.at(i).pt.x < 0 || pts_right_new.at(i).pt.y < 0 || (int)pts_right_new.at(i).pt.x >= img_right.cols ||
        (int)pts_right_new.at(i).pt.y >= img_right.rows)
      continue;
    // See if we have the same feature in the right
    bool added_already = (std::find(good_ids_right.begin(), good_ids_right.end(), ids_right_old.at(i)) != good_ids_right.end());
    // If it has not already been added as a good feature, add it as a mono track
    if (mask_rr[i] && !added_already) {
      good_right.push_back(pts_right_new.at(i));
      good_ids_right.push_back(ids_right_old.at(i));
      // PRINT_DEBUG("adding to right - %u \n", ids_right_old.at(i));
    }
  }

  // Update our feature database, with theses new observations.
  // NOTE: this path had NO defer branch, so with deferred writes armed (OV_OVERLAP_UPD /
  // OV_PIPELINE) it wrote Feature vectors while the update worker iterated them -- the
  // documented SIGSEGV class.  Latent today (use_stereo=false in every deployed and bench
  // config) but fixed here so it cannot go live by flipping one yaml key.
  ov_core::CamBase *_camL = calib_for(cam_id_left), *_camR = calib_for(cam_id_right);
  if (defer_runtime.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lkp(pending_mtx);
    for (size_t i = 0; i < good_left.size(); i++) {
      cv::Point2f npt_l = _camL->undistort_cv(good_left.at(i).pt);
      pending_obs.push_back({good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y});
    }
    for (size_t i = 0; i < good_right.size(); i++) {
      cv::Point2f npt_r = _camR->undistort_cv(good_right.at(i).pt);
      pending_obs.push_back({good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y});
    }
  } else {
    for (size_t i = 0; i < good_left.size(); i++) {
      cv::Point2f npt_l = _camL->undistort_cv(good_left.at(i).pt);
      database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                               npt_l.y);
    }
    for (size_t i = 0; i < good_right.size(); i++) {
      cv::Point2f npt_r = _camR->undistort_cv(good_right.at(i).pt);
      database->update_feature(good_ids_right.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                               npt_r.y);
    }
  }


  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_pyramid_last[cam_id_left] = imgpyr_left;
    img_pyramid_last[cam_id_right] = imgpyr_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
  }
  rT6 = boost::posix_time::microsec_clock::local_time();

  //  // Timing information
  PRINT_ALL("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for detection (%d detected)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)pts_last[cam_id_left].size() - pts_before_detect);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for stereo klt\n", (rT5 - rT4).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT6 - rT5).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-KLT]: %.4f seconds for total\n", (rT6 - rT1).total_microseconds() * 1e-6);
}

void TrackKLT::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, size_t cam_id) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less than grid_px_size points away then existing features
  cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                      (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  cv::Mat grid_polar = cv::Mat::zeros(1, std::max(1, polar_rings * polar_sectors), CV_16UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x);
    int y_grid = std::floor(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    }
    if (use_polar_grid) {
      int pb = polar_bin(cam_id, kpt.pt, img0pyr.at(0).size(), mask0);
      if (pb >= 0 && grid_polar.at<uint16_t>(pb) < 60000)
        grid_polar.at<uint16_t>(pb) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  // If we don't need any features, just return
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // This is old extraction code that would extract from the whole image
  // This can be slow as this will recompute extractions for grid areas that we have max features already
  // std::vector<cv::KeyPoint> pts0_ext;
  // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

  // We also check a downsampled mask such that we don't extract in areas where it is all masked!
  cv::Mat mask0_grid;
  cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

  // Create grids we need to extract from and then extract our features (use fast with griding)
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
  std::vector<std::pair<int, int>> valid_locs;
  if (use_polar_grid) {
    // Quota per (ring, sector) bin of equal solid angle. A rectangular extraction cell is
    // opened when the polar bin containing its centre is under-filled; extraction still
    // uses Grider_GRID's rectangular ROIs, only the QUOTA allocation is polar.
    int nbins = std::max(1, polar_rings * polar_sectors);
    int per_bin_req = std::max(1, (int)(min_feat_percent * ((double)num_features / nbins)));
    float csx = (float)img0pyr.at(0).cols / (float)grid_2d_grid.cols;
    float csy = (float)img0pyr.at(0).rows / (float)grid_2d_grid.rows;
    for (int x = 0; x < grid_2d_grid.cols; x++) {
      for (int y = 0; y < grid_2d_grid.rows; y++) {
        if ((int)mask0_grid.at<uint8_t>(y, x) == 255)
          continue;
        cv::Point2f cc((x + 0.5f) * csx, (y + 0.5f) * csy);
        int pb = polar_bin(cam_id, cc, img0pyr.at(0).size(), mask0);
        if (pb >= 0 && (int)grid_polar.at<uint16_t>(pb) < per_bin_req)
          valid_locs.emplace_back(x, y);
      }
    }
  } else {
    for (int x = 0; x < grid_2d_grid.cols; x++) {
      for (int y = 0; y < grid_2d_grid.rows; y++) {
        if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
  }
  std::vector<cv::KeyPoint> pts0_ext;
  { VPROF("1.track/c1_grider_FAST");
  Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true); }

  // Now, reject features that are close a current feature
  std::vector<cv::KeyPoint> kpts0_new;
  std::vector<cv::Point2f> pts0_new;
  for (auto &kpt : pts0_ext) {
    // Check that it is in bounds
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
      continue;
    // See if there is a point at this location
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else lets add it!
    kpts0_new.push_back(kpt);
    pts0_new.push_back(kpt.pt);
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Loop through and record only ones that are valid
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_new.size(); i++) {
    // update the uv coordinates
    kpts0_new.at(i).pt = pts0_new.at(i);
    // append the new uv coordinate
    pts0.push_back(kpts0_new.at(i));
    // move id foward and append this new point
    size_t temp = next_id(cam_id);
    ids0.push_back(temp);
    g_trk_stats.created++;
  }
}


void TrackKLT::flush_pending() {
  std::lock_guard<std::mutex> lkp(pending_mtx);
  size_t n_det = 0;
  for (size_t c = 0; c < PEND_NCAM; c++)
    n_det += pending_by_cam[c].size();
  if (pending_obs.empty() && n_det == 0)
    return;
  // ONE database mutex acquisition for the whole batch, matching the shipped
  // update_features_bulk path in feed_monocular (the per-observation update_feature loop this
  // replaces took the lock once per point -- ~11k acquisitions per 4-cam frame-set).
  std::vector<FeatureDatabase::BulkObs> bulk;
  bulk.reserve(pending_obs.size() + n_det);
  // Sensor-id order, then the (unused in det mode) shared vector.
  for (size_t c = 0; c < PEND_NCAM; c++) {
    for (auto &o : pending_by_cam[c])
      bulk.push_back({o.id, o.ts, o.cam, o.u, o.v, o.un, o.vn});
    pending_by_cam[c].clear();
  }
  for (auto &o : pending_obs)
    bulk.push_back({o.id, o.ts, o.cam, o.u, o.v, o.un, o.vn});
  database->update_features_bulk(bulk);
  pending_obs.clear();
}

// ---------------------------------------------------------------------------------------
// OV_PIPELINE calibration snapshot.  The tracker and the filter share the SAME CamBase
// objects (VioManager hands the tracker state->_cam_intrinsics_cameras), and with
// calib_cam_intrinsics=true StateHelper::ekf_update_impl calls CamBase::set_value on every
// update -- a non-atomic write of an 8-vector + a Matx33d + a Vec4d.  Overlapping tracking
// with the update therefore reads a torn calibration.  These two functions give the tracker
// a private copy refreshed only at the drain point (update worker idle, tracking not
// running), so every reader sees one self-consistent calibration.
// ---------------------------------------------------------------------------------------
bool TrackKLT::init_calib_snapshot() {
  calib_snap.clear();
  for (auto const &kv : camera_calib) {
    std::shared_ptr<ov_core::CamBase> c;
    const ov_core::CamBase *src = kv.second.get();
    if (dynamic_cast<const ov_core::CamRadtan *>(src) != nullptr)
      c = std::make_shared<ov_core::CamRadtan>(kv.second->w(), kv.second->h());
    else if (dynamic_cast<const ov_core::CamEqui *>(src) != nullptr)
      c = std::make_shared<ov_core::CamEqui>(kv.second->w(), kv.second->h());
    else if (dynamic_cast<const ov_core::CamDS *>(src) != nullptr)
      c = std::make_shared<ov_core::CamDS>(kv.second->w(), kv.second->h());
    else {
      std::fprintf(stderr, "[pipe]: camera %zu has an unrecognised model -- cannot snapshot\n", kv.first);
      calib_snap.clear();
      return false;
    }
    c->set_value(kv.second->get_value());
    calib_snap[kv.first] = c;
  }
  return !calib_snap.empty();
}

void TrackKLT::snapshot_calib() {
  for (auto &kv : calib_snap)
    kv.second->set_value(camera_calib.at(kv.first)->get_value());
}


void TrackKLT::perform_detection_submit(size_t cam_id, const cv::Mat &mask0, bool due_cadence) {
  // Duplicate of perform_detection_monocular_gpu's bookkeeping on a scratch copy of pts_last
  // (cheap, deterministic); queues the detect kernels async so all cameras' GPU work is in
  // flight before the parallel completion threads even start.
  std::vector<cv::KeyPoint> pts0 = pts_last[cam_id];
  const int w = img_curr.at(cam_id).cols, h = img_curr.at(cam_id).rows;
  cv::Size size_close((int)((float)w / (float)min_px_dist), (int)((float)h / (float)min_px_dist));
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)w / (float)grid_x, size_y = (float)h / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y);
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  std::vector<cv::KeyPoint> kept;
  kept.reserve(pts0.size());
  for (auto &kpt : pts0) {
    int x = (int)kpt.pt.x, y = (int)kpt.pt.y, edge = 10;
    if (x < edge || x >= w - edge || y < edge || y >= h - edge) continue;
    int xc = (int)(kpt.pt.x / (float)min_px_dist), yc = (int)(kpt.pt.y / (float)min_px_dist);
    if (xc < 0 || xc >= size_close.width || yc < 0 || yc >= size_close.height) continue;
    int xg = (int)(kpt.pt.x / size_x), yg = (int)(kpt.pt.y / size_y);
    if (xg < 0 || xg >= size_grid.width || yg < 0 || yg >= size_grid.height) continue;
    if (grid_2d_close.at<uint8_t>(yc, xc) > 127) continue;
    if (!mask0.empty() && mask0.at<uint8_t>(y, x) > 127) continue;
    grid_2d_close.at<uint8_t>(yc, xc) = 255;
    if (grid_2d_grid.at<uint8_t>(yg, xg) < 255) grid_2d_grid.at<uint8_t>(yg, xg) += 1;
    kept.push_back(kpt);
  }
  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)kept.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;   // completion side will early-out identically; nothing queued
  // The OV_DETECT_ADAPT burst is EXACTLY predictable one frame ahead: it compares the pruned
  // live-track count -- which is kept.size(), already computed above and frozen until the next
  // feed -- against the EMA as it stood at the end of this frame's gate. Read the EMA, never
  // write it; the consumption side remains the sole owner of that state machine. Without this,
  // every burst detection (measured ~46% of all detections) still ran inline on the critical
  // path, which is the population the whole lever is aimed at.
  if (!due_cadence) {
    const double af = det_adapt_frac();
    std::lock_guard<std::mutex> lk(g_det_mtx);
    const double m = g_det_ema.count(cam_id) ? g_det_ema[cam_id] : 0.0;
    if (!(af > 0.0 && m > 0.0 && (double)kept.size() < af * m))
      return;   // neither cadence nor burst -> this frame will not detect
    // MANDATORY mirror of the consuming gate's OV_DETECT_BURST_MIN rate limit. If the consume
    // side is going to refuse this burst, we must NOT queue the kernels: det_submitted would
    // stay true and the NEXT frame would take the STALE branch, i.e. a synchronous inline
    // gpu_detect on the critical path -- the change would make the tail WORSE. Read-only: the
    // consuming gate stays the sole writer of g_det_last_burst. This submit runs at the end of
    // frame g_frame_no and is consumed on g_frame_no + 1, hence the +1.
    const int bmin = det_burst_min();
    if (bmin > 0) {
      const long fn_next = g_frame_no.load(std::memory_order_relaxed) + 1;
      auto itb = g_det_last_burst.find(cam_id);
      if (itb != g_det_last_burst.end() && fn_next - itb->second < (long)bmin)
        return;
    }
  }
  cv::Mat mask0_grid;
  if (!mask0.empty()) cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);
  int nfg = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int nfg_req = std::max(1, (int)(min_feat_percent * nfg));
  std::vector<int> want((size_t)grid_x * grid_y, 0);
  for (int x = 0; x < grid_2d_grid.cols; x++)
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      bool masked = (!mask0_grid.empty() && (int)mask0_grid.at<uint8_t>(y, x) == 255);
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < nfg_req && !masked)
        want[(size_t)y * grid_x + x] = nfg;
    }
  std::vector<float> occ_x, occ_y;
  occ_x.reserve(kept.size()); occ_y.reserve(kept.size());
  for (auto &k : kept) { occ_x.push_back(k.pt.x); occ_y.push_back(k.pt.y); }
  std::vector<float> dx, dy;
  if (ov_core::gpu_detect(cam_id, false, occ_x.data(), occ_y.data(), (int)occ_x.size(), min_px_dist,
                          want.data(), grid_x, grid_y, threshold, dx, dy, true))
    det_submitted[cam_id] = true;
}

void TrackKLT::perform_detection_monocular_gpu(size_t cam_id, bool on_current, int w, int h, const cv::Mat &mask0,
                                               std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0) {

  // ---- identical bookkeeping to the CPU path: prune, then build the occupancy grids ----
  cv::Size size_close((int)((float)w / (float)min_px_dist), (int)((float)h / (float)min_px_dist));
  cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
  float size_x = (float)w / (float)grid_x;
  float size_y = (float)h / (float)grid_y;
  cv::Size size_grid(grid_x, grid_y);
  cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x, y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= w - edge || y < edge || y >= h - edge) { it0 = pts0.erase(it0); it1 = ids0.erase(it1); continue; }
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
      it0 = pts0.erase(it0); it1 = ids0.erase(it1); continue; }
    int x_grid = (int)(kpt.pt.x / size_x);
    int y_grid = (int)(kpt.pt.y / size_y);
    if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
      it0 = pts0.erase(it0); it1 = ids0.erase(it1); continue; }
    if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) { it0 = pts0.erase(it0); it1 = ids0.erase(it1); continue; }
    if (!mask0.empty() && mask0.at<uint8_t>(y, x) > 127) { it0 = pts0.erase(it0); it1 = ids0.erase(it1); continue; }
    grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
    it0++;
    it1++;
  }

  double min_feat_percent = 0.50;
  int num_featsneeded = num_features - (int)pts0.size();
  if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
    return;

  // Adaptive cadence: with ~98% KLT survival the per-frame top-off mostly re-finds the same
  // saturated cells. OV_DETECT_EVERY=N runs the (GPU) detection every Nth frame per camera.
  const int det_every = det_every_cfg();
  // Safety net for long cadences: a sudden drop in live tracks (aggressive motion, occlusion)
  // forces an immediate detection instead of waiting out the cadence. EMA tracks the
  // steady-state count per camera; OV_DETECT_ADAPT is the drop fraction (default 0.8).
  const double adapt_frac = det_adapt_frac();
  if (det_every > 1) {
    // OV_DETECT_STAGGER=1: offset each camera's cadence phase so cameras take turns detecting
    // instead of all four hitting the GPU on the same frame. Same per-camera rate, same
    // per-camera spacing -- only the alignment changes. Kills the every-Nth-frame latency
    // spike (4x detect kernels serialized on the device) that real-time pacing exposed.
    static const bool stagger = [] { const char *e = std::getenv("OV_DETECT_STAGGER"); return e && *e == '1'; }();
    std::lock_guard<std::mutex> lk(g_det_mtx);
    double &m = g_det_ema[cam_id];
    const double npts = (double)pts0.size();
    bool burst = adapt_frac > 0.0 && m > 0.0 && npts < adapt_frac * m;
    // OV_DETECT_BURST_MIN hysteresis. Sole writer of g_det_last_burst; the lock is already held.
    // Suppressing only makes `burst` false, so control falls through to the ORDINARY cadence
    // test below -- the cadence and its OV_DETECT_STAGGER phase are untouched.
    const int bmin = det_burst_min();
    if (burst && bmin > 0) {
      const long fn = g_frame_no.load(std::memory_order_relaxed);
      auto itb = g_det_last_burst.find(cam_id);
      if (itb != g_det_last_burst.end() && fn - itb->second < (long)bmin) {
        burst = false;
        g_det_burst_sup.fetch_add(1, std::memory_order_relaxed);
      } else {
        g_det_last_burst[cam_id] = fn;
      }
    }
    if (burst) g_det_burst.fetch_add(1, std::memory_order_relaxed);
    m = (m == 0.0) ? npts : 0.95 * m + 0.05 * npts;
    const int phase = stagger ? (int)((cam_id * (size_t)det_every) / 4) : 0;
    // OV_DETECT_BACKLOG_MS=X: while the estimator is draining a convoy (this frame waited
    // more than X ms between arrival and processing), postpone the detection top-off -- the
    // cadence counter pauses, so detection resumes at normal cadence once caught up. The
    // EMA burst safety above still forces an immediate detect on a real track collapse.
    static const double backlog_ms = [] { const char *e = std::getenv("OV_DETECT_BACKLOG_MS"); return e ? atof(e) : 0.0; }();
    if (det_phase_abs()) {
      // Absolute frame index: the burst stays an EXTRA detect and no longer re-aligns the
      // cadence, so a yaw burst on the hardware-synced cameras cannot phase-lock all four.
      if (backlog_ms > 0.0 && g_rt_behind_ms > backlog_ms && !burst)
        return;
      if (!burst && ((g_frame_no.load(std::memory_order_relaxed) + phase) % det_every) != 0)
        return;
    } else if (burst) {
      g_det_ctr[cam_id] = -phase; // detect now, restart cadence (keeping this cam's phase)
    } else if (backlog_ms > 0.0 && g_rt_behind_ms > backlog_ms) {
      return;
    } else if ((++g_det_ctr[cam_id] + phase) % det_every != 0) {
      return;
    }
  }

  cv::Mat mask0_grid;
  if (!mask0.empty())
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);
  int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
  int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));

  // per-cell deficit; 0 means "this cell is full or fully masked, skip it on the device"
  std::vector<int> want((size_t)grid_x * grid_y, 0);
  for (int x = 0; x < grid_2d_grid.cols; x++) {
    for (int y = 0; y < grid_2d_grid.rows; y++) {
      bool masked = (!mask0_grid.empty() && (int)mask0_grid.at<uint8_t>(y, x) == 255);
      if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && !masked)
        want[(size_t)y * grid_x + x] = num_features_grid;
    }
  }

  // ---- the only part that runs on the device ----
  std::vector<float> occ_x, occ_y;
  occ_x.reserve(pts0.size()); occ_y.reserve(pts0.size());
  for (auto &k : pts0) { occ_x.push_back(k.pt.x); occ_y.push_back(k.pt.y); }
  std::vector<float> cx, cy;
  // Single-submitter mode: feed_new_camera already queued this camera's detect kernels from
  // one thread (async); here we only complete. Otherwise submit+complete inline as before.
  { VPROF("1.track/c1_gpu_fast");
  bool okd;
  if (det_submitted[cam_id]) {
    det_submitted[cam_id] = false;
    const long want_gen = det_pending_gen[cam_id];
    det_pending_gen[cam_id] = -1;
    if (want_gen == g_frame_no.load(std::memory_order_relaxed)) {
      g_det_hit.fetch_add(1, std::memory_order_relaxed);
      okd = ov_core::gpu_detect_complete(cam_id, cx, cy);
    } else {
      // STALE: the frame this was predicted for did not consume it (early-out on
      // num_featsneeded, or an adaptive burst re-ordered the cadence). d_prev and pts_last have
      // both moved on, so those candidates are in a dead coordinate frame -- drain the device
      // buffers and detect inline against THIS frame's real occupancy/mask.
      // NOTE (round 17): this drain is MANDATORY, not cosmetic -- gpu_detect_complete ends in
      // cudaStreamSynchronize(cs_det), which is what retires the queued kernels before the
      // inline re-detect below reuses the same device buffers.  It is NOT sufficient on its
      // own, because the stale branch is only reached once perform_detection_monocular_gpu
      // gets past its own early returns; the cross-stream events in gpu_track.cu cover the
      // frames in between.
      g_det_stale.fetch_add(1, std::memory_order_relaxed);
      std::vector<float> junk_x, junk_y;
      ov_core::gpu_detect_complete(cam_id, junk_x, junk_y);
      okd = ov_core::gpu_detect(cam_id, on_current, occ_x.data(), occ_y.data(), (int)occ_x.size(), min_px_dist,
                                want.data(), grid_x, grid_y, threshold, cx, cy);
    }
  } else {
    g_det_inline.fetch_add(1, std::memory_order_relaxed);
    okd = ov_core::gpu_detect(cam_id, on_current, occ_x.data(), occ_y.data(), (int)occ_x.size(), min_px_dist,
                              want.data(), grid_x, grid_y, threshold, cx, cy);
  }
  if (!okd) return; }
  if (ov_core_trk_dump_enabled()) {
    ov_core::g_trk_ndet.fetch_add(1, std::memory_order_relaxed);
    ov_core::g_trk_npts_new.fetch_add((long)cx.size(), std::memory_order_relaxed);
  }

  // ---- greedy dedup at min_px_dist, exactly as the CPU path does ----
  for (size_t i = 0; i < cx.size(); i++) {
    int x_grid = (int)(cx[i] / (float)min_px_dist);
    int y_grid = (int)(cy[i] / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height) continue;
    if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127) continue;
    grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    cv::KeyPoint kpt;
    kpt.pt.x = cx[i];
    kpt.pt.y = cy[i];
    pts0.push_back(kpt);
    ids0.push_back(next_id(cam_id));
    g_trk_stats.created++;
  }
}

void TrackKLT::perform_detection_stereo(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, const cv::Mat &mask0,
                                        const cv::Mat &mask1, size_t cam_id_left, size_t cam_id_right, std::vector<cv::KeyPoint> &pts0,
                                        std::vector<cv::KeyPoint> &pts1, std::vector<size_t> &ids0, std::vector<size_t> &ids1) {

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size_close0((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                       (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
  cv::Mat grid_2d_close0 = cv::Mat::zeros(size_close0, CV_8UC1);
  float size_x0 = (float)img0pyr.at(0).cols / (float)grid_x;
  float size_y0 = (float)img0pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid0(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid0 = cv::Mat::zeros(size_grid0, CV_8UC1);
  cv::Mat mask0_updated = mask0.clone();
  auto it0 = pts0.begin();
  auto it1 = ids0.begin();
  while (it0 != pts0.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close0.width || y_close < 0 || y_close >= size_close0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x0);
    int y_grid = std::floor(kpt.pt.y / size_y0);
    if (x_grid < 0 || x_grid >= size_grid0.width || y_grid < 0 || y_grid >= size_grid0.height) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d_close0.at<uint8_t>(y_close, x_close) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask0.at<uint8_t>(y, x) > 127) {
      it0 = pts0.erase(it0);
      it1 = ids0.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close0.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid0.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid0.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // First compute how many more features we need to extract from this image
  double min_feat_percent = 0.50;
  int num_featsneeded_0 = num_features - (int)pts0.size();

  // LEFT: if we need features we should extract them in the current frame
  // LEFT: we will also try to track them from this frame over to the right frame
  // LEFT: in the case that we have two features that are the same, then we should merge them
  if (num_featsneeded_0 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts0_ext;
    // Grider_FAST::perform_griding(img0pyr.at(0), mask0_updated, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid0, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid0.cols; x++) {
      for (int y = 0; y < grid_2d_grid0.rows; y++) {
        if ((int)grid_2d_grid0.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_GRID::perform_griding(img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close0.width || y_grid < 0 || y_grid >= size_close0.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close0.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      grid_2d_close0.at<uint8_t>(y_grid, x_grid) = 255;
      kpts0_new.push_back(kpt);
      pts0_new.push_back(kpt.pt);
    }

    // TODO: Project points from the left frame into the right frame
    // TODO: This will not work for large baseline systems.....
    // TODO: If we had some depth estimates we could do a better projection
    // TODO: Or project and search along the epipolar line??
    std::vector<cv::KeyPoint> kpts1_new;
    std::vector<cv::Point2f> pts1_new;
    kpts1_new = kpts0_new;
    pts1_new = pts0_new;

    // If we have points, find their stereo correspondence in the right image
    if (!pts0_new.empty()) {

      std::vector<uchar> mask;
      // KLT from left to right (big window since the same-pixel init guess may be far off).
      // Original: KLT from left to right (big window since the same-pixel init guess may be far off).
      std::vector<float> error;
      cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
      cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0_new, pts1_new, mask, error, win_size, pyr_levels, term_crit,
                               cv::OPTFLOW_USE_INITIAL_FLOW);

      // Loop through and record only ones that are valid
      for (size_t i = 0; i < pts0_new.size(); i++) {

        // Check to see if the feature is out of bounds (oob) in either image
        bool oob_left = ((int)pts0_new.at(i).x < 0 || (int)pts0_new.at(i).x >= img0pyr.at(0).cols || (int)pts0_new.at(i).y < 0 ||
                         (int)pts0_new.at(i).y >= img0pyr.at(0).rows);
        bool oob_right = ((int)pts1_new.at(i).x < 0 || (int)pts1_new.at(i).x >= img1pyr.at(0).cols || (int)pts1_new.at(i).y < 0 ||
                          (int)pts1_new.at(i).y >= img1pyr.at(0).rows);

        // Check to see if it there is already a feature in the right image at this location
        //  1) If this is not already in the right image, then we should treat it as a stereo
        //  2) Otherwise we will treat this as just a monocular track of the feature
        // TODO: we should check to see if we can combine this new feature and the one in the right
        // TODO: seems if reject features which overlay with right features already we have very poor tracking perf
        if (!oob_left && !oob_right && mask[i] == 1) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          kpts1_new.at(i).pt = pts1_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          pts1.push_back(kpts1_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
          ids1.push_back(temp);
        } else if (!oob_left) {
          // update the uv coordinates
          kpts0_new.at(i).pt = pts0_new.at(i);
          // append the new uv coordinate
          pts0.push_back(kpts0_new.at(i));
          // move id forward and append this new point
          size_t temp = ++currid;
          ids0.push_back(temp);
        }
      }
    }
  }

  // RIGHT: Now summarise the number of tracks in the right image
  // RIGHT: We will try to extract some monocular features if we have the room
  // RIGHT: This will also remove features if there are multiple in the same location
  cv::Size size_close1((int)((float)img1pyr.at(0).cols / (float)min_px_dist), (int)((float)img1pyr.at(0).rows / (float)min_px_dist));
  cv::Mat grid_2d_close1 = cv::Mat::zeros(size_close1, CV_8UC1);
  float size_x1 = (float)img1pyr.at(0).cols / (float)grid_x;
  float size_y1 = (float)img1pyr.at(0).rows / (float)grid_y;
  cv::Size size_grid1(grid_x, grid_y); // width x height
  cv::Mat grid_2d_grid1 = cv::Mat::zeros(size_grid1, CV_8UC1);
  cv::Mat mask1_updated = mask0.clone();
  it0 = pts1.begin();
  it1 = ids1.begin();
  while (it0 != pts1.end()) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = *it0;
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int edge = 10;
    if (x < edge || x >= img1pyr.at(0).cols - edge || y < edge || y >= img1pyr.at(0).rows - edge) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate mask coordinates for close points
    int x_close = (int)(kpt.pt.x / (float)min_px_dist);
    int y_close = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_close < 0 || x_close >= size_close1.width || y_close < 0 || y_close >= size_close1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Calculate what grid cell this feature is in
    int x_grid = std::floor(kpt.pt.x / size_x1);
    int y_grid = std::floor(kpt.pt.y / size_y1);
    if (x_grid < 0 || x_grid >= size_grid1.width || y_grid < 0 || y_grid >= size_grid1.height) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Check if this keypoint is near another point
    // NOTE: if it is *not* a stereo point, then we will not delete the feature
    // NOTE: this means we might have a mono and stereo feature near each other, but that is ok
    bool is_stereo = (std::find(ids0.begin(), ids0.end(), *it1) != ids0.end());
    if (grid_2d_close1.at<uint8_t>(y_close, x_close) > 127 && !is_stereo) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Now check if it is in a mask area or not
    // NOTE: mask has max value of 255 (white) if it should be
    if (mask1.at<uint8_t>(y, x) > 127) {
      it0 = pts1.erase(it0);
      it1 = ids1.erase(it1);
      continue;
    }
    // Else we are good, move forward to the next point
    grid_2d_close1.at<uint8_t>(y_close, x_close) = 255;
    if (grid_2d_grid1.at<uint8_t>(y_grid, x_grid) < 255) {
      grid_2d_grid1.at<uint8_t>(y_grid, x_grid) += 1;
    }
    // Append this to the local mask of the image
    if (x - min_px_dist >= 0 && x + min_px_dist < img1pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img1pyr.at(0).rows) {
      cv::Point pt1(x - min_px_dist, y - min_px_dist);
      cv::Point pt2(x + min_px_dist, y + min_px_dist);
      cv::rectangle(mask1_updated, pt1, pt2, cv::Scalar(255), -1);
    }
    it0++;
    it1++;
  }

  // RIGHT: if we need features we should extract them in the current frame
  // RIGHT: note that we don't track them to the left as we already did left->right tracking above
  int num_featsneeded_1 = num_features - (int)pts1.size();
  if (num_featsneeded_1 > std::min(20, (int)(min_feat_percent * num_features))) {

    // This is old extraction code that would extract from the whole image
    // This can be slow as this will recompute extractions for grid areas that we have max features already
    // std::vector<cv::KeyPoint> pts1_ext;
    // Grider_FAST::perform_griding(img1pyr.at(0), mask1_updated, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask1_grid;
    cv::resize(mask1, mask1_grid, size_grid1, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid1.cols; x++) {
      for (int y = 0; y < grid_2d_grid1.rows; y++) {
        if ((int)grid_2d_grid1.at<uint8_t>(y, x) < num_features_grid_req && (int)mask1_grid.at<uint8_t>(y, x) != 255) {
          valid_locs.emplace_back(x, y);
        }
      }
    }
    std::vector<cv::KeyPoint> pts1_ext;
    Grider_GRID::perform_griding(img1pyr.at(0), mask1_updated, valid_locs, pts1_ext, num_features, grid_x, grid_y, threshold, true);

    // Now, reject features that are close a current feature
    for (auto &kpt : pts1_ext) {
      // Check that it is in bounds
      int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
      int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
      if (x_grid < 0 || x_grid >= size_close1.width || y_grid < 0 || y_grid >= size_close1.height)
        continue;
      // See if there is a point at this location
      if (grid_2d_close1.at<uint8_t>(y_grid, x_grid) > 127)
        continue;
      // Else lets add it!
      pts1.push_back(kpt);
      size_t temp = ++currid;
      ids1.push_back(temp);
      grid_2d_close1.at<uint8_t>(y_grid, x_grid) = 255;
    }
  }
}

// Exact squared-Sampson classification of ALL points against a fundamental matrix F.
// Arithmetic is lifted verbatim from the pre-existing OV_RANSAC_SUB classify loop, which is
// itself OpenCV's FMEstimatorCallback::computeError test. Returns the inlier count.
static int sampson_mask(const double *f, const std::vector<cv::Point2f> &p0, const std::vector<cv::Point2f> &p1, double thr2,
                        std::vector<uchar> &m) {
  m.assign(p0.size(), 0);
  int nin = 0;
  for (size_t i = 0; i < p0.size(); i++) {
    const double x = p0[i].x, y = p0[i].y, xp = p1[i].x, yp = p1[i].y;
    const double a = f[0] * x + f[1] * y + f[2], b = f[3] * x + f[4] * y + f[5], c = f[6] * x + f[7] * y + f[8];
    const double ap = f[0] * xp + f[3] * yp + f[6], bp = f[1] * xp + f[4] * yp + f[7];
    const double num = xp * a + yp * b + c;
    const double den = a * a + b * b + ap * ap + bp * bp;
    if (den > 0 && (num * num) / den < thr2) {
      m[i] = 1;
      nin++;
    }
  }
  return nin;
}

void TrackKLT::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out,
                                const std::vector<uchar> *pre_status, int n_old,
                                std::vector<cv::Point2f> *pts1_norm_out) {

  // Never hand the caller a stale vector: every early return below leaves it EMPTY, and the
  // caller's reuse is guarded on size, so an early return can only fall back to today's path.
  if (pts1_norm_out != nullptr)
    pts1_norm_out->clear();

  // We must have equal vectors
  assert(kpts0.size() == kpts1.size());

  // Return if we don't have any points
  if (kpts0.empty() || kpts1.empty())
    return;

  // Convert keypoints into points (stupid opencv stuff)
  std::vector<cv::Point2f> pts0, pts1;
  pts0.reserve(kpts0.size());
  pts1.reserve(kpts1.size());
  for (size_t i = 0; i < kpts0.size(); i++) {
    pts0.push_back(kpts0.at(i).pt);
    pts1.push_back(kpts1.at(i).pt);
  }

  // If we don't have enough points for ransac just return empty
  // We set the mask to be all zeros since all points failed RANSAC
  if (pts0.size() < 10) {
    for (size_t i = 0; i < pts0.size(); i++)
      mask_out.push_back((uchar)0);
    return;
  }

  // Now do KLT tracking to get the valid new points
  std::vector<uchar> mask_klt;
  std::vector<float> error;
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
  if (pre_status != nullptr) {
    // GPU already tracked these; pts1 carries the result and pre_status the success flags.
    mask_klt = *pre_status;
    mask_klt.resize(pts0.size(), (uchar)0);
  } else {
    VPROF("1.track/d1_calcOpticalFlowPyrLK");
    cv::calcOpticalFlowPyrLK(img0pyr, img1pyr, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
  }


  // Normalize these points, so we can then do ransac
  // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
  std::vector<cv::Point2f> pts0_n, pts1_n;
  const bool _tmd = ov_core_trk_dump_enabled();
  const double _tm0 = _tmd ? trk_now_s() : 0.0;
  ov_core::CamBase *_cam0 = calib_for(id0), *_cam1 = calib_for(id1);
  const bool _ub = undist_batch_on();
  { VPROF("1.track/d2_undistort");
  if (_ub) {
    static_assert(sizeof(cv::Point2f) == 2 * sizeof(float), "cv::Point2f must be 2 packed floats");
    if (g_ub_banner.exchange(1) == 0)
      fprintf(stderr, "[undist]: BATCH ENABLED (verify=%d)\n", (int)undist_verify_on());
    _cam0->undistort_cv_batch(pts0, pts0_n);
    _cam1->undistort_cv_batch(pts1, pts1_n);
    g_ub_calls.fetch_add(2, std::memory_order_relaxed);
    g_ub_pts.fetch_add((long)(pts0.size() + pts1.size()), std::memory_order_relaxed);
    if (undist_verify_on()) {
      // IN-BINARY EXACTNESS PROOF: recompute every point through the per-point path, in THIS
      // process, on THIS input, and memcmp the float pairs. Not an argument -- evidence.
      long bad = 0;
      for (size_t i = 0; i < pts0.size(); i++) {
        const cv::Point2f r0 = _cam0->undistort_cv(pts0.at(i));
        const cv::Point2f r1 = _cam1->undistort_cv(pts1.at(i));
        if (std::memcmp(&r0, &pts0_n[i], sizeof(cv::Point2f)) != 0) { bad++; ub_maxdiff(std::fabs(r0.x - pts0_n[i].x)); ub_maxdiff(std::fabs(r0.y - pts0_n[i].y)); }
        if (std::memcmp(&r1, &pts1_n[i], sizeof(cv::Point2f)) != 0) { bad++; ub_maxdiff(std::fabs(r1.x - pts1_n[i].x)); ub_maxdiff(std::fabs(r1.y - pts1_n[i].y)); }
      }
      g_ub_n.fetch_add((long)(pts0.size() + pts1.size()), std::memory_order_relaxed);
      if (bad) g_ub_bad.fetch_add(bad, std::memory_order_relaxed);
    }
  } else {
    pts0_n.reserve(pts0.size());
    pts1_n.reserve(pts1.size());
    for (size_t i = 0; i < pts0.size(); i++) {
      pts0_n.push_back(_cam0->undistort_cv(pts0.at(i)));
      pts1_n.push_back(_cam1->undistort_cv(pts1.at(i)));
    }
  } }
  if (_tmd) trk_add(ov_core::g_trk_und_ms, 1000.0 * (trk_now_s() - _tm0));

  // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
  std::vector<uchar> mask_rsc;
  double max_focallength_img0 = std::max(_cam0->get_K()(0, 0), _cam0->get_K()(1, 1));
  double max_focallength_img1 = std::max(_cam1->get_K()(0, 0), _cam1->get_K()(1, 1));
  double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
  // Seed RNG per-call for cross-thread determinism (findFundamentalMat uses cv::theRNG, thread-local).
  // Seed depends only on point coordinates so the same input gives the same RANSAC choices.
  // Per-call seed = FNV-1a(point coords) XOR OV_RNG_SEED. Same (input, seed) -> identical RANSAC
  // (cross-thread determinism preserved), but a different OV_RNG_SEED genuinely changes the RANSAC
  // realization, so robustness-to-RNG can be measured across seeds (not a fixed lucky seed).
  // Hoisted to function scope (was an inner block) so the exact path can be re-seeded to exactly
  // the state today's code installs here. Value is unchanged.
  const uint64_t rng_seed = [&] {
    static const uint64_t rng_salt = [] {
      const char *e = std::getenv("OV_RNG_SEED");
      return (e && *e) ? std::strtoull(e, nullptr, 10) : 0ULL;
    }();
    uint64_t s = 0xcbf29ce484222325ULL;
    for (const auto &p : pts0_n) {
      uint32_t bx, by;
      std::memcpy(&bx, &p.x, 4);
      std::memcpy(&by, &p.y, 4);
      s ^= (uint64_t)bx; s *= 0x100000001b3ULL;
      s ^= (uint64_t)by; s *= 0x100000001b3ULL;
    }
    s ^= rng_salt; s *= 0x100000001b3ULL;
    return s ? s : 42ULL;
  }();
  cv::theRNG().state = rng_seed;
  const double _tr0 = _tmd ? trk_now_s() : 0.0;
  { VPROF("1.track/d3_ransac_findF");
  // OV_USAC=1: OpenCV's USAC (SPRT + local optimization) reaches the same confidence with far
  // fewer model evaluations than classic FM_RANSAC. Inlier sets differ slightly -> ATE-gated.
  static const bool usac = [] { const char *e = std::getenv("OV_USAC"); return e && *e == '1'; }();
  static const double rconf = [] { const char *e = std::getenv("OV_RANSAC_CONF"); return e ? atof(e) : 0.999; }();
  static const int rsub = [] { const char *e = std::getenv("OV_RANSAC_SUB"); return e ? atoi(e) : 0; }();
  // OV_RANSAC_TAIL=N (0 = OFF, default): two-stage probe. Stage 1 runs FM_RANSAC on an N-point
  // deterministic stride subsample and classifies ALL points exactly (Sampson) to get the true
  // inlier ratio w. OpenCV's own adaptive-iteration formula (cv::RANSACUpdateNumIters, 7-point
  // model) then predicts how many iterations the FULL call would need. If that prediction is
  // inside OV_RANSAC_TAIL_ITERS, we restore the exact RNG state and run today's call verbatim ->
  // bit-identical mask. Only the calls predicted to burn the whole maxIters=1000 cap take the
  // probe model. Default OFF; nothing below runs unless rtail > 0.
  static const int rtail = [] { const char *e = std::getenv("OV_RANSAC_TAIL"); return e ? atoi(e) : 0; }();
  static const int rtail_b = [] { const char *e = std::getenv("OV_RANSAC_TAIL_ITERS"); return e ? atoi(e) : 150; }();
  static const int rtail_v = [] { const char *e = std::getenv("OV_RANSAC_TAIL_VERIFY"); return e ? atoi(e) : 0; }();
  static const bool rhist = [] { const char *e = std::getenv("OV_RANSAC_HIST"); return e && *e == '1'; }();
  const double rthr = 2.0 / max_focallength;
  if (rhist) {
    const size_t n = pts0_n.size();
    g_rh_calls++; g_rh_pts += (long)n;
    g_rh_b[n < 64 ? 0 : n < 128 ? 1 : n < 192 ? 2 : n < 256 ? 3 : n < 384 ? 4 : 5]++;
  }
  // OV_RANSAC_OLDFIT=1 (default OFF): on a camera that just harvested a detection the RANSAC input
  // is dominated by brand-new keypoints that have never been tracked. They collapse the inlier
  // ratio w, and cv::RANSACUpdateNumIters (7-point model) turns that into the iteration count:
  // 6 iters at w=0.95, 137 at 0.65, 882 at 0.50, and the maxIters=1000 cap at w<=0.45 -- which is
  // the measured tail. Fit F on the OLD-TRACK PREFIX ONLY (features that already survived RANSAC
  // on previous frames, so w~0.95), then classify ALL N points with the EXACT same Sampson test at
  // the EXACT same threshold today's call applies with its winning model. Only the MODEL-SEARCH SET
  // changes. Calls with no detection this frame have n_fit == pts0_n.size() and fall straight
  // through to today's code -- bit-identical.
  // This is NOT the rejected OV_RANSAC_TAIL/OV_RANSAC_SUB: those took a STRIDE subsample, which
  // PRESERVES the poor inlier ratio, so they fitted through the same junk and returned a LOOSER F
  // that admitted +2.6% more inliers -> more measurement rows -> fleet-adverse. Selecting by TRACK
  // AGE inverts the mechanism: the fit set is the highest-inlier-ratio subset available.
  static const bool oldfit = [] { const char *e = std::getenv("OV_RANSAC_OLDFIT"); return e && *e == '1'; }();
  static const int oldmin = [] { const char *e = std::getenv("OV_RANSAC_OLDFIT_MIN"); return e ? atoi(e) : 60; }();
  static const bool oldfit_v = [] { const char *e = std::getenv("OV_RANSAC_OLDFIT_VERIFY"); return e && *e == '1'; }();
  const int n_fit = (n_old < 0) ? (int)pts0_n.size() : std::min(n_old, (int)pts0_n.size());
  const bool oldfit_take = oldfit && n_fit < (int)pts0_n.size() && n_fit >= oldmin;
  // Census of the calls the gate did NOT take, kept OUT of the if-chain so that a passthrough can
  // never shadow OV_USAC / OV_RANSAC_TAIL / OV_RANSAC_SUB -- those branches stay exactly as they
  // are today and remain reachable.
  if (oldfit && n_old >= 0 && !oldfit_take) {
    if (n_fit >= (int)pts0_n.size())
      g_of_passthru++;
    else
      g_of_small++;
  }
  if (oldfit_take) {
    g_of_calls++;
    g_of_pts += (long)pts0_n.size();
    g_of_nfit += (long)n_fit;
    g_of_b[n_fit < 60 ? 0 : n_fit < 100 ? 1 : n_fit < 150 ? 2 : n_fit < 200 ? 3 : n_fit < 300 ? 4 : 5]++;
    // contiguous prefix -- no gather, no reorder
    std::vector<cv::Point2f> s0(pts0_n.begin(), pts0_n.begin() + n_fit);
    std::vector<cv::Point2f> s1(pts1_n.begin(), pts1_n.begin() + n_fit);
    cv::theRNG().state = rng_seed; // same seeding discipline as every other branch
    cv::Mat F = cv::findFundamentalMat(s0, s1, cv::FM_RANSAC, rthr, rconf);
    if (F.rows == 3) {
      sampson_mask(F.ptr<double>(0), pts0_n, pts1_n, rthr * rthr, mask_rsc); // EXACT, all N points
      g_of_approx++;
      if (oldfit_v) {
        // In-binary census: recompute the reference in the SAME process on the SAME input, so this
        // is a proof about this lever's calls and not a comparison across runs.
        std::vector<uchar> ref;
        cv::theRNG().state = rng_seed;
        cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, ref);
        ref.resize(pts0_n.size(), 0);
        long f01 = 0, f10 = 0, r1 = 0, c1 = 0;
        for (size_t i = 0; i < pts0_n.size(); i++) {
          const bool r = ref[i] != 0, c = mask_rsc[i] != 0;
          r1 += r; c1 += c; f01 += (!r && c); f10 += (r && !c);
        }
        g_of_vcalls++; g_of_vpts += (long)pts0_n.size();
        g_of_vref1 += r1; g_of_vcand1 += c1; g_of_vflip01 += f01; g_of_vflip10 += f10;
      }
    } else {
      // Degenerate prefix -> today's call verbatim, from the exact RNG state today installs.
      cv::theRNG().state = rng_seed;
      cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, mask_rsc);
      g_of_degen++;
    }
  } else if (rtail > 0 && rsub == 0 && !usac && (int)pts0_n.size() > 2 * rtail) {
    g_rt_calls++;
    const double thr2 = rthr * rthr;
    // --- stage 1: deterministic stride probe (identical construction to the OV_RANSAC_SUB branch)
    std::vector<cv::Point2f> s0, s1;
    s0.reserve(rtail); s1.reserve(rtail);
    const size_t stride = pts0_n.size() / rtail;
    for (size_t i = 0; i < pts0_n.size(); i += stride) { s0.push_back(pts0_n[i]); s1.push_back(pts1_n[i]); }
    cv::theRNG().state = rng_seed;
    cv::Mat F = cv::findFundamentalMat(s0, s1, cv::FM_RANSAC, rthr, rconf);
    std::vector<uchar> m_sub;
    const int nin = (F.rows == 3) ? sampson_mask(F.ptr<double>(0), pts0_n, pts1_n, thr2, m_sub) : 0;
    const double w = (double)nin / (double)pts0_n.size();
    // OpenCV's cv::RANSACUpdateNumIters with modelPoints=7, ep=1-w, p=rconf.
    const double den = std::log(1.0 - std::pow(w, 7.0));
    const int pred = (w <= 0.0 || !(den < 0.0)) ? 1000 : (int)std::min(1000.0, std::ceil(std::log(1.0 - rconf) / den));
    // F.rows != 3 means the probe found nothing at all; taking its (all-zero) mask would reject
    // every match on this camera-call, so fall back to the exact path there. Strictly safer than
    // the approximate branch and it keeps those calls bit-identical to today.
    const bool take_exact = (F.rows != 3) || (rtail_b > 0 && pred <= rtail_b);
    if (F.rows != 3)
      g_rt_degen++;
    if (take_exact) {
      // EASY: the full RANSAC self-terminates well inside the cap. Restore the exact RNG state
      // that today's code installs and run today's call verbatim.
      cv::theRNG().state = rng_seed;
      cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, mask_rsc);
      g_rt_exact++;
      if (rtail_v >= 1) {
        // In-binary exactness proof: recompute the reference in the SAME process, same state.
        std::vector<uchar> ref;
        cv::theRNG().state = rng_seed;
        cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, ref);
        if (ref != mask_rsc)
          g_rt_mismatch++;
      }
    } else {
      // HARD: today's call would run all 1000 iterations scoring every point per model.
      if (rtail_v >= 2) {
        // Quantify the measurement-set change on exactly the calls this lever alters.
        std::vector<uchar> ref;
        cv::theRNG().state = rng_seed;
        cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, ref);
        ref.resize(pts0_n.size(), 0);
        long f01 = 0, f10 = 0, r1 = 0, c1 = 0;
        for (size_t i = 0; i < pts0_n.size(); i++) {
          const bool r = ref[i] != 0, c = m_sub[i] != 0;
          r1 += r; c1 += c; f01 += (!r && c); f10 += (r && !c);
        }
        g_rt_vcalls++; g_rt_vpts += (long)pts0_n.size();
        g_rt_vref1 += r1; g_rt_vcand1 += c1; g_rt_vflip01 += f01; g_rt_vflip10 += f10;
      }
      mask_rsc.swap(m_sub);
      g_rt_approx++;
    }
  } else if (rsub > 0 && (int)pts0_n.size() > 2 * rsub) {
    // Model search on a subsample (same confidence, ~half the per-iteration cost), then exact
    // Sampson-distance classification of ALL points against the found model.
    std::vector<cv::Point2f> s0, s1;
    s0.reserve(rsub); s1.reserve(rsub);
    const size_t stride = pts0_n.size() / rsub;
    for (size_t i = 0; i < pts0_n.size(); i += stride) { s0.push_back(pts0_n[i]); s1.push_back(pts1_n[i]); }
    cv::Mat F = cv::findFundamentalMat(s0, s1, cv::FM_RANSAC, rthr, rconf);
    if (F.rows == 3)
      sampson_mask(F.ptr<double>(0), pts0_n, pts1_n, rthr * rthr, mask_rsc);
    else
      mask_rsc.assign(pts0_n.size(), 0);
  } else if (usac)
    cv::findFundamentalMat(pts0_n, pts1_n, cv::USAC_FAST, rthr, rconf, 1000, mask_rsc);
  else
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, rthr, rconf, mask_rsc); }
  if (_tmd) trk_add(ov_core::g_trk_rsc_ms, 1000.0 * (trk_now_s() - _tr0));

  // Loop through and record only ones that are valid
  for (size_t i = 0; i < mask_klt.size(); i++) {
    auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
    mask_out.push_back(mask);
  }
  if (const char *dv = std::getenv("OV_MATCH_DEBUG")) {
    if (*dv == '1') {
      int nk = 0, nr = 0, nb = 0;
      for (size_t i = 0; i < mask_klt.size(); i++) {
        bool a = mask_klt[i] != 0, b = (i < mask_rsc.size() && mask_rsc[i] != 0);
        nk += a; nr += b; nb += (a && b);
      }
      fprintf(stderr, "[match]: cam %zu in=%zu klt_ok=%d ransac_ok=%d both=%d%s\n",
              id0, pts0.size(), nk, nr, nb, (nb == 0 ? "   <<< ALL REJECTED" : ""));
    }
  }

  // OV_UNDIST_BATCH: hand back the normalized pts1. Placed HERE, immediately before the
  // copy-back below, so the invariant the caller relies on -- kpts1[i].pt == pts1[i] and
  // pts1_norm[i] == undistort(pts1[i]) -- is established by the very next statement.
  if (_ub && pts1_norm_out != nullptr)
    *pts1_norm_out = pts1_n;

  // Copy back the updated positions
  for (size_t i = 0; i < pts0.size(); i++) {
    kpts0.at(i).pt = pts0.at(i);
    kpts1.at(i).pt = pts1.at(i);
  }
}

