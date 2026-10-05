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

#include <omp.h>
#include "VioManager.h"

#include <chrono>
#include <atomic>
#include <mutex>
#include <cstdio>

#include <cstdlib>   // std::exit, for the use_aruco guard below

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/OvLogger.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/PreJac.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/vprof.h"
#include <thread>
#include <condition_variable>
#include <functional>
#include <deque>
#include <set>
#include <omp.h>

namespace {
// Frontend/backend overlap: the EKF update for frame N runs on this worker while the caller
// decodes and tracks frame N+1. Join points: (a) before ZUPT/tracking state access at the top
// of track_image_and_update, (b) at process exit. FeatureDatabase and Propagator are mutexed;
// the ordering change (tracking N+1 may insert measurements before update N's cleanup) alters
// which measurements survive cleanup -- statistically neutral, validated by ATE.
struct UpdWorker {
  std::thread t;
  void join() { if (t.joinable()) t.join(); }
  ~UpdWorker() { join(); }
};
UpdWorker g_upd;
inline bool async_update() {
  static const bool e = [] {
    const char *v = std::getenv("OV_ASYNC_UPDATE");
    return v && *v == '1';
  }();
  return e;
}

// =========================================================================================
// OV_PIPELINE: ONE persistent worker thread that runs the EKF update of frame N while the
// estimator thread tracks frame N+1.  Persistent, not per-frame, for two measured reasons:
//   * libgomp caches its team in the MASTER thread -- a fresh std::thread per frame would
//     build a new 4-thread OMP team every frame;
//   * the OV_EKF_SCRATCH / OV_MSCKF_SCRATCH gates rely on `static thread_local` scratch
//     buffers, which a fresh thread would reallocate every frame, turning both shipped
//     gates into pessimisations.
// The worker inherits the process affinity mask, so OV_AFFINITY's critical set applies to it
// exactly as it does to the estimator thread.
// =========================================================================================
struct PipeWorker {
  std::thread t;
  std::mutex m;
  std::condition_variable cv;
  std::function<void()> job;
  bool have = false, busy = false, quit = false, started = false;
  // published by the worker at completion, consumed (and cleared) by drain()
  double p_upd_ms = 0;
  int p_upd_n = 0;
  // [pipe] exit line: the evidence that the update ACTUALLY overlapped
  long n_submit = 0, n_drain = 0, n_wait = 0;
  double max_wait_ms = 0, sum_wait_ms = 0;

  static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void start() {
    if (started)
      return;
    started = true;
    t = std::thread([this] {
      // Build this thread's libgomp team ONCE, here, so the first update does not pay for it
      // and so the members are born with this thread's affinity mask.
      { volatile double sink = 0.0;
#pragma omp parallel reduction(+ : sink)
        { sink += omp_get_thread_num(); }
        (void)sink; }
      for (;;) {
        std::function<void()> j;
        {
          std::unique_lock<std::mutex> lk(m);
          cv.wait(lk, [&] { return have || quit; });
          if (quit)
            return;
          j = std::move(job);
          have = false;
          busy = true;
        }
        j();
        {
          std::lock_guard<std::mutex> lk(m);
          busy = false;
        }
        cv.notify_all();
      }
    });
  }
  void submit(std::function<void()> j) {
    start();
    std::unique_lock<std::mutex> lk(m);
    job = std::move(j);
    have = true;
    n_submit++;
    lk.unlock();
    cv.notify_one();
  }
  /// Wait for the in-flight job (if any) and hand back its published stats, clearing them so
  /// a later drain with nothing in flight reports zero rather than a stale value.
  void drain(double &upd_ms, int &upd_n) {
    const double t0 = now_s();
    std::unique_lock<std::mutex> lk(m);
    n_drain++;
    if (have || busy) {
      cv.wait(lk, [&] { return !have && !busy; });
      const double w = 1000.0 * (now_s() - t0);
      n_wait++;
      sum_wait_ms += w;
      if (w > max_wait_ms)
        max_wait_ms = w;
    }
    upd_ms = p_upd_ms;
    upd_n = p_upd_n;
    p_upd_ms = 0;
    p_upd_n = 0;
  }
  void stop() {
    { std::lock_guard<std::mutex> lk(m); quit = true; }
    cv.notify_all();
    if (t.joinable())
      t.join();
  }
  ~PipeWorker() {
    if (started) {
      std::fprintf(stderr, "[pipe]: submits=%ld drains=%ld waited=%ld (%.1f%%) wait_ms mean=%.2f max=%.2f\n", n_submit, n_drain, n_wait,
                   n_drain ? 100.0 * (double)n_wait / (double)n_drain : 0.0, n_wait ? sum_wait_ms / (double)n_wait : 0.0, max_wait_ms);
    }
    stop();
  }
};
PipeWorker g_pipe;
} // namespace

namespace {
// OV_ASYNC_EMIT (round 6): publish an immutable snapshot of everything the decoupled pose
// emitter needs.  Called ONLY from the thread that owns the state, at points where no
// StateHelper mutation is in flight, and only after every write of the current update.  The
// emitter never touches `state`; see PoseSnap in Propagator.h for why that matters.
inline void ov_publish_snap(const std::shared_ptr<ov_msckf::State> &state, const Eigen::Vector3d &gravity) {
  if (!ov_msckf::ov_async_emit_enabled())
    return;
  static std::atomic<long> seq{0};
  auto s = std::make_shared<ov_msckf::PoseSnap>();
  s->t = state->_timestamp;
  s->t_off = state->_calib_dt_CAMtoIMU->value()(0);
  s->imu = state->_imu->value();
  s->Dw = ov_msckf::State::Dm(state->_options.imu_model, state->_calib_imu_dw->value());
  s->Da = ov_msckf::State::Dm(state->_options.imu_model, state->_calib_imu_da->value());
  s->Tg = ov_msckf::State::Tg(state->_calib_imu_tg->value());
  s->R_A2I = state->_calib_imu_ACCtoIMU->Rot();
  s->R_G2I = state->_calib_imu_GYROtoIMU->Rot();
  s->gravity = gravity;
  s->seq = ++seq;
  ov_msckf::ov_pose_snap_publish(s);
}
} // namespace

#include <unordered_set>
#include <algorithm>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// Wall-clock breakdown of feed_measurement_camera. record_timing_information covers
// only frames where an EKF update fires (update_min_dt gates it), so it misses the
// tracking done on every other frame -- on a fleet recording that was 285 of 451
// frames and ~26 s of a 55 s pass. These buckets cover every call.
namespace ov_msckf {
std::map<std::string, double> g_vio_stage_secs;
std::map<std::string, long> g_vio_stage_calls;
} // namespace ov_msckf
namespace {
inline double _now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct VScoped {
  const char *k; double t0;
  explicit VScoped(const char *key) : k(key), t0(_now()) {}
  ~VScoped() { ov_msckf::g_vio_stage_secs[k] += _now() - t0; ov_msckf::g_vio_stage_calls[k] += 1; }
};
} // namespace

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  omp_set_max_active_levels(1);

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
    {
      auto _tk = std::dynamic_pointer_cast<TrackKLT>(trackFEATS);
      if (_tk) { _tk->use_polar_grid = params.use_polar_grid;
                 _tk->polar_rings = params.polar_rings;
                 _tk->polar_sectors = params.polar_sectors; }
    }
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }


  // Aruco tag tracking was removed with the TrackAruco class. It was the only thing in
  // this fork needing opencv2/aruco.hpp, which OpenCV moved out of contrib into
  // objdetect in 4.7 with a different API. JetPack 6.2.1 (L4T R36.4.4) ships OpenCV
  // 4.8.0 and has no opencv2/aruco.hpp at all; installing Ubuntu's
  // libopencv-contrib-dev to get it conflicts with the NVIDIA libopencv-dev that
  // cv_bridge, the OAK driver and depth estimation are all built against. Neither
  // operating point uses it -- use_aruco is false in both configs -- so the dependency
  // bought nothing and cost the drone build entirely.
  //
  // trackARUCO stays declared and permanently null. Every use of it is already
  // null-guarded, so no other call site changes. Fail loudly rather than silently
  // ignoring a config that asks for tracking it cannot do.
  if (params.use_aruco) {
    PRINT_ERROR(RED "use_aruco is true, but aruco tracking was removed from this fork.\n" RESET);
    PRINT_ERROR(RED "  Set use_aruco: false in your config. See PRUNED.txt.\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }

  // ---------------- OV_PIPELINE: arm, or refuse with a named reason --------------------
  // Every refusal below is a state the pipeline provably cannot make safe.  A refusal
  // DISARMS the gate and says so on stderr; it never runs a half-safe pipeline.
  {
    const char *pe = std::getenv("OV_PIPELINE");
    if (pe && *pe == '1') {
      auto refuse = [](const char *why) {
        std::fprintf(stderr, "[pipe]: REFUSED -- %s\n", why);
        return false;
      };
      bool ok = true;
      const char *ge = std::getenv("OV_GPU_EKF");
      const char *pj = std::getenv("OV_PREJAC");
      const char *ou = std::getenv("OV_OVERLAP_UPD");
      const char *au = std::getenv("OV_ASYNC_UPDATE");
      if (ge && *ge == '1')
        ok = refuse("OV_GPU_EKF=1 (update would issue CUDA on the default stream concurrently with the tracker streams)");
      else if (pj && *pj == '1')
        ok = refuse("OV_PREJAC=1 (its worker reads _upd_queue and is joined from inside do_feature_propagate_update)");
      else if (ou && *ou == '1')
        ok = refuse("OV_OVERLAP_UPD=1 (mutually exclusive intra-frame overlap)");
      else if (au && *au == '1')
        ok = refuse("OV_ASYNC_UPDATE=1 (retired; unreachable under OV_HYBRID_UPDATE)");
      else if (params.use_stereo)
        ok = refuse("use_stereo=true (stereo defer path is untested)");
      else if (!params.use_klt)
        ok = refuse("use_klt=false (calibration snapshot is implemented for TrackKLT only)");
      else if (!trackFEATS->init_calib_snapshot())
        ok = refuse("camera model cannot be snapshotted (would leave a torn intrinsics read)");
      _pipe_on = ok;
      std::fprintf(stderr, "[pipe]: armed=%d\n", (int)_pipe_on);
      // OV_PUB_WHEN_READY: only exists on top of the armed pipeline -- without it the caller
      // reads the state inline right after the update, which already is "when ready".
      const char *pr = std::getenv("OV_PUB_WHEN_READY");
      _pipe_pub_ready = _pipe_on && pr && *pr == '1';
      std::fprintf(stderr, "[pipe]: pub_when_ready=%d\n", (int)_pipe_pub_ready);
    }
  }
}

void VioManager::pipe_submit(const std::vector<ov_core::CameraData> &subs) {
  if (subs.empty())
    return;
  _pipe_inflight.store(true, std::memory_order_release);
  const std::vector<ov_core::CameraData> batch = subs; // by value: worker owns its own copy
  g_pipe.submit([this, batch] {
    const double t0 = PipeWorker::now_s();
    int n = 0;
    for (size_t i = 0; i < batch.size(); i++) {
      do_feature_propagate_update(batch[i]);
      n++;
    }
    const double ms = 1000.0 * (PipeWorker::now_s() - t0);
    // OV_PUB_WHEN_READY: the batch's writes are done and the worker still owns the state, so
    // the pose leaves NOW rather than after the next frame-set has been tracked.  Once per
    // batch, AFTER its last sub-update: that is exactly the state the post-drain sink would
    // read, so the emitted pose is the same one, only earlier.
    if (_pipe_pub_ready && pose_sink)
      pose_sink();
    std::lock_guard<std::mutex> lk(g_pipe.m);
    g_pipe.p_upd_ms = ms;
    g_pipe.p_upd_n = n;
  });
}

void VioManager::pipeline_flush() {
  if (!_pipe_on)
    return;
  double ms = 0;
  int n = 0;
  g_pipe.drain(ms, n);
  _pipe_inflight.store(false, std::memory_order_release);
  if (pose_sink)
    pose_sink();
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time;
  if (_pipe_inflight.load(std::memory_order_acquire)) {
    // OV_PIPELINE: the update worker is live and StateHelper::augment_clone INSERTS into
    // state->_clones_IMU WITHOUT state->_mutex_state, so iterating that map here (which is
    // what margtimestep() does) is a container race.  Use the value cached at the last
    // quiescent point.  It is always <= the live value -- clones only ever marginalize
    // forward in time -- so the propagator keeps a slightly LONGER IMU tail and can never
    // discard a measurement the filter still needs.  No filter arithmetic depends on the
    // length of that tail; select_imu_readings picks by time window.
    oldest_time = _pipe_oldest_time;
  } else {
    oldest_time = state->margtimestep();
    if (oldest_time > state->_timestamp) {
      oldest_time = -1;
    }
    if (!is_initialized_vio) {
      oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
    }
    _pipe_oldest_time = oldest_time;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  if (async_update()) {
    ov_core::CameraData mcopy = message;   // cv::Mats are refcounted; cheap
    g_upd.t = std::thread([this, mcopy] { do_feature_propagate_update(mcopy); });
  } else {
    do_feature_propagate_update(message);
  }
}

// Per-frame attribution for the realtime latency investigation (read by run_serial_msckf).
namespace ov_msckf {
double g_dbg_track_ms = 0, g_dbg_upd_ms = 0;
double g_dbg_sys_ms = 0, g_dbg_cmp_ms = 0, g_dbg_ekfu_ms = 0;
// OV_SYSPROF=1: sub-phase attribution inside UpdaterMSCKF step 4 (rT2->rT3).
double g_sp_alloc = 0, g_sp_jac = 0, g_sp_null = 0, g_sp_marg = 0, g_sp_chi2 = 0, g_sp_cpy = 0, g_sp_stack = 0, g_sp_rsz = 0;
long g_dbg_rows = 0, g_dbg_cols = 0;
int g_dbg_upd_n = 0;
// OV_PREJAC storage (declared in update/PreJac.h)
std::unordered_map<size_t, PreJacEntry> g_prejac;
std::mutex g_prejac_mtx;
std::atomic<uint64_t> g_state_epoch{1};
std::atomic<long> g_prejac_hit{0}, g_prejac_try{0}, g_prejac_made{0};
// OV_SLAMSTAT census counters (bench instrumentation, bumped unconditionally -- they are three
// relaxed atomic increments per SLAM promotion attempt / marginalization, i.e. ~8/s, and are
// only ever PRINTED when OV_SLAMSTAT=1).
std::atomic<long> g_slam_init_ok{0}, g_slam_init_fail{0}, g_slam_marg{0};
std::mutex g_pj_vmtx;
double g_pj_vmax_H = 0, g_pj_vmax_r = 0, g_pj_vmax_p = 0, g_pj_vrel_H = 0, g_pj_vrel_r = 0;
long g_pj_vn = 0, g_pj_vbad_order = 0, g_pj_vbad_tri = 0, g_pj_vbad_dim = 0;
void prejac_verify_report() {
  std::lock_guard<std::mutex> lk(g_pj_vmtx);
  printf("[prejac-verify] features_compared=%ld  max|dH_x|=%.3e (rel %.3e)  max|dres|=%.3e (rel %.3e)  "
         "max|dp_FinG|=%.3e  order_mismatch=%ld  tri_mismatch=%ld  dim_mismatch=%ld\n",
         g_pj_vn, g_pj_vmax_H, g_pj_vrel_H, g_pj_vmax_r, g_pj_vrel_r, g_pj_vmax_p, g_pj_vbad_order, g_pj_vbad_tri,
         g_pj_vbad_dim);
  fflush(stdout);
}
} // namespace ov_msckf

// OV_PREJAC=1: precompute the NEXT queued OV_SPREAD_UPD sub-update's MSCKF linear system on a
// worker during the following frame's tracking window. Exact (see update/PreJac.h).
static const bool prejac_on = [] { const char *e = std::getenv("OV_PREJAC"); return e && *e == '1'; }();

ov_msckf::VioManager::~VioManager() {
  // The worker's job captures `this`; it MUST NOT outlive the manager.
  if (_pipe_on) {
    double ms = 0;
    int n = 0;
    g_pipe.drain(ms, n);
    g_pipe.stop();
    _pipe_inflight.store(false, std::memory_order_release);
  }
  prejac_join();
}

void ov_msckf::VioManager::prejac_join() {
  if (_prejac_thread.joinable())
    _prejac_thread.join();
}

void ov_msckf::VioManager::prejac_launch(int lead_cam, const std::vector<std::shared_ptr<ov_core::Feature>> &lost_all,
                                         const std::unordered_set<size_t> &marg_ids) {
  // Snapshot on THIS thread (the tracker is idle right now), so the worker only ever reads
  // private copies -- the feature database is never touched concurrently with tracking.
  _prejac_snap.clear();
  for (const auto &f : lost_all) {
    if (f->to_delete)
      continue;                                             // consumed by the update that just ran
    if (marg_ids.find(f->featid) != marg_ids.end())
      continue;                                             // mirrors the feats_lost/feats_marg prune
    if (f->uvs.find((size_t)lead_cam) == f->uvs.end())
      continue;                                             // mirrors the per-camera feats_lost prune
    auto c = std::make_shared<ov_core::Feature>();
    c->featid = f->featid;
    c->to_delete = false;
    c->uvs = f->uvs;
    c->uvs_norm = f->uvs_norm;
    c->timestamps = f->timestamps;
    c->anchor_cam_id = f->anchor_cam_id;
    c->anchor_clone_timestamp = f->anchor_clone_timestamp;
    c->p_FinA = f->p_FinA;
    c->p_FinG = f->p_FinG;
    _prejac_snap.push_back(c);
  }
  {
    std::lock_guard<std::mutex> lk(ov_msckf::g_prejac_mtx);
    ov_msckf::g_prejac.clear();
  }
  if (_prejac_snap.empty())
    return;
  _prejac_epoch = ov_msckf::g_state_epoch.load(std::memory_order_relaxed);
  _prejac_thread = std::thread([this] { updaterMSCKF->prejac_precompute(state, _prejac_snap, _prejac_epoch); });
}
using ov_msckf::g_dbg_track_ms;
using ov_msckf::g_dbg_upd_ms;
using ov_msckf::g_dbg_upd_n;

// OV_TRK_DUMP per-frame tracking attribution (defined in ov_core/TrackKLT.cpp). Zeroed with
// g_dbg_track_ms so every counter covers exactly one frame-set.
namespace ov_core {
extern std::atomic<double> g_trk_det_ms, g_trk_klt_ms, g_trk_rsc_ms, g_trk_und_ms, g_trk_db_ms;
extern std::atomic<long> g_trk_npts, g_trk_ndet, g_trk_npts_new, g_trk_ncam;
} // namespace ov_core

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {
  g_dbg_track_ms = 0; g_dbg_upd_ms = 0; g_dbg_upd_n = 0;
  ov_core::g_trk_det_ms.store(0.0, std::memory_order_relaxed);
  ov_core::g_trk_klt_ms.store(0.0, std::memory_order_relaxed);
  ov_core::g_trk_rsc_ms.store(0.0, std::memory_order_relaxed);
  ov_core::g_trk_und_ms.store(0.0, std::memory_order_relaxed);
  ov_core::g_trk_db_ms.store(0.0, std::memory_order_relaxed);
  ov_core::g_trk_npts.store(0, std::memory_order_relaxed);
  ov_core::g_trk_ndet.store(0, std::memory_order_relaxed);
  ov_core::g_trk_npts_new.store(0, std::memory_order_relaxed);
  ov_core::g_trk_ncam.store(0, std::memory_order_relaxed);
  auto _dbg_now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  (void)_dbg_now;
  // Async mode: tracking only touches the (mutexed) feature database, so the previous frame's
  // update may still be running while we track -- that is the overlap. The join happens right
  // after tracking, before anything touches the state (mask refresh reads intrinsics, ZUPT
  // writes state). Sync mode joins here (no-op when no worker was ever launched).
  const bool do_async = async_update() && is_initialized_vio;
  if (!do_async) g_upd.join();
  // OV_PIPELINE: the update of the PREVIOUS frame's tick is running on the worker right now.
  // It is drained immediately after this frame's tracking (below), so exactly one update is
  // ever in flight and nothing downstream of the drain can see a half-updated state.
  const bool do_pipe = _pipe_on && is_initialized_vio;

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // If auto fisheye-disk mask is dynamic, regenerate it now from the latest
  // online-refined intrinsics so the tracker sees a mask that tracks any cx,cy
  // drift. Updates params.masks (for the next visualizer fetch) AND message.masks
  // (for this frame's tracker).
  if (!async_update() && !do_pipe) maybe_refresh_fisheye_masks(message);

  // Intra-frame overlap (OV_OVERLAP_UPD=1, hybrid): the EKF update for this frame's firing
  // lead only needs that ONE camera's fresh tracks. So: track the lead, launch its update on
  // a worker, track the other cameras concurrently with DB writes deferred (the async-mode
  // machinery), then join + flush before anything is logged. The pose still lands within this
  // frame's processing -- nothing becomes late; only the idle cores during the update get used.
  static const bool overlap_upd = [] {
    const char *e = std::getenv("OV_OVERLAP_UPD");
    return e && *e == '1';
  }();
  static const bool hybrid_upd_pre = [] {
    const char *e = std::getenv("OV_HYBRID_UPDATE");
    return e && *e == '1';
  }();
  if (overlap_upd && hybrid_upd_pre && !async_update() && is_initialized_vio && message.sensor_ids.size() > 1 &&
      params.update_min_dt > 0) {
    int fire_idx = -1;
    for (size_t ci = 0; ci < message.sensor_ids.size(); ci++) {
      int lead2 = (int)message.sensor_ids[ci];
      auto it2 = _last_update_time_decim.find(lead2);
      if (!(it2 != _last_update_time_decim.end() && message.timestamp < it2->second + params.update_min_dt)) {
        fire_idx = (int)ci;
        break;
      }
    }
    if (fire_idx >= 0) {
      ov_core::CameraData m_lead, m_rest;
      m_lead.timestamp = m_rest.timestamp = message.timestamp;
      for (size_t i = 0; i < message.sensor_ids.size(); i++) {
        ov_core::CameraData &dst = ((int)i == fire_idx) ? m_lead : m_rest;
        dst.sensor_ids.push_back(message.sensor_ids[i]);
        dst.images.push_back(message.images[i]);
        dst.masks.push_back(message.masks[i]);
      }
      { VPROF("1.track/TOTAL"); VScoped _s("KLT tracking"); trackFEATS->feed_new_camera(m_lead); }
      // ZUPT decision must precede the update, exactly as in the sequential path below.
      if (updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
        if (state->_timestamp != message.timestamp) {
          { VScoped _s("ZUPT try_update"); did_zupt_update = updaterZUPT->try_update(state, message.timestamp); }
        }
        if (did_zupt_update) {
          assert(state->_timestamp == message.timestamp);
          { VPROF("1.track/TOTAL"); VScoped _s("KLT tracking"); trackFEATS->feed_new_camera(m_rest); }
          propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
          updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
          propagator->invalidate_cache();
          ov_publish_snap(state, propagator->gravity());
          return;
        }
      }
      _last_update_time_decim[(int)m_lead.sensor_ids[0]] = message.timestamp;
      std::thread upd([this, &m_lead] { do_feature_propagate_update(m_lead); });
      trackFEATS->set_defer(true);
      { VPROF("1.track/TOTAL"); VScoped _s("KLT tracking"); trackFEATS->feed_new_camera(m_rest); }
      upd.join();
      trackFEATS->set_defer(false);
      trackFEATS->flush_pending();
      // any further leads whose gate also expired this frame (rare): sequential, as before
      for (size_t ci = 0; ci < m_rest.sensor_ids.size(); ci++) {
        int lead2 = (int)m_rest.sensor_ids[ci];
        auto it2 = _last_update_time_decim.find(lead2);
        if (it2 != _last_update_time_decim.end() && message.timestamp < it2->second + params.update_min_dt)
          continue;
        _last_update_time_decim[lead2] = message.timestamp;
        ov_core::CameraData sub;
        sub.timestamp = message.timestamp;
        sub.sensor_ids = {m_rest.sensor_ids[ci]};
        sub.images = {m_rest.images[ci]};
        sub.masks = {m_rest.masks[ci]};
        do_feature_propagate_update(sub);
      }
      return;
    }
    // no gate fires this frame: plain grouped tracking below (hybrid loop will fire nothing)
  }

  // Perform our feature tracking!
  // OV_PIPELINE: hold the tracker's database writes in pending_obs for the whole overlap
  // window.  Feature::uvs / uvs_norm / timestamps are unmutexed std::vectors and the updater
  // iterates the very Features the tracker would push_back onto (both feature queries run
  // with remove=false) -- that is the documented SIGSEGV class.  Committing after the drain
  // makes the database evolution the IDENTICAL serial sequence, cleanup() ordering included.
  // OV_DETERMINISTIC: deferral must be UNCONDITIONAL, not just while the pipeline is armed.
  // do_pipe is false until is_initialized_vio, so during initialisation the four camera threads
  // write the FeatureDatabase concurrently and the insertion order -- which fixes unordered_map
  // bucket order for the whole run -- is set by thread interleaving. Deferring always routes
  // every write through the single-threaded flush below.
  static const bool det_defer = [] { const char *e = std::getenv("OV_DETERMINISTIC"); return e && *e == '1'; }();
  if (do_pipe || det_defer) trackFEATS->set_defer(true);
  { const double _t0 = _dbg_now();
    { VPROF("1.track/TOTAL"); VScoped _s("KLT tracking"); trackFEATS->feed_new_camera(message); }
    g_dbg_track_ms += 1000.0 * (_dbg_now() - _t0); }
  // OV_PREJAC: the precompute worker launched by the previous update overlapped this frame's
  // tracking. Join BEFORE anything downstream can mutate the state (ZUPT / init / the update).
  if (prejac_on) prejac_join();
  if (do_pipe) {
    // Queue the NEXT frame's device prepare before blocking, so the GPU has work while we wait.
    if (post_track_hook) post_track_hook(message.timestamp);
    double _ums = 0;
    int _un = 0;
    { VScoped _s("pipe drain"); g_pipe.drain(_ums, _un); }
    _pipe_inflight.store(false, std::memory_order_release);
    // Columns 4/5 of OV_LAT_DUMP: the update batch that overlapped THIS frame's tracking.
    g_dbg_upd_ms = _ums;
    g_dbg_upd_n = _un;
    trackFEATS->set_defer(false);
    trackFEATS->flush_pending();
    // ---- from here the worker is idle: the state may be read and written again ----
    _pipe_oldest_time = state->margtimestep();
    if (_pipe_oldest_time > state->_timestamp) _pipe_oldest_time = -1;
    maybe_refresh_fisheye_masks(message);  // one frame late; mask drift is slow (flight-only)
    trackFEATS->snapshot_calib();          // race-free: nothing is writing CamBase now
    if (pose_sink) pose_sink();            // the pose of the update that just completed
  }
  if (!do_pipe && det_defer) {           // OV_DETERMINISTIC: commit the per-camera buckets
    trackFEATS->set_defer(false);
    trackFEATS->flush_pending();
  }
  if (async_update()) {
    g_upd.join();                          // update(N) finishes while we tracked N+1 (no-op pre-init)
    trackFEATS->flush_pending();           // commit deferred measurements (also during init!)
    maybe_refresh_fisheye_masks(message);  // one frame late in async mode; mask drift is slow
  }

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  // Tracking's GPU work has synced; the EKF update below leaves the device idle. Let the host
  // pipeline queue the NEXT frame's upload+CLAHE+pyramid into that window.
  if (!do_pipe && post_track_hook) post_track_hook(message.timestamp);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      { VScoped _s("ZUPT try_update"); did_zupt_update = updaterZUPT->try_update(state, message.timestamp); }
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      ov_publish_snap(state, propagator->gravity()); // ZUPT corrected the state too
      if (do_pipe && pose_sink) pose_sink();         // worker is drained; ZUPT moved the state
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (is_initialized_vio) ov_publish_snap(state, propagator->gravity());
    if (is_initialized_vio && _pipe_on && pose_sink) pose_sink();
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Update-decimation (config update_min_dt, seconds): track features on EVERY frame fed
  // to us, but run the EKF propagate+clone+update only at the reduced rate. Per-frame KLT
  // displacement halves at 30Hz (better track survival on aggressive motion) while clones
  // stay at the 15Hz parallax-optimal spacing. Observations at skipped (non-clone) times
  // are dropped inside the updaters via clean_old_measurements(clonetimes). <=0 = every frame.
  // HYBRID: grouped tracking (parallel across cameras, the 42->17 ms win) + sequential-style
  // per-camera updates (the shipped shape: ~29 ms each, on the frame, per-lead staggered
  // gates). Grouped-batch updates were pathological (175-222 ms each regardless of rate).
  static const bool hybrid_upd = [] {
    const char *e = std::getenv("OV_HYBRID_UPDATE");
    return e && *e == '1';
  }();
  if (hybrid_upd && message.sensor_ids.size() > 1) {
    // OV_UPDATE_STAGGER=1: the cameras are hardware-synced, so all four leads share the same
    // timestamps and their gates all expire on the SAME frame -- updates land as 4-at-once
    // bursts (~4x update cost every ~min_dt) instead of the intended rotation. Offsetting each
    // lead's FIRST gate by ci*min_dt/ncam keeps the per-lead rate and spacing identical but
    // spreads the updates one-per-frame, which is what a real-time deadline needs.
    static const bool upd_stagger = [] {
      const char *e = std::getenv("OV_UPDATE_STAGGER");
      return e && *e == '1';
    }();
    // OV_SPREAD_UPD=1: at a clone tick all four leads' gates expire together (synced cameras)
    // and the 4-update burst costs 40-100 ms -- the real-time latency tail. Run the first
    // sub-update now and QUEUE the rest, executing ONE per subsequent frame at the ORIGINAL
    // tick timestamp: the clone already exists, so the clone economy (the thing gate-staggering
    // destroyed) is untouched. Poses stay on-frame; three cams' tick measurements are ingested
    // 1-3 frames later. Queue drains in 4 frames against a 4.6-frame tick spacing.
    static const bool spread_upd = [] {
      const char *e = std::getenv("OV_SPREAD_UPD");
      return e && *e == '1';
    }();
    std::deque<std::pair<double, int>> &upd_queue = _upd_queue;
    if (spread_upd && params.update_min_dt > 0) {
      for (size_t ci = 0; ci < message.sensor_ids.size(); ci++) {
        int lead2 = (int)message.sensor_ids[ci];
        auto it2 = _last_update_time_decim.find(lead2);
        if (it2 != _last_update_time_decim.end() && message.timestamp < it2->second + params.update_min_dt)
          continue;
        _last_update_time_decim[lead2] = message.timestamp;
        upd_queue.push_back({message.timestamp, lead2});
      }
      // OV_SPREAD_ADAPT=1: look-ahead-gated drain. The shipped policy pops one queued
      // sub-update on EVERY frame, including frames that are already convoying -- so a frame
      // whose own tracking overran pays for a 5-50 ms sub-update while the next frame-set is
      // already sitting in the arrival queue. When a LATER frame has already arrived
      // (g_frames_queued > 0, set only in OV_REALTIME) we skip the drain and let the queue
      // absorb it, bounded two ways: a hard depth (OV_SPREAD_HARD) and an age deadline
      // (OV_SPREAD_DL_S) past which we force-drain TWO per frame. Arithmetic per sub-update is
      // untouched -- same call, same message.timestamp, same FIFO order -- only WHICH frame
      // executes it moves. Gate off => byte-identical to the shipped path; and with
      // OV_REALTIME off the counter is pinned at 0 so offline sweeps are unaffected either way.
      static const bool spread_adapt = [] { const char *e = std::getenv("OV_SPREAD_ADAPT"); return e && *e == '1'; }();
      static const double spread_dl_s = [] { const char *e = std::getenv("OV_SPREAD_DL_S"); return e ? atof(e) : 0.14; }();
      static const size_t spread_hard = [] { const char *e = std::getenv("OV_SPREAD_HARD"); return e ? (size_t)atoi(e) : 8; }();
      static const bool spread_stats = [] { const char *e = std::getenv("OV_SPREAD_STATS"); return e && *e == '1'; }();
      int budget;
      if (!spread_adapt) {
        budget = (upd_queue.size() > 8) ? 2 : 1; // shipped behaviour, bit-identical
      } else if (upd_queue.empty()) {
        budget = 0;
      } else {
        const bool forced = (upd_queue.size() > spread_hard) || (message.timestamp - upd_queue.front().first > spread_dl_s);
        const bool behind = ov_core::g_frames_queued.load(std::memory_order_relaxed) > 0;
        budget = forced ? 2 : (behind ? 0 : 1);
      }
      if (spread_stats) { // sanity envelope + mechanism counters, dumped once at exit
        struct SpreadStats {
          size_t max_q = 0, max_q_st = 0, n_tick = 0, n_behind = 0, n_forced = 0, n_skip = 0, n_two = 0;
          double max_lag = 0.0, max_lag_st = 0.0;
          std::vector<double> lags, lags_st;
          static double p99of(std::vector<double> &v) {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            return v[(size_t)(0.99 * (v.size() - 1))];
          }
          ~SpreadStats() {
            std::fprintf(stderr,
                         "[spreadq]: frames=%zu ALL maxq=%zu maxlag=%.3f p99lag=%.3f | STEADY(t>35) n=%zu maxq=%zu maxlag=%.3f p99lag=%.3f"
                         " | behind=%zu forced=%zu skipped=%zu two=%zu\n",
                         n_tick, max_q, max_lag, p99of(lags), lags_st.size(), max_q_st, max_lag_st, p99of(lags_st),
                         n_behind, n_forced, n_skip, n_two);
          }
        };
        static SpreadStats ss;
        static double first_ts = message.timestamp;
        const bool steady = (message.timestamp - first_ts) > 35.0; // same window rtbench2/score use
        ss.n_tick++;
        if (!upd_queue.empty()) {
          const double lag = message.timestamp - upd_queue.front().first;
          if (upd_queue.size() > ss.max_q) ss.max_q = upd_queue.size();
          if (lag > ss.max_lag) ss.max_lag = lag;
          ss.lags.push_back(lag);
          if (steady) { // the init transient stalls the feed thread for ~1-2 s and inflates both
            if (upd_queue.size() > ss.max_q_st) ss.max_q_st = upd_queue.size();
            if (lag > ss.max_lag_st) ss.max_lag_st = lag;
            ss.lags_st.push_back(lag);
          }
          if (ov_core::g_frames_queued.load(std::memory_order_relaxed) > 0) ss.n_behind++;
          if (budget == 0) ss.n_skip++;
          if (budget == 2) { ss.n_forced++; ss.n_two++; }
        }
      }
      if (do_pipe) {
        // OV_PIPELINE: pop the SAME items the serial path would pop, then hand them to the
        // worker BY VALUE.  Same measurements, same clone timestamps, same FIFO order --
        // only the thread and the wall-clock window change.  _upd_queue and
        // _last_update_time_decim stay estimator-thread-only.
        std::vector<ov_core::CameraData> _subs;
        while (budget-- > 0 && !upd_queue.empty()) {
          auto pr = upd_queue.front();
          upd_queue.pop_front();
          ov_core::CameraData sub;
          sub.timestamp = pr.first;
          sub.sensor_ids = {pr.second};
          sub.images = {cv::Mat()};
          sub.masks = {cv::Mat()};
          _subs.push_back(sub);
        }
        pipe_submit(_subs);
        return;
      }
      while (budget-- > 0 && !upd_queue.empty()) {
        auto pr = upd_queue.front();
        upd_queue.pop_front();
        ov_core::CameraData sub;
        sub.timestamp = pr.first;
        sub.sensor_ids = {pr.second};
        sub.images = {cv::Mat()};
        sub.masks = {cv::Mat()};
        { const double _t0 = _dbg_now();
          do_feature_propagate_update(sub);
          g_dbg_upd_ms += 1000.0 * (_dbg_now() - _t0);
          g_dbg_upd_n++; }
      }
      return;
    }
    std::vector<ov_core::CameraData> _pipe_subs;
    for (size_t ci = 0; ci < message.sensor_ids.size(); ci++) {
      ov_core::CameraData sub;
      sub.timestamp = message.timestamp;
      sub.sensor_ids = {message.sensor_ids[ci]};
      sub.images = {message.images[ci]};
      sub.masks = {message.masks[ci]};
      if (params.update_min_dt > 0) {
        int lead2 = (int)sub.sensor_ids[0];
        auto it2 = _last_update_time_decim.find(lead2);
        if (upd_stagger) {
          // Drift-free schedule: each lead's gate advances by EXACT multiples of min_dt from a
          // staggered seed, so the four phases can never re-align (snapping the gate to frame
          // timestamps let them collapse back into 4-at-once bursts within seconds).
          if (it2 == _last_update_time_decim.end()) {
            _last_update_time_decim[lead2] =
                sub.timestamp - params.update_min_dt + params.update_min_dt * (double)(ci + 1) / (double)message.sensor_ids.size();
            continue;
          }
          double &last = it2->second;
          if (sub.timestamp < last + params.update_min_dt)
            continue;
          last += params.update_min_dt * std::floor((sub.timestamp - last) / params.update_min_dt);
        } else {
          if (it2 != _last_update_time_decim.end() && sub.timestamp < it2->second + params.update_min_dt)
            continue;
          _last_update_time_decim[lead2] = sub.timestamp;
        }
      }
      if (do_pipe) {
        _pipe_subs.push_back(sub);
        continue;
      }
      { const double _t0 = _dbg_now();
        do_feature_propagate_update(sub);
        g_dbg_upd_ms += 1000.0 * (_dbg_now() - _t0);
        g_dbg_upd_n++; }
    }
    if (do_pipe) pipe_submit(_pipe_subs);
    return;
  }

  {
    const double update_min_dt = params.update_min_dt;
    if (update_min_dt > 0) {
      int lead = (int)message.sensor_ids.at(0);
      // Grouped multi-camera messages have ONE lead, so the per-lead decimation that let each
      // of 4 sequential messages through its own gate (4x the nominal rate -- the rate the
      // shipped accuracy was tuned at) would cut updates 4x. Compensate by scaling the gate.
      static const double dt_div = [] {
        const char *e = std::getenv("OV_UPDATE_DT_DIV");
        return e ? atof(e) : 1.0;
      }();
      const double eff_dt = update_min_dt / dt_div;  // 1=natural grouped rate; 4 exploded QR (222 ms/update)
      auto it = _last_update_time_decim.find(lead);
      if (it != _last_update_time_decim.end() && message.timestamp < it->second + eff_dt) {
        return; // tracked this frame, but skip the EKF update/clone
      }
      _last_update_time_decim[lead] = message.timestamp;
    }
  }

  // Call on our propagate and update function
  if (do_pipe) {
    std::vector<ov_core::CameraData> _subs(1, message);
    pipe_submit(_subs);
  } else if (do_async) {
    ov_core::CameraData mcopy = message;
    g_upd.t = std::thread([this, mcopy] { do_feature_propagate_update(mcopy); });
  } else {
    do_feature_propagate_update(message);
  }
}

void VioManager::preseed_next_frame(const ov_core::CameraData &m) {
  if (params.downsample_cameras) return;
  auto t = std::dynamic_pointer_cast<ov_core::TrackKLT>(trackFEATS);
  if (t) t->preseed_gpu(m);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {
  VPROF("0.do_feature_propagate_update(all)");
  // Safety net: nothing below may run concurrently with the precompute worker (the normal join
  // is right after tracking; this catches the >1-sub-update-per-frame drain path).
  if (prejac_on) prejac_join();
  // slow-update tracer: prints a phase breakdown for any update over 35 ms (stderr, cheap)
  struct UpdTrace {
    long pjh0 = ov_msckf::g_prejac_hit.load(), pjt0 = ov_msckf::g_prejac_try.load();
    UpdTrace() { g_dbg_sys_ms = g_dbg_cmp_ms = g_dbg_ekfu_ms = 0; g_dbg_rows = g_dbg_cols = 0;
                 g_sp_alloc = g_sp_jac = g_sp_null = g_sp_marg = g_sp_chi2 = g_sp_cpy = g_sp_stack = g_sp_rsz = 0; }
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double last = 0;
    double ph[6] = {0, 0, 0, 0, 0, 0};
    double mark(int i) {
      double now = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ph[i] += now - last;
      last = now;
      return now;
    }
    ~UpdTrace() {
      double tot = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      static const double thr = [] { const char *e = std::getenv("OV_SLOWUPD_MS"); return e ? atof(e) : 35.0; }();
      if (tot > thr)
        std::fprintf(stderr,
                     "[slowupd]: tot=%.1f sel=%.1f msckf=%.1f slam=%.1f init=%.1f marg=%.1f rest=%.1f "
                     "| sys=%.1f cmp=%.1f ekfu=%.1f rows=%ld cols=%ld pjhit=%ld/%ld"
                     " | sp alloc=%.1f jac=%.1f null=%.1f marg=%.1f chi2=%.1f cpy=%.1f stack=%.1f rsz=%.1f\n",
                     tot, ph[0], ph[1], ph[2], ph[3], ph[4], tot - ph[0] - ph[1] - ph[2] - ph[3] - ph[4],
                     g_dbg_sys_ms, g_dbg_cmp_ms, g_dbg_ekfu_ms, g_dbg_rows, g_dbg_cols,
                     ov_msckf::g_prejac_hit.load() - pjh0, ov_msckf::g_prejac_try.load() - pjt0,
                     g_sp_alloc, g_sp_jac, g_sp_null, g_sp_marg, g_sp_chi2, g_sp_cpy, g_sp_stack, g_sp_rsz);
    }
  } _ut;
  VScoped _sall("do_feature_propagate_update (all)");

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    { VPROF("2.propagate_clone"); VScoped _s("propagate+clone"); propagator->propagate_and_clone(state, message.timestamp); }
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // Disparity-aware (two-way) clone marginalization decision (VINS-Mono style)
  //===================================================================================
  // Decide NOW (before any feats_marg selection / cleanup / marginalization)

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  VPROF("B.select_and_sort");
  { VPROF("9.db/features_not_containing_newer");
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true); }
  // OV_PREJAC: keep the UNFILTERED lost set + the marg ids so the producer at the end of this
  // call can rebuild the NEXT sub-update's feats_lost without a second database scan. The lost
  // set cannot grow between now and then (a lost feature never gets a new observation, and a
  // still-tracked one has an observation at state->_timestamp so it stays out of it).
  std::vector<std::shared_ptr<Feature>> _pj_lost_all;
  std::unordered_set<size_t> _pj_marg_ids;
  if (prejac_on) _pj_lost_all = feats_lost;

  // Get the features that will be marginalized out of the sliding window.
  if (((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5)) {
    { VPROF("9.db/features_containing");
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true); }
    if (prejac_on)
      for (const auto &f : feats_marg)
        _pj_marg_ids.insert(f->featid);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  // erase-remove instead of erase-in-loop: the latter is O(n^2) because every vector::erase
  // shifts the tail. Same predicate, same surviving order.
  { VPROF("9.db/prune_lost_by_camid");
  feats_lost.erase(std::remove_if(feats_lost.begin(), feats_lost.end(),
                                  [&](const std::shared_ptr<Feature> &f) {
                                    for (const auto &camuvpair : f->uvs)
                                      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(),
                                                    camuvpair.first) != message.sensor_ids.end())
                                        return false;
                                    return true;
                                  }),
                   feats_lost.end()); }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  // was O(|feats_lost| * |feats_marg|) linear scans plus O(n^2) erase; hash-set makes it O(n).
  { VPROF("9.db/prune_lost_in_marg");
  std::unordered_set<std::shared_ptr<Feature>> marg_set(feats_marg.begin(), feats_marg.end());
  feats_lost.erase(std::remove_if(feats_lost.begin(), feats_lost.end(),
                                  [&](const std::shared_ptr<Feature> &f) { return marg_set.count(f) != 0; }),
                   feats_lost.end()); }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // ===== OV_SLAMSTAT (census only, no behaviour change) ==================================
  // Answers the three things the SLAM-promotion analysis could NOT close from artifacts:
  //  (a) |feats_maxtracks| per sub-update -- the CANDIDATE SUPPLY the cap is throttling;
  //  (b) whether sub-updates 2-4 of an OV_SPREAD_UPD tick contribute any candidates at all;
  //  (c) promotion attempts vs successes vs SLAM marginalizations, i.e. the true rate and tau.
  static const bool slamstat = [] { const char *e = std::getenv("OV_SLAMSTAT"); return e && *e == '1'; }();
  struct SlamStat {
    // do_feature_propagate_update runs on the OV_PIPELINE worker in the ship config, but the
    // OV_ASYNC_UPDATE / group-cam paths can call it from a spawned thread. One mutex per
    // sub-update (~20/s) removes any doubt for zero measurable cost.
    std::mutex mtx;
    size_t subupd = 0, ticks = 0, cand_sum = 0, cand_max = 0, attempts = 0, sub1_cand = 0, sub234_cand = 0;
    std::vector<int> cand, nslam;
    static int pct(std::vector<int> &v, double q) {
      if (v.empty()) return 0;
      std::sort(v.begin(), v.end());
      return v[(size_t)(q * (double)(v.size() - 1))];
    }
    ~SlamStat() {
      std::fprintf(stderr,
                   "[slamstat]: subupd=%zu ticks=%zu cand_p50=%d cand_p90=%d cand_max=%zu cand_sum=%zu "
                   "attempts=%zu init_ok=%ld init_fail=%ld margslam=%ld nslam_p50=%d nslam_p90=%d nslam_max=%d "
                   "sub1_cand=%zu sub234_cand=%zu\n",
                   subupd, ticks, pct(cand, 0.50), pct(cand, 0.90), cand_max, cand_sum, attempts,
                   ov_msckf::g_slam_init_ok.load(), ov_msckf::g_slam_init_fail.load(), ov_msckf::g_slam_marg.load(),
                   pct(nslam, 0.50), pct(nslam, 0.90), nslam.empty() ? 0 : nslam.back(), sub1_cand, sub234_cand);
    }
  };
  static SlamStat slamss;
  // A clone tick is the sub-update that will marginalize the oldest clone (StateHelper::
  // marginalize_old_clone fires under exactly this predicate). Sub-updates 2-4 of an
  // OV_SPREAD_UPD tick share the tick's timestamp and see clones == max, so they are not leads.
  const bool is_tick_lead = ((int)state->_clones_IMU.size() > state->_options.max_clone_size);
  if (slamstat) {
    std::lock_guard<std::mutex> _lk(slamss.mtx);
    slamss.subupd++;
    if (is_tick_lead) slamss.ticks++;
    const size_t c = feats_maxtracks.size();
    slamss.cand.push_back((int)c);
    slamss.cand_sum += c;
    if (c > slamss.cand_max) slamss.cand_max = c;
    (is_tick_lead ? slamss.sub1_cand : slamss.sub234_cand) += c;
    slamss.nslam.push_back((int)state->_features_SLAM.size());
  }

  // ===== OV_SLAM_PICK_BEST (default OFF): promote the LONGEST track, not a hash-order one ====
  // feats_maxtracks inherits its order from FeatureDatabase::features_containing, which iterates
  // an std::unordered_map -- hash-bucket order, uncorrelated with track length, camera count or
  // baseline. The promotion below takes the vector's TAIL, so today the single promoted landmark
  // per clone tick is chosen AT RANDOM from the eligible pool, while the MSCKF path already
  // sorts by the identical criterion (compare_feat) before its own cap. Sorting ASCENDING by
  // total observations makes the tail the BEST candidate. ZERO state-dimension change, and the
  // leftovers handed to featsup_MSCKF are unchanged AS A SET (that vector is re-sorted by the
  // same comparator anyway), so the MSCKF path is arithmetically unaffected except for which
  // single element left it.
  static const bool pick_best = [] { const char *e = std::getenv("OV_SLAM_PICK_BEST"); return e && *e == '1'; }();
  if (pick_best && feats_maxtracks.size() > 1) {
    std::sort(feats_maxtracks.begin(), feats_maxtracks.end(),
              [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
                size_t as = 0, bs = 0;
                for (const auto &pr : a->timestamps)
                  as += pr.second.size();
                for (const auto &pr : b->timestamps)
                  bs += pr.second.size();
                return as < bs;
              });
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    // OV_MAX_INIT_PER_UP: ramp cap on SLAM promotions per update. Right after initialization
    // amount_to_add is (max_slam - 0) = hundreds, and each delayed init costs ~4-8 ms -- the
    // measured 100-760 ms post-init update storm. Steady-state churn is <1 promotion/update,
    // so a cap of ~8 binds ONLY during the storm; it just spreads the ramp over ~1 s of frames.
    static const int max_init_per_up = [] {
      const char *e = std::getenv("OV_MAX_INIT_PER_UP");
      return e ? atoi(e) : 0;
    }();
    // OV_MAX_INIT_STEADY=k (default 0 = OFF => byte-identical to today). The comment above is
    // WRONG about steady state and the measurement says so: under OV_SPREAD_UPD the four
    // sub-updates of a clone tick share one timestamp, sub-update 1 marginalizes the clone and
    // flags every feats_marg/feats_maxtracks leftover to_delete, so the promotion budget is spent
    // ONCE PER TICK -- 8/s, not 30/s -- against a candidate pool many times larger. The
    // post-init storm the cap was added for lives entirely inside OV_RAMP_SECS, which
    // OV_RAMP_MSCKF already fences, so the cap can be tight there and loose afterwards.
    static const int max_init_steady = [] {
      const char *e = std::getenv("OV_MAX_INIT_STEADY");
      return e ? atoi(e) : 0;
    }();
    static const double ramp_secs_i = [] {
      const char *e = std::getenv("OV_RAMP_SECS");
      return e ? atof(e) : 0.0;
    }();
    int init_cap = max_init_per_up;
    if (max_init_steady > 0 && (ramp_secs_i <= 0.0 || message.timestamp - startup_time >= ramp_secs_i))
      init_cap = max_init_steady;
    static std::atomic<int> _slaminit_banner{0};
    if (_slaminit_banner.exchange(1) == 0)
      PRINT_INFO("[slaminit]: per_up=%d steady=%d ramp=%.1f pick_best=%d\n", max_init_per_up, max_init_steady, ramp_secs_i,
                 (int)pick_best);
    if (init_cap > 0 && amount_to_add > init_cap)
      amount_to_add = init_cap;
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    if (slamstat && valid_amount > 0) {
      std::lock_guard<std::mutex> _lk(slamss.mtx);
      slamss.attempts += (size_t)valid_amount;
    }
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  { VScoped _s("marginalize_slam"); StateHelper::marginalize_slam(state); }

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // OV_SLAM_UP_CAP: cap SLAM features measurement-updated per cycle. SLAM features persist in
  // the state; a deferred feature's observations stay in the DB (within the clone window) and
  // it is re-selected next update. The un-capped SLAM update was the residual 44-65 ms spike
  // the MSCKF row-budget could not touch.
  static const int slam_up_cap = [] { const char *e = std::getenv("OV_SLAM_UP_CAP"); return e ? atoi(e) : 0; }();
  if (slam_up_cap > 0 && (int)feats_slam_UPDATE.size() > slam_up_cap)
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin() + slam_up_cap, feats_slam_UPDATE.end());

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);


  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  // OV_RAMP_SECS/OV_RAMP_MSCKF: right after initialization the DB holds every feature tracked
  // during the pre-init phase; the first updates ingest hundreds at once (measured 100-760 ms).
  // For the first RAMP_SECS after startup, cap the batch (longest tracks are kept -- the erase
  // drops from the front). Steady state is untouched.
  static const double ramp_secs = [] { const char *e = std::getenv("OV_RAMP_SECS"); return e ? atof(e) : 0.0; }();
  static const int ramp_msckf = [] { const char *e = std::getenv("OV_RAMP_MSCKF"); return e ? atoi(e) : 40; }();
  // OV_MSCKF_CAP: steady-state per-update MSCKF batch cap. Feature death is bursty in flight
  // (a yaw kills a whole camera's tracks at once -> 50-70 ms single updates). Capped-out
  // features stay in the database and are ingested on the following updates -- deferral, not
  // loss; the 6.6 s clone window gives ample slack.
  static const int msckf_cap_env = [] { const char *e = std::getenv("OV_MSCKF_CAP"); return e ? atoi(e) : 0; }();
  int msckf_cap = state->_options.max_msckf_in_update;
  if (msckf_cap_env > 0 && msckf_cap_env < msckf_cap)
    msckf_cap = msckf_cap_env;
  if (ramp_secs > 0.0 && message.timestamp - startup_time < ramp_secs && ramp_msckf < msckf_cap)
    msckf_cap = ramp_msckf;
  if ((int)featsup_MSCKF.size() > msckf_cap)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - msckf_cap);

  // OV_UPD_BUDGET_MS: per-frame time-budget scheduler. Total update work fits the frame rate
  // (offline average 18.3 ms/frame vs a 34.5 ms budget) -- the latency tail is VARIANCE, not
  // volume. After tracking (cost already measured in g_dbg_track_ms) the MSCKF batch is sized
  // by a row-cost model to fill the frame's remaining budget: yaw-burst feature deaths smear
  // over the next few updates instead of one 50-90 ms whale. Deferred features are LOST tracks
  // (no new observations accrue), so this avoids the count-cap backfire. Feature order is
  // OpenVINS's own (sorted, best-last); we keep the tail.
  static const double upd_budget = [] { const char *e = std::getenv("OV_UPD_BUDGET_MS"); return e ? atof(e) : 0.0; }();
  // Cost model: msckf_ms ~= (F + nf*Cf + rows*K) * scale. The per-FEATURE term Cf is the
  // dominant one (triangulation Gauss-Newton over up to 50 clone poses; measured 27-78 ms
  // updates were ~0.4-0.8 ms/feature, not row-bound). `scale` self-calibrates online.
  static const double cf_ms = [] { const char *e = std::getenv("OV_UPD_FEAT_MS"); return e ? atof(e) : 0.4; }();
  static const double row_k = [] { const char *e = std::getenv("OV_UPD_ROW_MS"); return e ? atof(e) : 0.02; }();
  static const int age_force = [] { const char *e = std::getenv("OV_DEFER_AGE"); return e ? atoi(e) : 6; }();
  static double ema_F = 6.0, cal_scale = 1.0;
  static double g_sched_pred = -1.0;
  static std::map<size_t, int> defer_age; // featid -> updates deferred (starvation guard)
  if (upd_budget > 0.0 && !featsup_MSCKF.empty()) {
    double avail = upd_budget - g_dbg_track_ms - g_dbg_upd_ms - ema_F;
    if (avail < 2.0) avail = 2.0; // always make some progress
    // Marg-boundary features are alive: deferring them makes them come back BIGGER next update
    // and leaks their oldest observations each cycle (measured: standing backlog, every frame
    // saturated at budget). They are always ingested; only LOST (dead) features are deferrable.
    // Marg features have exactly one tick (~4 frames) of slack before their oldest clone is
    // marginalized: budget-defer them up to 3 frames (ZERO obs loss), force-include at the
    // deadline. This smooths the periodic marg whale into per-frame chunks.
    static std::map<size_t, int> marg_age;
    std::set<Feature *> must_do;
    for (auto &f : feats_marg)
      if (++marg_age[f->featid] >= 3)
        must_do.insert(f.get());
    std::vector<std::shared_ptr<Feature>> keepv;
    keepv.reserve(featsup_MSCKF.size());
    double cost = 0;
    auto fcost = [&](const std::shared_ptr<Feature> &f) {
      long nobs = 0;
      for (auto &kv : f->timestamps) nobs += (long)kv.second.size();
      // NOTE: width-aware pricing (rows x (nobs/15)^2) was tested and REJECTED: pricing whales
      // 9x starves long-track ingestion behind the must-do marg set (ATE 2.65 -> 10.2). Flat.
      return (cf_ms + row_k * std::max<long>(1, 2 * nobs - 3)) * cal_scale;
    };
    int aged_in = 0;
    for (auto &f : featsup_MSCKF) {
      const bool aged = [&] {
        auto it = defer_age.find(f->featid);
        return it != defer_age.end() && it->second >= age_force && aged_in < 5;
      }();
      if (must_do.count(f.get()) || aged) {
        cost += fcost(f);
        keepv.push_back(f);
        if (aged) aged_in++;
      }
    }
    std::set<Feature *> is_marg;
    for (auto &f : feats_marg)
      is_marg.insert(f.get());
    for (int pass = 0; pass < 2; pass++) {
      for (size_t i = featsup_MSCKF.size(); i > 0; i--) {
        auto &f = featsup_MSCKF[i - 1];
        if ((pass == 0) != (is_marg.count(f.get()) > 0))
          continue; // pass 0: marg (deadline-near) first; pass 1: lost
        if (std::find(keepv.begin(), keepv.end(), f) != keepv.end())
          continue;
        const double c = fcost(f);
        if (!keepv.empty() && cost + c > avail)
          goto budget_done;
        cost += c;
        keepv.push_back(f);
        if (cost > avail)
          goto budget_done;
      }
    }
  budget_done:;
    for (auto &f : featsup_MSCKF) {
      if (std::find(keepv.begin(), keepv.end(), f) == keepv.end())
        defer_age[f->featid]++;
      else {
        defer_age.erase(f->featid);
        marg_age.erase(f->featid);
      }
    }
    if (marg_age.size() > 4096) marg_age.clear();
    if (defer_age.size() > 4096) defer_age.clear(); // stale-id backstop
    g_sched_pred = cost;
    featsup_MSCKF.swap(keepv);
  }
  // OV_THIN_OBS: wide-span LOST features (consumed and deleted after this update) get every
  // 2nd observation dropped: halves whale rows AND involved width at an explicit, fleet-gated
  // accuracy cost. Marg/alive features are never touched. NOTE: used to live INSIDE the
  // budget-scheduler block, so without OV_UPD_BUDGET_MS it silently did nothing -- now
  // standalone (P9 bundled both; JSOB unknowingly re-tested the no-thin config).
  {
    static const int thin_over = [] { const char *e = std::getenv("OV_THIN_OBS"); return e ? atoi(e) : 0; }();
    if (thin_over > 0 && !featsup_MSCKF.empty()) {
      std::set<Feature *> is_marg2;
      for (auto &f : feats_marg)
        is_marg2.insert(f.get());
      for (auto &f : featsup_MSCKF) {
        if (is_marg2.count(f.get()))
          continue;
        long nobs = 0;
        for (auto &kv : f->timestamps) nobs += (long)kv.second.size();
        if (nobs <= thin_over)
          continue;
        for (auto &kv : f->timestamps) {
          auto &ts = kv.second;
          auto &uv = f->uvs[kv.first];
          auto &un = f->uvs_norm[kv.first];
          size_t w = 0;
          for (size_t r2 = 0; r2 < ts.size(); r2++)
            if (r2 % 2 == 0 || r2 + 1 == ts.size()) {
              ts[w] = ts[r2]; uv[w] = uv[r2]; un[w] = un[r2]; w++;
            }
          ts.resize(w); uv.resize(w); un.resize(w);
        }
      }
    }
  }
  { VScoped _s("MSCKF update"); {
    _ut.mark(0);
    const auto _tm0 = std::chrono::steady_clock::now();
    updaterMSCKF->update(state, featsup_MSCKF);
    const double _tm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _tm0).count();
    if (upd_budget > 0.0 && g_sched_pred > 0.5)
      cal_scale = std::min(4.0, std::max(0.25, 0.9 * cal_scale + 0.1 * (cal_scale * _tm / g_sched_pred)));
  } }
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    _ut.mark(1);
    { VScoped _s("SLAM update"); updaterSLAM->update(state, featsup_TEMP); }
    _ut.mark(2);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  _ut.mark(2);
  { VScoped _s("SLAM delayed_init"); updaterSLAM->delayed_init(state, feats_slam_DELAYED); }
  _ut.mark(3);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    { VPROF("A.retriangulate_active");
    static const bool skip_active = [] { const char *e = std::getenv("OV_SKIP_ACTIVE_TRACKS"); return e && *e == '1'; }();
    if (!skip_active) retriangulate_active_tracks(message); }

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  { VPROF("9.db/cleanup");
  trackFEATS->get_feature_database()->cleanup(); }
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);


  // Cleanup any features older than the marginalization time.
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    { VPROF("9.db/cleanup_measurements");
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep()); }
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed.
  { VPROF("9.db/marginalize_old_clone"); VScoped _s("marginalize_old_clone"); StateHelper::marginalize_old_clone(state); }
  _ut.mark(4);

  // OV_ASYNC_EMIT: last state write of this sub-update is done -- hand the emitter a snapshot.
  ov_publish_snap(state, propagator->gravity());

  // OV_PREJAC: hand the NEXT queued sub-update's per-feature system to a worker. From here to
  // the join (right after the next frame's tracking) the state is frozen and the snapshot's
  // observation sets are frozen, so the result is bit-identical to the inline computation.
  if (prejac_on && is_initialized_vio && !_upd_queue.empty())
    prejac_launch(_upd_queue.front().second, _pj_lost_all, _pj_marg_ids);

  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // [VIZ LOG] per-frame snapshot
  {
    auto& L = ov_msckf::OvLogger::Get();
    if (L.enabled()) {
      // Current pose: state->_imu has q_GtoI (xyzw) and p_IinG.
      // We want T_GtoI inverted → T_ItoG which gives camera-in-world (w,t are
      // both already in world frame in OV: p_IinG is IMU origin in world).
      Eigen::Vector4d q_GtoI = state->_imu->quat();  // (qx,qy,qz,qw)
      Eigen::Vector3d p_IinG = state->_imu->pos();
      // For visualization we want world-from-body, so the quaternion is
      // q_GtoI's inverse. Output as xyzw.
      double twc[7] = {p_IinG.x(), p_IinG.y(), p_IinG.z(),
                       -q_GtoI(0), -q_GtoI(1), -q_GtoI(2), q_GtoI(3)};
      std::vector<long> feat_ids;
      // Observed features = MSCKF features marginalized this update + SLAM
      // features currently in state. Both are tracked from `featsup_MSCKF` /
      // `state->_features_SLAM`. We just dump SLAM IDs here (cheap to get).
      feat_ids.reserve(state->_features_SLAM.size());
      for (auto const &f : state->_features_SLAM) feat_ids.push_back((long)f.first);
      L.log_frame(message.timestamp, (int)(message.timestamp * 1e6),
                  twc, feat_ids);

      // Active clones (the "window") + their world poses
      std::vector<long> clone_ids;
      std::vector<double> clone_twc;
      clone_ids.reserve(state->_clones_IMU.size());
      clone_twc.reserve(state->_clones_IMU.size() * 7);
      for (auto const &c : state->_clones_IMU) {
        // ID = ts in nanoseconds (so it sorts naturally)
        long cid = (long)(c.first * 1e9);
        clone_ids.push_back(cid);
        Eigen::Vector4d q = c.second->quat();
        Eigen::Vector3d t = c.second->pos();
        clone_twc.push_back(t.x()); clone_twc.push_back(t.y()); clone_twc.push_back(t.z());
        clone_twc.push_back(-q(0)); clone_twc.push_back(-q(1));
        clone_twc.push_back(-q(2)); clone_twc.push_back(q(3));
      }
      L.log_window(message.timestamp, clone_ids);
      L.log_clone_snapshot(message.timestamp, clone_ids, clone_twc);

      // SLAM features snapshot (world positions). Each feature has p_FinG()
      std::vector<long> mp_ids;
      std::vector<double> mp_xyz;
      mp_ids.reserve(state->_features_SLAM.size());
      mp_xyz.reserve(state->_features_SLAM.size() * 3);
      for (auto const &f : state->_features_SLAM) {
        Eigen::Vector3d p = f.second->get_xyz(false);
        mp_ids.push_back((long)f.first);
        mp_xyz.push_back(p.x()); mp_xyz.push_back(p.y()); mp_xyz.push_back(p.z());
      }
      L.log_feat_snapshot(message.timestamp, mp_ids, mp_xyz);
    }
  }

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
}

void VioManager::maybe_refresh_fisheye_masks(ov_core::CameraData &message) {
  if (!params.use_mask) return;
  if (!params.mask_fisheye_auto) return;
  if (!params.mask_fisheye_dynamic) return;
  if (state == nullptr) return;

  for (size_t k = 0; k < message.sensor_ids.size(); k++) {
    int cam_id = (int)message.sensor_ids.at(k);
    auto it_cam = state->_cam_intrinsics_cameras.find(cam_id);
    if (it_cam == state->_cam_intrinsics_cameras.end()) continue;
    auto cam = it_cam->second;
    Eigen::VectorXd v = cam->get_value();
    if (v.size() < 4) continue;
    double fx = v(0), fy = v(1), cx = v(2), cy = v(3);

    auto it_cache = _mask_intr_cache.find(cam_id);
    bool need_rebuild = false;
    if (it_cache == _mask_intr_cache.end()) {
      need_rebuild = true;
    } else {
      double d = std::abs(it_cache->second(0) - fx) + std::abs(it_cache->second(1) - fy) +
                 std::abs(it_cache->second(2) - cx) + std::abs(it_cache->second(3) - cy);
      if (d > params.mask_fisheye_dyn_thresh_px) need_rebuild = true;
    }
    if (!need_rebuild) continue;

    int W = cam->w();
    int H = cam->h();
    double f = 0.5 * (fx + fy);
    double theta_max_i = params.mask_fisheye_theta_max;
    auto it_tm = params.mask_fisheye_theta_max_per_cam.find(cam_id);
    if (it_tm != params.mask_fisheye_theta_max_per_cam.end()) theta_max_i = it_tm->second;
    double r_pix = f * theta_max_i - params.mask_fisheye_inset_px;
    if (r_pix < 1.0) r_pix = 1.0;

    cv::Mat mask(H, W, CV_8UC1, cv::Scalar(255));
    cv::circle(mask, cv::Point((int)std::round(cx), (int)std::round(cy)),
               (int)std::round(r_pix), cv::Scalar(0), -1);
    // Bottom-strip mask: per-cam override or global default
    double bottom_frac_i = params.mask_bottom_frac;
    auto it_bf = params.mask_bottom_frac_per_cam.find(cam_id);
    if (it_bf != params.mask_bottom_frac_per_cam.end()) bottom_frac_i = it_bf->second;
    int y_cut = (int)std::round(H * (1.0 - bottom_frac_i));
    if (bottom_frac_i > 0.0 && y_cut < H) {
      cv::rectangle(mask, cv::Point(0, y_cut), cv::Point(W, H), cv::Scalar(255), -1);
    }
    params.masks[(size_t)cam_id] = mask;
    if (k < message.masks.size()) message.masks.at(k) = mask;

    Eigen::Vector4d cur;
    cur << fx, fy, cx, cy;
    _mask_intr_cache[cam_id] = cur;

    PRINT_INFO("VioManager: refreshed fisheye mask cam%d cx,cy=(%.1f,%.1f) f=%.2f -> r=%.1f px\n",
               cam_id, cx, cy, f, r_pix);
  }
}
