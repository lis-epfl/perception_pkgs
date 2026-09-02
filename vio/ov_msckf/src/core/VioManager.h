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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <functional>
#include <deque>
#include <thread>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include "VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class Feature;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  /// Fires right after feature tracking, before the EKF update -- lets the host pipeline
  /// preseed the NEXT frame's GPU work into the update's idle window. Arg: current frame ts.
  void set_post_track_hook(std::function<void(double)> f) { post_track_hook = std::move(f); }

  // ================= OV_PIPELINE (default OFF) ==========================================
  // Overlap the EKF update of frame N with the GPU tracking of frame N+1.  The update runs
  // on ONE persistent worker thread; the estimator thread drains it immediately after the
  // next frame's tracking, so exactly one update is ever in flight and the sequence of
  // filter operations (which measurements, which clone times, which order) is UNCHANGED --
  // only which thread and which wall-clock window executes them.
  //
  // Consequence for anyone reading the state: between the submit at the end of a frame and
  // the drain in the middle of the next one, `state` is being MUTATED by the worker and must
  // not be read.  That is why the pose is emitted through pose_sink, which fires only at
  // points where the worker is provably idle.
  /// Fires whenever the filter state is quiescent and may have advanced (post-drain, post-ZUPT,
  /// post-init).  Only used when the pipeline is armed; the caller logs the pose from here
  /// instead of reading the live state after feed_measurement_camera returns.
  void set_pose_sink(std::function<void()> f) { pose_sink = std::move(f); }
  /// True only if OV_PIPELINE=1 AND every refusal check passed.  Read it back from the
  /// "[pipe]: armed=..." line in the run log; never assume the env var took effect.
  bool pipeline_on() const { return _pipe_on; }
  /// Drain the worker and fire pose_sink.  MUST be called once after the last frame, before
  /// anything reads the state (final .tum row, calibration dump, destructors).
  void pipeline_flush();
  /// Hand a batch of sub-updates to the persistent worker.  The batch is captured BY VALUE so
  /// the worker never touches _upd_queue or _last_update_time_decim.
  void pipe_submit(const std::vector<ov_core::CameraData> &subs);
  std::function<void()> pose_sink;
  bool _pipe_on = false;
  /// margtimestep()-derived IMU trim time cached at the last quiescent point.  See
  /// feed_measurement_imu: state->_clones_IMU is inserted into WITHOUT the state mutex
  /// (StateHelper::augment_clone), so the producer must not iterate it while the worker runs.
  double _pipe_oldest_time = -1.0;
  std::atomic<bool> _pipe_inflight{false};
  /// Queue next frame's device prepare (async). No-op for non-KLT trackers or downsampling.
  void preseed_next_frame(const ov_core::CameraData &m);

  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /// Joins the OV_PREJAC precompute worker. Without this, a still-joinable std::thread member
  /// is destroyed at shutdown and std::terminate() aborts the process (which also truncated the
  /// buffered latency dump).
  ~VioManager();

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// If we are initialized or not
  bool initialized() { return is_initialized_vio && timelastupdate != -1; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image();

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// Refresh the auto-fisheye masks if state's current intrinsics drifted from
  /// the values used to build the cached mask. Updates params.masks AND the
  /// current message's masks vector in-place.
  void maybe_refresh_fisheye_masks(ov_core::CameraData &message);

  /// Per-cam cache of (fx,fy,cx,cy) used at the last mask rebuild.
  std::map<int, Eigen::Vector4d> _mask_intr_cache;

  /// Per-lead-cam time of the last EKF update (OV_UPDATE_MIN_DT decimation).
  std::map<int, double> _last_update_time_decim;

  /// OV_SPREAD_UPD queue of pending sub-updates (tick timestamp, lead cam id). Member rather
  /// than a function-static so the producer at the end of do_feature_propagate_update can see
  /// which sub-update runs next.
  std::deque<std::pair<double, int>> _upd_queue;

  //=================================================================================
  // OV_PREJAC: hoist the next queued sub-update's triangulation + Jacobian + nullspace
  // projection onto a worker that runs during the next frame's tracking window. Exact.
  //=================================================================================
  std::thread _prejac_thread;
  /// Snapshot (deep copies) of the features the worker is allowed to touch. Copies, so the
  /// worker never reads a Feature the tracker may be writing to.
  std::vector<std::shared_ptr<ov_core::Feature>> _prejac_snap;
  uint64_t _prejac_epoch = 0;
  /// Join the precompute worker (no-op if none running). MUST be called before anything can
  /// mutate the state or the feature database.
  void prejac_join();
  /// Snapshot + launch. `lost_all` is the unfiltered feats_lost of the update that just ran.
  void prejac_launch(int lead_cam, const std::vector<std::shared_ptr<ov_core::Feature>> &lost_all,
                     const std::unordered_set<size_t> &marg_ids);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;
  std::function<void(double)> post_track_hook;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;

  // If we did a zero velocity update
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
