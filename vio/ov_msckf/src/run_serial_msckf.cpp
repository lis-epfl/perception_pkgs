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
#include "core/VioManagerOptions.h" // pulls in ov_core::YamlParser + Printer
#include "utils/bag_source.h"       // merged multi-file reader + deferred decode
#include "utils/dataset_reader.h"   // pulls in ov_core::CameraData / ImuData
#include "state/State.h"
#include "state/Propagator.h"

namespace ov_msckf {
// Defined in VioManager.cpp: per-stage wall clock inside feed_measurement_camera.
extern std::map<std::string, double> g_vio_stage_secs;
extern std::map<std::string, long> g_vio_stage_calls;
} // namespace ov_msckf

using namespace ov_msckf;

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

int main(int argc, char **argv) {
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
  params.use_multi_threading_subs = false; // serial / synchronous
  auto sys = std::make_shared<VioManager>(params);

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
  };
  std::map<double, Frame> pending; // keyed by stamp, sorted ascending
  double latest_imu_t = -1;

  std::vector<std::string> lines;
  lines.reserve(6000);
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
    std::fprintf(cf, "}\n");
  };

  auto write_calib = [&](const std::string &path) {
    FILE *cf = std::fopen(path.c_str(), "w");
    if (!cf) return;
    write_calib_to(cf);
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
  auto feed_frame = [&](const Frame &fr) {
    bool fed_any = false;
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
    } else {
      for (auto const &kv : fr.payloads) {
        int cid = kv.first;
        if (throttled(cid)) continue;
        groups.push_back({cid});
      }
    }
    if (groups.empty()) return;

    // 2. Decode only what those groups need, concurrently. Each decode is an
    //    independent pure function of its own payload, so this does not disturb
    //    the bit-identical-across-runs property the serial runner exists for.
    std::vector<int> need;
    for (auto const &g : groups)
      for (int c : g)
        if (std::find(need.begin(), need.end(), c) == need.end())
          need.push_back(c);
    std::vector<cv::Mat> decoded(need.size());
    const double _tdec = now_s();
    cv::parallel_for_(cv::Range(0, (int)need.size()), [&](const cv::Range &rng) {
      for (int i = rng.start; i < rng.end; i++)
        decoded[i] = fr.payloads.at(need[i]).decode();
    });
    SW.add("  jpeg decode", now_s() - _tdec);
    std::map<int, cv::Mat> imgs;
    for (size_t i = 0; i < need.size(); i++) {
      if (decoded[i].empty()) {
        PRINT_WARNING(YELLOW "[serial]: cam%d frame at %.6f failed to decode; skipping\n" RESET, need[i], fr.ts);
        continue;
      }
      imgs[need[i]] = decoded[i];
    }
    if (imgs.empty()) return;
    const int rows = imgs.begin()->second.rows;
    const int cols = imgs.begin()->second.cols;

    // 3. Feed.
    for (auto const &g : groups) {
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
    if (fed_any && sys->initialized() && (sys->get_state()->_timestamp - sys->initialized_time()) >= 1.0 &&
        sys->get_state()->_timestamp != last_logged_ts) {
      auto state = sys->get_state();
      last_logged_ts = state->_timestamp;
      Eigen::Vector4d q = state->_imu->quat(); // q_GtoI, JPL xyzw (matches odomimu)
      Eigen::Vector3d p = state->_imu->pos();  // p_IinG
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f",
                    state->_timestamp, p(0), p(1), p(2), q(0), q(1), q(2), q(3));
      lines.emplace_back(buf);
      if (series != nullptr) {
        const double ts = state->_timestamp;
        if (series_next < 0)
          series_next = ts;              // first snapshot as soon as the filter is up
        if (ts >= series_next) {
          std::fprintf(series, "{\"t\":%.6f,\"calib\":", ts);
          write_calib_to(series);
          series_next = ts + series_dt;
        }
      }
    }
  };

  // Flush any pending frame whose stamp the IMU has now passed (IMU available
  // through the image time => propagation/update is well-posed).
  auto flush_ready = [&]() {
    while (!pending.empty() && pending.begin()->first <= latest_imu_t) {
      feed_frame(pending.begin()->second);
      pending.erase(pending.begin());
    }
  };

  SW.add("setup+config+masks", now_s() - _wall0);

  size_t n_imu = 0, n_img = 0, n_frames = 0;
  for (;;) {
    double _t = now_s();
    BagRecord rec = source.next();
    SW.add("bag read+deserialize", now_s() - _t);
    if (rec.kind == BagRecord::NONE) break;

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
      }
    }
  }
  // EOF: flush whatever remains (IMU stream ended).
  latest_imu_t = std::numeric_limits<double>::infinity();
  { Scoped _s("flush->feed_frame"); flush_ready(); }

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
