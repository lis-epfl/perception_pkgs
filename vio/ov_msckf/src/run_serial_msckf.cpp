/*
 * Deterministic OFFLINE OpenVINS runner.
 *
 * Reads a rosbag2 (sqlite3) directly and feeds IMU + camera measurements into
 * VioManager *synchronously, in timestamp order, in the main thread*. There is
 * no rclcpp executor, no real-time bag playback, and no async update thread.
 * Given a fixed OV_RNG_SEED this produces bit-identical results across runs,
 * eliminating the message-timing / frame-dropping variance of the live
 * subscribe path (`ros2 bag play | run_subscribe_msckf`).
 *
 * Usage:
 *   ros2 run ov_msckf run_serial_msckf <config.yaml> <bag_dir> <out_tum> \
 *        --ros-args -p use_stereo:=true -p max_cameras:=4
 *
 * Output: TUM trajectory file (t x y z qx qy qz qw), one pose per camera frame,
 * matching exactly what /ov_msckf/odomimu would publish (fast_state_propagate to
 * the frame timestamp, gated by initialized() && (t - init_time) >= 1).
 */

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
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h" // pulls in ov_core::YamlParser + Printer
#include "utils/dataset_reader.h"   // pulls in ov_core::CameraData / ImuData
#include "state/State.h"
#include "state/Propagator.h"

using namespace ov_msckf;

int main(int argc, char **argv) {

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
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path;
  storage_options.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";
  rosbag2_cpp::Reader reader;
  reader.open(storage_options, converter_options);

  rclcpp::Serialization<sensor_msgs::msg::Imu> imu_ser;
  rclcpp::Serialization<sensor_msgs::msg::Image> img_ser;

  // A fully-assembled camera frame: all images at one shared stamp.
  struct Frame {
    double ts;
    std::map<int, cv::Mat> imgs; // cam_id -> mono8 image
  };
  std::map<double, Frame> pending; // keyed by stamp, sorted ascending
  double latest_imu_t = -1;

  std::vector<std::string> lines;
  lines.reserve(6000);
  double last_logged_ts = -1; // dedupe: with OV_UPDATE_MIN_DT decimation, skipped
                              // frames leave state->_timestamp unchanged

  // Harvest the online-calibration state at the end of the run (the tool reads this).
  auto write_calib = [&](const std::string &path) {
    auto st = sys->get_state();
    FILE *cf = std::fopen(path.c_str(), "w");
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
    std::fclose(cf);
  };

  auto cam_id_from_topic = [](const std::string &t) -> int {
    // "/cam3/image_raw" -> 3
    auto p = t.find("cam");
    if (p == std::string::npos) return -1;
    return std::atoi(t.c_str() + p + 3);
  };

  // track_frequency throttle (replicates ROS2Visualizer::callback_stereo lines
  // 554-558): per lead-camera, skip a frame whose timestamp is < last_tracked +
  // 1/track_frequency. With 30Hz cams and track_frequency=14.5 this tracks every
  // ~3rd frame -> ~100ms parallax baseline (the dominant accuracy lever per the
  // tuning history). track_frequency<=0 disables throttling (track every frame).
  std::map<int, double> cam_last_track;
  const double track_dt = (params.track_frequency > 1e-6) ? 1.0 / params.track_frequency : 0.0;

  // Feed one assembled frame's measurements, then log the pose once.
  auto feed_frame = [&](const Frame &fr) {
    int rows = fr.imgs.begin()->second.rows;
    int cols = fr.imgs.begin()->second.cols;
    bool fed_any = false;
    auto throttled = [&](int lead) {
      auto it = cam_last_track.find(lead);
      if (it != cam_last_track.end() && fr.ts < it->second + track_dt) return true;
      cam_last_track[lead] = fr.ts;
      return false;
    };
    if (use_stereo && ncam % 2 == 0) {
      for (int p = 0; p < ncam / 2; p++) {
        int l = 2 * p, r = 2 * p + 1;
        if (!fr.imgs.count(l) || !fr.imgs.count(r)) continue;
        if (throttled(l)) continue;
        ov_core::CameraData msg;
        msg.timestamp = fr.ts;
        msg.sensor_ids = {l, r};
        msg.images = {fr.imgs.at(l), fr.imgs.at(r)};
        msg.masks = {get_mask(l, rows, cols), get_mask(r, rows, cols)};
        sys->feed_measurement_camera(msg);
        fed_any = true;
      }
    } else {
      for (auto const &kv : fr.imgs) {
        int cid = kv.first;
        if (throttled(cid)) continue;
        ov_core::CameraData msg;
        msg.timestamp = fr.ts;
        msg.sensor_ids = {cid};
        msg.images = {kv.second};
        msg.masks = {get_mask(cid, rows, cols)};
        sys->feed_measurement_camera(msg);
        fed_any = true;
      }
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

  size_t n_imu = 0, n_img = 0, n_frames = 0;
  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    const std::string &topic = bag_msg->topic_name;
    rclcpp::SerializedMessage extracted(*bag_msg->serialized_data);

    if (topic == "/imu0") {
      sensor_msgs::msg::Imu m;
      imu_ser.deserialize_message(&extracted, &m);
      ov_core::ImuData imu;
      imu.timestamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      imu.wm << m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z;
      imu.am << m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z;
      sys->feed_measurement_imu(imu);
      latest_imu_t = imu.timestamp;
      n_imu++;
      flush_ready();
    } else if (topic.rfind("/cam", 0) == 0) {
      int cid = cam_id_from_topic(topic);
      if (cid < 0 || cid >= ncam) continue; // ignore cams beyond max_cameras
      sensor_msgs::msg::Image m;
      img_ser.deserialize_message(&extracted, &m);
      double ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      cv::Mat img;
      if (m.encoding == "mono8") {
        cv::Mat tmp(m.height, m.width, CV_8UC1, m.data.data(), m.step);
        img = tmp.clone();
      } else {
        // robust fallback for other encodings
        cv::Mat tmp(m.height, m.width, CV_8UC1, m.data.data(), m.step);
        img = tmp.clone();
      }
      pending[ts].ts = ts;
      pending[ts].imgs[cid] = img;
      n_img++;
      if ((int)pending[ts].imgs.size() == ncam) {
        n_frames++;
      }
    }
  }
  // EOF: flush whatever remains (IMU stream ended).
  latest_imu_t = std::numeric_limits<double>::infinity();
  flush_ready();

  // Write TUM
  FILE *f = std::fopen(out_path.c_str(), "w");
  if (!f) { PRINT_ERROR(RED "[serial]: cannot open out %s\n" RESET, out_path.c_str()); return EXIT_FAILURE; }
  for (auto const &l : lines) std::fprintf(f, "%s\n", l.c_str());
  std::fclose(f);

  // Dump OV's CONVERGED calibration: per-cam intrinsics + body_P_cam extrinsic (R_CtoI, p_CinI)
  // + cam-imu timeoffset + biases. (Same writer used for the mid-run snapshots above.)
  write_calib(out_path + ".calib.json");

  PRINT_INFO(GREEN "[serial]: done. imu=%zu img=%zu frames=%zu poses=%zu -> %s\n" RESET,
             n_imu, n_img, n_frames, lines.size(), out_path.c_str());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
