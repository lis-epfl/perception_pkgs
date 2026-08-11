#ifndef OV_MSCKF_BAG_SOURCE_H
#define OV_MSCKF_BAG_SOURCE_H

/*
 * BagSource -- the one place that knows how a recording is laid out on disk.
 *
 * The offline runner assumes a single stream of measurements in timestamp order.
 * A study recording satisfies that with one sqlite3 bag holding /imu0 and
 * /cam0../camN. A swarm-nxt flight recording does not: IMU and cameras live in
 * two separate mcap files (<log>/bag/ written by ros2 bag, <log>/cams/recording.mcap
 * written by oak_compressed_recorder), each internally ordered but unordered with
 * respect to the other, and on the fleet the camera file starts ~5 s after the
 * IMU file. Concatenating them breaks the runner in both directions: IMU-first
 * drains the filter's IMU before any image arrives, camera-first buffers every
 * decoded frame in RAM.
 *
 * So this merges N readers into one ordered stream, keyed on the SENSOR stamp --
 * never the bag log time, which is a different clock in the camera file.
 *
 * DEFAULTS REPRODUCE THE PREVIOUS BEHAVIOUR EXACTLY: a single sqlite3 URI, IMU on
 * /imu0 as sensor_msgs/Imu, cameras on the /cam<N> prefix as sensor_msgs/Image.
 * The study path is therefore unchanged unless a config opts into something else.
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#ifdef OV_HAVE_PX4_MSGS
#include <px4_msgs/msg/sensor_combined.hpp>
#endif

#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/swarmnxt_msgs.h"

namespace ov_msckf {

/**
 * @brief One camera image still in its on-the-wire form.
 *
 * Decoding is deferred on purpose. The runner's track_frequency throttle discards
 * roughly every other frame (track_frequency 29.0 against 30 Hz cameras), and a
 * discarded frame must never cost a JPEG decode. Call decode() only once the frame
 * is known to be fed to the estimator.
 *
 * This lives here rather than in ov_core because deferring a decode is a
 * bag-reader concern: a live subscriber is handed an already-deserialized message
 * by its callback and has no undecoded payload to hold on to.
 */
struct CamPayload {
  int cam_id = -1;
  double ts = -1.0;
  bool compressed = false;                  //!< true: `data` is a JPEG/PNG blob
  std::vector<uint8_t> data;                //!< encoded blob, or raw mono8 rows
  uint32_t width = 0, height = 0, step = 0; //!< meaningful only when !compressed

  /// -> mono8 cv::Mat, or an empty Mat if the payload cannot be decoded.
  cv::Mat decode() const {
    if (compressed) {
      // IMREAD_GRAYSCALE lets libjpeg emit luma directly -- markedly cheaper than
      // decoding to BGR and converting afterwards.
      return cv::imdecode(data, cv::IMREAD_GRAYSCALE);
    }
    if (width == 0 || height == 0 || data.empty())
      return cv::Mat();
    const size_t stride = (step != 0) ? step : width;
    if (data.size() < stride * height)
      return cv::Mat();
    cv::Mat wrapped(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
                    const_cast<uint8_t *>(data.data()), stride);
    return wrapped.clone();
  }
};

struct BagSourceOptions {
  std::vector<std::string> uris;              //!< one or more bag URIs
  std::string imu_topic = "/imu0";
  std::string imu_msg_type = "sensor_msgs/msg/Imu";
  std::string cam_msg_type = "sensor_msgs/msg/Image";
  std::map<std::string, int> cam_topics;      //!< explicit topic -> cam index; empty = "/cam<N>" rule
};

/// One measurement pulled off the merged stream.
struct BagRecord {
  enum Kind { NONE, IMU, CAM } kind = NONE;
  double ts = -1.0;
  ov_core::ImuData imu;
  CamPayload cam;
};

class BagSource {
public:
  explicit BagSource(BagSourceOptions opt) : _opt(std::move(opt)) {
    std::vector<std::string> topics = wanted_topics();
    for (const auto &uri : _opt.uris) {
      auto st = std::make_unique<Stream>();
      st->uri = resolve_uri(uri);
      rosbag2_storage::StorageOptions so;
      so.uri = st->uri;
      so.storage_id = detect_storage(st->uri);
      rosbag2_cpp::ConverterOptions co;
      co.input_serialization_format = "cdr";
      co.output_serialization_format = "cdr";
      st->reader = std::make_unique<rosbag2_cpp::Reader>();
      st->reader->open(so, co);
      // Filter at the storage layer. A fleet bag carries ~60 topics and 170k
      // messages of which we want one; without this every message is fetched and
      // its payload copied before being discarded on a string compare.
      if (!topics.empty()) {
        rosbag2_storage::StorageFilter filter;
        filter.topics = topics;
        st->reader->set_filter(filter);
      }
      PRINT_INFO("[bag]: opened %s (storage=%s)\n", st->uri.c_str(), so.storage_id.c_str());
      _streams.push_back(std::move(st));
    }
    prime();
  }

  /// -> next record in timestamp order; kind == NONE once every stream is done.
  BagRecord next() {
    int best = -1;
    for (size_t i = 0; i < _streams.size(); i++) {
      if (_streams[i]->head.kind == BagRecord::NONE)
        continue;
      if (best < 0 || _streams[i]->head.ts < _streams[best]->head.ts)
        best = static_cast<int>(i);
    }
    if (best < 0)
      return BagRecord();
    BagRecord out = std::move(_streams[best]->head);
    advance(*_streams[best]);
    return out;
  }

private:
  struct Stream {
    std::string uri;
    std::unique_ptr<rosbag2_cpp::Reader> reader;
    BagRecord head;
  };

  std::vector<std::string> wanted_topics() const {
    std::vector<std::string> t{_opt.imu_topic};
    for (const auto &kv : _opt.cam_topics)
      t.push_back(kv.first);
    // With the legacy "/cam<N>" prefix rule we cannot enumerate topics up front,
    // so no filter is applied and the topic test happens per message as before.
    if (_opt.cam_topics.empty())
      return {};
    return t;
  }

  /// A rosbag2 directory is only openable if it carries metadata.yaml. Some
  /// recordings ship a bare bag_N.mcap in a directory without one; rosbag2 cannot
  /// open the directory but can open the file, so point at the file.
  static std::string resolve_uri(const std::string &uri) {
    struct stat st;
    if (::stat(uri.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
      return uri;
    std::ifstream meta(uri + "/metadata.yaml");
    if (meta.good())
      return uri;
    std::vector<std::string> mcaps;
    if (DIR *d = ::opendir(uri.c_str())) {
      while (struct dirent *e = ::readdir(d)) {
        std::string n(e->d_name);
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".mcap") == 0)
          mcaps.push_back(uri + "/" + n);
      }
      ::closedir(d);
    }
    std::sort(mcaps.begin(), mcaps.end());
    // Several splits with no metadata: their order is unrecoverable, so refuse
    // rather than silently reading a fraction of the recording.
    if (mcaps.size() == 1)
      return mcaps[0];
    if (mcaps.size() > 1)
      PRINT_WARNING(YELLOW "[bag]: %s holds %zu .mcap splits but no metadata.yaml; "
                           "their order is unknown\n" RESET,
                    uri.c_str(), mcaps.size());
    return uri;
  }

  /// mcap vs sqlite3, decided from the recording itself rather than assumed.
  static std::string detect_storage(const std::string &uri) {
    if (uri.size() > 5 && uri.compare(uri.size() - 5, 5, ".mcap") == 0)
      return "mcap";
    std::ifstream meta(uri + "/metadata.yaml");
    if (meta) {
      std::string line;
      while (std::getline(meta, line)) {
        auto p = line.find("storage_identifier:");
        if (p == std::string::npos)
          continue;
        std::string v = line.substr(p + 19);
        // strip whitespace and quotes
        v.erase(0, v.find_first_not_of(" \t\"'"));
        auto e = v.find_last_not_of(" \t\"'\r\n");
        if (e != std::string::npos)
          v.erase(e + 1);
        if (!v.empty())
          return v;
      }
    }
    return "sqlite3";
  }

  int cam_id_for(const std::string &topic) const {
    auto it = _opt.cam_topics.find(topic);
    if (it != _opt.cam_topics.end())
      return it->second;
    if (!_opt.cam_topics.empty())
      return -1;
    // Legacy rule: "/cam3/image_raw" -> 3. Case-sensitive, as before.
    if (topic.rfind("/cam", 0) != 0)
      return -1;
    auto p = topic.find("cam");
    return std::atoi(topic.c_str() + p + 3);
  }

  /// Fill every stream's head so next() has something to compare.
  void prime() {
    for (auto &st : _streams)
      advance(*st);
  }

  void advance(Stream &st) {
    st.head = BagRecord();
    while (st.reader->has_next()) {
      auto msg = st.reader->read_next();
      const std::string &topic = msg->topic_name;

      const bool is_imu = (topic == _opt.imu_topic);
      const int cid = is_imu ? -1 : cam_id_for(topic);
      if (!is_imu && cid < 0)
        continue; // not a topic we consume -- skip BEFORE copying the payload

      BagRecord rec;
      if (is_imu) {
        if (!decode_imu(*msg->serialized_data, rec))
          continue;
      } else if (!decode_cam(*msg->serialized_data, cid, rec)) {
        continue;
      }

      st.head = std::move(rec);
      return;
    }
  }

  bool decode_imu(const rcutils_uint8_array_t &blob, BagRecord &rec) {
    if (_opt.imu_msg_type == "px4_msgs/msg/SensorCombined") {
#ifdef OV_HAVE_PX4_MSGS
      // px4_msgs is present (it always is when built inside ros2_swarmnxt_ws), so
      // use the generated type: a definition change becomes a compile error rather
      // than a silent misread.
      rclcpp::SerializedMessage ser(blob);
      px4_msgs::msg::SensorCombined m;
      _sc_ser.deserialize_message(&ser, &m);
      ov_core::sensor_combined_to_imu(m, rec.imu);
#else
      if (!ov_core::sensor_combined_to_imu(blob.buffer, blob.buffer_length, rec.imu)) {
        if (!_warned_imu) {
          _warned_imu = true;
          PRINT_WARNING(YELLOW "[bag]: SensorCombined payload not parseable (%zu bytes, expected >= %zu) -- "
                               "has the px4_msgs definition changed?\n" RESET,
                        (size_t)blob.buffer_length, ov_core::SENSOR_COMBINED_CDR_SIZE);
        }
        return false;
      }
#endif
    } else {
      rclcpp::SerializedMessage ser(blob);
      sensor_msgs::msg::Imu m;
      _imu_ser.deserialize_message(&ser, &m);
      rec.imu.timestamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      rec.imu.wm << m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z;
      rec.imu.am << m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z;
    }
    rec.kind = BagRecord::IMU;
    rec.ts = rec.imu.timestamp;
    return true;
  }

  bool decode_cam(const rcutils_uint8_array_t &blob, int cid, BagRecord &rec) {
    rclcpp::SerializedMessage ser(blob);
    rec.cam.cam_id = cid;
    if (_opt.cam_msg_type == "sensor_msgs/msg/CompressedImage") {
      sensor_msgs::msg::CompressedImage m;
      _cimg_ser.deserialize_message(&ser, &m);
      rec.cam.ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      rec.cam.compressed = true;
      rec.cam.data = std::move(m.data); // still encoded; decode() runs after the throttle
    } else {
      sensor_msgs::msg::Image m;
      _img_ser.deserialize_message(&ser, &m);
      rec.cam.ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      rec.cam.compressed = false;
      rec.cam.width = m.width;
      rec.cam.height = m.height;
      rec.cam.step = m.step;
      rec.cam.data = std::move(m.data);
    }
    rec.kind = BagRecord::CAM;
    rec.ts = rec.cam.ts;
    return true;
  }

  BagSourceOptions _opt;
  std::vector<std::unique_ptr<Stream>> _streams;
  bool _warned_imu = false;
  rclcpp::Serialization<sensor_msgs::msg::Imu> _imu_ser;
  rclcpp::Serialization<sensor_msgs::msg::Image> _img_ser;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> _cimg_ser;
#ifdef OV_HAVE_PX4_MSGS
  rclcpp::Serialization<px4_msgs::msg::SensorCombined> _sc_ser;
#endif
};

} // namespace ov_msckf

#endif // OV_MSCKF_BAG_SOURCE_H
