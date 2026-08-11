#ifndef OV_CORE_SWARMNXT_MSGS_H
#define OV_CORE_SWARMNXT_MSGS_H

/*
 * px4_msgs/msg/SensorCombined -> ov_core::ImuData.
 *
 * THIS FILE IS SHARED ON PURPOSE, and it is the reason it sits in ov_core rather
 * than beside the offline bag reader that currently uses it. Both the offline
 * runner and any live subscriber MUST take their IMU timestamp from here.
 *
 * Why: self-calibration estimates `timeshift_cam_imu`, a property of the
 * timestamping pipeline rather than of the optics. If the flight path and the
 * calibration path disagree about which field carries the sensor time, the
 * calibration is fitted for a pipeline that does not exist in flight and nothing
 * downstream flags it. The two plausible mistakes a live-node author would make
 * are node->now() and header.stamp -- and SensorCombined has no header at all.
 * Measured on the fleet, the PX4<->host timesync offset drifts 58 us over 74 s
 * (8 us max step), so t_d really is constant and really does transfer; that only
 * holds if both paths read `timestamp` and scale it identically.
 *
 * `timestamp` is microseconds and is ALREADY converted to the host epoch by
 * uxrce_dds_client's timesync before publication. It is therefore on the same
 * clock as the camera header stamps. Do not add offsets of your own.
 */

#include <cstdint>
#include <cstring>

#include "sensor_data.h"

#ifdef OV_HAVE_PX4_MSGS
#include <px4_msgs/msg/sensor_combined.hpp>
#endif

namespace ov_core {

/// Wire size of px4_msgs/msg/SensorCombined: 4-byte CDR encapsulation + 48-byte body.
static constexpr size_t SENSOR_COMBINED_CDR_SIZE = 52;

/*
 * FRD -> FLU, applied in both conversions below as (x, y, z) -> (x, -y, -z).
 *
 * PX4 publishes the body frame as Forward-Right-Down; ROS/REP-103, the seed
 * extrinsics and the estimator all use Forward-Left-Up. The two differ by a 180 deg
 * rotation about x.
 *
 * Feeding raw FRD does NOT fail loudly. The filter still converges the calibration
 * states self-consistently, so every certificate except ATE passes and the run
 * reports a confident wrong answer: measured on this fleet, the trajectory diverged
 * progressively (3 m at t+18 s, 161 m at t+45 s, 14 km at t+182 s) while every gate
 * stayed green. gates.py's gravity check backstops this.
 */

#ifdef OV_HAVE_PX4_MSGS
/// Preferred path: the generated type, so a px4_msgs change is a compile error.
inline void sensor_combined_to_imu(const px4_msgs::msg::SensorCombined &m, ImuData &out) {
  out.timestamp = static_cast<double>(m.timestamp) * 1e-6;
  out.wm << m.gyro_rad[0], -m.gyro_rad[1], -m.gyro_rad[2];
  out.am << m.accelerometer_m_s2[0], -m.accelerometer_m_s2[1], -m.accelerometer_m_s2[2];
}
#endif

/**
 * @brief Fallback: parse the CDR payload directly, for builds without px4_msgs.
 *
 * ov_msckf must stay buildable for a rig that has nothing to do with PX4 -- the
 * package documents validation on TUM-VI, EuRoC and UZH-FPV -- so px4_msgs is an
 * optional dependency and this is what runs when it is absent.
 *
 * Layout, validated byte-exact against 2000 real fleet messages by cross-checking
 * every field against the schema-decoded values:
 *
 *   [0..4)  CDR encapsulation header, little-endian = 00 01 00 00
 *   body offsets, natural alignment relative to the start of the body:
 *     +0    uint64  timestamp                        (microseconds, host epoch)
 *     +8    float32 gyro_rad[3]                      (rad/s, FRD body frame)
 *     +20   uint32  gyro_integral_dt
 *     +24   int32   accelerometer_timestamp_relative
 *     +28   float32 accelerometer_m_s2[3]            (m/s^2, FRD body frame)
 *     +40   uint32  accelerometer_integral_dt
 *     +44   uint8   accelerometer_clipping, gyro_clipping,
 *                   accel_calibration_count, gyro_calibration_count
 *
 * Returns false rather than plausible garbage if the payload is not that shape.
 */
inline bool sensor_combined_to_imu(const uint8_t *buf, size_t len, ImuData &out) {
  if (buf == nullptr || len < SENSOR_COMBINED_CDR_SIZE)
    return false;
  // Big-endian CDR would need byte swaps and has never been produced by this stack.
  if (!(buf[0] == 0x00 && buf[1] == 0x01))
    return false;
  const uint8_t *b = buf + 4;
  uint64_t stamp_us;
  float g[3], a[3];
  std::memcpy(&stamp_us, b + 0, 8);
  std::memcpy(g, b + 8, 12);
  std::memcpy(a, b + 28, 12);
  out.timestamp = static_cast<double>(stamp_us) * 1e-6;
  // FRD -> FLU, identical to the typed path above. Keep the two in step.
  out.wm << g[0], -g[1], -g[2];
  out.am << a[0], -a[1], -a[2];
  return true;
}

} // namespace ov_core

#endif // OV_CORE_SWARMNXT_MSGS_H
