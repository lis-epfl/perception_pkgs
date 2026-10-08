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

#ifndef OV_CORE_CAM_BASE_H
#define OV_CORE_CAM_BASE_H

#include <Eigen/Eigen>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

namespace ov_core {

/**
 * @brief Base pinhole camera model class
 *
 * This is the base class for all our camera models.
 * All these models are pinhole cameras, thus just have standard reprojection logic.
 * See each derived class for detailed examples of each model.
 */
class CamBase {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamBase(int width, int height) : _width(width), _height(height) {}

  virtual ~CamBase() {}

  /**
   * @brief This will set and update the camera calibration values.
   * This should be called on startup for each camera and after update!
   * @param calib Camera calibration information (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
   */
  virtual void set_value(const Eigen::MatrixXd &calib) {

    // Assert we are of size eight
    assert(calib.rows() == 8);
    camera_values = calib;

    // Camera matrix
    cv::Matx33d tempK;
    tempK(0, 0) = calib(0);
    tempK(0, 1) = 0;
    tempK(0, 2) = calib(2);
    tempK(1, 0) = 0;
    tempK(1, 1) = calib(1);
    tempK(1, 2) = calib(3);
    tempK(2, 0) = 0;
    tempK(2, 1) = 0;
    tempK(2, 2) = 1;
    camera_k_OPENCV = tempK;

    // Distortion parameters
    cv::Vec4d tempD;
    tempD(0) = calib(4);
    tempD(1) = calib(5);
    tempD(2) = calib(6);
    tempD(3) = calib(7);
    camera_d_OPENCV = tempD;
  }

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  virtual Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) = 0;

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  Eigen::Vector2d undistort_d(const Eigen::Vector2d &uv_dist) {
    Eigen::Vector2f ept1, ept2;
    ept1 = uv_dist.cast<float>();
    ept2 = undistort_f(ept1);
    return ept2.cast<double>();
  }

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  cv::Point2f undistort_cv(const cv::Point2f &uv_dist) {
    Eigen::Vector2f ept1, ept2;
    ept1 << uv_dist.x, uv_dist.y;
    ept2 = undistort_f(ept1);
    cv::Point2f pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Batched undistortion of many raw uv points at once.
   *
   * The DEFAULT implementation is exactly today's per-point loop, so any model that does not
   * override this is byte-for-byte unchanged. Models whose per-point path pays a large
   * fixed per-CALL cost (an OpenCV entry, heap-allocated 1-point cv::Mat headers) override it
   * to amortise that cost over N points. Overrides MUST be bit-exact per point: the OpenCV
   * undistortion kernels process each point independently, so batching may change only how
   * many times the per-call preamble is paid.
   *
   * @param in  Raw uv coordinates
   * @param out Normalized coordinates (resized to in.size())
   */
  virtual void undistort_cv_batch(const std::vector<cv::Point2f> &in, std::vector<cv::Point2f> &out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++)
      out[i] = undistort_cv(in[i]);
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  virtual Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) = 0;

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  Eigen::Vector2d distort_d(const Eigen::Vector2d &uv_norm) {
    Eigen::Vector2f ept1, ept2;
    ept1 = uv_norm.cast<float>();
    ept2 = distort_f(ept1);
    return ept2.cast<double>();
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  cv::Point2f distort_cv(const cv::Point2f &uv_norm) {
    Eigen::Vector2f ept1, ept2;
    ept1 << uv_norm.x, uv_norm.y;
    ept2 = distort_f(ept1);
    cv::Point2f pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Computes the derivative of raw distorted to normalized coordinate.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
   */
  virtual void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) = 0;

  /// Gets the complete intrinsic vector
  Eigen::MatrixXd get_value() { return camera_values; }

  /**
   * @brief Sets the FIXED non-radial terms (tangential p1, p2 and skew s) of the camera model.
   *
   * They are constants: never estimated, never part of the 8-vector of intrinsics. Only the
   * fisheye model (CamEqui) uses them, the other models ignore them. All three zero (the
   * default) keeps every function on its unmodified code path.
   * @param q (p1, p2, s)
   */
  void set_nonradial(const Eigen::Vector3d &q) {
    camera_nonradial = q;
    has_nonradial = nonradial_estimated || (q(0) != 0.0 || q(1) != 0.0 || q(2) != 0.0);
  }

  /**
   * @brief Declares the non-radial terms CALIBRATION STATES of the filter (OV_PRIOR_NONRAD_SIG > 0).
   * The model functions then take the non-radial code path even while all three terms are zero
   * (their Jacobian is needed there). Never called when the terms are fixed.
   */
  void set_nonradial_estimated(bool on) {
    nonradial_estimated = on;
    has_nonradial = on || (camera_nonradial(0) != 0.0 || camera_nonradial(1) != 0.0 || camera_nonradial(2) != 0.0);
  }

  /// True if the non-radial terms are calibration states
  bool get_nonradial_estimated() const { return nonradial_estimated; }

  /**
   * @brief OV_NONRAD_SKEW_FIXED set to anything but empty or 0: while the non-radial terms are calibration states,
   * the skew (third term) is held at its start value. Done by zeroing its column of the measurement Jacobian
   * (UpdaterHelper), so that its Kalman gain is exactly zero; the state vector keeps its layout. Unset = unchanged.
   */
  static bool nonradial_skew_fixed_from_env() {
    static const bool fixed = [] {
      const char *e = std::getenv("OV_NONRAD_SKEW_FIXED");
      return !(e == nullptr || *e == '\0' || std::string(e) == "0");
    }();
    return fixed;
  }

  /**
   * @brief Prior standard deviation of the non-radial terms when they are calibration states.
   * OV_PRIOR_NONRAD_SIG unset, empty, zero or negative = 0 = the terms are fixed constants.
   */
  static double nonradial_prior_sigma_from_env() {
    static const double sig = [] {
      const char *e = std::getenv("OV_PRIOR_NONRAD_SIG");
      if (e == nullptr || *e == '\0')
        return 0.0;
      char *end = nullptr;
      double v = std::strtod(e, &end);
      if (end == e || !std::isfinite(v)) {
        std::fprintf(stderr, "[nonradial]: OV_PRIOR_NONRAD_SIG ('%s') is not a number -- aborting\n", e);
        std::exit(EXIT_FAILURE);
      }
      return v > 0.0 ? v : 0.0;
    }();
    return sig;
  }

  /**
   * @brief Jacobian of the distorted pixel with respect to the non-radial terms (p1, p2, s).
   * Zero for the models that have no such terms.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dq 2x3 derivative of the raw pixel in respect to (p1, p2, s)
   */
  virtual void compute_nonradial_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dq) {
    (void)uv_norm;
    H_dz_dq = Eigen::MatrixXd::Zero(2, 3);
  }

  /// Gets the fixed non-radial terms (p1, p2, s)
  Eigen::Vector3d get_nonradial() const { return camera_nonradial; }

  /**
   * @brief Non-radial terms of one camera from the environment.
   *
   * OV_NONRADIAL="p1,p2,s;p1,p2,s;p1,p2,s;p1,p2,s" lists cameras 0, 1, 2, ... in order. Unset or
   * empty = all zero; a camera that is not listed (or an empty entry) = zero. The variable is
   * parsed once per process. A malformed entry stops the process: a silently ignored typo would
   * fly the plain model while the operator believes otherwise.
   * @param cam_id Camera index
   * @return (p1, p2, s)
   */
  static Eigen::Vector3d nonradial_from_env(size_t cam_id) {
    static const std::vector<Eigen::Vector3d> table = [] {
      std::vector<Eigen::Vector3d> t;
      const char *e = std::getenv("OV_NONRADIAL");
      if (e == nullptr)
        return t;
      std::string all(e);
      size_t a = 0;
      while (a <= all.size()) {
        size_t b = all.find(';', a);
        if (b == std::string::npos)
          b = all.size();
        std::string ent = all.substr(a, b - a);
        Eigen::Vector3d q = Eigen::Vector3d::Zero();
        if (ent.find_first_not_of(" \t\r\n") != std::string::npos) {
          const char *p = ent.c_str();
          for (int j = 0; j < 3; j++) {
            char *end = nullptr;
            q(j) = std::strtod(p, &end);
            bool ok = (end != p);
            while (ok && (*end == ' ' || *end == '\t'))
              end++;
            ok = ok && (j < 2 ? *end == ',' : *end == '\0') && std::isfinite(q(j));
            if (!ok) {
              std::fprintf(stderr, "[nonradial]: OV_NONRADIAL entry %zu ('%s') is not 'p1,p2,s' -- aborting\n", t.size(), ent.c_str());
              std::exit(EXIT_FAILURE);
            }
            p = end + 1;
          }
        }
        t.push_back(q);
        a = b + 1;
      }
      return t;
    }();
    return cam_id < table.size() ? table[cam_id] : Eigen::Vector3d(Eigen::Vector3d::Zero());
  }

  /// Gets the camera matrix
  cv::Matx33d get_K() { return camera_k_OPENCV; }

  /// Gets the camera distortion
  cv::Vec4d get_D() { return camera_d_OPENCV; }

  /// Gets the width of the camera images
  int w() { return _width; }

  /// Gets the height of the camera images
  int h() { return _height; }

protected:
  // Cannot construct the base camera class, needs a distortion model
  CamBase() = default;

  /// Raw set of camera intrinic values (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
  Eigen::MatrixXd camera_values;

  /// Camera intrinsics in OpenCV format
  cv::Matx33d camera_k_OPENCV;

  /// Camera distortion in OpenCV format
  cv::Vec4d camera_d_OPENCV;

  /// Fixed non-radial terms (p1, p2, s), see set_nonradial()
  Eigen::Vector3d camera_nonradial = Eigen::Vector3d::Zero();

  /// True if any non-radial term is non-zero (then, and only then, the model functions leave their original path)
  bool has_nonradial = false;

  /// True if the non-radial terms are calibration states of the filter (see set_nonradial_estimated())
  bool nonradial_estimated = false;

  /// Width of the camera (raw pixels)
  int _width;

  /// Height of the camera (raw pixels)
  int _height;
};

} // namespace ov_core

#endif /* OV_CORE_CAM_BASE_H */