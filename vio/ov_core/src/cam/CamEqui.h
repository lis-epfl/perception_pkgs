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

#ifndef OV_CORE_CAM_EQUI_H
#define OV_CORE_CAM_EQUI_H

#include "CamBase.h"

namespace ov_core {

/**
 * @brief Fisheye / equadistant model pinhole camera model class
 *
 * As fisheye or wide-angle lenses are widely used in practice, we here provide mathematical derivations
 * of such distortion model as in [OpenCV fisheye](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#details).
 *
 * \f{align*}{
 * \begin{bmatrix} u \\ v \end{bmatrix}:= \mathbf{z}_k &= \mathbf h_d(\mathbf{z}_{n,k}, ~\boldsymbol\zeta)
 * = \begin{bmatrix}  f_x * x + c_x \\
 * f_y * y + c_y \end{bmatrix}\\[1em]
 * \empty
 * {\rm where}~~
 * x &= \frac{x_n}{r} * \theta_d \\
 * y &= \frac{y_n}{r} * \theta_d \\
 * \theta_d &= \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8) \\
 * \quad r^2 &= x_n^2 + y_n^2 \\
 * \theta &= atan(r)
 * \f}
 *
 * where \f$ \mathbf{z}_{n,k} = [ x_n ~ y_n ]^\top\f$ are the normalized coordinates of the 3D feature
 * and u and v are the distorted image coordinates on the image plane.
 * Clearly, the following distortion intrinsic parameters are used in the above model:
 *
 * \f{align*}{
 * \boldsymbol\zeta = \begin{bmatrix} f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4 \end{bmatrix}^\top
 * \f}
 *
 * In analogy to the previous radial distortion (see @ref ov_core::CamRadtan) case, the following Jacobian for these
 * parameters is needed for intrinsic calibration:
 * \f{align*}{
 * \frac{\partial \mathbf h_d (\cdot)}{\partial \boldsymbol\zeta} =
 * \begin{bmatrix}
 * x_n & 0  & 1 & 0 & f_x*(\frac{x_n}{r}\theta^3) & f_x*(\frac{x_n}{r}\theta^5) & f_x*(\frac{x_n}{r}\theta^7) & f_x*(\frac{x_n}{r}\theta^9)
 * \\[5pt] 0  & y_n & 0 & 1 & f_y*(\frac{y_n}{r}\theta^3) & f_y*(\frac{y_n}{r}\theta^5) & f_y*(\frac{y_n}{r}\theta^7) &
 * f_y*(\frac{y_n}{r}\theta^9) \end{bmatrix} \f}
 *
 * Similarly, with the chain rule of differentiation,
 * we can compute the following Jacobian with respect to the normalized coordinates:
 *
 * \f{align*}{
 * \frac{\partial \mathbf h_d(\cdot)}{\partial \mathbf{z}_{n,k}}
 * &=
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial x_ny_n}+
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial r}\frac{\partial r}{\partial x_ny_n}+
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial \theta_d}\frac{\partial \theta_d}{\partial \theta}\frac{\partial
 * \theta}{\partial r}\frac{\partial r}{\partial x_ny_n} \\[1em] \empty
 * {\rm where}~~~~
 * \frac{\partial uv}{\partial xy} &= \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial x_ny_n} &= \begin{bmatrix} \theta_d/r & 0 \\ 0 & \theta_d/r \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial r} &= \begin{bmatrix} -\frac{x_n}{r^2}\theta_d \\ -\frac{y_n}{r^2}\theta_d \end{bmatrix} \\
 * \empty
 * \frac{\partial r}{\partial x_ny_n} &= \begin{bmatrix} \frac{x_n}{r} & \frac{y_n}{r} \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial \theta_d} &= \begin{bmatrix} \frac{x_n}{r} \\ \frac{y_n}{r} \end{bmatrix} \\
 * \empty
 * \frac{\partial \theta_d}{\partial \theta} &= \begin{bmatrix} 1 + 3k_1 \theta^2 + 5k_2 \theta^4 + 7k_3 \theta^6 + 9k_4
 * \theta^8\end{bmatrix} \\ \empty \frac{\partial \theta}{\partial r} &= \begin{bmatrix} \frac{1}{r^2+1} \end{bmatrix} \f}
 *
 * To equate this to one of Kalibr's models, this is what you would use for `pinhole-equi`.
 *
 * FIXED NON-RADIAL TERMS (optional, see CamBase::set_nonradial). With tangential terms p1, p2 and a
 * skew s the measured normalised coordinates (x, y) = ((u - cx)/fx, (v - cy)/fy) differ from the
 * ideal Kannala-Brandt ones (x_i, y_i) (the x, y of the equations above) by
 *
 *   x = x_i + N_x(x, y),   N_x = 2 p1 x y + p2 (r^2 + 2 x^2) + s y
 *   y = y_i + N_y(x, y),   N_y = p1 (r^2 + 2 y^2) + 2 p2 x y,        r^2 = x^2 + y^2
 *
 * Un-projection: subtract N (explicit, it is a function of the measured point) and hand the ideal
 * pixel to the same OpenCV call. Projection: Kannala-Brandt, then solve x = x_i + N(x) by fixed-point
 * iteration. Jacobians: d(x, y)/d(x_i, y_i) = (I - dN/d(x, y))^-1 multiplies every derivative that
 * goes through (x_i, y_i); the focal-length columns use the measured (x, y). The terms are constants,
 * they are not columns of the intrinsics Jacobian. When all three are zero the four functions below
 * run their original bodies.
 */
class CamEqui : public CamBase {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamEqui(int width, int height) : CamBase(width, height) {}

  ~CamEqui() {}

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override {

    // Determine what camera parameters we should use
    cv::Matx33d camK = camera_k_OPENCV;
    cv::Vec4d camD = camera_d_OPENCV;

    // Fixed non-radial terms: remove them from the raw pixel (in double), then the same OpenCV call
    if (has_nonradial) {
      cv::Mat mat64(1, 1, CV_64FC2);
      mat64.at<cv::Vec2d>(0, 0) = remove_nonradial(camK, (double)uv_dist(0), (double)uv_dist(1));
      cv::fisheye::undistortPoints(mat64, mat64, camK, camD);
      Eigen::Vector2f pt_nr;
      pt_nr(0) = (float)mat64.at<cv::Vec2d>(0, 0)[0];
      pt_nr(1) = (float)mat64.at<cv::Vec2d>(0, 0)[1];
      return pt_nr;
    }

    // Convert point to opencv format
    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = uv_dist(0);
    mat.at<float>(0, 1) = uv_dist(1);
    mat = mat.reshape(2); // Nx1, 2-channel

    // Undistort it!
    cv::fisheye::undistortPoints(mat, mat, camK, camD);

    // Construct our return vector
    Eigen::Vector2f pt_out;
    mat = mat.reshape(1); // Nx2, 1-channel
    pt_out(0) = mat.at<float>(0, 0);
    pt_out(1) = mat.at<float>(0, 1);
    return pt_out;
  }

  /**
   * @brief BIT-EXACT batched form of undistort_cv.
   *
   * cv::fisheye::undistortPoints (calib3d/src/fisheye.cpp) does ALL of its setup -- the
   * CV_Asserts, the f/c extraction from K, the R/P handling, the TermCriteria decode -- BEFORE
   * `for (size_t i = 0; i < n; i++)`, and the loop body reads only srcf[i] and writes only
   * dstf[i] with no cross-point state. Batching therefore changes only how many times that
   * per-CALL preamble is paid; every point's arithmetic (fp64 Newton, default
   * TermCriteria(MAX_ITER|EPS, 10, 1e-8), fp32 store) is identical.
   *
   * The per-point path this replaces allocated a cv::Mat(1,2,CV_32F) -- two fastMalloc/free
   * pairs plus a UMatData -- reshaped it twice and entered cv::fisheye::undistortPoints, PER
   * POINT. Measured 0.838 us/call = ~1660 A78AE cycles for a ~10-iteration Newton.
   *
   * Exactness is not argued, it is PROVEN in-binary: OV_UNDIST_BATCH_VERIFY=1 recomputes every
   * point through undistort_cv in the same process on the same input and memcmps the floats.
   */
  void undistort_cv_batch(const std::vector<cv::Point2f> &in, std::vector<cv::Point2f> &out) override {
    out.resize(in.size());
    if (in.empty())
      return;
    // Fixed non-radial terms: the same per-point arithmetic as undistort_f (so batch == per-point, bit for bit)
    if (has_nonradial) {
      const cv::Matx33d camK = camera_k_OPENCV;
      const cv::Vec4d camD = camera_d_OPENCV;
      cv::Mat mat64((int)in.size(), 1, CV_64FC2);
      for (size_t i = 0; i < in.size(); i++)
        mat64.at<cv::Vec2d>((int)i, 0) = remove_nonradial(camK, (double)in[i].x, (double)in[i].y);
      cv::fisheye::undistortPoints(mat64, mat64, camK, camD);
      for (size_t i = 0; i < in.size(); i++) {
        out[i].x = (float)mat64.at<cv::Vec2d>((int)i, 0)[0];
        out[i].y = (float)mat64.at<cv::Vec2d>((int)i, 0)[1];
      }
      return;
    }
    // Zero-copy Nx1 CV_32FC2 views over the two vectors (cv::Point2f is two contiguous floats).
    cv::Mat src((int)in.size(), 1, CV_32FC2, (void *)in.data());
    cv::Mat dst((int)out.size(), 1, CV_32FC2, (void *)out.data());
    cv::fisheye::undistortPoints(src, dst, camera_k_OPENCV, camera_d_OPENCV);
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override {

    // Fixed non-radial terms present: separate function, so that the body below stays the original one
    if (has_nonradial)
      return distort_f_nonradial(uv_norm);

    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;

    // Calculate distorted coordinates for fisheye
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                     cam_d(7) * std::pow(theta, 9);

    // Handle when r is small (meaning our xy is near the camera center)
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

    // Calculate distorted coordinates for fisheye
    Eigen::Vector2f uv_dist;
    double x1 = uv_norm(0) * cdist;
    double y1 = uv_norm(1) * cdist;
    uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
    uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
    return uv_dist;
  }

  /**
   * @brief Computes the derivative of raw distorted to normalized coordinate.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
   */
  void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override {

    // Fixed non-radial terms present: separate function, so that the body below stays the original one
    if (has_nonradial) {
      compute_distort_jacobian_nonradial(uv_norm, H_dz_dzn, H_dz_dzeta);
      return;
    }

    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;

    // Calculate distorted coordinates for fisheye
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                     cam_d(7) * std::pow(theta, 9);

    // Handle when r is small (meaning our xy is near the camera center)
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

    // Jacobian of distorted pixel to "normalized" pixel
    Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
    duv_dxy << cam_d(0), 0, 0, cam_d(1);

    // Jacobian of "normalized" pixel to normalized pixel
    Eigen::Matrix<double, 2, 2> dxy_dxyn = Eigen::Matrix<double, 2, 2>::Zero();
    dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;

    // Jacobian of "normalized" pixel to r
    Eigen::Matrix<double, 2, 1> dxy_dr = Eigen::Matrix<double, 2, 1>::Zero();
    dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;

    // Jacobian of r pixel to normalized xy
    Eigen::Matrix<double, 1, 2> dr_dxyn = Eigen::Matrix<double, 1, 2>::Zero();
    dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

    // Jacobian of "normalized" pixel to theta_d
    Eigen::Matrix<double, 2, 1> dxy_dthd = Eigen::Matrix<double, 2, 1>::Zero();
    dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

    // Jacobian of theta_d to theta
    double dthd_dth = 1 + 3 * cam_d(4) * std::pow(theta, 2) + 5 * cam_d(5) * std::pow(theta, 4) + 7 * cam_d(6) * std::pow(theta, 6) +
                      9 * cam_d(7) * std::pow(theta, 8);

    // Jacobian of theta to r
    double dth_dr = 1 / (r * r + 1);

    // Total Jacobian wrt normalized pixel coordinates
    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

    // Calculate distorted coordinates for fisheye
    double x1 = uv_norm(0) * cdist;
    double y1 = uv_norm(1) * cdist;

    // Compute the Jacobian in respect to the intrinsics
    H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
    H_dz_dzeta(0, 0) = x1;
    H_dz_dzeta(0, 2) = 1;
    H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 3);
    H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 5);
    H_dz_dzeta(0, 6) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 7);
    H_dz_dzeta(0, 7) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 9);
    H_dz_dzeta(1, 1) = y1;
    H_dz_dzeta(1, 3) = 1;
    H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 3);
    H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 5);
    H_dz_dzeta(1, 6) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 7);
    H_dz_dzeta(1, 7) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 9);
  }

  /**
   * @brief Jacobian of the raw pixel with respect to the non-radial terms q = (p1, p2, s).
   *
   * measured m = (x, y) solves m = m_i + N(m; q), N linear in q, so
   *   dm/dq = (I - dN/dm)^-1 * dN/dq,   dN/dq = [ 2xy, r^2 + 2x^2, y ; r^2 + 2y^2, 2xy, 0 ]  at the measured (x, y)
   * and the pixel is (fx x + cx, fy y + cy).
   */
  void compute_nonradial_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dq) override {

    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;
    const double p1 = camera_nonradial(0), p2 = camera_nonradial(1), sk = camera_nonradial(2);

    // Kannala-Brandt part, as in compute_distort_jacobian
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                     cam_d(7) * std::pow(theta, 9);
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

    // Ideal and measured normalised coordinates
    double xi = uv_norm(0) * cdist;
    double yi = uv_norm(1) * cdist;
    double x1, y1;
    add_nonradial(xi, yi, x1, y1);
    const double r2 = x1 * x1 + y1 * y1;

    Eigen::Matrix<double, 2, 2> dN;
    dN << 2.0 * p1 * y1 + 6.0 * p2 * x1, 2.0 * p1 * x1 + 2.0 * p2 * y1 + sk, 2.0 * p1 * x1 + 2.0 * p2 * y1, 6.0 * p1 * y1 + 2.0 * p2 * x1;
    Eigen::Matrix<double, 2, 2> dm_dmi = (Eigen::Matrix<double, 2, 2>::Identity() - dN).inverse();
    Eigen::Matrix<double, 2, 3> dN_dq;
    dN_dq << 2.0 * x1 * y1, r2 + 2.0 * x1 * x1, y1, r2 + 2.0 * y1 * y1, 2.0 * x1 * y1, 0.0;
    Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
    duv_dxy << cam_d(0), 0, 0, cam_d(1);
    H_dz_dq = Eigen::MatrixXd::Zero(2, 3);
    H_dz_dq = duv_dxy * dm_dmi * dN_dq;
  }

protected:
  /**
   * @brief Raw pixel -> the pixel an ideal Kannala-Brandt camera with the same 8 intrinsics would have measured.
   * (x, y) = ((u - cx)/fx, (v - cy)/fy) are the measured normalised coordinates; N(x, y) is subtracted.
   */
  cv::Vec2d remove_nonradial(const cv::Matx33d &camK, double u, double v) const {
    const double p1 = camera_nonradial(0), p2 = camera_nonradial(1), sk = camera_nonradial(2);
    const double fx = camK(0, 0), fy = camK(1, 1), cx = camK(0, 2), cy = camK(1, 2);
    const double x = (u - cx) / fx, y = (v - cy) / fy;
    const double r2 = x * x + y * y;
    const double nx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + sk * y;
    const double ny = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    return cv::Vec2d(fx * (x - nx) + cx, fy * (y - ny) + cy);
  }

  /**
   * @brief Ideal normalised coordinates -> measured ones: solves (x, y) = (xi, yi) + N(x, y).
   * Fixed-point iteration from (xi, yi); N is of order 1e-3, so every iteration gains about two digits.
   * At least 4 iterations, stops when the step is below 1e-15, never more than 20.
   */
  void add_nonradial(double xi, double yi, double &x, double &y) const {
    const double p1 = camera_nonradial(0), p2 = camera_nonradial(1), sk = camera_nonradial(2);
    x = xi;
    y = yi;
    for (int it = 0; it < 20; it++) {
      const double r2 = x * x + y * y;
      const double xn = xi + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + sk * y);
      const double yn = yi + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);
      const double step = std::abs(xn - x) + std::abs(yn - y);
      x = xn;
      y = yn;
      if (it >= 3 && step < 1e-15)
        break;
    }
  }

  /// distort_f with the fixed non-radial terms
  Eigen::Vector2f distort_f_nonradial(const Eigen::Vector2f &uv_norm) const {

    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;

    // Kannala-Brandt part, as in distort_f
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                     cam_d(7) * std::pow(theta, 9);
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;
    double xi = uv_norm(0) * cdist;
    double yi = uv_norm(1) * cdist;

    // Non-radial part: measured = ideal + N(measured)
    double x1, y1;
    add_nonradial(xi, yi, x1, y1);

    Eigen::Vector2f uv_dist;
    uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
    uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
    return uv_dist;
  }

  /// compute_distort_jacobian with the fixed non-radial terms (exact chain rule through the fixed point)
  void compute_distort_jacobian_nonradial(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) const {

    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;
    const double p1 = camera_nonradial(0), p2 = camera_nonradial(1), sk = camera_nonradial(2);

    // Kannala-Brandt part, as in compute_distort_jacobian
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double theta = std::atan(r);
    double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                     cam_d(7) * std::pow(theta, 9);
    double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

    // Ideal and measured normalised coordinates
    double xi = uv_norm(0) * cdist;
    double yi = uv_norm(1) * cdist;
    double x1, y1;
    add_nonradial(xi, yi, x1, y1);

    // (x1, y1) = (xi, yi) + N(x1, y1)  =>  d(x1, y1)/d(xi, yi) = (I - dN/d(x1, y1))^-1
    Eigen::Matrix<double, 2, 2> dN;
    dN << 2.0 * p1 * y1 + 6.0 * p2 * x1, 2.0 * p1 * x1 + 2.0 * p2 * y1 + sk, 2.0 * p1 * x1 + 2.0 * p2 * y1, 6.0 * p1 * y1 + 2.0 * p2 * x1;
    Eigen::Matrix<double, 2, 2> dm_dmi = (Eigen::Matrix<double, 2, 2>::Identity() - dN).inverse();

    // Jacobian of distorted pixel to measured normalised coordinates
    Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
    duv_dxy << cam_d(0), 0, 0, cam_d(1);

    // Jacobians of the ideal normalised coordinates, as in compute_distort_jacobian
    Eigen::Matrix<double, 2, 2> dxy_dxyn = Eigen::Matrix<double, 2, 2>::Zero();
    dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;
    Eigen::Matrix<double, 2, 1> dxy_dr = Eigen::Matrix<double, 2, 1>::Zero();
    dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;
    Eigen::Matrix<double, 1, 2> dr_dxyn = Eigen::Matrix<double, 1, 2>::Zero();
    dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;
    Eigen::Matrix<double, 2, 1> dxy_dthd = Eigen::Matrix<double, 2, 1>::Zero();
    dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;
    double dthd_dth = 1 + 3 * cam_d(4) * std::pow(theta, 2) + 5 * cam_d(5) * std::pow(theta, 4) + 7 * cam_d(6) * std::pow(theta, 6) +
                      9 * cam_d(7) * std::pow(theta, 8);
    double dth_dr = 1 / (r * r + 1);

    // Total Jacobian wrt normalized pixel coordinates
    Eigen::Matrix<double, 2, 2> duv_dmi = duv_dxy * dm_dmi;
    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    H_dz_dzn = duv_dmi * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

    // Jacobian in respect to the 8 intrinsics: focal lengths see the measured coordinates,
    // k1..k4 act on the ideal ones and go through d(measured)/d(ideal)
    H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
    H_dz_dzeta(0, 0) = x1;
    H_dz_dzeta(0, 2) = 1;
    H_dz_dzeta(1, 1) = y1;
    H_dz_dzeta(1, 3) = 1;
    for (int j = 0; j < 4; j++) {
      Eigen::Matrix<double, 2, 1> dmi_dk;
      dmi_dk << uv_norm(0) * inv_r * std::pow(theta, 3 + 2 * j), uv_norm(1) * inv_r * std::pow(theta, 3 + 2 * j);
      H_dz_dzeta.block(0, 4 + j, 2, 1) = duv_dmi * dmi_dk;
    }
  }
};

} // namespace ov_core

#endif /* OV_CORE_CAM_EQUI_H */