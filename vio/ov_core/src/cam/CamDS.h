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

#ifndef OV_CORE_CAM_DS_H
#define OV_CORE_CAM_DS_H

#include "CamBase.h"

namespace ov_core {

/**
 * @brief Double-Sphere camera model
 *
 * Implementation of the double-sphere model from
 * Usenko, Demmel, Cremers, "The Double Sphere Camera Model", 3DV 2018
 * (arXiv 1807.08957). DS has six intrinsic parameters per camera:
 *
 * \f{align*}{
 * \boldsymbol\zeta = \begin{bmatrix} f_x & f_y & c_x & c_y & \xi & \alpha \end{bmatrix}^\top
 * \f}
 *
 * To remain compatible with the existing 8-vector `CamBase` storage layout used
 * by `CamRadtan` / `CamEqui`, we pack the parameters as
 * `(fx, fy, cx, cy, xi, alpha, 0, 0)`; entries 6 and 7 are unused and held to
 * zero by `set_value`.
 *
 * Projection (Usenko Eq. 4-7):
 *
 * \f{align*}{
 * d_1 &= \sqrt{x^2 + y^2 + z^2} \\
 * d_2 &= \sqrt{x^2 + y^2 + (\xi d_1 + z)^2} \\
 * w   &= \alpha d_2 + (1-\alpha)(\xi d_1 + z) \\
 * u   &= f_x \frac{x}{w} + c_x \\
 * v   &= f_y \frac{y}{w} + c_y
 * \f}
 *
 * Unprojection (Usenko Eq. 11-13):
 *
 * \f{align*}{
 * m_x &= (u - c_x)/f_x \\
 * m_y &= (v - c_y)/f_y \\
 * r^2 &= m_x^2 + m_y^2 \\
 * m_z &= \frac{1 - \alpha^2 r^2}{\alpha\sqrt{1 - (2\alpha-1)r^2} + 1 - \alpha} \\
 * s   &= \frac{m_z \xi + \sqrt{m_z^2 + (1-\xi^2) r^2}}{m_z^2 + r^2} \\
 * \text{bearing} &= s\begin{bmatrix}m_x\\m_y\\m_z\end{bmatrix} - \begin{bmatrix}0\\0\\\xi\end{bmatrix}
 * \f}
 *
 * To equate this to one of Kalibr's models, this is what you would use for `ds`
 * / `pinhole-ds` (the `pinhole-` prefix is just a hint; the model itself is not
 * a pinhole). On Kalibr forks that expose double-sphere, the intrinsics order
 * (`fx, fy, cx, cy, xi, alpha`) matches Kalibr's convention.
 *
 * Online refinement: by default all 6 DS intrinsics (fx, fy, cx, cy, xi, alpha)
 * are refined when `calib_cam_intrinsics: true`; entries 6 and 7 of the
 * intrinsic vector are dummy zeros and their Jacobian columns are zeroed.
 */
class CamDS : public CamBase {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamDS(int width, int height) : CamBase(width, height) {}

  ~CamDS() {}

  /**
   * @brief Set DS intrinsics from an 8-vector
   *
   * Expected layout: `(fx, fy, cx, cy, xi, alpha, 0, 0)`. We forcibly zero the
   * last two entries so that they cannot accidentally interact with the model.
   * @param calib 8-vector of intrinsic values
   */
  void set_value(const Eigen::MatrixXd &calib) override {
    assert(calib.rows() == 8);
    Eigen::MatrixXd calib_ds = calib;
    // Force unused entries to zero (DS only uses indices 0..5).
    calib_ds(6) = 0.0;
    calib_ds(7) = 0.0;
    CamBase::set_value(calib_ds);
  }

  /**
   * @brief Given a raw uv point, this will undistort it into normalized coords.
   *
   * Implements DS unprojection (Usenko 2018, Sec. III.D), then drops to a
   * normalized perspective representation (x/z, y/z) on the unit-bearing ray so
   * the result can plug into the rest of the pinhole-style OV pipeline.
   *
   * If the input pixel falls outside the DS valid domain (i.e.
   * `1 - (2\alpha-1) r^2 < 0`, or the recovered bearing has `bz <= 0`), we
   * clamp to the input center and return the previous-frame value to avoid
   * NaNs propagating into the filter. The caller is responsible for not
   * tracking points outside the FOV in the first place.
   */
  Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override {
    Eigen::MatrixXd cam_d = camera_values;
    const double fx = cam_d(0), fy = cam_d(1), cx = cam_d(2), cy = cam_d(3);
    const double xi = cam_d(4), alpha = cam_d(5);

    const double mx = ((double)uv_dist(0) - cx) / fx;
    const double my = ((double)uv_dist(1) - cy) / fy;
    const double r2 = mx * mx + my * my;

    // Valid domain check on mz (Usenko Eq. 12 denominator radicand).
    const double inner = 1.0 - (2.0 * alpha - 1.0) * r2;
    if (inner < 0.0) {
      // Outside DS valid domain - return safe perspective normalization.
      Eigen::Vector2f out;
      out(0) = (float)mx;
      out(1) = (float)my;
      return out;
    }
    const double denom = alpha * std::sqrt(inner) + (1.0 - alpha);
    if (std::abs(denom) < 1e-12) {
      Eigen::Vector2f out;
      out(0) = (float)mx;
      out(1) = (float)my;
      return out;
    }
    const double mz = (1.0 - alpha * alpha * r2) / denom;

    // Bearing direction (s factor; Usenko Eq. 13).
    const double sq_inner = mz * mz + (1.0 - xi * xi) * r2;
    if (sq_inner < 0.0) {
      Eigen::Vector2f out;
      out(0) = (float)mx;
      out(1) = (float)my;
      return out;
    }
    const double s_denom = mz * mz + r2;
    if (std::abs(s_denom) < 1e-12) {
      Eigen::Vector2f out;
      out(0) = (float)mx;
      out(1) = (float)my;
      return out;
    }
    const double s = (mz * xi + std::sqrt(sq_inner)) / s_denom;

    // Bearing in camera frame.
    double bx = s * mx;
    double by = s * my;
    double bz = s * mz - xi;

    // Convert to perspective normalized (x/z, y/z) for the OV pipeline.
    if (bz <= 1e-12) {
      Eigen::Vector2f out;
      out(0) = (float)mx;
      out(1) = (float)my;
      return out;
    }
    Eigen::Vector2f out;
    out(0) = (float)(bx / bz);
    out(1) = (float)(by / bz);
    return out;
  }

  /**
   * @brief Given a normalized perspective coordinate `(x/z, y/z)`, distort to
   * raw pixels using the DS projection.
   *
   * Implements Usenko 2018 Eq. 4-7. We treat the input as the bearing of a
   * point with `z = 1`, so `P = (x_n, y_n, 1)`.
   */
  Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override {
    Eigen::MatrixXd cam_d = camera_values;
    const double fx = cam_d(0), fy = cam_d(1), cx = cam_d(2), cy = cam_d(3);
    const double xi = cam_d(4), alpha = cam_d(5);

    const double x = (double)uv_norm(0);
    const double y = (double)uv_norm(1);
    const double z = 1.0;

    const double d1 = std::sqrt(x * x + y * y + z * z);
    const double t = xi * d1 + z;
    const double d2 = std::sqrt(x * x + y * y + t * t);
    const double w = alpha * d2 + (1.0 - alpha) * t;

    // Guard against (near-)singular projection at the pole behind the cam.
    if (std::abs(w) < 1e-12) {
      Eigen::Vector2f out;
      out(0) = (float)cx;
      out(1) = (float)cy;
      return out;
    }

    Eigen::Vector2f uv_dist;
    uv_dist(0) = (float)(fx * x / w + cx);
    uv_dist(1) = (float)(fy * y / w + cy);
    return uv_dist;
  }

  /**
   * @brief Computes the analytical Jacobians of the DS projection.
   *
   * Two outputs:
   *  - `H_dz_dzn`: 2x2, partial of `(u,v)` w.r.t. normalized `(x_n,y_n)`
   *    (treating `z=1`); used by the filter's residual-Jacobian chain rule.
   *  - `H_dz_dzeta`: 2x8, partial of `(u,v)` w.r.t. the 8-vector intrinsics.
   *    Columns 0..5 correspond to (fx, fy, cx, cy, xi, alpha); columns 6 and 7
   *    are zero (unused DS slots).
   *
   * Derivation sketch (matches projection above; let `t = xi*d1 + z`):
   *  - `dd1/dx = x/d1`, `dd1/dy = y/d1`
   *  - `dt/dx  = xi*x/d1`, `dt/dy = xi*y/d1`
   *  - `dd2/dx = (x + t * xi*x/d1) / d2`, etc.
   *  - `dw/dx  = alpha * dd2/dx + (1-alpha) * dt/dx`
   *  - `du/dx  = fx * (w - x * dw/dx) / w^2`, etc.
   *
   * For intrinsic Jacobians we differentiate `u = fx*x/w + cx` directly:
   *  - `du/dfx = x/w`, `du/dcx = 1`
   *  - `du/dxi = -fx*x * dw/dxi / w^2` with `dw/dxi = alpha * (t*d1)/d2 + (1-alpha)*d1`
   *  - `du/dalpha = -fx*x * (d2 - t) / w^2`
   */
  void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override {
    Eigen::MatrixXd cam_d = camera_values;
    const double fx = cam_d(0), fy = cam_d(1);
    const double xi = cam_d(4), alpha = cam_d(5);

    const double x = uv_norm(0);
    const double y = uv_norm(1);
    const double z = 1.0;
    const double xy_norm2 = x * x + y * y;
    const double d1 = std::sqrt(xy_norm2 + z * z);
    const double t = xi * d1 + z;
    const double d2 = std::sqrt(xy_norm2 + t * t);
    const double w = alpha * d2 + (1.0 - alpha) * t;

    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);

    if (std::abs(d1) < 1e-12 || std::abs(d2) < 1e-12 || std::abs(w) < 1e-12) {
      // Degenerate; fall back to a pinhole-only Jacobian.
      H_dz_dzn(0, 0) = fx;
      H_dz_dzn(1, 1) = fy;
      H_dz_dzeta(0, 0) = x;
      H_dz_dzeta(0, 2) = 1.0;
      H_dz_dzeta(1, 1) = y;
      H_dz_dzeta(1, 3) = 1.0;
      return;
    }

    const double inv_d1 = 1.0 / d1;
    const double inv_d2 = 1.0 / d2;
    const double inv_w = 1.0 / w;
    const double inv_w2 = inv_w * inv_w;

    // d/dx_n of d1
    const double dd1_dx = x * inv_d1;
    const double dd1_dy = y * inv_d1;
    // d/dx_n of t = xi*d1 + z (z does not depend on x_n)
    const double dt_dx = xi * dd1_dx;
    const double dt_dy = xi * dd1_dy;
    // d/dx_n of d2 = sqrt(x^2 + y^2 + t^2)
    const double dd2_dx = (x + t * dt_dx) * inv_d2;
    const double dd2_dy = (y + t * dt_dy) * inv_d2;
    // d/dx_n of w
    const double dw_dx = alpha * dd2_dx + (1.0 - alpha) * dt_dx;
    const double dw_dy = alpha * dd2_dy + (1.0 - alpha) * dt_dy;

    // u = fx * x / w + cx  ->  du/dx_n = fx * (delta * w - x * dw/dx_n) / w^2
    // delta_xx = 1, delta_xy = 0
    H_dz_dzn(0, 0) = fx * (w - x * dw_dx) * inv_w2;
    H_dz_dzn(0, 1) = fx * (-x * dw_dy) * inv_w2;
    H_dz_dzn(1, 0) = fy * (-y * dw_dx) * inv_w2;
    H_dz_dzn(1, 1) = fy * (w - y * dw_dy) * inv_w2;

    // Intrinsic Jacobians (columns 0..5 are fx, fy, cx, cy, xi, alpha).
    // Re-evaluate projected normalized values to fill column-0/1 entries.
    const double x_over_w = x * inv_w;
    const double y_over_w = y * inv_w;

    // u, v wrt fx, fy
    H_dz_dzeta(0, 0) = x_over_w;
    H_dz_dzeta(1, 1) = y_over_w;
    // u, v wrt cx, cy
    H_dz_dzeta(0, 2) = 1.0;
    H_dz_dzeta(1, 3) = 1.0;

    // dw/dxi    : t depends on xi (dt/dxi = d1); d2 depends on xi via t.
    // dt/dxi    = d1
    // dd2/dxi   = (t * dt/dxi) / d2 = t*d1/d2
    // dw/dxi    = alpha * t*d1/d2 + (1-alpha) * d1
    const double dt_dxi = d1;
    const double dd2_dxi = (t * dt_dxi) * inv_d2;
    const double dw_dxi = alpha * dd2_dxi + (1.0 - alpha) * dt_dxi;
    H_dz_dzeta(0, 4) = -fx * x * dw_dxi * inv_w2;
    H_dz_dzeta(1, 4) = -fy * y * dw_dxi * inv_w2;

    // dw/dalpha : w = alpha*d2 + (1-alpha)*t  ->  dw/dalpha = d2 - t.
    const double dw_dalpha = d2 - t;
    H_dz_dzeta(0, 5) = -fx * x * dw_dalpha * inv_w2;
    H_dz_dzeta(1, 5) = -fy * y * dw_dalpha * inv_w2;

    // Columns 6 and 7 are zero by construction (DS has no k3/k4 / p1/p2).
  }
};

} // namespace ov_core

#endif /* OV_CORE_CAM_DS_H */
