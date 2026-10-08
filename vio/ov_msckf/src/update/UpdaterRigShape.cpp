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

#include "UpdaterRigShape.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "update/PreJac.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

// Root mean square and largest absolute value of the rows, in millimetres
double rms_mm(const Eigen::VectorXd &h) { return 1000.0 * std::sqrt(h.squaredNorm() / (double)h.rows()); }
double max_mm(const Eigen::VectorXd &h) { return 1000.0 * h.cwiseAbs().maxCoeff(); }

// Lengths in metres as a comma-separated list in millimetres ("none" if there are none)
std::string list_mm(const std::vector<double> &v, const char *format) {
  std::string s;
  char buf[64];
  for (double x : v) {
    std::snprintf(buf, sizeof(buf), format, 1000.0 * x);
    s += (s.empty() ? "" : ",") + std::string(buf);
  }
  return s.empty() ? "none" : s;
}

} // namespace

std::shared_ptr<UpdaterRigShape> UpdaterRigShape::from_env() {

  const char *env_dist = std::getenv("OV_RIG_DIST"), *env_planar = std::getenv("OV_RIG_PLANAR");
  const std::string str_dist = env_dist ? env_dist : "", str_planar = env_planar ? env_planar : "";
  if (str_dist.empty() && str_planar.empty()) {
    if (std::getenv("OV_RIG_RADIAL_DEG"))
      std::fprintf(stderr, "[rig-warn] OV_RIG_RADIAL_DEG needs the cameras of OV_RIG_PLANAR or OV_RIG_DIST: the rig shape is not used\n");
    if (std::getenv("OV_RIG_TILT_SIG_DEG"))
      std::fprintf(stderr, "[rig-warn] OV_RIG_TILT_SIG_DEG needs the cameras of OV_RIG_PLANAR or OV_RIG_DIST: the rig shape is not used\n");
    return nullptr;
  }

  // "i-j:millimetres,..." and "a,b,c,d[,...]" with camera ids; the positions in _cams come once all ids are known
  std::vector<Pair> pairs;
  std::vector<size_t> planar;
  std::set<size_t> ids;
  std::string token;
  for (std::stringstream ss(str_dist); std::getline(ss, token, ',');) {
    int i, j;
    double mm;
    char rest;
    if (std::sscanf(token.c_str(), "%d-%d:%lf%c", &i, &j, &mm, &rest) != 3 || i < 0 || j < 0 || i == j || !(mm > 0) || !std::isfinite(mm)) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_DIST: '%s' is not 'i-j:millimetres' with two different cameras and a finite length above 0: the rig "
                           "shape is not used\n",
                   token.c_str());
      return nullptr;
    }
    pairs.push_back({(size_t)i, (size_t)j, 1e-3 * mm});
    ids.insert((size_t)i);
    ids.insert((size_t)j);
  }
  for (std::stringstream ss(str_planar); std::getline(ss, token, ',');) {
    int a;
    char rest;
    if (std::sscanf(token.c_str(), "%d%c", &a, &rest) != 1 || a < 0 || std::find(planar.begin(), planar.end(), (size_t)a) != planar.end()) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_PLANAR: '%s' is not a camera id, or is listed twice: the rig shape is not used\n", token.c_str());
      return nullptr;
    }
    planar.push_back((size_t)a);
    ids.insert((size_t)a);
  }
  if (!planar.empty() && planar.size() < 4) {
    std::fprintf(stderr, "[rig-warn] OV_RIG_PLANAR needs four or more cameras (three are always in one plane): the rig shape is not used\n");
    return nullptr;
  }

  auto rig = std::shared_ptr<UpdaterRigShape>(new UpdaterRigShape());
  rig->_cams.assign(ids.begin(), ids.end());
  auto position = [&rig](size_t id) { return (size_t)(std::find(rig->_cams.begin(), rig->_cams.end(), id) - rig->_cams.begin()); };
  for (const Pair &p : pairs)
    rig->_pairs.push_back({position(p.i), position(p.j), p.dist});
  for (size_t id : planar)
    rig->_planar.push_back(position(id));

  if (const char *e = std::getenv("OV_RIG_SIG_MM")) {
    if (atof(e) > 0)
      rig->_sigma = 1e-3 * atof(e);
    else
      std::fprintf(stderr, "[rig-warn] OV_RIG_SIG_MM='%s' is not a positive number: %.3f mm is used\n", e, 1000.0 * rig->_sigma);
  }
  if (const char *e = std::getenv("OV_RIG_EVERY"))
    rig->_every = std::max(0, atoi(e));
  const char *e_snap = std::getenv("OV_RIG_SNAP"), *e_stats = std::getenv("OV_RIG_STATS"), *e_test = std::getenv("OV_RIG_SELFTEST");
  rig->_do_snap = !(e_snap && *e_snap == '0');
  rig->_stats = e_stats && *e_stats == '1';
  rig->_selftest = e_test && *e_test == '1';

  // OV_RIG_REAPPLY: unset or 0 (default) = the rows once, with the first application, afterwards only the mean-only snap of the shape
  // (update_again()); 1 = the rows again at every later application. Any other value is not guessed at: one warning, then as 0.
  if (const char *e_reapply = std::getenv("OV_RIG_REAPPLY")) {
    const std::string value = e_reapply;
    if (value == "1")
      rig->_reapply = true;
    else if (value != "0")
      std::fprintf(stderr, "[rig-warn] OV_RIG_REAPPLY='%s' is neither 0 nor 1: 0 is used (the rows once, then the mean-only snap of the shape)\n",
                   e_reapply);
  }
  // OV_RIG_AFTER_ZUPT is read where it acts (VioManager.cpp, same rule: exactly 1 = on); here it is only checked and shown.
  const char *e_zupt = std::getenv("OV_RIG_AFTER_ZUPT");
  const std::string v_zupt = e_zupt ? e_zupt : "";
  if (!v_zupt.empty() && v_zupt != "0" && v_zupt != "1")
    std::fprintf(stderr, "[rig-warn] OV_RIG_AFTER_ZUPT='%s' is neither 0 nor 1: 0 is used (no call after a zero-velocity update)\n", e_zupt);
  char reapply[48] = "";
  std::snprintf(reapply, sizeof(reapply), " reapply=%d after_zupt=%d", (int)rig->_reapply, (int)(v_zupt == "1"));

  // The cameras of the turn row and of the tilt rows (and of the two angles in the lines of the default mode): those of the plane;
  // without one, all cameras of the pairs
  rig->_radial_cams = rig->_planar;
  for (size_t k = 0; rig->_planar.empty() && k < rig->_cams.size(); k++)
    rig->_radial_cams.push_back(k);

  // OV_RIG_RADIAL_DEG: the cameras of the plane (without one: all cameras of the pairs) look radially outward on average
  const char *e_radial = std::getenv("OV_RIG_RADIAL_DEG");
  char radial[128] = "";
  if (e_radial && *e_radial) {
    char *end = nullptr;
    const double deg = std::strtod(e_radial, &end);
    if (end == e_radial || *end != '\0' || !std::isfinite(deg)) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_RADIAL_DEG='%s' is not a number: the viewing directions are not used\n", e_radial);
    } else if ((rig->_planar.empty() ? rig->_cams.size() : rig->_planar.size()) < 3) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_RADIAL_DEG needs three or more cameras: the viewing directions are not used\n");
    } else {
      rig->_radial = true;
      rig->_radial_target = deg * M_PI / 180.0;
      // The noise of the row: a number (0 or negative = no row, the snap only). Anything else is reported and the default stays.
      if (const char *e = std::getenv("OV_RIG_RADIAL_SIG_DEG")) {
        char *end_sig = nullptr;
        const double sig = std::strtod(e, &end_sig);
        if (end_sig == e || *end_sig != '\0' || !std::isfinite(sig))
          std::fprintf(stderr, "[rig-warn] OV_RIG_RADIAL_SIG_DEG='%s' is not a number: %.4f deg is used\n", e, rig->_radial_sigma * 180.0 / M_PI);
        else
          rig->_radial_sigma = sig * M_PI / 180.0;
      }
      std::snprintf(radial, sizeof(radial), " radial_deg=%.4f radial_sig_deg=%.4f radial_cameras=%zu radial_row=%d", deg,
                    rig->_radial_sigma * 180.0 / M_PI, rig->_radial_cams.size(), (int)(rig->_radial_sigma > 0));
    }
  }

  // OV_RIG_TILT_SIG_DEG: the plane of the same cameras' centres contains their optical axes on average
  const char *e_tilt = std::getenv("OV_RIG_TILT_SIG_DEG");
  char tilted[128] = "";
  if (e_tilt && *e_tilt) {
    char *end = nullptr;
    const double deg = std::strtod(e_tilt, &end);
    if (end == e_tilt || *end != '\0' || !std::isfinite(deg) || deg < 0) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_TILT_SIG_DEG='%s' is not a number from 0 up: the plane of the optical axes is not used\n", e_tilt);
    } else if ((rig->_planar.empty() ? rig->_cams.size() : rig->_planar.size()) < 3) {
      std::fprintf(stderr, "[rig-warn] OV_RIG_TILT_SIG_DEG needs three or more cameras: the plane of the optical axes is not used\n");
    } else {
      rig->_tilt = true;
      rig->_tilt_sigma = deg * M_PI / 180.0;
      std::snprintf(tilted, sizeof(tilted), " tilt_sig_deg=%.4f tilt_cameras=%zu tilt_rows=%d", deg, rig->_radial_cams.size(), deg > 0 ? 2 : 0);
    }
  }

  std::fprintf(stderr, "[rig-cfg] pairs=%zu plane_rows=%zu cameras=%zu sig_mm=%.3f every=%d snap=%d%s%s%s\n", rig->_pairs.size(),
               rig->_planar.empty() ? (size_t)0 : rig->_planar.size() - 3, rig->_cams.size(), 1000.0 * rig->_sigma, rig->_every,
               (int)rig->_do_snap, radial, tilted, reapply);
  return rig;
}

UpdaterRigShape::~UpdaterRigShape() {
  if (!_stats)
    return;

  // Where the centres are now: the distances of the listed pairs and the plane rows
  std::vector<double> sides, plane;
  if (_state != nullptr) {
    const std::vector<Eigen::Vector3d> c = centres(_state);
    Eigen::VectorXd h;
    for (const Pair &p : _pairs)
      sides.push_back((c[p.i] - c[p.j]).norm());
    if (measure(c, h, nullptr)) {
      for (int r = (int)_pairs.size(); r < h.rows(); r++)
        plane.push_back(h(r));
    }
  }

  // OV_RIG_RADIAL_DEG / OV_RIG_TILT_SIG_DEG: how the rig is turned against the viewing directions now, and how its plane is tilted
  // against the plane of the optical axes. In the default mode (OV_RIG_REAPPLY=0) both are printed whether their switches are set or not.
  const bool listed = !_reapply && _radial_cams.size() >= 3;
  char turned[64] = "", tilted[64] = "", reapply[64] = "";
  double angle = 0;
  Tilt t;
  if ((_radial || listed) && _state != nullptr && turn(centres(_state), axes(_state), angle))
    std::snprintf(turned, sizeof(turned), " turn_deg=%.6f", angle * 180.0 / M_PI);
  if ((_tilt || listed) && _state != nullptr && tilt(centres(_state), axes(_state), t))
    std::snprintf(tilted, sizeof(tilted), " tilt_deg=%.6f", t.angle * 180.0 / M_PI);
  std::snprintf(reapply, sizeof(reapply), " reapply=%d snaps=%ld after_zupt=%ld", (int)_reapply, _snaps, _after_zupt);

  // rms_mm_before_mean and max_mm_before are over the EKF applications and, in the default mode, over the snaps that followed.
  // after_zupt = how many of the later snaps followed a zero-velocity update, not a feature update (default mode with
  // OV_RIG_AFTER_ZUPT=1 only; with OV_RIG_REAPPLY=1 nothing follows a zero-velocity update).
  const long uses = _applications + _snaps;
  std::fprintf(stderr,
               "[rig-total] applications=%ld snap_moved_mm=%s rms_mm_before_mean=%.6f sides_mm=%s plane_mm=%s max_mm_before=%.6f%s%s%s\n",
               _applications, list_mm(_snap_moved, "%.3f").c_str(), uses > 0 ? _sum_rms_before_mm / (double)uses : 0.0,
               list_mm(sides, "%.4f").c_str(), list_mm(plane, "%.4f").c_str(), _max_before_mm, turned, tilted, reapply);
}

bool UpdaterRigShape::update_first(std::shared_ptr<State> state) {

  if (_first_done)
    return false;
  _first_done = true;

  // The rows act on the camera poses, which have to be calibration states of the filter
  if (!state->_options.do_calib_camera_pose) {
    std::fprintf(stderr, "[rig-warn] the camera poses are not calibration states (calib_cam_extrinsics): the rig shape is not applied\n");
    return false;
  }
  for (size_t id : _cams) {
    if (state->_calib_IMUtoCAM.find(id) == state->_calib_IMUtoCAM.end()) {
      std::fprintf(stderr, "[rig-warn] camera %zu of OV_RIG_DIST / OV_RIG_PLANAR does not exist (%d cameras): the rig shape is not applied\n", id,
                   state->_options.num_cameras);
      return false;
    }
  }

  // A given length far from the distance its pair starts with is a wrong unit (metres typed for millimetres) or a wrong pair, not a
  // correction of the start values: the snap would pull the centres together or apart by that much with nothing to stop it. Limit:
  // 50 % of the given length (the start values of the fleet are within a quarter of the given lengths; a wrong unit is a factor of ten
  // or more). Every such pair is named.
  {
    const std::vector<Eigen::Vector3d> c = centres(state);
    bool plausible = true;
    for (const Pair &p : _pairs) {
      const double start = (c[p.i] - c[p.j]).norm();
      if (!(std::abs(start - p.dist) <= 0.50 * p.dist)) {
        std::fprintf(stderr, "[rig-warn] OV_RIG_DIST: pair %zu-%zu is given as %.6g mm but starts at %.6g mm, more than 50 %% of the given length "
                             "away (the unit is millimetres): the rig shape is not used in this run\n",
                     _cams[p.i], _cams[p.j], 1000.0 * p.dist, 1000.0 * start);
        plausible = false;
      }
    }
    if (!plausible)
      return false;
  }
  _active = true;
  _state = state;
  _snap_moved.assign(_cams.size(), 0.0);

  // The self-test comes first, on the values the filter starts with (it adds a displaced test point of its own)
  if (_selftest)
    selftest(state);

  // A snap that was asked for and does not settle has left the centres at their start values (its own warning says why). The EKF
  // update is then not made either: with rows that are not finite it would put NaN into every state.
  bool snapped = false;
  if (_do_snap) {
    snapped = snap(state);
    if (!snapped) {
      std::fprintf(stderr, "[rig-warn] the snap of the first application failed: no EKF update is made, the rig shape is not used in this run\n");
      _active = false;
      return false;
    }
  }
  return apply(state) || snapped;
}

bool UpdaterRigShape::update_again(std::shared_ptr<State> state, bool after_zupt) {

  // Default mode (OV_RIG_REAPPLY=0): the rows were applied once, with the first application; their noise set how sure the filter is of
  // the shape there and is no tolerance of the result. From then on there is no EKF update: the centres are only put back onto the
  // shape (mean only; the turn and the tilt are left to the pictures), so the result has exactly the given shape and size. That adds
  // no information, so it is done after every call that gets here: the sub-updates of one clone tick and the zero-velocity updates
  // included.
  if (!_reapply) {
    if (!_active || _every <= 0 || ++_calls % _every != 0)
      return false;
    if (!snap(state, true)) {
      std::fprintf(stderr, "[rig-warn] ts=%.9f: OV_RIG_REAPPLY=0: the snap failed, the rig shape is not held any more\n", state->_timestamp);
      _active = false;
      return false;
    }
    _after_zupt += after_zupt ? 1 : 0;
    return true;
  }
  // OV_RIG_REAPPLY=1: the rows are applied again at feature updates only. At standstill a zero-velocity update comes with every
  // picture, and applying the rows after each would count the same shape hundreds of times before takeoff.
  if (!_active || _every <= 0 || after_zupt || state->_timestamp == _last_timestamp)
    return false;
  _last_timestamp = state->_timestamp;
  if (++_new_timestamps % _every != 0)
    return false;
  return apply(state);
}

std::vector<Eigen::Vector3d> UpdaterRigShape::centres(std::shared_ptr<State> state) const {
  std::vector<Eigen::Vector3d> c;
  for (size_t id : _cams) {
    const std::shared_ptr<PoseJPL> &calib = state->_calib_IMUtoCAM.at(id);
    c.push_back(-calib->Rot().transpose() * calib->pos());
  }
  return c;
}

bool UpdaterRigShape::measure(const std::vector<Eigen::Vector3d> &c, Eigen::VectorXd &h, Eigen::MatrixXd *G) const {

  const int rows = (int)_pairs.size() + (_planar.empty() ? 0 : (int)_planar.size() - 3);
  h = Eigen::VectorXd::Zero(rows);
  if (G != nullptr)
    *G = Eigen::MatrixXd::Zero(rows, 3 * c.size());
  int r = 0;

  // Distance rows: h = |c_i - c_j| - L; dh/dc_i = u^T and dh/dc_j = -u^T with u the unit vector from j to i
  for (const Pair &p : _pairs) {
    const Eigen::Vector3d d = c[p.i] - c[p.j];
    const double len = d.norm();
    if (!(len > 1e-6))
      return false;
    h(r) = len - p.dist;
    if (G != nullptr) {
      G->block(r, 3 * p.i, 1, 3) = d.transpose() / len;
      G->block(r, 3 * p.j, 1, 3) = -d.transpose() / len;
    }
    r++;
  }

  // Plane rows: with e1 = c_b - c_a, e2 = c_c - c_a, n = e1 x e2 and w = c_d - c_a the row is h = n . w / |n|.
  // dh = n^ . dw + (w - h n^) . dn / |n| with n^ = n / |n| and dn = de1 x e2 + e1 x de2; (w - h n^) is the part of w in the plane.
  for (size_t k = 3; k < _planar.size(); k++) {
    const size_t a = _planar[0], b = _planar[1], cc = _planar[2], d = _planar[k];
    const Eigen::Vector3d e1 = c[b] - c[a], e2 = c[cc] - c[a], w = c[d] - c[a];
    const Eigen::Vector3d n = e1.cross(e2);
    const double len = n.norm();
    if (!(len > 1e-9))
      return false;
    const Eigen::Vector3d n_unit = n / len;
    h(r) = n_unit.dot(w);
    if (G != nullptr) {
      const Eigen::Vector3d w_in = w - h(r) * n_unit;
      const Eigen::Vector3d g_b = e2.cross(w_in) / len, g_c = w_in.cross(e1) / len;
      G->block(r, 3 * b, 1, 3) = g_b.transpose();
      G->block(r, 3 * cc, 1, 3) = g_c.transpose();
      G->block(r, 3 * d, 1, 3) = n_unit.transpose();
      G->block(r, 3 * a, 1, 3) = -(g_b + g_c + n_unit).transpose();
    }
    r++;
  }
  return true;
}

bool UpdaterRigShape::linearize(std::shared_ptr<State> state, Eigen::VectorXd &h, Eigen::MatrixXd &H) const {

  Eigen::MatrixXd G;
  if (!measure(centres(state), h, &G))
    return false;

  // PoseJPL::update() does R <- (I - [dtheta]x) R and p <- p + dp, so that the centre c = -R^T p moves by
  // R^T [p]x dtheta - R^T dp (first order). The self-test checks exactly this against the real update().
  H = Eigen::MatrixXd::Zero(h.rows(), 6 * _cams.size());
  for (size_t k = 0; k < _cams.size(); k++) {
    const std::shared_ptr<PoseJPL> &calib = state->_calib_IMUtoCAM.at(_cams[k]);
    const Eigen::Matrix3d R_CtoI = calib->Rot().transpose();
    H.block(0, 6 * k, h.rows(), 3) = G.block(0, 3 * k, h.rows(), 3) * R_CtoI * skew_x(calib->pos());
    H.block(0, 6 * k + 3, h.rows(), 3) = -G.block(0, 3 * k, h.rows(), 3) * R_CtoI;
  }
  return true;
}

std::vector<Eigen::Vector3d> UpdaterRigShape::axes(std::shared_ptr<State> state) const {
  std::vector<Eigen::Vector3d> a;
  for (size_t id : _cams)
    a.push_back(state->_calib_IMUtoCAM.at(id)->Rot().transpose().col(2));
  return a;
}

bool UpdaterRigShape::turn(const std::vector<Eigen::Vector3d> &c, const std::vector<Eigen::Vector3d> &a, double &angle, Eigen::Vector3d *normal,
                           Eigen::Vector3d *middle) const {

  // Middle of the listed centres and unit normal of the plane through the first three of them
  Eigen::Vector3d m = Eigen::Vector3d::Zero();
  for (size_t k : _radial_cams)
    m += c[k] / (double)_radial_cams.size();
  const Eigen::Vector3d n = (c[_radial_cams[1]] - c[_radial_cams[0]]).cross(c[_radial_cams[2]] - c[_radial_cams[0]]);
  const double len = n.norm();
  if (!(len > 1e-9))
    return false;
  const Eigen::Vector3d n_unit = n / len;

  // Per camera the angle from the radial direction to the optical axis, counter-clockwise about n, both without their part along n
  angle = 0;
  for (size_t k : _radial_cams) {
    const Eigen::Vector3d u = c[k] - m;
    const Eigen::Vector3d u_in = u - n_unit.dot(u) * n_unit, a_in = a[k] - n_unit.dot(a[k]) * n_unit;
    if (!(u_in.norm() > 1e-6) || !(a_in.norm() > 1e-6))
      return false;
    angle += std::atan2(n_unit.dot(u_in.cross(a_in)), u_in.dot(a_in)) / (double)_radial_cams.size();
  }
  if (normal != nullptr)
    *normal = n_unit;
  if (middle != nullptr)
    *middle = m;
  return true;
}

bool UpdaterRigShape::linearize_turn(std::shared_ptr<State> state, double &h, Eigen::MatrixXd &H) const {

  const double eps = 1e-6;
  const std::vector<Eigen::Vector3d> c_before = centres(state), a_before = axes(state);
  H = Eigen::MatrixXd::Zero(1, 6 * _cams.size());
  if (!turn(c_before, a_before, h))
    return false;
  h -= _radial_target;

  // The mean angle depends on a camera's rotation twice (through the centre c = -R^T p and through the axis) and on its position.
  // Each of the six columns is a central difference through the variable's own update(), on a copy of its value that is put back
  // each time, as in the self-test: no sign or frame convention is assumed. 48 evaluations of a few cross products per application.
  bool defined = true;
  for (size_t k : _radial_cams) {
    const std::shared_ptr<PoseJPL> &calib = state->_calib_IMUtoCAM.at(_cams[k]);
    const Eigen::MatrixXd base = calib->value();
    for (int m = 0; m < 6; m++) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
      dx(m) = eps;
      double h_plus = 0, h_minus = 0;
      calib->update(dx);
      defined = turn(centres(state), axes(state), h_plus) && defined;
      calib->set_value(base);
      calib->update(-dx);
      defined = turn(centres(state), axes(state), h_minus) && defined;
      calib->set_value(base);
      H(0, 6 * k + m) = (h_plus - h_minus) / (2.0 * eps);
    }
  }

  // The poses have to be bit for bit what they were
  return defined && centres(state) == c_before && axes(state) == a_before;
}

bool UpdaterRigShape::tilt(const std::vector<Eigen::Vector3d> &c, const std::vector<Eigen::Vector3d> &a, Tilt &t) const {

  // Middle of the listed centres and unit normal of the plane through the first three of them, as in turn()
  t.middle.setZero();
  for (size_t k : _radial_cams)
    t.middle += c[k] / (double)_radial_cams.size();
  const Eigen::Vector3d n = (c[_radial_cams[1]] - c[_radial_cams[0]]).cross(c[_radial_cams[2]] - c[_radial_cams[0]]);
  const double len = n.norm();
  if (!(len > 1e-9))
    return false;
  t.normal = n / len;

  // Basis of the rig plane: e1 = the radial direction of the first listed camera without its part along the normal, e2 = n x e1
  const Eigen::Vector3d u = c[_radial_cams[0]] - t.middle;
  const Eigen::Vector3d u_in = u - t.normal.dot(u) * t.normal;
  if (!(u_in.norm() > 1e-6))
    return false;
  t.e1 = u_in / u_in.norm();
  t.e2 = t.normal.cross(t.e1);

  // Plane of the optical axes: the unit vector n_axes with the smallest sum of (n_axes . a_i)^2, which is the right singular vector
  // with the smallest singular value of the matrix with one axis per row. The axes have to span a plane (the second singular value
  // well above the third). Its sign is that of the rig normal.
  Eigen::MatrixXd A(_radial_cams.size(), 3);
  for (size_t k = 0; k < _radial_cams.size(); k++)
    A.row(k) = a[_radial_cams[k]].transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd values = svd.singularValues();
  if (!(values(1) - values(2) > 1e-3 * values(0)))
    return false;
  t.normal_axes = svd.matrixV().col(2);
  if (t.normal_axes.dot(t.normal) < 0)
    t.normal_axes = -t.normal_axes;

  t.h1 = t.normal_axes.dot(t.e1);
  t.h2 = t.normal_axes.dot(t.e2);
  t.angle = std::atan2(t.normal.cross(t.normal_axes).norm(), t.normal.dot(t.normal_axes));
  return true;
}

bool UpdaterRigShape::linearize_tilt(std::shared_ptr<State> state, Eigen::VectorXd &h, Eigen::MatrixXd &H) const {

  const double eps = 1e-6;
  const std::vector<Eigen::Vector3d> c_before = centres(state), a_before = axes(state);
  Tilt t, t_plus, t_minus;
  h = Eigen::VectorXd::Zero(2);
  H = Eigen::MatrixXd::Zero(2, 6 * _cams.size());
  if (!tilt(c_before, a_before, t))
    return false;
  h << t.h1, t.h2;

  // As in linearize_turn(): the rows depend on a camera's rotation through its centre (rig normal, basis) and through its axis (plane
  // of the axes), and on its position. Central differences through the variable's own update(), its value put back each time.
  bool defined = true;
  for (size_t k : _radial_cams) {
    const std::shared_ptr<PoseJPL> &calib = state->_calib_IMUtoCAM.at(_cams[k]);
    const Eigen::MatrixXd base = calib->value();
    for (int m = 0; m < 6; m++) {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
      dx(m) = eps;
      calib->update(dx);
      defined = tilt(centres(state), axes(state), t_plus) && defined;
      calib->set_value(base);
      calib->update(-dx);
      defined = tilt(centres(state), axes(state), t_minus) && defined;
      calib->set_value(base);
      H(0, 6 * k + m) = (t_plus.h1 - t_minus.h1) / (2.0 * eps);
      H(1, 6 * k + m) = (t_plus.h2 - t_minus.h2) / (2.0 * eps);
    }
  }

  // Bit for bit, as there
  return defined && centres(state) == c_before && axes(state) == a_before;
}

bool UpdaterRigShape::apply(std::shared_ptr<State> state) {

  Eigen::VectorXd h;
  Eigen::MatrixXd H;
  if (!linearize(state, h, H)) {
    std::fprintf(stderr, "[rig-warn] ts=%.9f: a row is not defined (two centres coincide, or the first three cameras of the plane are in line): "
                         "the rig shape is not applied any more\n",
                 state->_timestamp);
    _active = false;
    return false;
  }

  // Turn row (OV_RIG_RADIAL_DEG): h = mean signed angle - target in radians, with noise _radial_sigma. EKFUpdate() below takes ONE
  // noise for all rows (_sigma, metres), so this row and its Jacobian are multiplied by _sigma / _radial_sigma: the same measurement,
  // now with noise _sigma. Without a noise (snap only) the angle is read for the statistics and no row is added.
  Eigen::VectorXd res = -h;
  double turn_before = 0;
  if (_radial) {
    double h_turn = 0;
    Eigen::MatrixXd H_turn;
    if (!(_radial_sigma > 0 ? linearize_turn(state, h_turn, H_turn) : turn(centres(state), axes(state), turn_before))) {
      std::fprintf(stderr, "[rig-warn] ts=%.9f: the mean angle between the radial directions and the optical axes is not defined, or the camera "
                           "poses did not come back unchanged from its difference quotients: OV_RIG_RADIAL_DEG is not applied any more\n",
                   state->_timestamp);
      _radial = false;
    } else if (_radial_sigma > 0) {
      const double scale = _sigma / _radial_sigma;
      turn_before = h_turn + _radial_target;
      H.conservativeResize(h.rows() + 1, Eigen::NoChange);
      H.row(h.rows()) = scale * H_turn;
      res.conservativeResize(h.rows() + 1);
      res(h.rows()) = -scale * h_turn;
    }
  }

  // Tilt rows (OV_RIG_TILT_SIG_DEG): h = (n_axes . e1, n_axes . e2), for a small tilt two angles in radians, with noise _tilt_sigma
  // each. Multiplied by _sigma / _tilt_sigma for the one noise of EKFUpdate(), as the turn row. Without a noise (snap only) the
  // angle between the two normals is read for the statistics and no rows are added.
  double tilt_before = 0;
  if (_tilt) {
    Tilt t;
    Eigen::VectorXd h_tilt;
    Eigen::MatrixXd H_tilt;
    if (!tilt(centres(state), axes(state), t) || (_tilt_sigma > 0 && !linearize_tilt(state, h_tilt, H_tilt))) {
      std::fprintf(stderr, "[rig-warn] ts=%.9f: the plane of the optical axes against the rig plane is not defined, or the camera poses did "
                           "not come back unchanged from its difference quotients: OV_RIG_TILT_SIG_DEG is not applied any more\n",
                   state->_timestamp);
      _tilt = false;
    } else if (_tilt_sigma > 0) {
      const double scale = _sigma / _tilt_sigma;
      const int rows = (int)res.rows();
      H.conservativeResize(rows + 2, Eigen::NoChange);
      H.block(rows, 0, 2, H.cols()) = scale * H_tilt;
      res.conservativeResize(rows + 2);
      res.segment(rows, 2) = -scale * h_tilt;
    }
    tilt_before = t.angle;
  }

  // The drawing says that every row is zero, so the residual is 0 - h
  std::vector<std::shared_ptr<Type>> Hx_order;
  for (size_t id : _cams)
    Hx_order.push_back(state->_calib_IMUtoCAM.at(id));
  StateHelper::EKFUpdate(state, Hx_order, H, res, _sigma * _sigma);

  _applications++;
  _sum_rms_before_mm += rms_mm(h);
  _max_before_mm = std::max(_max_before_mm, max_mm(h));
  if (_stats) {
    Eigen::VectorXd h_after = h;
    measure(centres(state), h_after, nullptr);

    // rows = all rows of the update; the millimetre values are those of the shape rows, the turn row has its own two fields
    char turned[96] = "";
    double turn_after = 0;
    if (_radial && turn(centres(state), axes(state), turn_after))
      std::snprintf(turned, sizeof(turned), " turn_deg=%.6f turn_deg_after=%.6f", turn_before * 180.0 / M_PI, turn_after * 180.0 / M_PI);
    char tilted[96] = "";
    Tilt tilt_after;
    if (_tilt && tilt(centres(state), axes(state), tilt_after))
      std::snprintf(tilted, sizeof(tilted), " tilt_deg=%.6f tilt_deg_after=%.6f", tilt_before * 180.0 / M_PI, tilt_after.angle * 180.0 / M_PI);
    std::fprintf(stderr, "[rig] ts=%.9f rows=%d rms_mm_before=%.6f rms_mm_after=%.6f max_mm_before=%.6f%s%s\n", state->_timestamp,
                 (int)res.rows(), rms_mm(h), rms_mm(h_after), max_mm(h), turned, tilted);
  }
  return true;
}

bool UpdaterRigShape::snap(std::shared_ptr<State> state, bool again) {

  const std::vector<Eigen::Vector3d> c_start = centres(state);
  std::vector<Eigen::Vector3d> c = c_start;
  Eigen::VectorXd h, h_start;
  Eigen::MatrixXd G;
  int iterations = 0;
  double step = INFINITY;
  while (!(step < 1e-9) && iterations < 50 && measure(c, h, &G)) {
    if (iterations == 0)
      h_start = h;

    // The smallest correction of the centre coordinates that zeroes the linearised rows. The rigid motions of the rig change no row,
    // and when there are more rows than the shape has freedoms one combination of rows cannot be reached: both show up as zero
    // singular values and stay out. Rows that contradict each other are then met in the least-squares sense.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(G, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(1e-9);
    const Eigen::VectorXd d = svd.solve(-h);
    for (size_t k = 0; k < c.size(); k++)
      c[k] += d.segment(3 * k, 3);
    step = d.cwiseAbs().maxCoeff();
    iterations++;
  }
  if (!(step < 1e-9) || !measure(c, h, nullptr)) {
    std::fprintf(stderr, "[rig-warn] the snap did not settle within %d iterations (last step %.3e m): the centres stay at their start values\n",
                 iterations, step);
    return false;
  }

  // OV_RIG_TILT_SIG_DEG: nor do they say how the rig plane lies against the viewing directions. All centres are rotated about the
  // middle, about the axis n x n_axes by the angle between the two normals, which brings the rig normal onto n_axes (the axes stay,
  // and so does every row above). This comes before the turn, which is about the new normal and keeps it.
  char tilted[96] = "";
  if (_tilt && !again) {
    const std::vector<Eigen::Vector3d> a = axes(state);
    Tilt t, t_after;
    _snap_tilted = 0;
    if (tilt(c, a, t) && t.normal.cross(t.normal_axes).norm() > 1e-12) {
      const Eigen::Matrix3d R_tilt = Eigen::AngleAxisd(t.angle, t.normal.cross(t.normal_axes).normalized()).toRotationMatrix();
      for (Eigen::Vector3d &centre : c)
        centre = t.middle + R_tilt * (centre - t.middle);
      _snap_tilted = t.angle;
    }
    if (!tilt(c, a, t_after) || !measure(c, h, nullptr)) {
      std::fprintf(stderr, "[rig-warn] the snap could not tilt the rig: the plane of the optical axes against the rig plane is not defined\n");
      t_after.angle = NAN;
    }
    std::snprintf(tilted, sizeof(tilted), " tilted_deg=%.6f tilt_deg_after=%.9f", _snap_tilted * 180.0 / M_PI, t_after.angle * 180.0 / M_PI);
  }

  // OV_RIG_RADIAL_DEG: the rows above do not say how the rig is turned about its normal. Turning all centres by phi about the normal
  // through the middle takes phi off every signed angle (the axes stay, and so does every row above), hence phi = mean angle - target.
  // The second round only removes rounding.
  char turned[96] = "";
  if (_radial && !again) {
    const std::vector<Eigen::Vector3d> a = axes(state);
    Eigen::Vector3d normal, middle;
    double angle = 0;
    bool defined = true;
    _snap_turned = 0;
    for (int round = 0; round < 2 && (defined = turn(c, a, angle, &normal, &middle)); round++) {
      const Eigen::Matrix3d R_turn = Eigen::AngleAxisd(angle - _radial_target, normal).toRotationMatrix();
      for (Eigen::Vector3d &centre : c)
        centre = middle + R_turn * (centre - middle);
      _snap_turned += angle - _radial_target;
    }
    if (!defined || !turn(c, a, angle) || !measure(c, h, nullptr)) {
      std::fprintf(stderr, "[rig-warn] the snap could not turn the rig: the mean angle between the radial directions and the optical axes "
                           "is not defined\n");
      angle = NAN;
    }
    std::snprintf(turned, sizeof(turned), " turned_deg=%.6f turn_deg_after=%.9f", _snap_turned * 180.0 / M_PI, angle * 180.0 / M_PI);
  }

  // Mean only: p_IinC = -R_ItoC c with the rotation as it is; the covariance is not touched. The first-estimate copy follows, as it
  // would after a start from a chain that had these centres.
  for (size_t k = 0; k < _cams.size(); k++) {
    const std::shared_ptr<PoseJPL> &calib = state->_calib_IMUtoCAM.at(_cams[k]);
    const Eigen::Vector3d p_IinC = -calib->Rot() * c[k];
    Eigen::MatrixXd value = calib->value(), fej = calib->fej();
    value.block(4, 0, 3, 1) = p_IinC;
    fej.block(4, 0, 3, 1) = p_IinC;
    calib->set_value(value);
    calib->set_fej(fej);
    if (!again)
      _snap_moved[k] = (c[k] - c_start[k]).norm();
  }

  // A later snap of the default mode (OV_RIG_REAPPLY=0): no EKF update follows that would tell the precomputed systems (OV_PREJAC) that the state has
  // changed, so the state epoch is moved here, as StateHelper does with every change of the state. Its line is "[rig]": the row
  // values the pictures left behind (before) and after the snap, and the two angles whether their switches are set or not.
  if (again) {
    g_state_epoch.fetch_add(1, std::memory_order_relaxed);
    _snaps++;
    _sum_rms_before_mm += rms_mm(h_start);
    _max_before_mm = std::max(_max_before_mm, max_mm(h_start));
    if (_stats) {
      const std::vector<Eigen::Vector3d> a = axes(state);
      char angles[160] = "";
      double turn_before = 0, turn_after = 0;
      Tilt tilt_before, tilt_after;
      if (_radial_cams.size() >= 3 && turn(c_start, a, turn_before) && turn(c, a, turn_after) && tilt(c_start, a, tilt_before) &&
          tilt(c, a, tilt_after))
        std::snprintf(angles, sizeof(angles), " turn_deg=%.6f turn_deg_after=%.6f tilt_deg=%.6f tilt_deg_after=%.6f", turn_before * 180.0 / M_PI,
                      turn_after * 180.0 / M_PI, tilt_before.angle * 180.0 / M_PI, tilt_after.angle * 180.0 / M_PI);
      std::fprintf(stderr, "[rig] ts=%.9f rows=%d rms_mm_before=%.6f rms_mm_after=%.6f max_mm_before=%.6f%s\n", state->_timestamp, (int)h.rows(),
                   rms_mm(h_start), rms_mm(h), max_mm(h_start), angles);
    }
    return true;
  }
  if (_stats)
    std::fprintf(stderr, "[rig-snap] iterations=%d rms_mm_before=%.6f max_mm_before=%.6f rms_mm_after=%.9f max_mm_after=%.9f moved_mm=%s%s%s\n",
                 iterations, rms_mm(h_start), max_mm(h_start), rms_mm(h), max_mm(h), list_mm(_snap_moved, "%.3f").c_str(), turned, tilted);
  return true;
}

void UpdaterRigShape::selftest(std::shared_ptr<State> state) const {

  const double eps = 1e-6;
  const std::vector<Eigen::Vector3d> c_before = centres(state);
  std::vector<std::shared_ptr<PoseJPL>> calib;
  std::vector<Eigen::MatrixXd> saved;
  for (size_t id : _cams) {
    calib.push_back(state->_calib_IMUtoCAM.at(id));
    saved.push_back(calib.back()->value());
  }

  // Two test points: the state as it is, and the state with every camera pose displaced by centimetres and degrees, where no row
  // is near zero whatever the start values are (a wrong term that vanishes with the row values would otherwise pass).
  // worst[point] = largest absolute and relative difference of a rotation column, then of a position column
  double worst[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  double worst_turn[2] = {0, 0}, worst_tilt[2] = {0, 0};
  bool defined = true, turn_tested = true, tilt_tested = true;
  for (int point = 0; point < 2 && defined; point++) {
    if (point == 1) {
      for (size_t k = 0; k < calib.size(); k++) {
        Eigen::VectorXd dx(6);
        for (int m = 0; m < 6; m++)
          dx(m) = (m < 3 ? 0.05 : 0.02) * std::sin(1.0 + 2.3 * (double)k + 1.7 * (double)m);
        calib[k]->update(dx);
      }
    }

    Eigen::VectorXd h, h_plus, h_minus;
    Eigen::MatrixXd H;
    defined = linearize(state, h, H);
    if (!defined)
      break;

    // A column is compared relative to its own size, but to no less than a thousandth of the largest column of its kind
    // (a rotation about the lever arm itself moves no centre: that column is zero and its difference pure rounding)
    double largest[2] = {0, 0};
    for (int col = 0; col < H.cols(); col++)
      largest[col % 6 < 3 ? 0 : 1] = std::max(largest[col % 6 < 3 ? 0 : 1], H.col(col).cwiseAbs().maxCoeff());

    for (size_t k = 0; k < calib.size() && defined; k++) {
      const Eigen::MatrixXd base = calib[k]->value();
      for (int m = 0; m < 6 && defined; m++) {

        // Central difference through the variable's own update(), on a copy of its value that is put back each time
        Eigen::VectorXd dx = Eigen::VectorXd::Zero(6);
        dx(m) = eps;
        calib[k]->update(dx);
        defined = measure(centres(state), h_plus, nullptr);
        calib[k]->set_value(base);
        calib[k]->update(-dx);
        defined = measure(centres(state), h_minus, nullptr) && defined;
        calib[k]->set_value(base);
        if (!defined)
          break;

        const int kind = m < 3 ? 0 : 1;
        const Eigen::VectorXd column = H.col(6 * k + m);
        const double abs_diff = (column - (h_plus - h_minus) / (2.0 * eps)).cwiseAbs().maxCoeff();
        const double rel_diff = abs_diff / std::max(column.cwiseAbs().maxCoeff(), 1e-3 * largest[kind]);
        worst[point][2 * kind] = std::max(worst[point][2 * kind], abs_diff);
        worst[point][2 * kind + 1] = std::max(worst[point][2 * kind + 1], rel_diff);
      }
    }

    // The tilt rows (OV_RIG_TILT_SIG_DEG) are difference quotients, like the turn row below, and are tested in the same way. Moving
    // all cameras as one body changes neither row. Turning the centres alone by a small angle about the middle changes them by the
    // angle times: about e1 (0, + n_axes . n), since e1 stays and e2 = n x e1 tips towards n; about e2 (- n_axes . n, 0), since e1
    // tips away from n; about n (h2, -h1), since the basis turns in the plane.
    // worst_tilt = largest |row . direction| of the six body motions, and the largest difference to these six values, over both points
    if (_tilt && defined) {
      const std::vector<Eigen::Vector3d> c = centres(state);
      Tilt t;
      Eigen::VectorXd h_tilt;
      Eigen::MatrixXd H_tilt;
      tilt_tested = tilt_tested && linearize_tilt(state, h_tilt, H_tilt) && tilt(c, axes(state), t);
      if (tilt_tested) {
        for (int axis = 0; axis < 3; axis++) {
          Eigen::VectorXd dx_rot = Eigen::VectorXd::Zero(H_tilt.cols()), dx_shift = Eigen::VectorXd::Zero(H_tilt.cols());
          for (size_t k = 0; k < calib.size(); k++) {
            dx_rot.segment(6 * k, 3) = calib[k]->Rot().col(axis);
            dx_shift.segment(6 * k + 3, 3) = -calib[k]->Rot().col(axis);
          }
          const Eigen::VectorXd by_rot = H_tilt * dx_rot, by_shift = H_tilt * dx_shift;
          worst_tilt[0] = std::max(worst_tilt[0], std::max(by_rot.cwiseAbs().maxCoeff(), by_shift.cwiseAbs().maxCoeff()));
        }
        const double cosine = t.normal_axes.dot(t.normal);
        const Eigen::Vector3d about[3] = {t.e1, t.e2, t.normal};
        const double expected[3][2] = {{0.0, cosine}, {-cosine, 0.0}, {t.h2, -t.h1}};
        for (int j = 0; j < 3; j++) {
          Eigen::VectorXd dx = Eigen::VectorXd::Zero(H_tilt.cols());
          for (size_t k = 0; k < calib.size(); k++)
            dx.segment(6 * k + 3, 3) = -calib[k]->Rot() * about[j].cross(c[k] - t.middle);
          const Eigen::VectorXd by_turn = H_tilt * dx;
          worst_tilt[1] = std::max(worst_tilt[1], std::max(std::abs(by_turn(0) - expected[j][0]), std::abs(by_turn(1) - expected[j][1])));
        }
      }
    }

    // The turn row (OV_RIG_RADIAL_DEG) is a difference quotient itself, so it is tested against what holds for any rig: moving all
    // cameras as one body does not change the mean angle (three rotations w: dtheta = R w, dp = 0; three shifts v: dp = -R v), and
    // turning the centres alone about the normal, dp = -R (n x (c - middle)), lowers it one for one.
    // worst_turn = largest |row . direction| of the six body motions, and |row . direction + 1| of the turn, over both test points
    if (_radial && defined) {
      const std::vector<Eigen::Vector3d> c = centres(state);
      Eigen::Vector3d normal, middle;
      Eigen::MatrixXd H_turn;
      double h_turn = 0, angle = 0;
      turn_tested = turn_tested && linearize_turn(state, h_turn, H_turn) && turn(c, axes(state), angle, &normal, &middle);
      if (!turn_tested)
        continue;
      Eigen::VectorXd dx_turn = Eigen::VectorXd::Zero(H_turn.cols());
      for (int axis = 0; axis < 3; axis++) {
        Eigen::VectorXd dx_rot = Eigen::VectorXd::Zero(H_turn.cols()), dx_shift = Eigen::VectorXd::Zero(H_turn.cols());
        for (size_t k = 0; k < calib.size(); k++) {
          dx_rot.segment(6 * k, 3) = calib[k]->Rot().col(axis);
          dx_shift.segment(6 * k + 3, 3) = -calib[k]->Rot().col(axis);
        }
        worst_turn[0] = std::max(worst_turn[0], std::max(std::abs(H_turn.row(0).dot(dx_rot)), std::abs(H_turn.row(0).dot(dx_shift))));
      }
      for (size_t k = 0; k < calib.size(); k++)
        dx_turn.segment(6 * k + 3, 3) = -calib[k]->Rot() * normal.cross(c[k] - middle);
      worst_turn[1] = std::max(worst_turn[1], std::abs(H_turn.row(0).dot(dx_turn) + 1.0));
    }
  }

  // Put the poses back and make sure that the centres are bit for bit what they were
  for (size_t k = 0; k < calib.size(); k++)
    calib[k]->set_value(saved[k]);
  const bool unchanged = (centres(state) == c_before);
  if (!defined) {
    std::fprintf(stderr, "[rig-selftest] not done: a row is not defined at a test point; state_unchanged=%d\n", (int)unchanged);
    return;
  }
  const bool ok = unchanged && worst[0][1] < 1e-6 && worst[0][3] < 1e-6 && worst[1][1] < 1e-6 && worst[1][3] < 1e-6 &&
                  (!_radial || (turn_tested && worst_turn[0] < 1e-6 && worst_turn[1] < 1e-6)) &&
                  (!_tilt || (tilt_tested && worst_tilt[0] < 1e-6 && worst_tilt[1] < 1e-6));
  char turned[96] = "", tilted[96] = "";
  if (_tilt)
    std::snprintf(tilted, sizeof(tilted), " tilt_tested=%d tilt_body=%.3e tilt_centres=%.3e", (int)tilt_tested, worst_tilt[0], worst_tilt[1]);
  if (_radial)
    std::snprintf(turned, sizeof(turned), " turn_tested=%d turn_body=%.3e turn_centres=%.3e", (int)turn_tested, worst_turn[0], worst_turn[1]);
  std::fprintf(stderr,
               "[rig-selftest] rows=%d cols=%d eps=%.0e rot_abs=%.3e rot_rel=%.3e pos_abs=%.3e pos_rel=%.3e displaced_rot_abs=%.3e "
               "displaced_rot_rel=%.3e displaced_pos_abs=%.3e displaced_pos_rel=%.3e state_unchanged=%d%s%s ok=%d\n",
               (int)_pairs.size() + (_planar.empty() ? 0 : (int)_planar.size() - 3), 6 * (int)_cams.size(), eps, worst[0][0], worst[0][1],
               worst[0][2], worst[0][3], worst[1][0], worst[1][1], worst[1][2], worst[1][3], (int)unchanged, turned, tilted, (int)ok);
}
