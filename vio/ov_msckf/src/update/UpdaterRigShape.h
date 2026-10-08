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

#ifndef OV_MSCKF_UPDATER_RIGSHAPE_H
#define OV_MSCKF_UPDATER_RIGSHAPE_H

#include <Eigen/Eigen>
#include <cmath>
#include <memory>
#include <vector>

namespace ov_msckf {

class State;

/**
 * @brief The known shape of the camera rig as a measurement of the camera-IMU calibration states (default OFF).
 *
 * The optical centre of camera i in the IMU frame is c_i = -R_ItoC_i^T p_IinC_i. In a short flight all scene points are far, and
 * 1 mm of camera position moves a point at 5 m by 0.05 px, so the pictures cannot fix the distances between the cameras. The
 * mechanical drawing can. Two kinds of rows, each with noise OV_RIG_SIG_MM:
 * - a pair i-j at distance L:           h = |c_i - c_j| - L
 * - cameras a, b, c, d in one plane:    h = ((c_b - c_a) x (c_c - c_a)) . (c_d - c_a) / |(c_b - c_a) x (c_c - c_a)|,
 *   the distance of d from the plane of a, b, c (one row for every camera after the third)
 *
 * Only distances BETWEEN the centres enter, so the place and the orientation of the rig on the IMU stay free.
 *
 * One more row (OV_RIG_RADIAL_DEG, default OFF) says how the rig is turned about its normal against the viewing directions: the listed
 * cameras look radially outward on average. Listed = the cameras of OV_RIG_PLANAR; without it all cameras of OV_RIG_DIST, ascending.
 * With m the mean of their centres, n the unit normal of the plane through the first three of them ((c_b - c_a) x (c_c - c_a),
 * normalised), u_i = c_i - m and the optical axis a_i = R_ItoC_i^T e_z, both taken without their part along n:
 * - h = mean_i atan2(n . (u_i x a_i), u_i . a_i) - target     in radians, noise OV_RIG_RADIAL_SIG_DEG
 * Every angle is taken in (-180, 180] deg (the cameras have to look outward, not inward); its sign follows n, hence the order of the list.
 *
 * Two more rows (OV_RIG_TILT_SIG_DEG, default OFF) say how the rig plane is tilted against the viewing directions: it contains the
 * optical axes of the listed cameras on average. n_axes = unit normal of the least-squares plane through the origin of these axes (the
 * right singular vector with the smallest singular value of the matrix with one axis per row), with the sign of n. e1 = u of the first
 * listed camera without its part along n, normalised, and e2 = n x e1 span the rig plane:
 * - h = (n_axes . e1, n_axes . e2)     for a small tilt two angles in radians, noise OV_RIG_TILT_SIG_DEG each
 * The plane of the axes goes through the origin, so an elevation that all axes share is not a tilt.
 *
 * Switches (environment):
 * - OV_RIG_DIST      "i-j:millimetres,..."  pairs of cameras and the distance between their optical centres
 * - OV_RIG_PLANAR    "a,b,c,d[,...]"        four or more cameras whose centres lie in one plane
 * - OV_RIG_SIG_MM    noise of every row in millimetres (default 0.5). In the default mode of OV_RIG_REAPPLY it is NOT a tolerance of
 *                    the result (see there)
 * - OV_RIG_EVERY     n: the later calls act at the end of every n-th filter update (default 1; 0 = never again): every n-th call in
 *                    the default mode, every n-th new state timestamp with OV_RIG_REAPPLY=1
 * - OV_RIG_SNAP      1 (default): before the first application the centres are moved onto the constraints (mean only)
 * - OV_RIG_STATS     1: one "[rig]" line per application, "[rig-snap]" for the snap, "[rig-total]" at exit
 * - OV_RIG_SELFTEST  1: once, the Jacobian against finite differences through the state variables ("[rig-selftest]")
 * - OV_RIG_RADIAL_DEG      t: the mean signed angle from the radial direction to the optical axis is t degrees (unset = not used);
 *                          the snap then also turns the centres about n, so that it holds exactly at the start
 * - OV_RIG_RADIAL_SIG_DEG  noise of that row in degrees (default 0.5; 0 or negative = no row, the snap only)
 * - OV_RIG_TILT_SIG_DEG    s >= 0 (unset = not used; with or without OV_RIG_RADIAL_DEG): the snap also tilts the centres about their
 *                          middle so that n = n_axes, before it turns them; s > 0: the two rows above with noise s degrees
 * - OV_RIG_REAPPLY   0 or unset (default): the rows are applied once, with the first application, and every later call only puts the
 *                    centres back onto the shape: the snap of the distance and plane rows, mean only, no turn and no tilt. The
 *                    result then has EXACTLY the given shape and size, whatever OV_RIG_SIG_MM is: the pictures cannot correct a
 *                    wrong drawing value, and the noise only sets how sure the filter is of the shape at that single application
 *                    (its covariance; six distance rows from one drawing count as independent). 1: every later application is a
 *                    new EKF update with all rows, which pull with their noise each time (the mode a review found over-confident).
 *                    Any other value: one "[rig-warn]", then as 0
 * - OV_RIG_AFTER_ZUPT  1: the later call is also made after every zero-velocity update (read in VioManager.cpp). Default OFF: the
 *                    centres are then off the shape, by up to about 1 mm, from a zero-velocity update until the next feature update, and
 *                    a calibration taken at standstill after a landing is not exactly on the shape. The results reported for flight
 *                    windows of 20 to 60 s were made without it, the whole-flight runs with the shape held with it. With
 *                    OV_RIG_REAPPLY=1 it does nothing (the rows are applied again at feature updates only). Any value other
 *                    than 0 or 1: one "[rig-warn]", then as 0
 *
 * Input checks ("[rig-warn]", the rig shape is then not used in this run): a length of OV_RIG_DIST that is not finite and positive;
 * at the first application, a given length that is more than 50 % of itself away from the start distance of its pair (metres typed
 * for millimetres); a snap of the first application that was asked for and does not settle.
 */
class UpdaterRigShape {

public:
  /**
   * @brief The updater that the OV_RIG_* switches describe
   * @return nullptr if neither OV_RIG_DIST nor OV_RIG_PLANAR is set, or if they cannot be read (one warning on stderr)
   */
  static std::shared_ptr<UpdaterRigShape> from_env();

  /// Prints the "[rig-total]" line (OV_RIG_STATS=1)
  ~UpdaterRigShape();

  /**
   * @brief First application; has to run before the filter's first feature update. Every later call does nothing.
   *
   * Needs the camera poses as calibration states and every listed camera; otherwise one warning and the updater stays idle.
   * The same when a given length is more than 50 % of itself away from the start distance of its pair (a wrong unit).
   * Order: self-test (if asked for), snap (if not switched off), EKF update. If the snap was asked for and does not settle, there
   * is no EKF update: one warning and the updater goes idle.
   *
   * @param state State of the filter
   * @return True if the state was changed
   */
  bool update_first(std::shared_ptr<State> state);

  /**
   * @brief Later applications; to be called at the end of every feature update, and after a zero-velocity update only where
   * OV_RIG_AFTER_ZUPT=1 asks for it (VioManager.cpp).
   *
   * Default (OV_RIG_REAPPLY unset or 0): no EKF update, the mean-only snap of the shape on every OV_RIG_EVERY-th call, whatever its
   * timestamp.
   * With OV_RIG_REAPPLY=1: applies the rows on every OV_RIG_EVERY-th feature-update call whose state timestamp differs from that of
   * the call before (the sub-updates of one clone tick share a timestamp and count once); never after a zero-velocity update.
   *
   * @param state State of the filter
   * @param after_zupt The call follows a zero-velocity update: in the default mode it is counted for the "[rig-total]" line, with
   *                   OV_RIG_REAPPLY=1 nothing is done
   * @return True if the state was changed
   */
  bool update_again(std::shared_ptr<State> state, bool after_zupt = false);

protected:
  /// A pair of cameras (positions in _cams) and the distance between their centres in metres
  struct Pair {
    size_t i, j;
    double dist;
  };

  UpdaterRigShape() {}

  /// Optical centres of the cameras of _cams in the IMU frame
  std::vector<Eigen::Vector3d> centres(std::shared_ptr<State> state) const;

  /**
   * @brief Values of all rows for the given centres
   * @param c Centres, in the order of _cams
   * @param h Row values in metres (pairs first, then the plane rows)
   * @param G If not null: derivative of h with respect to the centre coordinates (rows x 3 per camera)
   * @return False if a row is not defined (two centres coincide, or the first three cameras of the plane are in line)
   */
  bool measure(const std::vector<Eigen::Vector3d> &c, Eigen::VectorXd &h, Eigen::MatrixXd *G) const;

  /// Row values at the current state and their Jacobian with respect to the error states of the camera poses (6 per camera of _cams)
  bool linearize(std::shared_ptr<State> state, Eigen::VectorXd &h, Eigen::MatrixXd &H) const;

  /// Optical axes of the cameras of _cams in the IMU frame (third row of R_ItoC)
  std::vector<Eigen::Vector3d> axes(std::shared_ptr<State> state) const;

  /**
   * @brief Mean signed angle from the radial direction to the optical axis over the cameras of _radial_cams, in the rig plane
   * @param c Centres, in the order of _cams
   * @param a Optical axes, in the order of _cams
   * @param angle Mean angle in radians
   * @param normal If not null: unit normal of the plane through the first three cameras of _radial_cams
   * @param middle If not null: mean of the centres of _radial_cams
   * @return False if it is not defined (the first three centres in line, a centre on the normal through the middle, an axis along it)
   */
  bool turn(const std::vector<Eigen::Vector3d> &c, const std::vector<Eigen::Vector3d> &a, double &angle, Eigen::Vector3d *normal = nullptr,
            Eigen::Vector3d *middle = nullptr) const;

  /**
   * @brief The turn row at the current state: h = mean angle - target (radians), not yet scaled
   * @param H Its Jacobian (1 x 6 per camera of _cams) by central differences through PoseJPL::update(); the poses are put back
   * @return False if the row is not defined, or if the poses did not come back bit for bit
   */
  bool linearize_turn(std::shared_ptr<State> state, double &h, Eigen::MatrixXd &H) const;

  /// The rig plane against the plane of the optical axes (OV_RIG_TILT_SIG_DEG), filled by tilt()
  struct Tilt {
    /// Mean of the listed centres, unit normal and basis of the rig plane, unit normal of the plane of the listed optical axes
    Eigen::Vector3d middle, normal, e1, e2, normal_axes;
    /// The two row values normal_axes . e1 and normal_axes . e2, and the angle between the two normals in radians
    double h1 = 0, h2 = 0, angle = 0;
  };

  /**
   * @brief The plane of the optical axes of the cameras of _radial_cams against the plane of their centres
   * @param c Centres, in the order of _cams
   * @param a Optical axes, in the order of _cams
   * @param t Result
   * @return False if it is not defined (the first three centres in line, the first centre on the normal through the middle, axes
   * that do not span a plane)
   */
  bool tilt(const std::vector<Eigen::Vector3d> &c, const std::vector<Eigen::Vector3d> &a, Tilt &t) const;

  /**
   * @brief The two tilt rows at the current state: h = (n_axes . e1, n_axes . e2), not yet scaled
   * @param H Their Jacobian (2 x 6 per camera of _cams) by central differences through PoseJPL::update(); the poses are put back
   * @return False if the rows are not defined, or if the poses did not come back bit for bit
   */
  bool linearize_tilt(std::shared_ptr<State> state, Eigen::VectorXd &h, Eigen::MatrixXd &H) const;

  /// One EKF update with all rows. False (and the updater goes idle, with one warning) if a row is not defined.
  bool apply(std::shared_ptr<State> state);

  /// Moves the centres onto the constraints by minimum-norm corrections (mean only, rotations kept). False if it did not converge.
  /// again = a later snap of the default mode (OV_RIG_REAPPLY=0): the shape only (no tilt, no turn), a "[rig]" line instead of
  /// "[rig-snap]", and the state epoch of the precomputed systems is moved, since no EKF update follows.
  bool snap(std::shared_ptr<State> state, bool again = false);

  /// Compares the Jacobian of linearize() with central finite differences through PoseJPL::update(); leaves the state as it was
  void selftest(std::shared_ptr<State> state) const;

  /// The distinct cameras of all rows, ascending. Their poses are the variables of the update, in this order.
  std::vector<size_t> _cams;

  /// Distance rows
  std::vector<Pair> _pairs;

  /// Cameras in one plane (positions in _cams); empty or at least four
  std::vector<size_t> _planar;

  /// Noise of every row in metres
  double _sigma = 0.5e-3;

  /// The later calls act on every n-th call (OV_RIG_REAPPLY=1: on every n-th new state timestamp); 0 = only the first application
  int _every = 1;

  bool _do_snap = true, _stats = false, _selftest = false;

  /// OV_RIG_RADIAL_DEG: in use, the cameras it averages over (positions in _cams), the target of the mean angle and the noise of the
  /// row, both in radians (noise 0 or negative = no row, the snap only), and the angle the snap turned the centres by (radians)
  bool _radial = false;
  std::vector<size_t> _radial_cams;
  double _radial_target = 0, _radial_sigma = 0.5 * M_PI / 180.0, _snap_turned = 0;

  /// OV_RIG_TILT_SIG_DEG: in use (over the cameras of _radial_cams), the noise of each of its two rows in radians (0 = no rows, the
  /// snap only), and the angle the snap tilted the centres by (radians)
  bool _tilt = false;
  double _tilt_sigma = 0, _snap_tilted = 0;

  /// OV_RIG_REAPPLY: false (default, 0) = the rows once and then the mean-only snap of the shape, true (1) = the rows again at every
  /// later application; the snaps and the calls of update_again() so far in the default mode
  bool _reapply = false;
  long _snaps = 0, _calls = 0;

  /// The calls of update_again() after a zero-velocity update that changed the state (snaps of the default mode; none with OV_RIG_REAPPLY=1)
  long _after_zupt = 0;

  /// update_first() has been called / it found the cameras and the rows are in use
  bool _first_done = false, _active = false;

  /// State timestamp of the last call of update_again() and the number of new timestamps it has seen
  double _last_timestamp = -1;
  long _new_timestamps = 0;

  /// For the lines of OV_RIG_STATS: the state (read once more at exit), the applications so far, the sum of the rms row values and
  /// the largest row value found before an update (millimetres), and how far the snap moved each centre (metres)
  std::shared_ptr<State> _state;
  long _applications = 0;
  double _sum_rms_before_mm = 0, _max_before_mm = 0;
  std::vector<double> _snap_moved;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_RIGSHAPE_H
