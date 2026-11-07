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

#include "FeatureInitializer.h"

#include <cmath>
#include <limits>
#include <vector>

#include "Feature.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;

// ===================================================================================================================================
// OV_KB_BEARING (default OFF; CamBase::bearing_cfg): features with at least one rim observation (bearing above
// OV_BEARING_SWITCH_DEG) are triangulated from unit bearings unprojected from the raw pixels with the current calibration, anchored
// on an observation within OV_BEARING_ANCHOR_DEG, and refined with pixel residuals. Features without rim observations, and
// everything with the switch off, take the original code below byte for byte.
// OV_BEARING_ALL=1 (read only with OV_KB_BEARING=1): EVERY feature takes the bearing path,
// and every observation's bearing is unprojected from its raw pixel with the current calibration (not only the rim ones); the clean
// step erases every observation without a bearing. Features without rim observations are counted in the [bearing-all] line.
// ===================================================================================================================================
namespace {
// the last bearing triangulation of this thread: lets the refinement count its outcome only when the linear step succeeded
struct BearingTriLast {
  size_t featid = (size_t)-1;
  bool ok = false;
  bool rim = true; // the feature had rim observations (counted in tri_rim*), else a core feature of OV_BEARING_ALL (tri_core*)
};
thread_local BearingTriLast g_btri_last;
} // namespace

int FeatureInitializer::count_rim(const Feature &feat) {
  if (!CamBase::kb_bearing())
    return 0;
  int n = 0;
  for (auto const &pair : feat.uvs_norm)
    for (auto const &un : pair.second)
      n += CamBase::bearing_is_rim(un(0), un(1)) ? 1 : 0;
  return n;
}

int FeatureInitializer::bearing_clean(Feature &feat, const std::unordered_map<size_t, std::shared_ptr<CamBase>> &cams) {
  if (!CamBase::kb_bearing())
    return 0;
  const bool all = CamBase::bearing_cfg().all; // OV_BEARING_ALL: every observation needs a bearing
  int erased = 0, erased_core = 0;
  for (auto &pair : feat.timestamps) {
    const size_t cam = pair.first;
    auto cit = cams.find(cam);
    if (cit == cams.end())
      continue;
    auto &ts = feat.timestamps[cam];
    auto &uv = feat.uvs[cam];
    auto &un = feat.uvs_norm[cam];
    size_t w = 0;
    for (size_t m = 0; m < ts.size(); m++) {
      bool keep = true;
      const bool rim = CamBase::bearing_is_rim(un[m](0), un[m](1));
      if (rim || all) {
        Eigen::Vector3d b;
        keep = cit->second->unproject_bearing(Eigen::Vector2d((double)uv[m](0), (double)uv[m](1)), b);
      }
      if (!keep) {
        erased++;
        if (!rim)
          erased_core++;
        continue;
      }
      if (w != m) {
        ts[w] = ts[m];
        uv[w] = uv[m];
        un[w] = un[m];
      }
      w++;
    }
    ts.resize(w);
    uv.resize(w);
    un.resize(w);
  }
  if (erased > erased_core)
    CamBase::bearing_stats().clean_erased.fetch_add(erased - erased_core, std::memory_order_relaxed);
  if (erased_core > 0)
    CamBase::bearing_stats().clean_erased_core.fetch_add(erased_core, std::memory_order_relaxed);
  return erased;
}

bool FeatureInitializer::bearing_path(const Feature &feat) const {
  if (!CamBase::kb_bearing() || _bcams.empty())
    return false;
  if (CamBase::bearing_cfg().all) // OV_BEARING_ALL: every feature
    return true;
  return count_rim(feat) > 0;
}

bool FeatureInitializer::single_triangulation_bearing(std::shared_ptr<Feature> feat,
                                                      std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {
  const CamBase::BearingCfg &cfg = CamBase::bearing_cfg();
  CamBase::BearingStats &st = CamBase::bearing_stats();
  g_btri_last.featid = feat->featid;
  g_btri_last.ok = false;
  g_btri_last.rim = true;

  // Base camera as today: the camera with the most observations (first one found in the map's order)
  size_t anchor_most_meas = 0;
  size_t most_meas = 0;
  for (auto const &pair : feat->timestamps) {
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }

  // Unit bearing of every observation in its camera frame: non-rim = normalize(uvs_norm, 1) as today; rim = from the raw pixel.
  // OV_BEARING_ALL: every observation from its raw pixel (current calibration).
  struct Obs {
    size_t cam;
    size_t m;
    double t;
    Eigen::Vector3d b;
    bool rim;
  };
  std::vector<Obs> obs;
  bool unproj_ok = true;
  int nrim = 0, ncore_fail = 0;
  for (auto const &pair : feat->timestamps) {
    for (size_t m = 0; m < pair.second.size(); m++) {
      Obs o;
      o.cam = pair.first;
      o.m = m;
      o.t = pair.second.at(m);
      const Eigen::VectorXf &un = feat->uvs_norm.at(pair.first).at(m);
      o.rim = CamBase::bearing_is_rim(un(0), un(1));
      if (o.rim)
        nrim++;
      if (o.rim || cfg.all) {
        auto cit = _bcams.find(pair.first);
        const Eigen::VectorXf &uv = feat->uvs.at(pair.first).at(m);
        if (cit == _bcams.end() || !cit->second->unproject_bearing(Eigen::Vector2d((double)uv(0), (double)uv(1)), o.b)) {
          unproj_ok = false;
          if (!o.rim)
            ncore_fail++;
          o.b = Eigen::Vector3d(un(0), un(1), 1.0).normalized();
        }
      } else {
        o.b = Eigen::Vector3d(un(0), un(1), 1.0).normalized();
      }
      obs.push_back(o);
    }
  }
  // the rim features are counted as before (tri_rim*); OV_BEARING_ALL's features without rim observations in tri_core*
  const bool has_rim = nrim > 0;
  g_btri_last.rim = has_rim;
  (has_rim ? st.tri_rim : st.tri_core).fetch_add(1, std::memory_order_relaxed);
  st.tri_rim_obs.fetch_add(nrim, std::memory_order_relaxed);
  if (ncore_fail > 0)
    st.core_unproj_fail.fetch_add(ncore_fail, std::memory_order_relaxed);
  if (obs.empty())
    return false;

  // Anchor: the most recent observation of the base camera with bearing <= OV_BEARING_ANCHOR_DEG; else the most recent such
  // observation of any camera; else the observation with the smallest bearing
  int ia = -1;
  for (size_t i = 0; i < obs.size(); i++)
    if (obs[i].cam == anchor_most_meas && obs[i].b(2) >= cfg.cos_anchor && (ia < 0 || obs[i].t > obs[(size_t)ia].t))
      ia = (int)i;
  if (ia < 0) {
    for (size_t i = 0; i < obs.size(); i++)
      if (obs[i].b(2) >= cfg.cos_anchor && (ia < 0 || obs[i].t > obs[(size_t)ia].t || (obs[i].t == obs[(size_t)ia].t && obs[i].cam < obs[(size_t)ia].cam)))
        ia = (int)i;
    if (ia >= 0)
      (has_rim ? st.tri_anchor_fallback : st.core_anchor_fallback).fetch_add(1, std::memory_order_relaxed);
  }
  bool anchor_wide = false;
  if (ia < 0) {
    anchor_wide = true;
    for (size_t i = 0; i < obs.size(); i++)
      if (ia < 0 || obs[i].b(2) > obs[(size_t)ia].b(2))
        ia = (int)i;
    (has_rim ? st.tri_anchor_rim : st.core_anchor_wide).fetch_add(1, std::memory_order_relaxed);
  }
  feat->anchor_cam_id = (int)obs[(size_t)ia].cam;
  feat->anchor_clone_timestamp = obs[(size_t)ia].t;
  // OV_BEARING_ALL statistics: a core feature whose anchor differs from the original rule's (base camera, its last observation)
  if (!has_rim && most_meas > 0 &&
      (obs[(size_t)ia].cam != anchor_most_meas || obs[(size_t)ia].t != feat->timestamps.at(anchor_most_meas).back()))
    st.core_anchor_moved.fetch_add(1, std::memory_order_relaxed);
  if (!unproj_ok) {
    (has_rim ? st.tri_rim_fail_lin : st.tri_core_fail_lin).fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  // Linear system: sum over the observations of (I - b b^T) (p_f - p_Ci) = 0, every bearing b expressed in the anchor frame
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();
  for (const Obs &o : obs) {
    const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(o.cam).at(o.t).Rot();
    const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(o.cam).at(o.t).pos();
    Eigen::Matrix<double, 3, 3> R_AtoCi;
    R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
    Eigen::Matrix<double, 3, 1> p_CiinA;
    p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
    Eigen::Vector3d b_i = R_AtoCi.transpose() * o.b;
    b_i = b_i / b_i.norm();
    Eigen::Matrix3d Bperp = skew_x(b_i);
    Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
    A += Ai;
    b += Ai * p_CiinA;
  }
  Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
  Eigen::MatrixXd singularValues;
  singularValues.resize(svd.singularValues().rows(), 1);
  singularValues = svd.singularValues();
  double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);
  bool bad = std::abs(condA) > _options.max_cond_number || std::isnan(p_f.norm());
  if (!bad) {
    if (!anchor_wide) {
      bad = p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist;
    } else {
      const Eigen::Vector3d bA = obs[(size_t)ia].b;
      const Eigen::Vector3d pf3 = p_f.block(0, 0, 3, 1);
      bad = pf3.dot(bA) < _options.min_dist || pf3.norm() > _options.max_dist;
    }
  }
  if (bad) {
    (has_rim ? st.tri_rim_fail_lin : st.tri_core_fail_lin).fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  feat->p_FinA = p_f;
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  g_btri_last.ok = true;
  return true;
}

double FeatureInitializer::compute_error_bearing(std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                                 std::shared_ptr<Feature> feat, double alpha, double beta, double rho) {
  const CamBase::BearingCfg &cfg = CamBase::bearing_cfg();
  const double wr = 1.0 / (cfg.rim_sigma * cfg.rim_sigma);
  double err = 0;
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();
  for (auto const &pair : feat->timestamps) {
    auto cit = _bcams.find(pair.first);
    if (cit == _bcams.end())
      return std::numeric_limits<double>::infinity();
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
      Eigen::Matrix<double, 3, 1> p_AinCi;
      p_AinCi.noalias() = -R_AtoCi * p_CiinA;
      const Eigen::Vector3d h = R_AtoCi * Eigen::Vector3d(alpha, beta, 1.0) + rho * p_AinCi;
      Eigen::Vector2d uvp;
      if (!cit->second->project_bearing(h, uvp))
        return std::numeric_limits<double>::infinity();
      const Eigen::VectorXf &uvm = feat->uvs.at(pair.first).at(m);
      const Eigen::Vector2d r((double)uvm(0) - uvp(0), (double)uvm(1) - uvp(1));
      const Eigen::VectorXf &un = feat->uvs_norm.at(pair.first).at(m);
      err += (CamBase::bearing_is_rim(un(0), un(1)) ? wr : 1.0) * r.squaredNorm();
    }
  }
  return err;
}

bool FeatureInitializer::single_gaussnewton_bearing(std::shared_ptr<Feature> feat,
                                                    std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {
  CamBase::BearingStats &st = CamBase::bearing_stats();
  const bool counted = (g_btri_last.featid == feat->featid && g_btri_last.ok);
  const bool counted_rim = g_btri_last.rim; // rim feature (tri_rim*) or OV_BEARING_ALL core feature (tri_core*)
  g_btri_last.featid = (size_t)-1;
  auto fail = [&]() {
    if (counted)
      (counted_rim ? st.tri_rim_fail_ref : st.tri_core_fail_ref).fetch_add(1, std::memory_order_relaxed);
    return false;
  };
  // the anchor must be a clone of this update (the linear step sets it first; a stale anchor = the linear step did not run)
  auto ait = clonesCAM.find((size_t)feat->anchor_cam_id);
  if (feat->anchor_cam_id < 0 || ait == clonesCAM.end() || ait->second.find(feat->anchor_clone_timestamp) == ait->second.end())
    return fail();
  if (!(feat->p_FinA.allFinite()) || feat->p_FinA(2) == 0.0)
    return fail();
  const CamBase::BearingCfg &cfg = CamBase::bearing_cfg();
  const double wr = 1.0 / cfg.rim_sigma;

  // Get into inverse depth (as today)
  double rho = 1 / feat->p_FinA(2);
  double alpha = feat->p_FinA(0) / feat->p_FinA(2);
  double beta = feat->p_FinA(1) / feat->p_FinA(2);

  double lam = _options.init_lamda;
  double eps = 10000;
  int runs = 0;
  bool recompute = true;
  Eigen::Matrix<double, 3, 3> Hess = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 1> grad = Eigen::Matrix<double, 3, 1>::Zero();
  double cost_old = compute_error_bearing(clonesCAM, feat, alpha, beta, rho);
  if (!std::isfinite(cost_old))
    return fail();
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  while (runs < _options.max_runs && lam < _options.max_lamda && eps > _options.min_dx) {
    if (recompute) {
      Hess.setZero();
      grad.setZero();
      bool jac_ok = true;
      for (auto const &pair : feat->timestamps) {
        auto cit = _bcams.find(pair.first);
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size() && jac_ok; m++) {
          const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).Rot();
          const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).pos();
          Eigen::Matrix<double, 3, 3> R_AtoCi;
          R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
          Eigen::Matrix<double, 3, 1> p_CiinA;
          p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
          Eigen::Matrix<double, 3, 1> p_AinCi;
          p_AinCi.noalias() = -R_AtoCi * p_CiinA;
          // h = R_AtoCi (alpha, beta, 1) + rho p_AinCi, a positive multiple of the point in camera i while rho > 0
          const Eigen::Vector3d h = R_AtoCi * Eigen::Vector3d(alpha, beta, 1.0) + rho * p_AinCi;
          Eigen::Vector2d uvp;
          Eigen::Matrix<double, 2, 3> Hp;
          if (cit == _bcams.end() || !cit->second->project_bearing(h, uvp) || !cit->second->project_bearing_jacobian(h, Hp, nullptr, nullptr)) {
            jac_ok = false;
            break;
          }
          Eigen::Matrix<double, 3, 3> dh;
          dh.col(0) = R_AtoCi.col(0);
          dh.col(1) = R_AtoCi.col(1);
          dh.col(2) = p_AinCi;
          Eigen::Matrix<double, 2, 3> H = Hp * dh; // d(predicted pixel) / d(alpha, beta, rho)
          const Eigen::VectorXf &uvm = feat->uvs.at(pair.first).at(m);
          Eigen::Vector2d res((double)uvm(0) - uvp(0), (double)uvm(1) - uvp(1));
          const Eigen::VectorXf &un = feat->uvs_norm.at(pair.first).at(m);
          if (CamBase::bearing_is_rim(un(0), un(1))) {
            H *= wr;
            res *= wr;
          }
          grad.noalias() += H.transpose() * res;
          Hess.noalias() += H.transpose() * H;
        }
        if (!jac_ok)
          break;
      }
      if (!jac_ok)
        return fail();
    }
    Eigen::Matrix<double, 3, 3> Hess_l = Hess;
    for (size_t r = 0; r < (size_t)Hess.rows(); r++)
      Hess_l(r, r) *= (1.0 + lam);
    Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
    double cost = compute_error_bearing(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));
    if (!std::isfinite(cost)) { // a failed projection counts as a cost increase
      recompute = false;
      lam = lam * _options.lam_mult;
      continue;
    }
    if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      eps = 0;
      break;
    }
    if (cost <= cost_old) {
      recompute = true;
      cost_old = cost;
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      runs++;
      lam = lam / _options.lam_mult;
      eps = dx.norm();
    } else {
      recompute = false;
      lam = lam * _options.lam_mult;
      continue;
    }
  }

  feat->p_FinA(0) = alpha / rho;
  feat->p_FinA(1) = beta / rho;
  feat->p_FinA(2) = 1 / rho;

  // Same validity checks as today (anchor-frame depth, distance, baseline ratio)
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
  Eigen::MatrixXd Q = qr.householderQ();
  double base_line_max = 0.0;
  for (auto const &pair : feat->timestamps) {
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      Eigen::Matrix<double, 3, 1> p_CiinA = R_GtoA * (p_CiinG - p_AinG);
      double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
      if (base_line > base_line_max)
        base_line_max = base_line;
    }
  }
  if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist ||
      (feat->p_FinA.norm() / base_line_max) > _options.max_baseline || std::isnan(feat->p_FinA.norm())) {
    return fail();
  }
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  if (counted)
    (counted_rim ? st.tri_rim_ok : st.tri_core_ok).fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool FeatureInitializer::single_triangulation(std::shared_ptr<Feature> feat,
                                              std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // OV_KB_BEARING: a feature with rim observations takes the bearing path
  if (bearing_path(*feat))
    return single_triangulation_bearing(feat, clonesCAM);

  // Total number of measurements
  // Also set the first measurement to be the anchor frame
  int total_meas = 0;
  size_t anchor_most_meas = 0;
  size_t most_meas = 0;
  for (auto const &pair : feat->timestamps) {
    total_meas += (int)pair.second.size();
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }
  feat->anchor_cam_id = anchor_most_meas;
  feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back();

  // Our linear system matrices
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();

  // Get the position of the anchor pose
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

      // Get the UV coordinate normal
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
      b_i = R_AtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
      A += Ai;
      b += Ai * p_CiinA;
    }
  }

  // Solve the linear system
  Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

  // Check A and p_f
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
  Eigen::MatrixXd singularValues;
  singularValues.resize(svd.singularValues().rows(), 1);
  singularValues = svd.singularValues();
  double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

  // std::stringstream ss;
  // ss << feat->featid << " - cond " << std::abs(condA) << " - z " << p_f(2, 0) << std::endl;
  // PRINT_DEBUG(ss.str().c_str());

  // If we have a bad condition number, or it is too close
  // Then set the flag for bad (i.e. set z-axis to nan)
  if (std::abs(condA) > _options.max_cond_number || p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist ||
      std::isnan(p_f.norm())) {
    return false;
  }

  // Store it in our feature object
  feat->p_FinA = p_f;
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

bool FeatureInitializer::single_triangulation_1d(std::shared_ptr<Feature> feat,
                                                 std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // OV_KB_BEARING: a feature with rim observations takes the (full) bearing triangulation
  if (bearing_path(*feat))
    return single_triangulation_bearing(feat, clonesCAM);

  // Total number of measurements
  // Also set the first measurement to be the anchor frame
  int total_meas = 0;
  size_t anchor_most_meas = 0;
  size_t most_meas = 0;
  for (auto const &pair : feat->timestamps) {
    total_meas += (int)pair.second.size();
    if (pair.second.size() > most_meas) {
      anchor_most_meas = pair.first;
      most_meas = pair.second.size();
    }
  }
  feat->anchor_cam_id = anchor_most_meas;
  feat->anchor_clone_timestamp = feat->timestamps.at(feat->anchor_cam_id).back();
  size_t idx_anchor_bearing = feat->timestamps.at(feat->anchor_cam_id).size() - 1;

  // Our linear system matrices
  double A = 0.0;
  double b = 0.0;

  // Get the position of the anchor pose
  ClonePose anchorclone = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp);
  const Eigen::Matrix<double, 3, 3> &R_GtoA = anchorclone.Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = anchorclone.pos();

  // Get bearing in anchor frame
  Eigen::Matrix<double, 3, 1> bearing_inA;
  bearing_inA << feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(0),
      feat->uvs_norm.at(feat->anchor_cam_id).at(idx_anchor_bearing)(1), 1;
  bearing_inA = bearing_inA / bearing_inA.norm();

  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {

    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      // Skip the anchor bearing
      if ((int)pair.first == feat->anchor_cam_id && m == idx_anchor_bearing)
        continue;

      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();

      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);

      // Get the UV coordinate normal
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << feat->uvs_norm.at(pair.first).at(m)(0), feat->uvs_norm.at(pair.first).at(m)(1), 1;
      b_i = R_AtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Vector3d BperpBanchor = Bperp * bearing_inA;
      A += BperpBanchor.dot(BperpBanchor);
      b += BperpBanchor.dot(Bperp * p_CiinA);
    }
  }

  // Solve the linear system
  double depth = b / A;
  Eigen::MatrixXd p_f = depth * bearing_inA;

  // Then set the flag for bad (i.e. set z-axis to nan)
  if (p_f(2, 0) < _options.min_dist || p_f(2, 0) > _options.max_dist || std::isnan(p_f.norm())) {
    return false;
  }

  // Store it in our feature object
  feat->p_FinA = p_f;
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

bool FeatureInitializer::single_gaussnewton(std::shared_ptr<Feature> feat,
                                            std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM) {

  // OV_KB_BEARING: a feature with rim observations is refined with pixel residuals
  if (bearing_path(*feat))
    return single_gaussnewton_bearing(feat, clonesCAM);

  // Get into inverse depth
  double rho = 1 / feat->p_FinA(2);
  double alpha = feat->p_FinA(0) / feat->p_FinA(2);
  double beta = feat->p_FinA(1) / feat->p_FinA(2);

  // Optimization parameters
  double lam = _options.init_lamda;
  double eps = 10000;
  int runs = 0;

  // Variables used in the optimization
  bool recompute = true;
  Eigen::Matrix<double, 3, 3> Hess = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 1> grad = Eigen::Matrix<double, 3, 1>::Zero();

  // Cost at the last iteration
  double cost_old = compute_error(clonesCAM, feat, alpha, beta, rho);

  // Get the position of the anchor pose
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  // Loop till we have either
  // 1. Reached our max iteration count
  // 2. System is unstable
  // 3. System has converged
  while (runs < _options.max_runs && lam < _options.max_lamda && eps > _options.min_dx) {

    // Triggers a recomputation of jacobians/information/gradients
    if (recompute) {

      Hess.setZero();
      grad.setZero();

      double err = 0;

      // Loop through each camera for this feature
      for (auto const &pair : feat->timestamps) {

        // Add CAM_I features
        for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

          //=====================================================================================
          //=====================================================================================

          // Get the position of this clone in the global
          const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).Rot();
          const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps[pair.first].at(m)).pos();
          // Convert current position relative to anchor
          Eigen::Matrix<double, 3, 3> R_AtoCi;
          R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
          Eigen::Matrix<double, 3, 1> p_CiinA;
          p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
          Eigen::Matrix<double, 3, 1> p_AinCi;
          p_AinCi.noalias() = -R_AtoCi * p_CiinA;

          //=====================================================================================
          //=====================================================================================

          // Middle variables of the system
          double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
          double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
          double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
          // Calculate jacobian
          double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
          double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
          double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
          double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
          double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
          double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
          Eigen::Matrix<double, 2, 3> H;
          H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
          // Calculate residual
          Eigen::Matrix<float, 2, 1> z;
          z << hi1 / hi3, hi2 / hi3;
          Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;

          //=====================================================================================
          //=====================================================================================

          // Append to our summation variables
          err += std::pow(res.norm(), 2);
          grad.noalias() += H.transpose() * res.cast<double>();
          Hess.noalias() += H.transpose() * H;
        }
      }
    }

    // Solve Levenberg iteration
    Eigen::Matrix<double, 3, 3> Hess_l = Hess;
    for (size_t r = 0; r < (size_t)Hess.rows(); r++) {
      Hess_l(r, r) *= (1.0 + lam);
    }

    Eigen::Matrix<double, 3, 1> dx = Hess_l.colPivHouseholderQr().solve(grad);
    // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

    // Check if error has gone down
    double cost = compute_error(clonesCAM, feat, alpha + dx(0, 0), beta + dx(1, 0), rho + dx(2, 0));

    // Debug print
    // std::stringstream ss;
    // ss << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;
    // PRINT_DEBUG(ss.str().c_str());

    // Check if converged
    if (cost <= cost_old && (cost_old - cost) / cost_old < _options.min_dcost) {
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      eps = 0;
      break;
    }

    // If cost is lowered, accept step
    // Else inflate lambda (try to make more stable)
    if (cost <= cost_old) {
      recompute = true;
      cost_old = cost;
      alpha += dx(0, 0);
      beta += dx(1, 0);
      rho += dx(2, 0);
      runs++;
      lam = lam / _options.lam_mult;
      eps = dx.norm();
    } else {
      recompute = false;
      lam = lam * _options.lam_mult;
      continue;
    }
  }

  // Revert to standard, and set to all
  feat->p_FinA(0) = alpha / rho;
  feat->p_FinA(1) = beta / rho;
  feat->p_FinA(2) = 1 / rho;

  // Get tangent plane to x_hat
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->p_FinA);
  Eigen::MatrixXd Q = qr.householderQ();

  // Max baseline we have between poses
  double base_line_max = 0.0;

  // Check maximum baseline
  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {
    // Loop through the other clones to see what the max baseline is
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {
      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 1> p_CiinA = R_GtoA * (p_CiinG - p_AinG);
      // Dot product camera pose and nullspace
      double base_line = ((Q.block(0, 1, 3, 2)).transpose() * p_CiinA).norm();
      if (base_line > base_line_max)
        base_line_max = base_line;
    }
  }
  // std::stringstream ss;
  // ss << feat->featid << " - max base " << (feat->p_FinA.norm() / base_line_max) << " - z " << feat->p_FinA(2) << std::endl;
  // PRINT_DEBUG(ss.str().c_str());

  // Check if this feature is bad or not
  // 1. If the feature is too close
  // 2. If the feature is invalid
  // 3. If the baseline ratio is large
  if (feat->p_FinA(2) < _options.min_dist || feat->p_FinA(2) > _options.max_dist ||
      (feat->p_FinA.norm() / base_line_max) > _options.max_baseline || std::isnan(feat->p_FinA.norm())) {
    return false;
  }

  // Finally get position in global frame
  feat->p_FinG = R_GtoA.transpose() * feat->p_FinA + p_AinG;
  return true;
}

double FeatureInitializer::compute_error(std::unordered_map<size_t, std::unordered_map<double, ClonePose>> &clonesCAM,
                                         std::shared_ptr<Feature> feat, double alpha, double beta, double rho) {

  // Total error
  double err = 0;

  // Get the position of the anchor pose
  const Eigen::Matrix<double, 3, 3> &R_GtoA = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).Rot();
  const Eigen::Matrix<double, 3, 1> &p_AinG = clonesCAM.at(feat->anchor_cam_id).at(feat->anchor_clone_timestamp).pos();

  // Loop through each camera for this feature
  for (auto const &pair : feat->timestamps) {
    // Add CAM_I features
    for (size_t m = 0; m < feat->timestamps.at(pair.first).size(); m++) {

      //=====================================================================================
      //=====================================================================================

      // Get the position of this clone in the global
      const Eigen::Matrix<double, 3, 3> &R_GtoCi = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).Rot();
      const Eigen::Matrix<double, 3, 1> &p_CiinG = clonesCAM.at(pair.first).at(feat->timestamps.at(pair.first).at(m)).pos();
      // Convert current position relative to anchor
      Eigen::Matrix<double, 3, 3> R_AtoCi;
      R_AtoCi.noalias() = R_GtoCi * R_GtoA.transpose();
      Eigen::Matrix<double, 3, 1> p_CiinA;
      p_CiinA.noalias() = R_GtoA * (p_CiinG - p_AinG);
      Eigen::Matrix<double, 3, 1> p_AinCi;
      p_AinCi.noalias() = -R_AtoCi * p_CiinA;

      //=====================================================================================
      //=====================================================================================

      // Middle variables of the system
      double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
      double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
      double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
      // Calculate residual
      Eigen::Matrix<float, 2, 1> z;
      z << hi1 / hi3, hi2 / hi3;
      Eigen::Matrix<float, 2, 1> res = feat->uvs_norm.at(pair.first).at(m) - z;
      // Append to our summation variables
      err += pow(res.norm(), 2);
    }
  }

  return err;
}