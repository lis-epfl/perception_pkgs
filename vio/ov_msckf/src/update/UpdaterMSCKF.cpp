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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"
#include "PreJac.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include "utils/vprof.h"
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <sched.h>

namespace ov_msckf {
extern double g_dbg_sys_ms, g_dbg_cmp_ms, g_dbg_ekfu_ms;
extern long g_dbg_rows, g_dbg_cols;
extern double g_sp_alloc, g_sp_jac, g_sp_null, g_sp_marg, g_sp_chi2, g_sp_cpy, g_sp_stack, g_sp_rsz;
}
#include <chrono>
#include <mutex>
#include <vector>
namespace {
// OV_SYSPROF=1: sub-phase attribution inside UpdaterMSCKF step 4. Default OFF; when off the
// only cost is one predicted-not-taken branch per phase.
static const bool g_sysprof = [] { const char *e = std::getenv("OV_SYSPROF"); return e && *e == '1'; }();
// OV_MSCKF_SCRATCH=1 (default OFF): bit-exact execution changes in UpdaterMSCKF step 4 --
// dead zero-fill removal, per-thread scratch, moves instead of copies, exact-size Hx_big and a
// parallel two-pass stacking. No arithmetic operation, operand or summation order changes.
static const bool g_sc = [] { const char *e = std::getenv("OV_MSCKF_SCRATCH"); return e && *e == '1'; }();
static const bool g_sc_verify = [] { const char *e = std::getenv("OV_MSCKF_SCRATCH_VERIFY"); return e && *e == '1'; }();
static const int g_msckf_threads = [] { const char *e = std::getenv("OV_MSCKF_THREADS"); int v = e ? atoi(e) : 4; return v > 0 ? v : 4; }();
}
namespace ov_msckf {
extern thread_local bool g_msckf_scratch_tl;
// per-update scratch: P_marg (c x c), T = H_x*P_marg (m x c), S (m x m)
static thread_local std::vector<double> g_sc_pm, g_sc_t1, g_sc_s;
// stacking buffers (feed thread only)
static std::mutex g_scv_mtx;
static long g_scv_nfeat = 0, g_scv_nfeat_bad = 0, g_scv_nup = 0, g_scv_nup_bad = 0, g_scv_nord_bad = 0;
static double g_scv_maxdiff = 0.0;
static inline double *sc_buf(std::vector<double> &v, size_t n) {
  if (v.size() < n) v.resize(n);
  return v.data();
}
static inline double sp_now() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace ov_msckf
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

// OV_PREJAC_VERIFY helper: compare a cached entry against the freshly computed inline system.
static void prejac_verify_one(const PreJacEntry &e, const Eigen::MatrixXd &H_x, const Eigen::VectorXd &res,
                              const std::vector<std::shared_ptr<Type>> &Hx_order) {
  std::lock_guard<std::mutex> lk(g_pj_vmtx);
  g_pj_vn++;
  if (e.Hx_order.size() != Hx_order.size()) {
    g_pj_vbad_order++;
    return;
  }
  for (size_t i = 0; i < Hx_order.size(); i++) {
    if (e.Hx_order[i] != Hx_order[i]) {
      g_pj_vbad_order++;
      return;
    }
  }
  if (e.H_x.rows() != H_x.rows() || e.H_x.cols() != H_x.cols() || e.res.rows() != res.rows()) {
    g_pj_vbad_dim++;
    return;
  }
  double dH = (e.H_x - H_x).cwiseAbs().maxCoeff();
  double dr = (e.res - res).cwiseAbs().maxCoeff();
  double nH = H_x.cwiseAbs().maxCoeff(), nr = res.cwiseAbs().maxCoeff();
  if (dH > g_pj_vmax_H)
    g_pj_vmax_H = dH;
  if (dr > g_pj_vmax_r)
    g_pj_vmax_r = dr;
  if (nH > 0 && dH / nH > g_pj_vrel_H)
    g_pj_vrel_H = dH / nH;
  if (nr > 0 && dr / nr > g_pj_vrel_r)
    g_pj_vrel_r = dr / nr;
}

void UpdaterMSCKF::prejac_precompute(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Feature>> &feats, uint64_t epoch) {

  if (feats.empty())
    return;

  // Same clone timestamps / camera clone poses the consuming update will build. The state is
  // frozen for the whole of this call (see update/PreJac.h), so these are the exact values.
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  const int NF = (int)feats.size();
  std::vector<PreJacEntry> out((size_t)NF);
  std::vector<char> ok((size_t)NF, 0);
  // 2 threads: leave the other 2 of the 4-thread budget to the tracking we are overlapping.
  static const int pj_thr = [] { const char *e = std::getenv("OV_PREJAC_THREADS"); return e ? atoi(e) : 2; }();

#pragma omp parallel for schedule(dynamic, 1) num_threads(pj_thr)
  for (int fi = 0; fi < NF; fi++) {
    auto fp = feats[(size_t)fi];

    // Mirrors step 1 of update(): clean to clone times, then the ct_meas/anchor guards. If the
    // consumer would drop the feature there we simply publish nothing (it never triangulates).
    fp->clean_old_measurements(clonetimes);
    size_t ct = 0;
    for (const auto &pair : fp->timestamps) {
      ct += fp->timestamps[pair.first].size();
    }
    if (ct < 2)
      continue;
    if (fp->anchor_clone_timestamp > 0 && state->_clones_IMU.find(fp->anchor_clone_timestamp) == state->_clones_IMU.end())
      continue;

    PreJacEntry e;
    e.epoch = epoch;
    e.nobs = ct;

    // Mirrors step 3 EXACTLY, including that single_gaussnewton is attempted even when the
    // linear triangulation failed.
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(fp, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(fp, clones_cam);
    }
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(fp, clones_cam);
    }
    if (!success_tri || !success_refine) {
      e.tri_ok = false;
      out[(size_t)fi] = std::move(e);
      ok[(size_t)fi] = 1;
      continue;
    }
    e.tri_ok = true;
    e.anchor_cam_id = fp->anchor_cam_id;
    e.anchor_clone_timestamp = fp->anchor_clone_timestamp;
    e.p_FinA = fp->p_FinA;
    e.p_FinG = fp->p_FinG;

    // Mirrors step 4's jacobian + nullspace projection (chi2 stays with the consumer: it needs
    // the live covariance).
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = fp->featid;
    feat.uvs = fp->uvs;
    feat.uvs_norm = fp->uvs_norm;
    feat.timestamps = fp->timestamps;
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = fp->anchor_cam_id;
      feat.anchor_clone_timestamp = fp->anchor_clone_timestamp;
      feat.p_FinA = fp->p_FinA;
      feat.p_FinA_fej = fp->p_FinA;
    } else {
      feat.p_FinG = fp->p_FinG;
      feat.p_FinG_fej = fp->p_FinG;
    }
    Eigen::MatrixXd H_f;
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, e.H_x, e.res, e.Hx_order);
    UpdaterHelper::nullspace_project_inplace(H_f, e.H_x, e.res);
    out[(size_t)fi] = std::move(e);
    ok[(size_t)fi] = 1;
  }

  std::lock_guard<std::mutex> lk(g_prejac_mtx);
  for (int fi = 0; fi < NF; fi++) {
    if (ok[(size_t)fi])
      g_prejac[feats[(size_t)fi]->featid] = std::move(out[(size_t)fi]);
  }
  g_prejac_made.fetch_add((long)g_prejac.size(), std::memory_order_relaxed);
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // OV_PREJAC: take ownership of whatever the precompute worker left for us. An entry is only
  // usable while the state epoch is unchanged (no EKF update / clone / marginalization since
  // the snapshot) and the cleaned measurement count still matches -- otherwise we fall back to
  // the inline computation, so a missed invalidation can only cost time, never correctness.
  static const bool prejac = [] { const char *e = std::getenv("OV_PREJAC"); return e && *e == '1'; }();
  static const bool prejac_dbg = [] { const char *e = std::getenv("OV_PREJAC_DBG"); return e && *e == '1'; }();
  static const bool prejac_verify = [] { const char *e = std::getenv("OV_PREJAC_VERIFY"); return e && *e == '1'; }();
  std::unordered_map<size_t, PreJacEntry> pj;
  if (prejac) {
    std::lock_guard<std::mutex> lk(g_prejac_mtx);
    pj.swap(g_prejac);
  }
  const uint64_t pj_epoch_now = g_state_epoch.load(std::memory_order_relaxed);

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  VPROF("3.msckf/TOTAL");
  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  { VPROF("3.msckf/a_clean_meas");
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // OV_PREJAC validity gate (the ONLY thing standing between the cache and a wrong answer)
    if (!pj.empty()) {
      auto pit = pj.find((*it0)->featid);
      if (pit != pj.end() && (pit->second.epoch != pj_epoch_now || pit->second.nobs != (size_t)ct_meas))
        pj.erase(pit);
    }

    // Defensive guard: drop features whose anchor clone has been marginalized.
    // This can happen during multi-stereo out-of-order queue processing. Without this
    // guard, UpdaterHelper's `_clones_IMU.at(anchor_clone_timestamp)` throws.
    bool anchor_missing = false;
    if ((*it0)->anchor_clone_timestamp > 0 &&
        state->_clones_IMU.find((*it0)->anchor_clone_timestamp) == state->_clones_IMU.end()) {
      anchor_missing = true;
    }

    // Remove if we don't have enough or anchor is missing
    if (ct_meas < 2 || anchor_missing) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  { VPROF("3.msckf/b_clone_poses");
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  }
  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  { VPROF("3.msckf/c_triangulate");
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // OV_PREJAC hit: the worker already ran exactly this triangulation on identical inputs.
    if (!pj.empty() && !prejac_verify) {
      auto pit = pj.find((*it1)->featid);
      if (pit != pj.end()) {
        if (!pit->second.tri_ok) {
          (*it1)->to_delete = true;
          it1 = feature_vec.erase(it1);
          pj.erase(pit);
          continue;
        }
        (*it1)->anchor_cam_id = pit->second.anchor_cam_id;
        (*it1)->anchor_clone_timestamp = pit->second.anchor_clone_timestamp;
        (*it1)->p_FinA = pit->second.p_FinA;
        (*it1)->p_FinG = pit->second.p_FinG;
        it1++;
        continue;
      }
    }

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // OV_PREJAC_VERIFY: compare the worker's triangulation against this inline one.
    if (prejac_verify && !pj.empty()) {
      auto pit = pj.find((*it1)->featid);
      if (pit != pj.end()) {
        std::lock_guard<std::mutex> lk(g_pj_vmtx);
        if (pit->second.tri_ok != (success_tri && success_refine)) {
          g_pj_vbad_tri++;
        } else if (pit->second.tri_ok) {
          double d = (pit->second.p_FinG - (*it1)->p_FinG).cwiseAbs().maxCoeff();
          if (d > g_pj_vmax_p)
            g_pj_vmax_p = d;
        }
      }
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  static const bool omp_msckf_ = [] { const char *e = std::getenv("OV_OMP_MSCKF"); return e && *e == '1'; }();
  const bool big_exact = g_sc && omp_msckf_; // size Hx_big AFTER the counts are known
  double sp_t0 = g_sysprof ? sp_now() : 0.0;
  Eigen::VectorXd res_big;
  Eigen::MatrixXd Hx_big;
  if (!big_exact) {
    res_big = Eigen::VectorXd::Zero(max_meas_size);
    Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  }
  if (g_sysprof) g_sp_alloc += sp_now() - sp_t0;
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject.
  // Parallel phase: per-feature Jacobian/nullspace/chi2 are independent, read-only on the
  // state. Sequential phase: append in the ORIGINAL order so Hx ordering and id-based feature
  // selection are unchanged.
  static const bool omp_msckf = [] { const char *e = std::getenv("OV_OMP_MSCKF"); return e && *e == '1'; }();
  if (omp_msckf) {
    const int NF = (int)feature_vec.size();
    // schedule(dynamic,1): per-feature cost spans ~3 orders of magnitude and median NF is ~26,
    // so chunk=4 gives 6-7 chunks for 4 threads. Results are stored by index -> bit-exact.
    const int nthr_ = g_sc ? g_msckf_threads : 4;
    std::vector<Eigen::MatrixXd> vH(NF);
    std::vector<Eigen::VectorXd> vr(NF);
    std::vector<std::vector<std::shared_ptr<Type>>> vo(NF);
    std::vector<char> keep(NF, 0);
    // OV_LPT&2 (LPT): with schedule(dynamic,1) the makespan of a load-imbalanced loop is
    // minimised by issuing the LONGEST jobs first. Per-feature cost is superlinear in its
    // observation count (S is (2*obs-3)^2, P_marg is 6*clones squared), so the descending total
    // observation count is the natural cost proxy. This is UNOBSERVABLE in the result: every
    // output (keep/vH/vr/vo) is written at index fi, no iteration reads another's slot, and the
    // serial passes below still walk fi = 0..NF-1. Only the issue ORDER changes -- and which
    // thread runs which feature was already nondeterministic under schedule(dynamic,1).
    static const int g_lpt = [] { const char *e = std::getenv("OV_LPT"); return e ? atoi(e) : 0; }();
    std::vector<int> ord((size_t)NF);
    std::iota(ord.begin(), ord.end(), 0);
    if (g_lpt & 2) {
      std::vector<long> cost((size_t)NF, 0);
      for (int i = 0; i < NF; i++) {
        long n = 0;
        for (const auto &pr : feature_vec[i]->timestamps) n += (long)pr.second.size();
        cost[i] = n;
      }
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cost[a] > cost[b]; });
    }
    // OV_UPD_PROBE=N: for the first N updates, report what the OMP team ACTUALLY got -- team size
    // and the set of CPUs its members ran on. This is the only honest test that a wide mask
    // reached libgomp's cached worker pool.
    static const int g_probe = [] { const char *e = std::getenv("OV_UPD_PROBE"); return e ? atoi(e) : 0; }();
    static std::atomic<int> g_probe_n{0};
    if (g_probe > 0 && g_probe_n.fetch_add(1) < g_probe) {
      std::atomic<unsigned> cpumask{0};
      int got = 0;
#pragma omp parallel num_threads(nthr_)
      {
        cpumask.fetch_or(1u << (sched_getcpu() & 31));
#pragma omp master
        got = omp_get_num_threads();
      }
      printf("[updprobe] msckf req=%d got=%d NF=%d cpus=0x%02x\n", nthr_, got, NF, cpumask.load());
      fflush(stdout);
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthr_)
    for (int kk = 0; kk < NF; kk++) {
      const int fi = ord[(size_t)kk];
      auto &fp = feature_vec[fi];
      g_msckf_scratch_tl = g_sc;
      double a_jac = 0, a_null = 0, a_marg = 0, a_chi2 = 0, a_cpy = 0, tA = 0;
      if (g_sysprof) tA = sp_now();
      Eigen::MatrixXd H_x;
      Eigen::VectorXd res;
      std::vector<std::shared_ptr<Type>> Hx_order;
      auto pit = pj.find(fp->featid);
      if (pit != pj.end() && !prejac_verify) {
        // OV_PREJAC hit: bit-identical to the two calls in the else-branch below.
        H_x = std::move(pit->second.H_x);
        res = std::move(pit->second.res);
        Hx_order = std::move(pit->second.Hx_order);
        g_prejac_hit.fetch_add(1, std::memory_order_relaxed);
      } else {
      UpdaterHelper::UpdaterHelperFeature feat;
      feat.featid = fp->featid;
      feat.uvs = fp->uvs;
      feat.uvs_norm = fp->uvs_norm;
      feat.timestamps = fp->timestamps;
      feat.feat_representation = state->_options.feat_rep_msckf;
      if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)
        feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
      if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
        feat.anchor_cam_id = fp->anchor_cam_id;
        feat.anchor_clone_timestamp = fp->anchor_clone_timestamp;
        feat.p_FinA = fp->p_FinA;
        feat.p_FinA_fej = fp->p_FinA;
      } else {
        feat.p_FinG = fp->p_FinG;
        feat.p_FinG_fej = fp->p_FinG;
      }
      Eigen::MatrixXd H_f;
      UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);
      if (g_sysprof) { double t = sp_now(); a_jac += t - tA; tA = t; }
      UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);
      if (g_sysprof) { double t = sp_now(); a_null += t - tA; tA = t; }
      if (g_sc_verify) {
        // OV_MSCKF_SCRATCH_VERIFY: run the gate-OFF path on the SAME feature and memcmp.
        g_msckf_scratch_tl = false;
        Eigen::MatrixXd H_f0, H_x0;
        Eigen::VectorXd res0;
        std::vector<std::shared_ptr<Type>> Hx_order0;
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f0, H_x0, res0, Hx_order0);
        UpdaterHelper::nullspace_project_inplace(H_f0, H_x0, res0);
        g_msckf_scratch_tl = g_sc;
        bool ok = (H_x0.rows() == H_x.rows() && H_x0.cols() == H_x.cols() && res0.rows() == res.rows() &&
                   Hx_order0.size() == Hx_order.size());
        if (ok)
          for (size_t q = 0; q < Hx_order.size(); q++)
            ok = ok && (Hx_order0[q] == Hx_order[q]);
        if (ok)
          ok = (std::memcmp(H_x0.data(), H_x.data(), sizeof(double) * (size_t)H_x.size()) == 0) &&
               (std::memcmp(res0.data(), res.data(), sizeof(double) * (size_t)res.size()) == 0);
        double dmax = 0.0;
        if (H_x0.rows() == H_x.rows() && H_x0.cols() == H_x.cols())
          dmax = (H_x0 - H_x).cwiseAbs().maxCoeff();
        {
          std::lock_guard<std::mutex> lk(g_scv_mtx);
          g_scv_nfeat++;
          if (!ok) g_scv_nfeat_bad++;
          if (dmax > g_scv_maxdiff) g_scv_maxdiff = dmax;
        }
      }
      if (prejac_verify && pit != pj.end())
        prejac_verify_one(pit->second, H_x, res, Hx_order);
      }
      g_prejac_try.fetch_add(1, std::memory_order_relaxed);
      if (g_sysprof) tA = sp_now();
      double chi2 = 0.0;
      if (g_sc) {
        // STEP 2: per-thread monotonic scratch for the three transients. Same shapes, same
        // products, same order -- only the ADDRESS of the buffers changes.
        size_t cN = 0;
        for (const auto &var : Hx_order) cN += (size_t)var->size();
        const size_t mN = (size_t)H_x.rows();
        Eigen::Map<Eigen::MatrixXd> P_marg(sc_buf(g_sc_pm, cN * cN), (long)cN, (long)cN);
        StateHelper::get_marginal_covariance_into(state, Hx_order, P_marg);
        if (g_sysprof) { double t = sp_now(); a_marg += t - tA; tA = t; }
        Eigen::Map<Eigen::MatrixXd> T(sc_buf(g_sc_t1, mN * cN), (long)mN, (long)cN);
        Eigen::Map<Eigen::MatrixXd> S(sc_buf(g_sc_s, mN * mN), (long)mN, (long)mN);
        T.noalias() = H_x * P_marg;      // same two products, same order as (H_x*P_marg)*H_x^T
        S.noalias() = T * H_x.transpose();
        S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
        chi2 = res.dot(S.llt().solve(res));
      } else {
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        if (g_sysprof) { double t = sp_now(); a_marg += t - tA; tA = t; }
        Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
        S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
        chi2 = res.dot(S.llt().solve(res));
      }
      if (g_sc_verify) {
        Eigen::MatrixXd P0 = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S0 = H_x * P0 * H_x.transpose();
        S0.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S0.rows());
        double chi2_0 = res.dot(S0.llt().solve(res));
        std::lock_guard<std::mutex> lk(g_scv_mtx);
        if (std::memcmp(&chi2_0, &chi2, sizeof(double)) != 0) g_scv_nfeat_bad++;
      }
      double chi2_check = (res.rows() < 500) ? chi_squared_table[res.rows()]
                                             : boost::math::quantile(boost::math::chi_squared(res.rows()), 0.95);
      if (g_sysprof) { double t = sp_now(); a_chi2 += t - tA; tA = t; }
      if (chi2 <= _options.chi2_multipler * chi2_check) {
        keep[fi] = 1;
        if (g_sc) { // STEP 3a: nothing reads H_x/res/Hx_order after this in the iteration
          vH[fi] = std::move(H_x); vr[fi] = std::move(res); vo[fi] = std::move(Hx_order);
        } else {
          vH[fi] = H_x; vr[fi] = res; vo[fi] = Hx_order;
        }
      }
      if (g_sysprof) {
        a_cpy += sp_now() - tA;
#pragma omp atomic
        g_sp_jac += a_jac;
#pragma omp atomic
        g_sp_null += a_null;
#pragma omp atomic
        g_sp_marg += a_marg;
#pragma omp atomic
        g_sp_chi2 += a_chi2;
#pragma omp atomic
        g_sp_cpy += a_cpy;
      }
      g_msckf_scratch_tl = false;
    }
    double sp_t1 = g_sysprof ? sp_now() : 0.0;
    if (big_exact) {
      // ---- STEP 4b pass A: ordering and offsets only. O(NF*|Hx_order|), no matrix traffic.
      // Walks the kept features in exactly the same order as the serial loop below, so
      // Hx_mapping / Hx_order_big / ct_jacob / ct_meas come out identical.
      std::vector<size_t> row_off((size_t)NF, 0), col_beg((size_t)NF + 1, 0), col_off;
      col_off.reserve((size_t)NF * 16);
      for (int fi = 0; fi < NF; fi++) {
        if (!keep[fi]) {
          feature_vec[fi]->to_delete = true;
          col_beg[fi + 1] = col_off.size();
          continue;
        }
        row_off[fi] = ct_meas;
        for (const auto &var : vo[fi]) {
          auto it = Hx_mapping.find(var);
          if (it == Hx_mapping.end()) {
            it = Hx_mapping.insert({var, ct_jacob}).first;
            Hx_order_big.push_back(var);
            ct_jacob += var->size();
          }
          col_off.push_back(it->second);
        }
        col_beg[fi + 1] = col_off.size();
        ct_meas += (size_t)vr[fi].rows();
      }
      // ---- STEP 4a: exact-size, UNINITIALISED. Kills the max_meas_size x max_hx_size zeroed
      // allocation (43 MB on a whale -- above DEFAULT_MMAP_THRESHOLD_MAX so glibc always
      // mmaps/munmaps and always pays the demand-zero fault path) AND the conservativeResize
      // copy further down. Only ct_meas*ct_jacob is ever touched, and only once.
      res_big.resize((long)ct_meas);
      Hx_big.resize((long)ct_meas, (long)ct_jacob);
      // ---- STEP 4b pass B: row bands are disjoint -> no races, byte-identical result.
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthr_)
      for (int fi = 0; fi < NF; fi++) {
        if (!keep[fi])
          continue;
        const Eigen::MatrixXd &H = vH[fi];
        const long m = H.rows();
        Hx_big.block((long)row_off[fi], 0, m, (long)ct_jacob).setZero();
        size_t ct_hx = 0, k = col_beg[fi];
        for (const auto &var : vo[fi]) {
          Hx_big.block((long)row_off[fi], (long)col_off[k], m, var->size()) = H.block(0, (long)ct_hx, m, var->size());
          ct_hx += (size_t)var->size();
          k++;
        }
        res_big.segment((long)row_off[fi], m) = vr[fi];
      }
      if (g_sc_verify) {
        // Rebuild with the ORIGINAL serial algorithm from the same vH/vr/vo and memcmp.
        std::unordered_map<std::shared_ptr<Type>, size_t> M2;
        std::vector<std::shared_ptr<Type>> O2;
        size_t cj = 0, cm = 0;
        Eigen::VectorXd r2 = Eigen::VectorXd::Zero(max_meas_size);
        Eigen::MatrixXd H2 = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
        for (int fi = 0; fi < NF; fi++) {
          if (!keep[fi]) continue;
          size_t ct_hx = 0;
          for (const auto &var : vo[fi]) {
            if (M2.find(var) == M2.end()) { M2.insert({var, cj}); O2.push_back(var); cj += var->size(); }
            H2.block(cm, M2[var], vH[fi].rows(), var->size()) = vH[fi].block(0, ct_hx, vH[fi].rows(), var->size());
            ct_hx += var->size();
          }
          r2.block(cm, 0, vr[fi].rows(), 1) = vr[fi];
          cm += vr[fi].rows();
        }
        r2.conservativeResize(cm, 1);
        H2.conservativeResize(cm, cj);
        bool ok = (cm == ct_meas && cj == ct_jacob && O2.size() == Hx_order_big.size());
        if (ok)
          for (size_t q = 0; q < O2.size(); q++)
            if (O2[q] != Hx_order_big[q]) { ok = false; break; }
        bool ord_ok = ok;
        if (ok)
          ok = (std::memcmp(H2.data(), Hx_big.data(), sizeof(double) * (size_t)Hx_big.size()) == 0) &&
               (std::memcmp(r2.data(), res_big.data(), sizeof(double) * (size_t)res_big.size()) == 0);
        std::lock_guard<std::mutex> lk(g_scv_mtx);
        g_scv_nup++;
        if (!ok) g_scv_nup_bad++;
        if (!ord_ok) g_scv_nord_bad++;
      }
    } else {
    for (int fi = 0; fi < NF; fi++) {
      if (!keep[fi]) { feature_vec[fi]->to_delete = true; continue; }
      size_t ct_hx = 0;
      for (const auto &var : vo[fi]) {
        if (Hx_mapping.find(var) == Hx_mapping.end()) {
          Hx_mapping.insert({var, ct_jacob});
          Hx_order_big.push_back(var);
          ct_jacob += var->size();
        }
        Hx_big.block(ct_meas, Hx_mapping[var], vH[fi].rows(), var->size()) = vH[fi].block(0, ct_hx, vH[fi].rows(), var->size());
        ct_hx += var->size();
      }
      res_big.block(ct_meas, 0, vr[fi].rows(), 1) = vr[fi];
      ct_meas += vr[fi].rows();
    }
    }
    // features not kept were erased-equivalent: rebuild vec of kept for the cleanup below
    std::vector<std::shared_ptr<Feature>> kept;
    for (int fi = 0; fi < NF; fi++) if (keep[fi]) kept.push_back(feature_vec[fi]);
    feature_vec.swap(kept);
    if (g_sysprof) g_sp_stack += sp_now() - sp_t1;
    goto after_feature_loop;
  }
  {
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;
    auto pit2 = pj.find((*it2)->featid);
    if (pit2 != pj.end() && !prejac_verify) {
      // OV_PREJAC hit: bit-identical to the inline path below.
      H_x = std::move(pit2->second.H_x);
      res = std::move(pit2->second.res);
      Hx_order = std::move(pit2->second.Hx_order);
      g_prejac_hit.fetch_add(1, std::memory_order_relaxed);
    } else {
    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    Eigen::MatrixXd H_f;

    // Get the Jacobian for this feature
    { VPROF("3.msckf/d_jacobian");
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order); }

    // Nullspace project
    { VPROF("3.msckf/e_nullspace");
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res); }
    if (prejac_verify && pit2 != pj.end())
      prejac_verify_one(pit2->second, H_x, res, Hx_order);
    }
    g_prejac_try.fetch_add(1, std::memory_order_relaxed);

    /// Chi2 distance check
    double chi2;
    { VPROF("3.msckf/f_chi2");
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    chi2 = res.dot(S.llt().solve(res)); }

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  }
after_feature_loop:;
  rT3 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  double sp_t2 = g_sysprof ? sp_now() : 0.0;
  if (!big_exact) {
    res_big.conservativeResize(ct_meas, 1);
    Hx_big.conservativeResize(ct_meas, ct_jacob);
  }
  if (g_sysprof) g_sp_rsz += sp_now() - sp_t2;
  assert((long)Hx_big.rows() == (long)ct_meas && (long)Hx_big.cols() == (long)ct_jacob);

  // 5. Perform measurement compression
  const long nrows_precmp = (long)Hx_big.rows();
  { VPROF("3.msckf/g_compress");
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big); }
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // OV_CHUNK_ROWS: sequential chunked EKF update, exact for isotropic R (see
  // StateHelper::EKFUpdateTriChunked). Gated to the COMPRESSED updates only -- those are the
  // whales, and only they have the square upper-triangular shape the split exploits. The fat
  // uncompressed updates are ~97% of calls and would pay k-fold overhead for ~0.1 ms.
  // Also requires OV_CHOL_DOWN: without it the invariant downdate term is 2*N^2*m and the win halves.
  static const int chunk_rows = [] { const char *e = std::getenv("OV_CHUNK_ROWS"); return e ? atoi(e) : 0; }();
  static const bool chunk_ok = [] { const char *e = std::getenv("OV_CHOL_DOWN"); return e && atoi(e) > 0; }();
  const bool compressed = (nrows_precmp > (long)Hx_big.cols()); // compress ran => Hx_big is upper triangular

  // 6. With all good features update the state
  { VPROF("3.msckf/h_ekf_update");
  if (chunk_rows > 0 && chunk_ok && compressed && Hx_big.rows() == Hx_big.cols() && Hx_big.rows() >= 2 * chunk_rows) {
    StateHelper::EKFUpdateTriChunked(state, Hx_order_big, Hx_big, res_big, _options.sigma_pix_sq, chunk_rows);
  } else {
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, _options.sigma_pix_sq);
  } }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // slow-update attribution: per-update sub-phase totals, printed by VioManager's UpdTrace
  g_dbg_sys_ms += (rT3 - rT2).total_microseconds() * 1e-3;
  g_dbg_cmp_ms += (rT4 - rT3).total_microseconds() * 1e-3;
  g_dbg_ekfu_ms += (rT5 - rT4).total_microseconds() * 1e-3;
  if ((long)Hx_big.cols() > g_dbg_cols) g_dbg_cols = (long)Hx_big.cols();
  if (nrows_precmp > g_dbg_rows) g_dbg_rows = nrows_precmp;

  if (g_sc_verify) {
    static long nup2 = 0;
    if ((++nup2 % 50) == 0) {
      std::lock_guard<std::mutex> lk(g_scv_mtx);
      printf("[msckfscratch] feats %ld/%ld BITEXACT | stack %ld/%ld BITEXACT | order_bad=%ld | maxHdiff=%.3e\n",
             g_scv_nfeat - g_scv_nfeat_bad, g_scv_nfeat, g_scv_nup - g_scv_nup_bad, g_scv_nup, g_scv_nord_bad, g_scv_maxdiff);
      fflush(stdout);
    }
  }

  if (prejac_dbg) {
    static long nup = 0;
    if ((++nup % 100) == 0) {
      printf("[prejac] updates=%ld feats_seen=%ld hits=%ld made=%ld\n", nup, g_prejac_try.load(), g_prejac_hit.load(),
             g_prejac_made.load());
      if (prejac_verify)
        prejac_verify_report();
    }
  }

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
