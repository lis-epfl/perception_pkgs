/*
 * OV_PREJAC: off-critical-path precompute of the MSCKF per-feature linear system.
 *
 * With OV_SPREAD_UPD the four per-camera sub-updates of one clone tick are executed one per
 * frame at the ORIGINAL tick timestamp. Between the end of sub-update N and the start of
 * sub-update N+1 the filter state is completely frozen (no propagate/clone -- the queued
 * sub-update runs at a timestamp the state is already at -- no marginalization, no EKF update)
 * and the observation set that sub-update N+1 will use is frozen too (its features are already
 * "lost", so tracking cannot add to them, and any measurement tracking does add lands at a
 * non-clone timestamp and is discarded by clean_old_measurements).
 *
 * So triangulation + get_feature_jacobian_full + nullspace_project_inplace for those features
 * can be computed on a worker during the next frame's tracking window and simply handed to the
 * update. This is EXACT, not a deferral or an approximation: on a cache hit the consumer uses
 * H_x/res/Hx_order that are bit-identical to what it would have computed inline.
 *
 * Safety property: a cache entry is accepted ONLY if the state epoch is unchanged since the
 * snapshot AND the cleaned measurement count matches. Any state mutation we failed to
 * instrument can therefore only cause a cache MISS (fall back to the inline computation),
 * never a wrong answer.
 */

#ifndef OV_MSCKF_PREJAC_H
#define OV_MSCKF_PREJAC_H

#include <Eigen/Eigen>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "types/Type.h"

namespace ov_msckf {

struct PreJacEntry {
  uint64_t epoch = 0;   ///< g_state_epoch snapshotted when this entry was produced
  size_t nobs = 0;      ///< measurement count after clean_old_measurements(clonetimes)
  bool tri_ok = false;  ///< single_triangulation (+ single_gaussnewton) succeeded
  int anchor_cam_id = -1;
  double anchor_clone_timestamp = -1;
  Eigen::Vector3d p_FinA = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
  Eigen::MatrixXd H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<ov_type::Type>> Hx_order;
};

/// featid -> precomputed system for the next queued sub-update.
extern std::unordered_map<size_t, PreJacEntry> g_prejac;
extern std::mutex g_prejac_mtx;

/// Bumped by EVERY StateHelper mutation of the filter state (mean, covariance, dimension or
/// clone set). A stale epoch invalidates the cache.
extern std::atomic<uint64_t> g_state_epoch;

/// Diagnostics (OV_PREJAC_DBG=1).
extern std::atomic<long> g_prejac_hit, g_prejac_try, g_prejac_made;

/// OV_PREJAC_VERIFY=1: take NO shortcut, compute everything inline as if the cache did not
/// exist, and compare the cached values against it. This proves the exactness claim directly,
/// which an end-to-end trajectory comparison cannot (the pipeline is not run-to-run
/// deterministic). Accumulated worst-case deviations, reported at the end of the run.
extern std::mutex g_pj_vmtx;
extern double g_pj_vmax_H, g_pj_vmax_r, g_pj_vmax_p, g_pj_vrel_H, g_pj_vrel_r;
extern long g_pj_vn, g_pj_vbad_order, g_pj_vbad_tri, g_pj_vbad_dim;
void prejac_verify_report();

} // namespace ov_msckf

#endif // OV_MSCKF_PREJAC_H
