/**
 * OvLogger — JSONL event dump for OpenVINS replay, mirroring the ORB-SLAM3
 * OrbslamLogger so the same Python rerun replay tool can render it.
 *
 * Enabled by env var OPENVINS_DUMP_PATH (no-op otherwise).
 *
 * OpenVINS doesn't have keyframes the same way ORB-SLAM3 does. We map:
 *   - clone insertion (new pose added to sliding window)  → "kf_create"
 *   - clone marginalization (oldest pose dropped)         → "kf_cull"
 *   - per-frame current pose                              → "frame"
 *   - SLAM feature creation                               → "mp_create"
 *   - SLAM feature deletion                               → "mp_destroy"
 *   - all features (MSCKF + SLAM) snapshot                → "mp_snap"
 *   - all alive clones                                    → "lba"  (the "window")
 * OpenVINS has no LC, so no lc_cand/loop_closed events.
 */
#pragma once
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace ov_msckf {

class OvLogger {
 public:
  static OvLogger& Get() {
    static OvLogger inst;
    return inst;
  }
  bool enabled() const { return fp_ != nullptr; }

  // Current pose (after each propagate+update)
  void log_frame(double ts, int frame_idx, const double twc[7],
                 const std::vector<long>& observed_feat_ids);

  // New clone inserted into the sliding window — OV analogue of KF create
  void log_clone_create(double ts, long clone_id, const double twc[7]);
  // Clone marginalized — analogue of KF cull
  void log_clone_cull(long clone_id);

  // SLAM feature created (just initialized into the state)
  void log_feat_create(long feat_id, const double pos[3], long clone_id);
  // SLAM feature destroyed (marginalized out or rejected)
  void log_feat_destroy(long feat_id);

  // Snapshot of all features currently in the filter (positions in world frame)
  void log_feat_snapshot(double ts, const std::vector<long>& ids,
                         const std::vector<double>& xyz);

  // Snapshot of all alive clones with their world poses
  void log_clone_snapshot(double ts, const std::vector<long>& ids,
                          const std::vector<double>& twc);

  // The "lba" event for OV is the current set of clones (no actual BA — but
  // matches the visualizer's expectation of which poses are active).
  void log_window(double ts, const std::vector<long>& clone_ids);

  void flush();

 private:
  OvLogger();
  ~OvLogger();
  OvLogger(const OvLogger&) = delete;
  OvLogger& operator=(const OvLogger&) = delete;
  std::FILE* fp_;
  std::mutex mu_;
};

}  // namespace ov_msckf
