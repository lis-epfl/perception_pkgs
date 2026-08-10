#!/usr/bin/env python3
"""One-time mount calibration (paper §6.2): solve the constant body/mount rotation that separates
the start-anchored alignment from the best-fit alignment, on ONE flight; apply it unchanged to
future flights so the *unaligned* (anchored) trajectory is accurate without per-flight alignment.

solve():  M = R_fit @ R_anchor^T on flight A  (dominantly a yaw offset on this platform)
apply():  on flight B, use R = M @ R_anchor_B — no fitting against B's ground truth.
The cross-flight test (docs/VALIDATION_MATRIX.md) is the falsifiable claim: if M were not constant,
applying A's M to B would not help."""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalate import umeyama, q_to_R, takeoff_time, load_aligned


def _anchored(te, pe, qe, pg, gt):
    idx = np.clip(np.searchsorted(gt[:, 0], te[0]), 0, len(gt) - 1)
    R_anchor = q_to_R(gt[idx, 4:8]) @ q_to_R(qe[0]).T
    return R_anchor


def _rmse_with_R(R, te, pe, pg, gt, post_takeoff=True):
    t = pg[0] - R @ pe[0]
    pea = (R @ pe.T).T + t
    if post_takeoff:
        m = te >= takeoff_time(gt)
        if m.sum() < 20:
            m = np.ones(len(te), bool)
        return float(np.sqrt(np.mean(np.linalg.norm(pea[m] - pg[m], axis=1) ** 2)))
    return float(np.sqrt(np.mean(np.linalg.norm(pea - pg, axis=1) ** 2)))


def solve(est_path, gt_path, toff=0.0, cam_end=None):
    """-> dict with M (3x3), mount_yaw_deg, mount_tilt_deg, and flight-A errors."""
    te, pe, qe, pg, gt = load_aligned(est_path, gt_path, toff, cam_end)
    R_anchor = _anchored(te, pe, qe, pg, gt)
    fl = te >= takeoff_time(gt)
    R_fit, _ = umeyama(pe[fl], pg[fl])
    M = R_fit @ R_anchor.T
    yaw = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))))
    return {'M': M.tolist(), 'mount_yaw_deg': round(yaw, 3), 'mount_tilt_deg': round(tilt, 3),
            'A_anchored_m': round(_rmse_with_R(R_anchor, te, pe, pg, gt), 4),
            'A_mounted_m': round(_rmse_with_R(M @ R_anchor, te, pe, pg, gt), 4),
            'A_bestfit_m': round(_rmse_with_R(R_fit, te, pe, pg, gt), 4)}


def apply(est_path, gt_path, M, toff=0.0, cam_end=None):
    """Apply flight-A's mount correction to flight B (no fitting on B). -> before/after/reference."""
    te, pe, qe, pg, gt = load_aligned(est_path, gt_path, toff, cam_end)
    R_anchor = _anchored(te, pe, qe, pg, gt)
    fl = te >= takeoff_time(gt)
    R_fit, _ = umeyama(pe[fl], pg[fl])
    M = np.asarray(M)
    return {'B_anchored_m': round(_rmse_with_R(R_anchor, te, pe, pg, gt), 4),
            'B_mounted_m': round(_rmse_with_R(M @ R_anchor, te, pe, pg, gt), 4),
            'B_bestfit_m': round(_rmse_with_R(R_fit, te, pe, pg, gt), 4)}


if __name__ == '__main__':
    a, gta, b, gtb = sys.argv[1:5]
    toff = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    s = solve(a, gta, toff)
    print('solve on A:', {k: v for k, v in s.items() if k != 'M'})
    print('apply on B:', apply(b, gtb, s['M'], toff))
