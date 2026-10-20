#!/usr/bin/env python3
"""One-time mount calibration (paper §6.2): solve the constant body/mount rotation that separates
the alignment anchored before takeoff from the best-fit alignment, on ONE flight; apply it unchanged to
future flights so the *unaligned* (anchored) trajectory is accurate without per-flight alignment.
Anchored on the last second before takeoff (evalate.pin), as vision_pose_node places the estimate: the
estimator's first poses can be far off while it starts.

Also the lever arm (lever_arm_imu_m): the IMU's position from the origin of the motion capture's rigid body, in the
IMU frame. The estimator tracks the IMU and the motion capture the origin, so without it the two differ by the lever
arm turned with the attitude (about 4 cm in the hard turns of nxt3, 15 Oct 2026). vision_pose_node of swarm-nxt
moves the estimate back by it onto the origin.

Also the room tilt (room_tilt_deg): the estimator's vertical as the motion capture's room sees it. vision_pose_node
places the estimate with a heading only, which takes the two verticals for one, and turns it by this tilt on top.

solve():  M = R_fit @ R_anchor^T on flight A  (dominantly a yaw offset on this platform)
apply():  on flight B, use R = M @ R_anchor_B — no fitting against B's ground truth.
The cross-flight test (docs/VALIDATION_MATRIX.md) is the falsifiable claim: if M were not constant,
applying A's M to B would not help."""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalate import umeyama, q_to_R, takeoff_time, load_aligned, pin, pin_window


def _anchored(te, pe, qe, pg, gt):
    return pin(te, pe, qe, pg, gt)[0]


def _rmse_with_R(R, te, pe, pg, gt, post_takeoff=True):
    w = pin_window(te, gt)
    t = pg[w].mean(axis=0) - R @ pe[w].mean(axis=0)
    pea = (R @ pe.T).T + t
    if post_takeoff:
        m = te >= takeoff_time(gt)
        if m.sum() < 20:
            m = np.ones(len(te), bool)
        return float(np.sqrt(np.mean(np.linalg.norm(pea[m] - pg[m], axis=1) ** 2)))
    return float(np.sqrt(np.mean(np.linalg.norm(pea - pg, axis=1) ** 2)))


# The lever arm: the estimator tracks the IMU, the motion capture the origin of the rigid body, and the two points differ
# by a constant vector in the body. It shows only where the attitude changes, so it is solved only on a flight that tilts,
# and written only when the three thirds of the flight agree on it.
LEVER_MIN_TILT_DEG = 15.0   # 90th percentile of the rigid body's tilt in flight
LEVER_MAX_SPREAD_M = 0.03   # largest distance of a third's lever arm from the whole flight's (calibrations of nxt3: 1.5-1.6 cm)


def _lever_fit(pe, Re, pg):
    """IMU position from the rigid body's origin, in the IMU frame, and the RMS left: the origin is p - Re l in the
    estimate's world, aligned rigidly to the motion capture. Alternates the alignment and the (linear) lever arm."""
    l = np.zeros(3)
    for _ in range(200):
        R, t = umeyama(pe - np.einsum('nij,j->ni', Re, l), pg)
        # R (p - Re l) + t = g   ->   Re l = p - R^T (g - t)
        l_new = np.mean(np.einsum('nji,nj->ni', Re, pe - (pg - t) @ R), axis=0)
        done = np.linalg.norm(l_new - l) < 1e-7
        l = l_new
        if done:
            break
    R, t = umeyama(pe - np.einsum('nij,j->ni', Re, l), pg)
    res = (pe - np.einsum('nij,j->ni', Re, l)) @ R.T + t - pg
    return l, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def lever_arm(te, pe, qe, pg, gt):
    """-> dict: lever_arm_imu_m (only when determined), the fit, its spread over the thirds of the flight, the RMS
    without and with it, the tilt that made it observable, and why it was left out (lever_arm_skipped)."""
    fl = te >= takeoff_time(gt)
    p, g = pe[fl], pg[fl]
    Re = np.array([q_to_R(q) for q in qe[fl]])
    idx = np.clip(np.searchsorted(gt[:, 0], te[fl]), 0, len(gt) - 1)
    tilt = np.degrees(np.arccos(np.clip([q_to_R(q)[2, 2] for q in gt[idx, 4:8]], -1.0, 1.0))) if fl.any() else np.zeros(1)
    out = {'lever_arm_tilt_p90_deg': round(float(np.percentile(tilt, 90)), 1)}
    if len(p) < 150:
        out['lever_arm_skipped'] = 'too little flight (%d poses)' % len(p)
        return out
    l, rms = _lever_fit(p, Re, g)
    R0, t0 = umeyama(p, g)
    rms0 = float(np.sqrt(np.mean(np.sum((p @ R0.T + t0 - g) ** 2, axis=1))))
    thirds = [_lever_fit(p[s], Re[s], g[s])[0] for s in np.array_split(np.arange(len(p)), 3)]
    spread = max(float(np.linalg.norm(x - l)) for x in thirds)
    out.update({'lever_arm_fit_m': [round(float(v), 4) for v in l], 'lever_arm_spread_m': round(spread, 4),
                'lever_arm_rms_m': [round(rms0, 4), round(rms, 4)]})
    if out['lever_arm_tilt_p90_deg'] < LEVER_MIN_TILT_DEG:
        out['lever_arm_skipped'] = ('the flight tilts too little to see it (90th percentile %.1f deg, %.0f needed)'
                                    % (out['lever_arm_tilt_p90_deg'], LEVER_MIN_TILT_DEG))
    elif spread > LEVER_MAX_SPREAD_M:
        out['lever_arm_skipped'] = ('the thirds of the flight disagree on it by up to %.1f cm (%.0f allowed)'
                                    % (100 * spread, 100 * LEVER_MAX_SPREAD_M))
    else:
        out['lever_arm_imu_m'] = out['lever_arm_fit_m']
    return out


# The room tilt: the rotation about the room's x and y axes (deg) that takes the room's vertical to the estimator's, from
# the best rigid fit of the flight's positions (moved by the lever arm when it is written, as vision_pose_node moves
# them), so it holds whatever the attitudes and the mount. On the flights of 19/20 Oct 2026 it was 0.5-1.1 deg, partly
# the room (the ground plane set in Motive) and partly the drone (nxt6 0.4 deg more than nxt3); placed without it, the
# estimate's height is off by about 1.4 cm per metre from the start.
def room_tilt(te, pe, qe, pg, gt, lever=None):
    """-> dict: room_tilt_deg ([x, y])."""
    fl = te >= takeoff_time(gt)
    p, g = pe[fl], pg[fl]
    if lever is not None:
        p = p - np.einsum('nij,j->ni', np.array([q_to_R(q) for q in qe[fl]]), np.asarray(lever, float))
    R, _ = umeyama(p, g)
    up = R[:, 2]                        # the estimator's vertical, in the room
    angle = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
    axis = np.cross([0.0, 0.0, 1.0], up)
    v = axis / np.linalg.norm(axis) * angle if angle > 0.0 else np.zeros(3)
    return {'room_tilt_deg': [round(float(np.degrees(v[0])), 3), round(float(np.degrees(v[1])), 3)]}


def solve(est_path, gt_path, toff=0.0, cam_end=None):
    """-> dict with M (3x3), mount_yaw_deg, mount_tilt_deg, and flight-A errors."""
    te, pe, qe, pg, gt = load_aligned(est_path, gt_path, toff, cam_end)
    R_anchor = _anchored(te, pe, qe, pg, gt)
    fl = te >= takeoff_time(gt)
    R_fit, _ = umeyama(pe[fl], pg[fl])
    M = R_fit @ R_anchor.T
    yaw = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))))
    out = {'M': M.tolist(), 'mount_yaw_deg': round(yaw, 3), 'mount_tilt_deg': round(tilt, 3),
           'A_anchored_m': round(_rmse_with_R(R_anchor, te, pe, pg, gt), 4),
           'A_mounted_m': round(_rmse_with_R(M @ R_anchor, te, pe, pg, gt), 4),
           'A_bestfit_m': round(_rmse_with_R(R_fit, te, pe, pg, gt), 4)}
    out.update(lever_arm(te, pe, qe, pg, gt))
    out.update(room_tilt(te, pe, qe, pg, gt, out.get('lever_arm_imu_m')))
    return out


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
