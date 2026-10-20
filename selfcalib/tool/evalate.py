#!/usr/bin/env python3
"""Trajectory-accuracy evaluation for the tool (conventions of the basin study's eval_all.py,
plus the toff clock correction est_t += toff)."""
import numpy as np


def umeyama(s, d):
    mu_s, mu_d = s.mean(0), d.mean(0)
    sc, dc = s - mu_s, d - mu_d
    cov = sc.T @ dc / max(len(s), 1)
    U, S, V = np.linalg.svd(cov)
    D = np.diag([1, 1, np.linalg.det(U @ V)])
    R = (U @ D @ V).T
    return R, mu_d - R @ mu_s


def q_to_R(q):
    x, y, z, w = q
    s = 1.0 / np.sqrt(x * x + y * y + z * z + w * w + 1e-18)
    x, y, z, w = x * s, y * s, z * s, w * s
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def takeoff_time(gt):
    disp = np.linalg.norm(gt[:, 1:4] - gt[0, 1:4], axis=1)
    idx = np.where(disp > 0.10)[0]
    return gt[idx[0], 0] if len(idx) else gt[0, 0]


def load_aligned(est_path, gt_path, toff=0.0, cam_end=None):
    est = np.loadtxt(est_path)
    est = est[np.isfinite(est).all(axis=1)]
    est[:, 0] += toff
    gt = np.loadtxt(gt_path, comments='#')
    t_hi = min(est[-1, 0], gt[-1, 0])
    if cam_end:
        t_hi = min(t_hi, est[0, 0] + cam_end)
    m = (est[:, 0] >= max(est[0, 0], gt[0, 0])) & (est[:, 0] <= t_hi)
    te, pe, qe = est[m, 0], est[m, 1:4], est[m, 4:8]
    pg = np.column_stack([np.interp(te, gt[:, 0], gt[:, c]) for c in (1, 2, 3)])
    return te, pe, qe, pg, gt


# The start is pinned on the last second before takeoff, as vision_pose_node of swarm-nxt places the estimate (on the
# ground, from its last second of poses). The estimator's first poses can be far off while it starts: nxt3, 20 Oct 2026,
# first pose 1 m from where it settled 0.3 s later, the drone still, which made NOUM 1.1 m and the mount's tilt 6.7 deg.
PIN_S = 1.0


def pin_window(te, gt):
    """-> indices of the estimate's poses in the last second before takeoff; the first pose with fewer than 3 there."""
    t_to = takeoff_time(gt)
    w = np.where((te >= t_to - PIN_S) & (te < t_to))[0]
    return w if len(w) >= 3 else np.array([0])


def pin(te, pe, qe, pg, gt):
    """-> (R, p_e, p_g): over pin_window(), the rotation from the estimate's world to the room that the attitudes give
    (mean, projected onto a rotation) and the mean positions of the two."""
    w = pin_window(te, gt)
    idx = np.clip(np.searchsorted(gt[:, 0], te[w]), 0, len(gt) - 1)
    S = sum(q_to_R(gt[g, 4:8]) @ q_to_R(qe[i]).T for g, i in zip(idx, w))
    U, _, Vt = np.linalg.svd(S)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, pe[w].mean(axis=0), pg[w].mean(axis=0)


def ate_metrics(est_path, gt_path, toff=0.0, cam_end=None):
    """-> dict(global_m, postg_m, noum_m): global Umeyama; post-takeoff Umeyama (POSTG);
    no-Umeyama (NOUM: the GT poses of the last second before takeoff fix R,t, see pin()), after takeoff."""
    te, pe, qe, pg, gt = load_aligned(est_path, gt_path, toff, cam_end)
    if len(te) < 50:
        return {'global_m': float('nan'), 'postg_m': float('nan'), 'noum_m': float('nan'), 'n': int(len(te))}
    R, t = umeyama(pe, pg)
    g = float(np.sqrt(np.mean(np.linalg.norm((R @ pe.T).T + t - pg, axis=1) ** 2)))
    t_to = takeoff_time(gt)
    fl = te >= t_to
    postg = float('nan')
    if fl.sum() >= 20:
        R2, t2 = umeyama(pe[fl], pg[fl])
        postg = float(np.sqrt(np.mean(np.linalg.norm((R2 @ pe[fl].T).T + t2 - pg[fl], axis=1) ** 2)))
    Ra, pe0, pg0 = pin(te, pe, qe, pg, gt)
    ta = pg0 - Ra @ pe0
    m = fl if fl.sum() >= 20 else np.ones(len(te), bool)
    noum = float(np.sqrt(np.mean(np.linalg.norm((Ra @ pe[m].T).T + ta - pg[m], axis=1) ** 2)))
    return {'global_m': g, 'postg_m': postg, 'noum_m': noum, 'n': int(len(te))}
