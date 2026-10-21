"""The 6-DoF output (relloc2 round 23, review23/o6maint/scripts/km6.py, variant ship1_6y): the peer's VIO world in the own one as a full
rotation R and translation t, instead of a rotation about gravity alone.

Why: the 4-DoF output takes the relative roll and pitch of the two VIO worlds as zero, so its tilt error is the two VIOs' tilt
disagreement. REFINE6 (the receiver's 6-DoF refinement, already in every solve) measures that tilt. Replay result (o6full, June flights):
lower e_1m in the first 30-60 s after take-off, while the VIOs disagree by 0.66-1.03 deg; no gain afterwards; none on TUM-VI / EuRoC (o6pub).

Definitions, copied from km6.py:
  solve        w6_closed: R_w6 = Rz(psi - yaw(M)) M, M = Ry(rel_pitch) Rx(rel_roll) Rz(-yB);  t_w6 = t_w4 + (Rz(psi) - R_w6) p_vB
               (psi, t_w4) = the solve's 4-DoF VIO-world estimate (rule15.vio_world_estimate), yB / p_vB = the peer's VIO yaw / position in
               its packet header, rel_roll / rel_pitch = REFINE6's relative tilt of the solve.
  lock         weighted chordal L2 mean of the lock members' R_w6 (weights = inliers), weighted mean of t_w6 (= rule15.fuse at zero tilt)
  update       M6: rule15.Maintenance with (R, t): same acceptance and z gate; innovation |R_v p + t_v - (R_e p + t_e)| at the peer's
               position p, withdraw above 0.20 m; rotation test 2 deg on the yaw difference only (gate 'yaw', the deployable test: the
               full-angle test tripled the withdrawals); blend R <- R_e Exp(0.7 Log(R_e^T R_v)), t <- 0.3 t_e + 0.7 t_v.
An estimate here is a 12-vector: R row-major, then t."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from . import paths
import rule15 as R15


def chordal(Rs, w):
    M = np.einsum('k,kij->ij', np.asarray(w, float), np.asarray(Rs, float))
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def w6_closed(v_yaw_deg, v_xyz, rr_deg, rp_deg, RvB_mat, pvB):
    """km6.w6_closed for one solve -> (R_w6 (3, 3), t_w6 (3,))."""
    rr = np.radians(np.nan_to_num(np.atleast_1d(np.asarray(rr_deg, float)), nan=0.0))
    rp = np.radians(np.nan_to_num(np.atleast_1d(np.asarray(rp_deg, float)), nan=0.0))
    RvB = Rot.from_matrix(np.asarray(RvB_mat, float).reshape(-1, 3, 3))
    yB = RvB.as_euler('ZYX')[:, 0]
    psi = np.radians(np.atleast_1d(np.asarray(v_yaw_deg, float)))
    M = Rot.from_euler('ZYX', np.stack([np.zeros_like(rr), rp, rr], 1)) * Rot.from_euler('z', -yB)
    Rw6 = Rot.from_euler('z', psi - M.as_euler('ZYX')[:, 0]) * M
    pvB = np.asarray(pvB, float).reshape(-1, 3)
    tw6 = np.asarray(v_xyz, float).reshape(-1, 3) + Rot.from_euler('z', psi).apply(pvB) - Rw6.apply(pvB)
    return Rw6.as_matrix()[0], tw6[0]


def w6_of_solve(peer_pose, v4, x6):
    """12-vector of one solve. peer_pose = (roll, pitch, yaw, x, y, z) of the peer's packet header, v4 = (v_yaw deg, v_x, v_y, v_z) of the
    same solve (rule15.vio_world_estimate), x6 = REFINE6's (roll, pitch, yaw, tx, ty, tz) in rad / m (None = a 4-DoF solve: zero tilt)."""
    rr, rp = (0.0, 0.0) if x6 is None else (float(np.degrees(x6[0])), float(np.degrees(x6[1])))
    RvB = Rot.from_euler('ZYX', [peer_pose[2], peer_pose[1], peer_pose[0]]).as_matrix()
    R, t = w6_closed(v4[0], v4[1:4], rr, rp, RvB, peer_pose[3:6])
    return np.concatenate([R.ravel(), t])


def lock6(members, est4=None):
    """the 6-DoF estimate of a new lock from its members (dicts with 'w6' and 'ninl'); members with a non-finite 'w6' are left out, and
    with none left the 4-DoF lock estimate est4 (yaw deg, x, y, z) is returned as a 6-DoF one (zero tilt)."""
    mem = [m for m in members if m.get('w6') is not None and np.isfinite(m['w6']).all()]
    if not mem:
        if est4 is None: raise ValueError('lock6: no member with a finite 6-DoF value')
        return np.concatenate([Rot.from_euler('z', est4[0], degrees=True).as_matrix().ravel(), np.asarray(est4[1:4], float)])
    w = np.array([m['ninl'] for m in mem], float)
    X = np.array([m['w6'] for m in mem], float)
    R = chordal(X[:, :9].reshape(-1, 3, 3), w)
    t = (X[:, 9:12] * w[:, None]).sum(0) / w.sum()
    return np.concatenate([R.ravel(), t])


def to4(e, p=None):
    """the 4-DoF transform (yaw deg, x, y, z) that maps the peer position p (its VIO world; None = origin) where the 6-DoF estimate e does."""
    R = np.asarray(e[:9], float).reshape(3, 3); t = np.asarray(e[9:12], float); p = np.zeros(3) if p is None else np.asarray(p, float)
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))); c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([yaw, *(R @ p + t - np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]))])


def tilt_deg(e):
    """angle between the 6-DoF rotation and a rotation about gravity (the relative tilt it carries), deg."""
    R = np.asarray(e[:9], float).reshape(3, 3); return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def _rotvec(R): return Rot.from_matrix(R).as_rotvec()
def _expm(r): return Rot.from_rotvec(r).as_matrix()


class M6(R15.Maintenance):
    """km6.M6 on rule15.Maintenance (the option-free ship1 path that kfsim's M16 runs without options): state est = 12-vector."""

    def __init__(self, blend='geo', gate='yaw', **kw):
        self.blend6 = blend; self.gate6 = gate
        super().__init__(**kw)

    @staticmethod
    def _at(e, p):
        e = np.asarray(e, float)
        return e[:9].reshape(3, 3) @ np.asarray(p, float) + e[9:12]

    def _ang(self, v):
        Rv = v[:9].reshape(3, 3); Re = self.est[:9].reshape(3, 3)
        if self.gate6 == 'full':
            return float(np.degrees(np.linalg.norm(_rotvec(Re.T @ Rv))))
        yv = np.degrees(np.arctan2(Rv[1, 0], Rv[0, 0])); ye = np.degrees(np.arctan2(Re[1, 0], Re[0, 0]))
        return abs((yv - ye + 180) % 360 - 180)

    def update(self, s, t, p_peer=None):
        if self.est is None or not R15._accepted(s, {'min_inl': s.get('min_inl', self.min_inl)}):
            return self.est
        v = np.asarray(s['w6'], float)
        if not np.isfinite(v).all(): return self.est
        p = np.zeros(3) if p_peer is None else np.asarray(p_peer, float)
        dy = self._ang(v)
        dtr = float(np.linalg.norm(self._at(v, p) - self._at(self.est, p)))
        gated = self.z_gate is None or R15.zmargin(s) >= self.z_gate
        self.innov = (self.innov + [dtr])[-10:]
        if gated:
            if self.wm is not None and (dtr > self.wm or dy > self.wd):
                self.withdrawn = True; self.est = None
                return None
            self._blend(v); self.t_gate = t; self.pending = None
        return self.est

    def _blend(self, v):
        Re = self.est[:9].reshape(3, 3); Rv = v[:9].reshape(3, 3); w = self.w
        Rn = Re @ _expm(w * _rotvec(Re.T @ Rv)) if self.blend6 == 'geo' else chordal(np.stack([Re, Rv]), [1.0 - w, w])
        tn = (1.0 - w) * self.est[9:12] + w * np.asarray(v[9:12], float)
        self.est = np.concatenate([Rn.ravel(), tn])
