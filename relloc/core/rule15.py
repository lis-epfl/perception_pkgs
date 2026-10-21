"""Round 15: the ONE implementation of the General Route lock rule, shared by the offline replay (replay.py) and the flight node.
Dependency-free apart from numpy (and scipy only for vio_world_estimate). The functions zmargin / _accepted / _locked / _fw / _ft are moved
verbatim from replay.py (round 14), so every replayed number is unchanged (golden test: test_rule15.py). FINAL is the deployed rule.
Streaming use (flight node): PeerLock.add(solve) after every accepted solve against one peer; it re-evaluates the rule on all accepted solves
since the search started, exactly as the replay does, and returns the fused estimate of the peer's VIO world in the own VIO world.
Known quirk kept for parity (review 09-28): the solo tiers are tested only on the member with the most agreeing solves / inliers (argmax);
rule key 'anysolo': True tests every accepted solve (measured within RANSAC-seed noise, so not the default)."""
import numpy as np

# round-14 rule (fp16-ray receiver; the golden test replays it). The deployed round-15 rules are RULES15 below.
FINAL = {'dmode': 'A', 'dy': 1.0, 'dt': 0.07, 'solo': [(340, 0.15), (225, 0.10), (135, 0.06)], 'solo_zmin': 2.5, 'z_cons': 1.5, 'amb_cons': None, 'min_inl': 57, 'chain': 'vio'}
# round 15: exact unit rays and a per-rig test threshold (rig15.RIGS thr_px) change every per-solve statistic, so each rig's rule is the
# round-14 rule carried by rank on that rig's tuned pairs (transfer15.py: the same solves' inlier / sigma_yaw / margin quantiles; no
# re-tuning on evaluated data). drones: 9 forward pairs at 7.5 px; tumvi: room2 x room3 at 1.5 px; euroc: V1_02 x V1_03 at 1.5 px;
# uzh (sealed test): the TUM-VI rule unchanged. = cache_md/final15f_cfg.json 'rules' (base config: final15_cfg.json).
_R15 = dict(dmode='A', dy=1.0, dt=0.07, amb_cons=None, chain='vio')
RULES15 = {   # final configuration (REFINE_Q=0), transferred on the _ws15f0 tuned pairs (cache_md/final15f_cfg.json)
    'drones': dict(_R15, solo=[(807, 0.3079), (605, 0.209), (398, 0.1436)], solo_zmin=4.21, z_cons=2.7, min_inl=176),
    'tumvi': dict(_R15, solo=[(222, 0.2676), (140, 0.1826), (82, 0.1038)], solo_zmin=3.43, z_cons=2.15, min_inl=33),
    'euroc': dict(_R15, solo=[(109, 0.5498), (69, 0.4362), (41, 0.3199)], solo_zmin=3.16, z_cons=1.94, min_inl=30),
}
RULES15['uzh'] = dict(RULES15['tumvi'])
RIG_OF = {'drone4': 'drones', 'tumvi': 'tumvi', 'euroc': 'euroc', 'uzh': 'uzh'}   # rig15.RIGS name -> rule

def zmargin(s, col='amb'):
    """margin of the best RANSAC mode over the best distinct mode (> 3 deg or 30 cm away; other radii in the amb_* columns of AMB_EXTRA dumps)
    in count standard deviations: (best - second) / sqrt(best + second), from the raw inlier count and the ambiguity ratio second / best."""
    b = float(s['ninl_raw']); a = float(s[col]) if col in s and np.isfinite(s[col]) else 1.0
    return b * (1.0 - a) / np.sqrt(max(b * (1.0 + a), 1.0))
def _accepted(s, rule):
    ok = s['ok'] > 0 and s['ninl'] >= rule.get('min_inl', 50) and np.isfinite(s['h_x'])
    if ok and rule.get('tilt_max') is not None and 'rel_roll' in s:   # round 14: a large estimated relative tilt flags a poor gravity estimate
        ok = bool(np.hypot(s['rel_roll'], s['rel_pitch']) <= rule['tilt_max'])
    return ok
def _locked(ev, rule):
    """returns (locked, index of the best cluster member, agree matrix row mask) for the event list ev."""
    m = len(ev)
    if m == 0: return False, -1, None
    ck = ('v_yaw', 'v_x', 'v_y', 'v_z') if rule.get('chain') == 'vio' else ('h_yaw', 'h_x', 'h_y', 'h_z')   # round 14: consensus through the drones' VIO (deployable) or the mocap chain
    Hb = np.array([[e[k] for k in ck] for e in ev]); nb = np.array([e['ninl'] for e in ev]); ij = np.array([(e['i'], e['j']) for e in ev])
    syb = np.array([e['sy'] for e in ev]); amb = np.array([e['amb'] for e in ev]); stb = np.array([e['st'] for e in ev])
    zb = np.array([zmargin(e, rule.get('amb_col', 'amb')) for e in ev])
    if rule.get('amb_col2'):   # round 14 test: a second margin (e.g. pose-radius ambiguity next to the inlier-overlap one), own thresholds
        z2 = np.array([zmargin(e, rule['amb_col2']) for e in ev]); zb = np.where((z2 >= rule.get('z_cons2', -1e9)), zb, -1e9)
        solo2 = z2 >= rule.get('solo_zmin2', -1e9)
    else: solo2 = np.ones(len(ev), bool)
    dy = np.abs((Hb[:, 0:1] - Hb[None, :, 0] + 180) % 360 - 180); dtv = np.linalg.norm(Hb[:, None, 1:4] - Hb[None, :, 1:4], axis=2); agree = (dy < rule['dy']) & (dtv < rule['dt'])
    dm_ = rule['dmode']
    if dm_ == 'both': same = (ij[:, 0:1] == ij[None, :, 0]) | (ij[:, 1:2] == ij[None, :, 1])
    elif dm_ == 'B': same = ij[:, 1:2] == ij[None, :, 1]                                   # round 14 test: distinct received keyframes
    elif dm_ == 'any': same = (ij[:, 0:1] == ij[None, :, 0]) & (ij[:, 1:2] == ij[None, :, 1])   # any two distinct solves
    else: same = ij[:, 0:1] == ij[None, :, 0]
    agree = agree & ~(same & ~np.eye(m, dtype=bool))
    if rule.get('sep') and all('tA' in e for e in ev):   # round 14 test: consensus members from own keyframes at least sep seconds apart (distinct viewpoints)
        tA_ = np.array([e['tA'] for e in ev]); agree = agree & ((np.abs(tA_[:, None] - tA_[None, :]) >= rule['sep']) | np.eye(m, dtype=bool))
    if rule.get('amb_cons') is not None:
        okc = np.nan_to_num(amb, nan=1.0) <= rule['amb_cons']; agree = agree & okc[:, None] & okc[None, :] | np.eye(m, dtype=bool)
    if rule.get('z_cons') is not None:           # round 14: consensus members must win their RANSAC by z_cons count-sigmas over the best distinct mode
        okc = zb >= rule['z_cons']; agree = agree & okc[:, None] & okc[None, :] | np.eye(m, dtype=bool)
    cnt = agree.sum(1); b = int(np.argmax(cnt * 1000 + nb))
    if rule.get('anysolo') and cnt[b] < 2:   # round 15 option: test the solo tiers on every accepted solve, fuse around the best qualifying one
        for k in np.argsort(-nb):
            if any(nb[k] >= t_[0] and syb[k] <= t_[1] for t_ in rule['solo']) and zb[k] >= rule.get('solo_zmin', -1.0): b = int(k); break
    if rule.get('z_sum') is not None:            # round 14: evidence rule — lock when the agreeing solves' margins sum to z_sum (one solve alone if its margin does)
        zc = np.clip(zb, 0, rule.get('z_cap', np.inf)); Sg = agree.astype(float) @ zc; b = int(np.argmax(Sg))
        return bool(Sg[b] >= rule['z_sum'] and cnt[b] >= rule.get('z_min_members', 1)), b, agree[b]
    solo = False
    for tier in rule['solo']:
        n_, s_ = tier[0], tier[1]; a_ = tier[2] if len(tier) > 2 else None; t_ = tier[3] if len(tier) > 3 else None
        if nb[b] >= n_ and syb[b] <= s_ and (a_ is None or (np.isfinite(amb[b]) and amb[b] <= a_)) and (t_ is None or stb[b] <= t_) and zb[b] >= rule.get('solo_zmin', -1.0) and solo2[b]: solo = True; break
    for tier in rule.get('z_solo', []):         # round 14: solo lock on the margin alone: (z, max sigma_yaw deg)
        z_, s_ = tier if isinstance(tier, (tuple, list)) else (tier, None)
        if zb[b] >= z_ and (s_ is None or syb[b] <= s_): solo = True; break
    if rule.get('warm_solo') is not None and solo:   # round 14: a lone solve on packets from a sender still settling after take-off does not lock
        agb = ev[b]['ageB'] if 'ageB' in ev[b] else 1e9; aga = ev[b]['ageA'] if 'ageA' in ev[b] else 1e9
        if agb < rule['warm_solo'] or (rule.get('warm_both') and aga < rule['warm_solo']): solo = False
    if rule.get('warm_n') is not None and (cnt[b] >= 2 or solo):   # settling sender: lock only on a larger consensus (warm_n agreeing solves)
        mem = np.where(agree[b])[0]
        if all(ev[k].get('ageB', 1e9) < rule['warm_s'] for k in mem):
            solo = False
            if cnt[b] < rule['warm_n']: cnt = cnt.copy(); cnt[b] = 1
    if rule.get('warm_cons') is not None and cnt[b] >= 2:   # consensus needs at least one member from a settled sender
        mem = np.where(agree[b])[0]
        if not any(ev[k].get('ageB', 1e9) >= rule['warm_cons'] and (not rule.get('warm_both') or ev[k].get('ageA', 1e9) >= rule['warm_cons']) for k in mem): cnt = cnt.copy(); cnt[b] = 1
    return (cnt[b] >= 2 or solo), b, agree[b]
def _fw(ev, sel, rule):
    """fusion weights of the agreeing solves: 'inl' inlier count (rounds 9-13), 'sig' inverse translation variance of the solve, 'one' equal."""
    mode = rule.get('fuse', 'inl')
    if mode == 'sig': return np.array([1.0 / max(ev[k]['st'], 1e-3) ** 2 if np.isfinite(ev[k]['st']) else 1.0 for k in sel], float)
    if mode == 'one': return np.ones(len(sel))
    return np.array([ev[k]['ninl'] for k in sel], float)
def _ft(Ht, w, rule):
    if rule.get('fuse_t', 'mean') == 'median' and len(Ht) >= 3: return np.median(Ht, axis=0)
    return (Ht * w[:, None]).sum(0) / w.sum()


def vio_world_estimate(own, peer, x6):
    """estimate of the peer's VIO world in the own VIO world from one solve. own / peer: VIO pose of each drone's rig (body) frame at the two
    keyframes, (roll, pitch, yaw, x, y, z) in its own VIO world (the packet header carries the peer's); x6: the solve (roll, pitch, yaw, tx,
    ty, tz) = rig B in gravity-aligned rig A (relative roll/pitch from the 6-DoF refinement, 0 for a 4-DoF solve). Returns
    (v_yaw deg, v_x, v_y, v_z) as regrade.body_grade stores it: VIO world B -> VIO world A, yaw about gravity."""
    from scipy.spatial.transform import Rotation as Rot
    RvA = Rot.from_euler('ZYX', [own[2], own[1], own[0]]); RvB = Rot.from_euler('ZYX', [peer[2], peer[1], peer[0]])
    tA = Rot.from_euler('ZYX', [0.0, own[1], own[0]]); tB = Rot.from_euler('ZYX', [0.0, peer[1], peer[0]])
    R6 = Rot.from_euler('ZYX', [x6[2], x6[1], x6[0]]); te = np.asarray(x6[3:6], float)
    Rbb = tA.inv() * R6 * tB; tbb = tA.inv().apply(te)
    Rw = RvA * Rbb * RvB.inv(); yw = Rw.as_euler('ZYX')[0]   # the VIO worlds share gravity: keep the yaw, and the translation consistent with it
    tw = np.asarray(own[3:6], float) + RvA.apply(tbb) - Rot.from_euler('z', yw).apply(np.asarray(peer[3:6], float))
    return float(np.degrees(yw)), float(tw[0]), float(tw[1]), float(tw[2])

def fuse(ev, sel, rule):
    """fused estimate (yaw deg, x, y, z) of the agreeing solves sel (inlier-weighted circular mean of yaw, weighted mean of t)."""
    w = _fw(ev, sel, rule); ck = ('v_yaw', 'v_x', 'v_y', 'v_z') if rule.get('chain') == 'vio' else ('h_yaw', 'h_x', 'h_y', 'h_z')
    a = np.radians(np.array([ev[k][ck[0]] for k in sel])); yaw = float(np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())))
    t = _ft(np.array([[ev[k][ck[1]], ev[k][ck[2]], ev[k][ck[3]]] for k in sel]), w, rule)
    return np.array([yaw, *t])

class PeerLock:
    """lock state against one peer. add(solve) with a dict carrying at least ok, ninl, ninl_raw, amb, sy, st, i (own keyframe id), j (peer
    keyframe id), h_x (finite for a valid pose), v_yaw, v_x, v_y, v_z (vio_world_estimate) and t (time). Returns None until locked, then the
    fused estimate (yaw deg, x, y, z) and the members. reset() on a peer VIO epoch change or after the maintenance withdraws the estimate."""
    # round 16 (REVIEW15 S1): the search keeps a sliding window of accepted solves (the last max_ev, none older than max_age s), so a peer
    # that never locks costs O(max_ev^2) per add instead of growing without bound; the replay's 5-s horizon never reaches either limit.
    def __init__(self, rule=FINAL, max_ev=300, max_age=60.0):
        self.rule = rule; self.ev = []; self.locked = False; self.estimate = None; self.members = []; self.max_ev = max_ev; self.max_age = max_age
    def reset(self):
        self.ev = []; self.locked = False; self.estimate = None; self.members = []
    def add(self, s):
        if self.locked or not _accepted(s, self.rule): return self.estimate if self.locked else None
        self.ev.append(dict(s))
        if self.max_age is not None and 't' in s: self.ev = [e for e in self.ev if e.get('t', s['t']) >= s['t'] - self.max_age]
        if self.max_ev is not None and len(self.ev) > self.max_ev: self.ev = self.ev[-self.max_ev:]
        ok, b, ag = _locked(self.ev, self.rule)
        if not ok: return None
        sel = np.where(ag)[0]; self.locked = True; self.members = [self.ev[k] for k in sel]; self.estimate = fuse(self.ev, sel, self.rule); return self.estimate


class Maintenance:
    """post-lock estimate of one peer's VIO world (yaw deg, x, y, z) from the maintenance solves (1 packet/s after the lock).
    Policy (round 14 deployed = weight 0.7 on every accepted solve; round 15 options chosen with integrity15.py):
      z_gate: a solve enters only with RANSAC margin z >= z_gate (the consensus-member gate of the lock rule); None = no gate
      fallback_s: after this long without a gated solve, the best ungated accepted solve enters (a drifting VIO is still followed)
      withdraw_m / withdraw_deg: a gated solve this far from the carried estimate withdraws the output; the pair searches again
      degraded_m: status 'degraded' when the median innovation of the last 10 solves exceeds it (no ground truth needed)."""
    # round-15 defaults = integrity15 'zdetect' on the round-15 data (integrity15_r15.py, seed 0): gate at the rig rule's consensus margin,
    # withdraw at 20 cm / 2 deg, no fallback. Maintained time > 20 cm: healthy drones 0.002 % (deployed EMA 0.056 %), TUM-VI held-out
    # 0.14 % (1.15 %), EuRoC 4.6 % (16.5 %); nxt1 pairs ~25 % under every policy (a drifting VIO; unavailable 6.5 % while re-searching).
    def __init__(self, weight=0.7, z_gate='cons', fallback_s=None, withdraw_m=0.20, withdraw_deg=2.0, degraded_m=0.10, rule=None):
        # rule: the peer's lock rule (acceptance floor min_inl; z_gate='cons' = its consensus gate z_cons). None = round-14 FINAL floor.
        self.rule = rule or FINAL; self.min_inl = self.rule['min_inl']
        self.w = weight; self.z_gate = self.rule['z_cons'] if z_gate == 'cons' else z_gate; self.fb = fallback_s; self.wm = withdraw_m; self.wd = withdraw_deg; self.dm = degraded_m; self.reset()
    def reset(self):
        self.est = None; self.t_gate = -np.inf; self.withdrawn = False; self.innov = []; self.pending = None
    def start(self, est, t):
        self.est = np.array(est, float); self.t_gate = t; self.withdrawn = False; self.innov = []; self.pending = None
    def _blend(self, v):
        a = np.radians([self.est[0], v[0]]); w = np.array([1.0 - self.w, self.w])
        yaw = np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()))
        self.est = np.array([yaw, *((1.0 - self.w) * self.est[1:] + self.w * np.asarray(v[1:], float))])
    @staticmethod
    def _at(e, p):   # peer VIO position p mapped into the own VIO world by estimate e = (yaw deg, x, y, z)
        c, s_ = np.cos(np.radians(e[0])), np.sin(np.radians(e[0])); return np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1], p[2]]) + e[1:4]
    def update(self, s, t, p_peer=None):
        """p_peer: the peer's current VIO position (packet header x, y, z). The innovation is the displacement of that point between the solve's
        and the carried estimate (integrity15.innov), which includes the lever of a yaw difference; None = at the peer's VIO origin."""
        if self.est is None or not _accepted(s, {'min_inl': s.get('min_inl', self.min_inl)}): return self.est
        v = np.array([s['v_yaw'], s['v_x'], s['v_y'], s['v_z']], float); p = np.zeros(3) if p_peer is None else np.asarray(p_peer, float)
        dy = abs((v[0] - self.est[0] + 180) % 360 - 180); dtr = float(np.linalg.norm(self._at(v, p) - self._at(self.est, p))); self.innov = (self.innov + [dtr])[-10:]
        gated = self.z_gate is None or zmargin(s) >= self.z_gate
        if gated:
            if self.wm is not None and (dtr > self.wm or dy > self.wd):
                self.withdrawn = True; self.est = None; return None
            self._blend(v); self.t_gate = t; self.pending = None
        else:
            if self.pending is None or s['ninl'] > self.pending[1]: self.pending = (v, s['ninl'])
            if self.fb is not None and t - self.t_gate >= self.fb: self._blend(self.pending[0]); self.t_gate = t; self.pending = None
        return self.est
    @property
    def degraded(self): return len(self.innov) >= 5 and float(np.median(self.innov)) > self.dm


class KalmanMaintenance(Maintenance):
    """round 20: Kalman filter on one peer's VIO-world transform with an adaptive solve rate (alternative to the 0.7 blend at 1 packet/s).
    State = Maintenance.est (yaw deg, x, y, z). The covariance P (deg, m, m, m) is kept at an ANCHOR, the image of the peer's latest VIO
    position p: (yaw, m = Rz(yaw) p + t). There a solve's yaw and position errors are treated as independent; a yaw error reaches the
    translation only through the lever of a later, different p (_reanchor).
    Model: between solves the transform is a random walk at the anchor (q_yaw deg and q_pos m per sqrt(s): the relative drift of the two
    VIOs and the slowly varying part of the solve error). A solve measures (yaw, m) with sigma k_yaw * sy (floor r_yaw, deg) and
    k_pos * st (floor r_pos, m). With q_adapt the process noise is scaled by qs = median of the last 5 normalised innovations over its
    expected value (1 .. q_max), so a peer whose VIO drifts faster than the model is followed faster.
    Gates on a solve with margin z >= z_gate (others are ignored, as in Maintenance), in this order: farther than withdraw_m /
    withdraw_deg from the carried estimate -> withdraw (the Maintenance rule); normalised innovation > nis_max -> skipped; else fused.
    confirm_after (s, None = off): on a lock at least this old, the first gated solve beyond the withdraw limits only makes the output
    'suspect' and forces the fast rate; the next gated solve decides (beyond the limits again -> withdraw, else the suspicion is dropped).
    Rate, no absolute accuracy threshold (the same rule for every rig): period(t) is `fast` (every 5-Hz keyframe) from each (re)lock
    until the filter has CONVERGED = at least n_min solves fused and the last one reduced neither sigma by more than 1 - conv_rel; then
    `slow`. It returns to `fast` while suspect and, when set, when no solve was fused for more than miss x slow or while qs >= q_hot.
    One fast phase lasts at most fast_max s and is then followed by at least `cool` s at the slow rate, which bounds the GPU share of
    a peer that never settles."""
    # noise defaults frozen on the TUNE drone flights (s1 + s2, seed 0; review20/kf/FREEZE.md part 1): the solve's own sigmas understate
    # its error about 3x (k_yaw, k_pos); q_pos 0.03 m/sqrt(s) covers the slowly varying part of the solve error and the VIO wander, which
    # do not average out. Rate / option defaults (confirm from the lock, q_adapt on, miss and q_hot off): FREEZE.md part 2, TUNE groups.
    def __init__(self, q_yaw=0.1, q_pos=0.03, k_yaw=3.0, k_pos=3.0, r_yaw=0.1, r_pos=0.01, s0_yaw=0.3, s0_pos=0.04, nis_max=18.5,
                 conv_rel=0.95, n_min=3, miss=None, fast=0.2, slow=1.0, fast_max=4.0, cool=4.0, confirm_after=0.0, q_adapt=True, q_max=16.0, q_hot=None, **kw):
        self.q = np.array([q_yaw, q_pos, q_pos, q_pos], float) ** 2; self.k = (k_yaw, k_pos); self.rmin = (r_yaw, r_pos); self.s0 = (s0_yaw, s0_pos); self.nis_max = nis_max
        self.conv_rel = conv_rel; self.n_min = n_min; self.miss = miss; self.fast = fast; self.slow = slow; self.fast_max = fast_max; self.cool = cool
        self.confirm_after = confirm_after; self.q_adapt = q_adapt; self.q_max = q_max; self.q_hot = q_hot
        super().__init__(**kw)
    def reset(self):
        super().reset(); self.P = None; self.p_ref = None; self.t_kf = None; self.t0 = None; self.n_fused = 0; self.nis = []; self.suspect = None; self.qs = 1.0
        self.is_fast = True; self.t_fast = None; self.t_cool = -np.inf; self.converged = False; self.s_prev = None
    def start(self, est, t):
        super().start(est, t); self.P = np.diag([self.s0[0], self.s0[1], self.s0[1], self.s0[1]]) ** 2; self.p_ref = None; self.t_kf = t; self.t0 = t
        self.n_fused = 0; self.nis = []; self.suspect = None; self.qs = 1.0; self.is_fast = True; self.t_fast = t; self.t_cool = -np.inf; self.converged = False; self.s_prev = self.s0
    @staticmethod
    def _lever(yaw, d):
        """d(Rz(yaw) d) / d(yaw in deg)"""
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw)); return np.radians(1.0) * np.array([-s_ * d[0] - c * d[1], c * d[0] - s_ * d[1], 0.0])
    def _reanchor(self, p):
        if self.p_ref is not None:
            J = np.eye(4); J[1:4, 0] = self._lever(self.est[0], p - self.p_ref); self.P = J @ self.P @ J.T
        self.p_ref = np.array(p, float)
    def _predict(self, t):
        self.P = self.P + np.diag(self.q) * self.qs * max(t - self.t_kf, 0.0); self.t_kf = max(t, self.t_kf)
    def sigma(self, t=None):
        """predicted 1-sigma at the anchor: (yaw deg, position m, the largest axis); None before a lock"""
        if self.est is None or self.P is None: return None
        d = np.diag(self.P) + self.q * self.qs * (max(t - self.t_kf, 0.0) if t is not None else 0.0); return float(np.sqrt(d[0])), float(np.sqrt(d[1:4].max()))
    def update(self, s, t, p_peer=None):
        if self.est is None or not _accepted(s, {'min_inl': s.get('min_inl', self.min_inl)}): return self.est
        v = np.array([s['v_yaw'], s['v_x'], s['v_y'], s['v_z']], float); p = np.zeros(3) if p_peer is None else np.asarray(p_peer, float)
        dys = float(_wrapd(v[0] - self.est[0])); dm = self._at(v, p) - self._at(self.est, p); dtr = float(np.linalg.norm(dm)); self.innov = (self.innov + [dtr])[-10:]
        if not (self.z_gate is None or zmargin(s) >= self.z_gate): return self.est
        if self.wm is not None and (dtr > self.wm or abs(dys) > self.wd):
            if self.confirm_after is not None and self.suspect is None and t - self.t0 >= self.confirm_after:
                self.suspect = t; return self.est
            self.withdrawn = True; self.est = None; return None
        self.suspect = None; self._reanchor(p); self._predict(t)
        sy = s.get('sy', np.nan); st = s.get('st', np.nan)
        ry = max(self.k[0] * sy, self.rmin[0]) if np.isfinite(sy) else self.s0[0]; rp = max(self.k[1] * st, self.rmin[1]) if np.isfinite(st) else self.s0[1]
        R = np.diag([ry, rp, rp, rp]) ** 2; nu = np.array([dys, *dm]); Si = np.linalg.inv(self.P + R); nis = float(nu @ Si @ nu); self.nis = (self.nis + [nis])[-10:]
        if self.q_adapt and len(self.nis) >= 3: self.qs = float(np.clip(np.median(self.nis[-5:]) / 3.36, 1.0, self.q_max))
        if self.nis_max is not None and nis > self.nis_max: return self.est
        K = self.P @ Si; dx = K @ nu; A = np.eye(4) - K; self.P = A @ self.P @ A.T + K @ R @ K.T
        yaw = float(_wrapd(self.est[0] + dx[0])); m = self._at(self.est, p) + dx[1:4]; c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        self.est = np.array([yaw, m[0] - (c * p[0] - s_ * p[1]), m[1] - (s_ * p[0] + c * p[1]), m[2] - p[2]]); self.t_gate = t; self.n_fused += 1; self.pending = None
        sg = self.sigma(); self.converged = self.n_fused >= self.n_min and sg[0] >= self.conv_rel * self.s_prev[0] and sg[1] >= self.conv_rel * self.s_prev[1]; self.s_prev = sg
        return self.est
    def _qhot(self): return self.q_hot is not None and self.qs >= self.q_hot
    def period(self, t):
        """seconds to the next maintenance solve wanted at time t (node15 asks once per admitted peer keyframe)"""
        if self.est is None or self.P is None or self.suspect is not None: return self.fast
        if self.is_fast:
            if self.converged and not self._qhot(): self.is_fast = False
            elif t - self.t_fast >= self.fast_max: self.is_fast = False; self.t_cool = t + self.cool
        elif t >= self.t_cool and ((self.miss is not None and t - self.t_gate > self.miss * self.slow) or self._qhot()):
            self.is_fast = True; self.t_fast = t; self.converged = False
        return self.fast if self.is_fast else self.slow


class DepthVote:
    """round 15 (proposal; vote15.py): swarm depth-consistency vote, ground-truth-free, needs >= 3 drones. Which drone's depth to trust
    where both triangulated a point: every accepted solve reports lsr = median log(sender depth / receiver depth) on shared inliers
    (orin_dump.depth_ratio). With per-drone log depth scales s, lsr(rx <- tx) = s_tx - s_rx. At time t each unordered pair's value is the
    mean of its directed medians (own solves, and the peers' summaries from vote messages, packet_codec.encode_vote), each over solves
    with both keyframes <= t (cumulative, or the last `win` s); with every pair of the swarm at >= nmin solves, s is fitted by least
    squares (sum zero) and a drone whose |s - median(s)| > tau is flagged. The receiver takes its own depth when the sender is flagged
    and it is not (use_receiver_depth); otherwise the sender's (the default). tau 0.015 / nmin 30 were set from whole-flight offsets
    (healthy <= 0.008, nxt1 0.031-0.038); 0.010-0.020 and a 10-s window gave the same lock statistics (seed 0)."""
    def __init__(self, tau=0.015, nmin=30, win=None):
        self.tau, self.nmin, self.win = tau, nmin, win; self.obs = {}; self._arr = {}
    def add(self, rx, tx, t, lsr):
        if np.isfinite(lsr): self.obs.setdefault((rx, tx), []).append((float(t), float(lsr))); self._arr.pop((rx, tx), None)
    def _sel(self, key, t):
        if key not in self._arr:
            a = np.array(sorted(self.obs.get(key, [])), float).reshape(-1, 2); self._arr[key] = a
        a = self._arr[key]; hi = np.searchsorted(a[:, 0], t, side='right'); lo = np.searchsorted(a[:, 0], t - self.win, side='right') if self.win else 0
        return a[lo:hi, 1]
    def summary(self, rx, t):
        """vote message content of drone rx at time t: [(tx, median lsr, count)] over its own solves."""
        out = []
        for (r, x) in self.obs:
            if r == rx:
                v = self._sel((r, x), t)
                if len(v): out.append((x, float(np.median(v)), int(len(v))))
        return out
    def merge(self, rx, entries, t_msg):
        """a peer rx's vote message (its summary at t_msg): stored as directed medians (latest message wins)."""
        for tx, med, n in entries: self.obs[(rx, tx)] = [(float(t_msg), float(med))] * int(n); self._arr.pop((rx, tx), None)
    def flags(self, t, drones):
        drones = list(drones); idx = {d: k for k, d in enumerate(drones)}; rows, rhs = [], []
        for x_, y_ in ((a, b) for k, a in enumerate(drones) for b in drones[k + 1:]):
            vs, n = [], 0
            for (r, s_), sg in (((x_, y_), 1.0), ((y_, x_), -1.0)):
                v = self._sel((r, s_), t)
                if len(v): vs.append(sg * float(np.median(v))); n += len(v)
            if n < self.nmin or not vs: return set()
            r_ = np.zeros(len(drones)); r_[idx[y_]] = 1.0; r_[idx[x_]] = -1.0; rows.append(r_); rhs.append(float(np.mean(vs)))   # lsr(x <- y) = s_y - s_x
        if len(drones) < 3 or not rows: return set()
        A = np.vstack(rows + [np.ones(len(drones))]); s_ = np.linalg.lstsq(A, np.array(rhs + [0.0]), rcond=None)[0]; dev = s_ - np.median(s_)
        return {d for d in drones if abs(dev[idx[d]]) > self.tau}
    def use_receiver_depth(self, rx, tx, t, drones):
        f = self.flags(t, drones); return tx in f and rx not in f


class ThresholdSelector:
    """round 15: ground-truth-free choice of the RANSAC / refinement test threshold for one peer (or one rig). With exact rays the optimum
    spans 10x in angle across rigs (drones 7.5 px, TUM-VI 1.5 px, EuRoC 1.0 px); on every rig it coincides with the threshold that
    maximises the median RANSAC margin z of accepted solves. The receiver scores its hypothesis table at every candidate threshold
    (orin_dump ZSET columns z_<px>), feeds the margins of each accepted solve here, and solves at `current`. Starts at `prior` (the rig's
    value, or rig15.THR_PX) and switches once `min_n` solves are in to the largest threshold whose median margin is within `tie` (relative) of the best; it stays
    put while its own median is within that band (no flapping)."""
    def __init__(self, candidates, prior, min_n=30, window=200, tie=0.02):
        self.c = sorted(candidates); self.current = prior; self.min_n = min_n; self.window = window; self.tie = tie; self.z = {c: [] for c in self.c}
    def add(self, zs):
        """zs: {candidate px: margin z of this solve's RANSAC table at that threshold}."""
        for c in self.c:
            if c in zs and np.isfinite(zs[c]): self.z[c] = (self.z[c] + [float(zs[c])])[-self.window:]
        n = min(len(v) for v in self.z.values())
        if n < self.min_n: return self.current
        med = {c: float(np.median(v)) for c, v in self.z.items()}; top = max(med.values())
        best = max(c for c in self.c if med[c] >= (1 - self.tie) * top)   # the largest threshold within `tie` of the best median margin (more support)
        if best != self.current and (self.current not in med or med[self.current] < (1 - self.tie) * top): self.current = best
        return self.current


# ---- round 16: two-direction pair protocol (review16/publish, review16/verify_publish policy C) ----
def _wrapd(a): return (a + 180.0) % 360.0 - 180.0
def _Rz(y):
    c, s = np.cos(np.radians(y)), np.sin(np.radians(y)); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
def est_inverse(e):
    """(yaw deg, x, y, z) of world B -> world A  ->  the same transform A -> B."""
    return np.array([-e[0], *(-(_Rz(-e[0]) @ np.asarray(e[1:4], float)))])
def est_agree(X, Y, p, m=0.20, d=2.0):
    """X: the own estimate (peer VIO world -> own), Y: the peer's own estimate (own world -> peer's). They agree when X o Y ~ identity
    at the peer's current VIO position p (peer world): |X(p) - Y^-1(p)| <= m and |yaw_X + yaw_Y| <= d (review16/verify_publish vsim.agree)."""
    if X is None or Y is None or p is None: return False
    a = _Rz(X[0]) @ p + X[1:4]; b = _Rz(-Y[0]) @ (p - Y[1:4])
    return bool(np.linalg.norm(a - b) <= m and abs(_wrapd(X[0] + Y[0])) <= d)

class PairLink:
    """round 16: the pair protocol between this drone and ONE peer, on top of the independent locks (both directions keep searching).
    Messages: packet_codec.Codec.encode_est / decode_est (60 B), sent on every (re)lock / maintenance update / withdraw / confirmation, when an ack is owed or requested, at
    `period` (the 1-Hz maintenance rate) while there is anything to say, and every `resend` s while the own (re)lock or withdraw is not
    acknowledged (lost messages / acks are recovered).
    Output, two streams: provisional = the own estimate from the own (re)lock on, or, before it, the peer's first lock ADOPTED (inverted)
    at its arrival; confirmed = the own estimate once the peer's own, independent estimate agrees through the VIO (est_agree, 20 cm / 2 deg
    at the peer's last header position), checked at every change of either estimate (not at repeated messages). Policy C (review16/verify_publish): every (re)lock is
    checked again; a relock never inherits `confirmed`; confirmation holds until the own epoch ends (withdraw / relock / VIO epoch).
    Crossing locks (the peer's lock arrives within msg_win of the own lock, both unconfirmed): the lower drone id's estimate is shown on
    the provisional stream of both drones (the loser shows the adopted one) until its own estimate is confirmed.
    Epochs: messages carry (sender VIO epoch, epoch of the receiver's VIO the estimate refers to); stale ones are ignored; a VIO epoch change
    on either side drops the peer estimate, the adoption and the confirmation."""
    def __init__(self, own_id, peer_id, m=0.20, d=2.0, period=1.0, resend=0.2, msg_win=0.05, auto_check=True):
        self.own_id, self.peer_id, self.m, self.d, self.period, self.resend, self.msg_win, self.auto = own_id, peer_id, m, d, period, resend, msg_win, auto_check
        self.own_ep = 0; self.peer_ep = None; self.p_peer = None; self.lock_id = 0; self.log = []
        self._clear_own(); self._clear_peer(); self.last_sent = -np.inf; self.urgent = False; self.acked = True; self.ack_owed = None
    def _clear_own(self):
        self.X = None; self.t_lock = None; self.confirmed = False; self.t_conf = None; self.status = 'none'
    def _clear_peer(self):
        self.Y = None; self.peer_lock_id = None; self.t_peer_lock = None; self.adopted = None
    # ---- own side (node15.Node.feed) ----
    def own_lock(self, t, est):
        self.lock_id = (self.lock_id + 1) & 0xFFFF; self.X = np.array(est, float); self.t_lock = t; self.confirmed = False; self.t_conf = None
        self.status = 'provisional'; self.urgent = True; self.acked = False
        if not (self.Y is not None and self.t_peer_lock is not None and t - self.t_peer_lock <= self.msg_win and self.peer_id < self.own_id): self.adopted = None
        self.log.append((t, 'provisional', 'own', self.lock_id)); self._auto(t)
    def own_update(self, t, est):
        if self.X is None: return
        self.X = np.array(est, float); self.urgent = True; self._auto(t)   # every change is sent (maintenance rate)
    def own_withdraw(self, t):
        if self.X is None: return
        self._clear_own(); self.status = 'withdrawn'; self.urgent = True; self.acked = False; self.log.append((t, 'withdrawn', 'own', self.lock_id))
    def set_own_epoch(self, e, t):
        if (e & 0xFF) == self.own_ep: return
        self.own_ep = e & 0xFF; self._clear_own(); self._clear_peer(); self.status = 'none'; self.log.append((t, 'own_epoch', None, None))
    # ---- peer side (node15.Peer headers, estimate datagrams) ----
    def peer_header(self, epoch, p, t):
        if self.peer_ep is not None and epoch != self.peer_ep:
            self._clear_own(); self._clear_peer(); self.status = 'none'; self.log.append((t, 'peer_epoch', None, None))
        self.peer_ep = epoch; self.p_peer = None if p is None else np.asarray(p, float)
    def on_msg(self, msg, t):
        if msg is None or msg['drone'] != self.peer_id or msg['peer'] != self.own_id: return False
        if msg['peer_epoch'] != self.own_ep or (self.peer_ep is not None and msg['epoch'] != self.peer_ep): return False   # stale epoch
        if msg['ack_id'] is not None and msg['ack_id'] == self.lock_id: self.acked = True
        if msg.get('need_ack'): self.urgent = True   # the peer has not heard our ack (lost): answer at once
        Y0 = self.Y
        if msg['status'] in ('provisional', 'confirmed') and msg['est'] is not None:
            new = msg['lock_id'] != self.peer_lock_id; self.Y = np.array(msg['est'], float)
            if new:
                self.peer_lock_id = msg['lock_id']; self.t_peer_lock = t; self.ack_owed = msg['lock_id']; self.urgent = True
                crossing = self.X is not None and not self.confirmed and t - self.t_lock <= self.msg_win and self.peer_id < self.own_id
                if self.X is None or crossing: self.adopted = est_inverse(self.Y); self.log.append((t, 'provisional', 'adopted', msg['lock_id']))
            elif self.adopted is not None: self.adopted = est_inverse(self.Y)   # follow the peer's maintenance
        elif msg['status'] == 'withdrawn':
            if msg['lock_id'] != self.peer_lock_id: self.ack_owed = msg['lock_id']; self.urgent = True
            self.peer_lock_id = msg['lock_id']; self.Y = None
            if self.adopted is not None: self.adopted = None; self.log.append((t, 'adoption_withdrawn', 'adopted', msg['lock_id']))
        # round 16 follow-up 4: agreement is re-checked only when the peer's estimate CHANGED (keepalives / resends / acks repeat it), so the
        # flight protocol is policy C exactly (review16/verify_publish vsim: checks at estimate changes only)
        if not (Y0 is None and self.Y is None) and (Y0 is None or self.Y is None or not np.array_equal(Y0, self.Y)): self._auto(t)
        return True
    # ---- confirmation ----
    def _auto(self, t):
        if self.auto: self.check(t)
    def check(self, t):
        if self.X is not None and not self.confirmed and est_agree(self.X, self.Y, self.p_peer, self.m, self.d):
            self.confirmed = True; self.t_conf = t; self.status = 'confirmed'; self.adopted = None; self.urgent = True
            self.log.append((t, 'confirmed', 'own', self.lock_id))
        return self.confirmed
    def output(self):
        """{'provisional': estimate or None, 'source': 'own' | 'adopted' | None, 'confirmed': estimate or None (never an adopted one)}"""
        if self.adopted is not None and not self.confirmed: return {'provisional': self.adopted, 'source': 'adopted', 'confirmed': None}
        if self.X is not None: return {'provisional': self.X, 'source': 'own', 'confirmed': self.X if self.confirmed else None}
        return {'provisional': None, 'source': None, 'confirmed': None}
    def message(self, t, force=False):
        """the estimate message due at t (dict for Codec.encode_est) or None."""
        due = force or self.urgent or (t - self.last_sent >= self.period and (self.X is not None or self.Y is not None or self.status == 'withdrawn')) \
              or (not self.acked and t - self.last_sent >= self.resend)
        if not due: return None
        self.urgent = False; self.last_sent = t; ack = self.ack_owed
        return {'t': t, 'peer': self.peer_id, 'epoch': self.own_ep, 'peer_epoch': self.peer_ep if self.peer_ep is not None else 0, 'lock_id': self.lock_id,
                'ack_id': ack, 'status': self.status, 'est': self.X, 'need_ack': not self.acked}
