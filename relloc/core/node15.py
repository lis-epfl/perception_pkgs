"""Reference receiver node (round 15, hardened in round 16): the General Route receiver logic as the flight node must run it, without ROS. It
closes the gap the 09-28 review found between what was measured (offline replay of export files) and what a drone can execute from the bytes
it receives:
  - packets arrive as packet_codec v3 datagrams; a keyframe is decoded once all its datagrams arrived, or at the expected end of its burst
    (round 16: projected from the arrival rate of the burst itself, plus a margin, capped at deadline_cap < the 0.2-s keyframe period)
  - rays and 3-D points of the peer are rebuilt from pixels, the peer's calibration (sent once) and the tilt in the packet header
  - each received keyframe is paired with every own keyframe within +-win while searching (round 16: all of them, as replay.py; round 15
    capped at 2), nearest first, and with the nearest only once locked; SolveQueue runs them in the replay's order with its COND2 skip
  - solves run through orin_dump.solve (the evaluated receiver) at the peer rig's test threshold (round 16: orin_dump.setup_rig per peer,
    focal = mean over both drones' cameras as the exports); estimates are mapped to VIO worlds with the peer pose in the header
  - the lock rule is rule15 (identical to the replay); maintenance is rule15.Maintenance at the evaluated 1 packet/s (round 16: receiver
    rate gate on the header time, phase-locked to the lock time as integrity15 schedules it) and the VIO-epoch reset
Round 16 hardening (REVIEW15 S1): malformed / foreign datagrams are validated and dropped (never raise), duplicate chunks ignored, the
keyframe store pruned by age, the epoch compared modulo 256 (late old-epoch datagrams dropped, no flip-flop), solves tagged with the peer's
and the own epoch (a queued old-epoch solve cannot lock), the clock offset a windowed minimum over finite header times, ncam = 0 / empty
keyframes skipped, per-peer rig (codec, rule, threshold), own-epoch reset and add_own_keyframe API, DIRSEL='health' wiring (header byte).
Usage: loopback15.py (bytes -> this node -> dump rows); test_node15.py (state machine); test_stream15.py (streaming parity with replay)."""
import numpy as np, torch, sys, heapq, collections
import packet_codec as PC, rule15
from scipy.spatial.transform import Rotation as Rot
import os
NP_DEQUANT = os.environ.get('NP_DEQUANT', '0') == '1'   # numpy descriptor reconstruction (bit-identical to the export emulation; GPU path differs by fp16 ulps)


class PeerCalib:
    """calibration a peer sends once: per camera the unprojection (callable uv -> unit rays in the camera frame) and the rig/body <- camera
    rotation and position. Rays of every packet pixel are then a lookup in a per-camera table (built lazily). fx (round 16): the focal
    lengths of the peer's cameras (px), for the pixel test threshold; None = the rig's rig15 value."""
    def __init__(self, unproject, R_body_cam, p_body_cam, width, height, mode='direct', fx=None):
        self.mode = mode; self.unproject = unproject; self.R = [np.asarray(r, float) for r in R_body_cam]; self.p = [np.asarray(x, float) for x in p_body_cam]
        self.W = width; self.H = height; self._tab = {}; self.fx = None if fx is None else [float(v) for v in np.atleast_1d(fx)]
    def rays_cam(self, c, kpts):
        if self.mode == 'direct':   # round 15: unproject the ~500-1000 packet pixels directly (no 12 MB table per peer camera)
            return np.asarray(self.unproject(c, np.round(np.asarray(kpts, np.float64))), np.float64)
        if c not in self._tab: self._tab[c] = PC.ray_table(lambda uv, _intr, _c=c: self.unproject(_c, uv), None, self.W, self.H)
        u = np.clip(np.round(kpts[:, 0]).astype(int), 0, self.W - 1); v = np.clip(np.round(kpts[:, 1]).astype(int), 0, self.H - 1)
        return self._tab[c][v, u].astype(np.float64)


def peer_side(dec, calib, codec, dev='cuda', image_size=None):
    """decoded v3 keyframe -> the per-camera dicts orin_dump.solve takes (side B): descriptors, rays and 3-D points in the peer's
    gravity-aligned rig frame (tilt from the header), camera centres, quality code (1 = fresh)."""
    h = dec['hdr']; Rt = Rot.from_euler('ZYX', [0.0, h['pitch'], h['roll']]).as_matrix(); out = {}
    for c, cm in enumerate(dec['cams']):
        if c >= len(calib.R): break   # round 16: a header announcing more cameras than the calibration has
        kp = cm['kpts']; n = len(kp)
        Rc = Rt @ calib.R[c]; cc = Rt @ calib.p[c]
        d = calib.rays_cam(c, kp) @ Rc.T if n else np.zeros((0, 3)); d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        X = np.full((n, 3), np.nan); dep = cm['depth']; m = np.isfinite(dep); X[m] = cc + dep[m, None] * d[m]
        desc = (torch.from_numpy(codec.dequant_desc(cm['q'], cm['step']).astype(np.float32)).to(dev).half() if NP_DEQUANT else codec.dequant_desc_torch(cm['q'], cm['step'], torch, dev)) if 'q' in cm else torch.from_numpy(cm['desc'].astype(np.float32)).to(dev).half()
        kpt = torch.from_numpy(kp.astype(np.float32)).to(dev)
        out[c] = {'dic': {'keypoints': kpt, 'descriptors': desc.float(), 'scores': None, 'image_size': image_size or (calib.W, calib.H)},
                  'gd': torch.nn.functional.normalize(desc.float(), dim=1), 'c': cc, 'd': d, 'X': X, 'q': np.ones(n, np.float32)}
    return out


def _newer(e, cur):
    """modular epoch order (uint8 on the wire): True when e is ahead of cur by 1..127."""
    return 0 < ((int(e) - int(cur)) & 0xFF) < 128


class Peer:
    def __init__(self, calib, rule=rule15.RULES15['drones'], maint=None, codec=None, rig='drone4', clock_win=10.0, kf_ttl=1.0, done_keep=5.0, link=None):
        self.calib = calib; self.rule = rule; self.lock = rule15.PeerLock(rule); self.maint = maint or rule15.Maintenance(rule=rule); self.epoch = None
        self.codec = codec; self.rig = rig
        self.kf = {}          # keyframe -> {'t_first', 't_last', 'dg': [datagrams], 'seen': {(cam, chunk)}, 'nck': {cam: chunks}, 'hdr', 'epoch'}
        self.done = {}        # round 16: keyframe -> time it was taken for decoding (late datagrams of it are dropped, not re-stored)
        self.events = []      # log of (time, kind, info)
        self.clock_off = None  # own clock - peer clock + minimum latency: minimum over the last clock_win s of received keyframe headers
        self._clk = collections.deque(); self.clock_win = clock_win; self.kf_ttl = kf_ttl; self.done_keep = done_keep
        self.next_maint = None; self.last_maint = None; self.t_lock = None
        self.buf = {}         # round 16 (behaviour A): decoded received keyframes still open for pairing with own keyframes processed later
        self.pend = None      # round 16: the maintenance keyframe waiting for a nearer own keyframe
        self.n_bad = 0; self.n_dup = 0; self.n_stale = 0; self.n_late = 0
        self.link = link      # round 16: rule15.PairLink (pair protocol, provisional / confirmed output), None = no protocol
    def _reset(self, t, why, info=None):
        self.lock.reset(); self.maint.reset(); self.next_maint = None; self.last_maint = None; self.t_lock = None; self.buf = {}; self.pend = None; self.events.append((t, why, info))
    def _prune(self, t):
        for k in [k for k, e in self.kf.items() if t - e['t_first'] > self.kf_ttl]: del self.kf[k]
        for k in [k for k, td in self.done.items() if t - td > self.done_keep]: del self.done[k]
    def finish(self, kf, t=None):
        e = self.kf.pop(kf, None)
        if e is not None: self.done[kf] = e['t_first'] if t is None else t
    def on_datagram(self, g, t_arr):
        """store one datagram. Returns its keyframe index, or None when it was dropped (malformed, foreign, duplicate, stale epoch, late)."""
        try:
            if self.codec is not None and not self.codec.validate(g): self.n_bad += 1; return None
            drone, kf, cam, ck, nck, has = PC.Codec.peek(g); h = PC.Codec.keyframe_header(g) if has else None
        except Exception:
            self.n_bad += 1; return None
        self._prune(t_arr)
        if h is not None:
            if not all(np.isfinite([h['t'], h['roll'], h['pitch'], h['yaw'], h['x'], h['y'], h['z']])): self.n_bad += 1; return None
            if self.epoch is not None and h['epoch'] != self.epoch:
                if not _newer(h['epoch'], self.epoch): self.n_stale += 1; return None   # a late datagram of an older VIO epoch
                self._reset(t_arr, 'epoch_reset', h['epoch'])                            # the peer's VIO re-initialised: its world moved
                self.kf = {k: e for k, e in self.kf.items() if e.get('epoch') is None}; self.done = {}
            self.epoch = h['epoch']
            if self.link is not None: self.link.peer_header(h['epoch'], (h['x'], h['y'], h['z']), t_arr)
            self._clk.append((t_arr, t_arr - h['t']))
            while self._clk and t_arr - self._clk[0][0] > self.clock_win: self._clk.popleft()
            self.clock_off = min(d for _, d in self._clk)
        if kf in self.done: self.n_late += 1; return None
        e = self.kf.setdefault(kf, {'t_first': t_arr, 't_last': t_arr, 'dg': [], 'seen': set(), 'nck': {}, 'hdr': None, 'epoch': None})
        if (cam, ck) in e['seen']: self.n_dup += 1; return None
        if h is not None and e['hdr'] is not None and e['epoch'] is not None and h['epoch'] != e['epoch']:   # index reuse across an epoch
            e.update(t_first=t_arr, dg=[], seen=set(), nck={}, hdr=None)
        e['seen'].add((cam, ck)); e['nck'][cam] = nck; e['dg'].append(g); e['t_last'] = t_arr
        if h is not None and e['hdr'] is None: e['hdr'] = h; e['epoch'] = h['epoch']
        return kf


class Node:
    """one receiver. own(k) -> (t, pose (roll, pitch, yaw, x, y, z), side-A dict for orin_dump.solve[, own health 0..1]);
    solve_fn(A, B) -> orin_dump.solve result. Round 16: rig = the own rig (rig15 name); peers may declare their own rig in add_peer
    (per-peer codec, lock rule and test threshold). When solve_fn comes from orin_dump, its setup_rig / set_health are wired automatically."""
    def __init__(self, codec, solve_fn, own, own_times, win=0.25, deadline=0.010, rule=None, clock_free=True, lat_min=0.09, pf_gate=0, prefilter=None,
                 own_id=None, vote=None, set_dir=None, rig='drone4', set_rig=None, set_health=None, own_fx=None, maint_period=1.0, maint_tol=0.1, maint_late=0.15,
                 deadline_cap=0.19, deadline_margin=0.005, cand_cap=None, clock_win=10.0, kf_ttl=1.0, repair=True, buf_max=64, pair_protocol=True, link_kw=None,
                 maint='blend', maint_kw=None):
        # round 20: maint = the post-lock estimator of every peer added without its own. 'blend' = rule15.Maintenance (0.7 blend, 1 packet/s;
        # the configuration of every replayed number and parity test). 'kalman' = rule15.KalmanMaintenance (filter with an adaptive solve
        # rate: every keyframe after a (re)lock until converged, then 1 packet/s). maint_kw: extra options of that class.
        assert maint in ('blend', 'kalman'), maint
        self.maint_kind = maint; self.maint_kw = maint_kw or {}
        # round 15 (proposal, >= 3 drones): vote = rule15.DepthVote shared by the swarm (own solves via feed, peers' via on_vote); set_dir('A'|'B')
        # selects the depth side of the next solve (orin_dump: lambda d: orin_dump._DIR.__setitem__(0, d)); without a vote the sender's depth is used
        self.own_id = own_id; self.vote = vote; self.set_dir = set_dir
        self.clock_free = clock_free; self.lat_min = lat_min; self.pf_gate = pf_gate; self.prefilter = prefilter   # prefilter(A, Bcodes) -> best mutual-NN count
        self.codec = codec; self.solve = solve_fn; self.own = own; self.own_t = np.asarray(own_times, float); self.win = win; self.deadline = deadline
        self.rig = rig; self.rule = rule or rule15.RULES15[rule15.RIG_OF[rig]]; self.peers = {}
        mod = sys.modules.get(getattr(solve_fn, '__module__', None) or '')
        self.set_rig = set_rig or getattr(mod, 'setup_rig', None); self.set_health = set_health or getattr(mod, 'set_health', None)
        if mod is not None and hasattr(mod, 'setup_rig'): assert getattr(mod, 'THR_PX', 0) > 0 or getattr(mod, 'THR', 0) > 0, 'orin_dump test threshold not set'
        self.own_fx = None if own_fx is None else [float(v) for v in np.atleast_1d(own_fx)]
        self.maint_period = maint_period; self.maint_tol = maint_tol; self.maint_late = maint_late; self.deadline_cap = deadline_cap; self.deadline_margin = deadline_margin
        self.cand_cap = cand_cap; self.clock_win = clock_win; self.kf_ttl = kf_ttl; self.own_epoch = 0
        self.repair = repair and not cand_cap; self.buf_max = buf_max; self.pair_protocol = pair_protocol; self.link_kw = link_kw or {}   # repair: behaviour A (own_ready); cand_cap emulates the round-15 node
    def add_peer(self, pid, calib, maint=None, rig=None, codec=None):
        rig = rig or self.rig; rule = self.rule if rig == self.rig else rule15.RULES15[rule15.RIG_OF[rig]]
        assert codec is not None or rig == self.rig, 'a peer of another rig needs its own codec (image size -> pixel bits)'
        maint = maint or (rule15.KalmanMaintenance if self.maint_kind == 'kalman' else rule15.Maintenance)(rule=rule, **self.maint_kw)
        self.peers[pid] = Peer(calib, rule, maint, codec or self.codec, rig, self.clock_win, self.kf_ttl,
                               link=rule15.PairLink(self.own_id if self.own_id is not None else 0, pid, **self.link_kw) if self.pair_protocol else None)
    def add_own_keyframe(self, t, t_now=None):
        """round 16: append an own keyframe time; returns its index (the k own(k) receives). With t_now (the time its processing finished)
        also returns own_ready's solves: (index, [(pid, decoded keyframe, index, rank)])."""
        self.own_t = np.append(self.own_t, float(t)); i = len(self.own_t) - 1
        return i if t_now is None else (i, self.own_ready(i, t_now))
    def _own_period(self):
        return float(np.median(np.diff(self.own_t[-6:]))) if len(self.own_t) >= 2 else 0.2
    def own_ready(self, i, t_now=None):
        """round 16 (behaviour A, review16/ownkf): own keyframe i has finished processing (available now). Pairs it with every buffered
        received keyframe of a searching peer within +-win it was not yet solved against (replay.py pairs a received keyframe with own
        keyframes up to +win later, available at max(tA, tB) + LAT); rank 0 when i is nearer than every own keyframe already paired with
        that keyframe. For a locked peer, a maintenance keyframe deferred for a nearer own keyframe is solved against the nearest now.
        Returns [(pid, decoded keyframe, own index, rank)] to push (SolveQueue.push_jobs) at t_now."""
        jobs = []; ti = float(self.own_t[i])
        for pid, p in self.peers.items():
            if p.lock.locked:
                if p.pend is not None:
                    tp = p.pend['tp']; d = np.abs(self.own_t - tp); k = int(np.argmin(d)); t_next = ti + self._own_period()
                    if abs(t_next - tp) < d[k] and t_next - tp <= self.win: continue   # the next own keyframe will be nearer still
                    if d[k] <= self.win: jobs.append((pid, p.pend, k, 0))
                    p.pend = None
                continue
            for kf, b in list(p.buf.items()):
                d = abs(ti - b['tp'])
                if d <= self.win and i not in b['solved']:
                    jobs.append((pid, b['dec'], i, 0 if d < b['best'] else 1)); b['best'] = min(b['best'], d); b['solved'].add(i)
                if ti > b['tp'] + self.win: del p.buf[kf]   # no later own keyframe can be within +-win
        return jobs
    def own_epoch_reset(self, t):
        """round 16: the own VIO re-initialised: every estimate is in a world that no longer exists; queued solves of the old epoch are dropped."""
        self.own_epoch += 1
        for p in self.peers.values():
            p._reset(t, 'own_epoch_reset', self.own_epoch)
            if p.link is not None: p.link.set_own_epoch(self.own_epoch, t)
    def _fx(self, p):
        return float(np.mean(self.own_fx + p.calib.fx)) if self.own_fx and p.calib.fx else None
    def keyframe_ready(self, pid, kf, t_now):
        """decode a received keyframe once every datagram has arrived, or at the projected end of its burst (arrival rate of the datagrams
        received so far x the datagrams announced) + margin, never before `deadline` and at the latest deadline_cap after its first datagram."""
        e = self.peers[pid].kf.get(kf)
        if e is None: return False
        el = t_now - e['t_first']
        if el >= self.deadline_cap: return True
        h = e['hdr']
        if h is None: return False
        per = e['nck']; n = len(e['seen']); nexp = sum(per.values()) + (h['ncam'] - len(per)) * max(per.values())
        if len(per) >= h['ncam'] and n >= sum(per.values()): return True
        if el < self.deadline or n < 2 or e['t_last'] <= e['t_first']: return False
        dur = (e['t_last'] - e['t_first']) * (nexp - 1) / (n - 1)
        return t_now >= e['t_first'] + dur * 1.25 + self.deadline_margin
    def _peer_time(self, p, h):
        return h['t'] + ((p.clock_off - self.lat_min) if (self.clock_free and p.clock_off is not None) else 0.0)   # peer keyframe time on the own clock
    def _maint_due(self, p, tp):
        """1 packet/s after the lock (integrity15's schedule: the peer keyframe nearest lock time + k s, none when > 0.15 s away), decided
        causally: the first peer keyframe at or after slot - maint_tol (half the 0.2-s keyframe period), up to slot + maint_late; the others
        are not decoded."""
        if p.next_maint is None: return True
        per = self._mper(p, tp)
        if p.last_maint is not None: p.next_maint = min(p.next_maint, p.last_maint + per)
        while p.next_maint + self.maint_late < tp: p.next_maint += per
        if tp >= p.next_maint - self.maint_tol: p.last_maint = p.next_maint; p.next_maint += per; return True
        return False
    def _mper(self, p, t):
        """round 20: maintenance period of one peer at time t. A Maintenance object with its own period(t) (rule15.KalmanMaintenance:
        every keyframe until its covariance has converged, then 1 packet/s) sets it; otherwise the fixed maint_period."""
        return p.maint.period(t) if hasattr(p.maint, 'period') else self.maint_period
    def requests(self, pid, kf):
        """own keyframes to solve this received keyframe against: every one within +-win while searching (nearest first), the nearest once
        locked, and only for the keyframe the maintenance rate admits. Returns (decoded keyframe or None, [own keyframe indices])."""
        p = self.peers[pid]; e = p.kf.get(kf)
        if e is None: return None, []
        h = e['hdr']
        if h is not None and p.lock.locked and not self._maint_due(p, self._peer_time(p, h)):
            p.finish(kf); return {'hdr': h, 'cams': [], 'missing': [], 'skipped': 'rate'}, []   # rate gate: header only, no decode
        try:
            dec = p.codec.decode_v3(e['dg'], reconstruct=False)
        except Exception:
            p.n_bad += 1; p.finish(kf); return None, []
        p.finish(kf)
        if dec['hdr'] is None or not dec['cams'] or sum(len(c['kpts']) for c in dec['cams']) == 0: return dec, []
        if p.epoch is not None and dec['hdr']['epoch'] != p.epoch: p.n_stale += 1; return dec, []
        tp = self._peer_time(p, dec['hdr']); dec['tp'] = tp
        dt = np.abs(self.own_t - tp); cand = [int(i) for i in np.argsort(dt, kind='stable') if dt[i] <= self.win]
        if p.lock.locked:
            # maintenance: the own keyframe nearest in time (integrity15's pairing); when the next own keyframe (expected one own period after
            # the last) will be nearer than every existing one, defer to own_ready instead of solving a worse pair now
            t_next = (self.own_t[-1] + self._own_period()) if len(self.own_t) else np.inf
            if self.repair and abs(t_next - tp) < (dt[cand[0]] if cand else np.inf) and t_next - tp <= self.win: p.pend = dec; return dec, []
            return dec, cand[:1]
        if self.cand_cap: return dec, cand[:self.cand_cap]
        if self.repair:
            p.buf[kf] = {'dec': dec, 'tp': tp, 'solved': set(cand), 'best': float(dt[cand[0]]) if cand else np.inf}
            while len(p.buf) > self.buf_max: del p.buf[next(iter(p.buf))]
        return dec, cand
    def solve_pair(self, pid, dec, i):
        p = self.peers[pid]; o = self.own(i); t_i, pose_i, A = o[0], o[1], o[2]; h_own = o[3] if len(o) > 3 else None
        if self.pf_gate > 0 and self.prefilter is not None:   # round 15: two-stage decode, descriptors first; skip weak pairs before rays and 3-D
            best = self.prefilter(A, dec)
            if best < self.pf_gate: return {'ok': 0, 'pf_max': best, 'early_reject': True}, None
        h = dec['hdr']
        try:
            B = peer_side(dec, p.calib, p.codec)
            if self.set_rig is not None: self.set_rig(p.rig, self._fx(p))
            if self.set_health is not None: self.set_health(h_own, h['health'] / 255.0)
            if self.vote is not None and self.set_dir is not None:
                self.set_dir('A' if self.vote.use_receiver_depth(self.own_id, pid, t_i, [self.own_id] + list(self.peers)) else 'B')
            st = self.solve(A, B)
        except Exception as ex:   # round 16: one bad keyframe must not kill the receiver loop
            p.events.append((None, 'solve_error', repr(ex))); return {'ok': 0, 'error': repr(ex)}, None
        finally:
            if self.set_health is not None: self.set_health(None, None)
        if self.vote is not None and st.get('ok', 0) > 0 and st.get('ninl', 0) >= p.rule['min_inl'] and np.isfinite(st.get('lsr', np.nan)) and st.get('nboth', 0) >= 20:
            self.vote.add(self.own_id, pid, t_i, st['lsr'])
        if st.get('ok', 0) <= 0 or st.get('x6') is None and st.get('v') is None: return st, None
        x6 = st['x6'] if st.get('x6') is not None else np.array([0.0, 0.0, np.arctan2(st['v'][1], st['v'][0]), *st['v'][2:5]])
        v = rule15.vio_world_estimate(pose_i, (h['roll'], h['pitch'], h['yaw'], h['x'], h['y'], h['z']), x6)
        s = {'ok': st['ok'], 'ninl': st['ninl'], 'ninl_raw': st['ninl_raw'], 'amb': st['amb'], 'sy': st['sy'], 'st': st['st'], 'i': i, 'j': h['kf'],
             'h_x': 0.0, 'v_yaw': v[0], 'v_x': v[1], 'v_y': v[2], 'v_z': v[3], 'p_peer': (h['x'], h['y'], h['z']), 'epoch': h['epoch'], 'own_epoch': self.own_epoch}
        return st, s
    def vote_message(self, t):
        """datagram with this drone's depth-ratio summaries (broadcast at ~1 Hz when a vote is used)."""
        return PC.Codec.encode_vote(self.own_id, t, self.vote.summary(self.own_id, t)) if self.vote is not None else None
    def on_vote(self, dgram):
        try:
            m = PC.Codec.decode_vote(dgram)
        except Exception:
            return
        if m is not None and self.vote is not None and m['drone'] != self.own_id: self.vote.merge(m['drone'], m['entries'], m['t'])
    def feed(self, pid, s, t):
        """an accepted solve: lock rule while searching, maintenance once locked. Returns the peer's current output (estimate or None).
        Round 16: a solve of an old peer or own VIO epoch (queued before a reset) is dropped."""
        p = self.peers[pid]
        if s is None: return p.maint.est if p.lock.locked else None
        if ('epoch' in s and p.epoch is not None and s['epoch'] != p.epoch) or s.get('own_epoch', self.own_epoch) != self.own_epoch:
            p.n_stale += 1; return p.maint.est if p.lock.locked else None
        s = dict(s); s['t'] = t
        if not p.lock.locked:
            est = p.lock.add(s)
            if est is not None:
                p.maint.start(est, t); p.t_lock = t; p.last_maint = t; p.next_maint = t + self._mper(p, t); p.buf = {}; p.events.append((t, 'lock', est.tolist()))
                if p.link is not None: p.link.own_lock(t, est)
            return est
        prev = None if p.maint.est is None else p.maint.est.copy(); out = p.maint.update(s, t, s.get('p_peer'))
        if p.maint.withdrawn:
            p.lock.reset(); p.next_maint = None; p.last_maint = None; p.t_lock = None; p.events.append((t, 'withdraw', None))
            if p.link is not None: p.link.own_withdraw(t)
        elif p.link is not None and out is not None and (prev is None or not np.array_equal(prev, out)): p.link.own_update(t, out)
        return out
    # ---- round 16: pair protocol (rule15.PairLink; packet_codec estimate datagrams) ----
    def on_estimate(self, dgram, t):
        """a peer's estimate datagram (Codec.decode_est); ignored unless addressed to this drone by a known peer. Returns True if used."""
        m = PC.Codec.decode_est(dgram)
        if m is None or m['drone'] not in self.peers or self.peers[m['drone']].link is None: return False
        return self.peers[m['drone']].link.on_msg(m, t)
    def estimate_messages(self, t):
        """the estimate datagrams due at t (call at ~5 Hz or on events): one per peer with something to say."""
        out = []
        for pid, p in self.peers.items():
            if p.link is None: continue
            p.link.own_ep = self.own_epoch & 0xFF; m = p.link.message(t)
            if m is not None: out.append(PC.Codec.encode_est(self.own_id if self.own_id is not None else 0, m))
        return out
    def output(self, pid):
        """the two output streams for one peer: {'provisional', 'source' ('own' | 'adopted'), 'confirmed'} (rule15.PairLink.output);
        without the protocol, provisional = the own estimate and confirmed = None."""
        p = self.peers[pid]
        if p.link is not None: return p.link.output()
        return {'provisional': p.maint.est if p.lock.locked else None, 'source': 'own' if p.lock.locked else None, 'confirmed': None}


class SolveQueue:
    """round 16: pending solves of one GPU in the replay's order: availability time, the nearest own keyframe before the others, then
    received keyframe and own keyframe index (replay.py's candidate order). pop() drops a non-nearest solve of a peer that locked before it
    starts (replay COND2) and a solve of an old peer epoch. The flight loop pushes requests() results; test_stream15.py drives it."""
    def __init__(self, node): self.node = node; self.q = []; self.n = 0
    def push(self, t_av, pid, dec, cand, ranks=None):
        for n_, i in enumerate(cand):
            rank = (0 if n_ == 0 else 1) if ranks is None else ranks[n_]
            heapq.heappush(self.q, (t_av(i) if callable(t_av) else t_av, rank, dec['hdr']['kf'], i, self.n, pid, dec)); self.n += 1
    def push_jobs(self, t_av, jobs):
        """own_ready's solves, all available at t_av."""
        for pid, dec, i, rank in jobs: self.push(t_av, pid, dec, [i], [rank])
    def t_next(self): return self.q[0][0] if self.q else np.inf
    def pop(self):
        while self.q:
            t_av, rank, kf, i, _, pid, dec = heapq.heappop(self.q); p = self.node.peers[pid]
            if rank > 0 and p.lock.locked: continue
            if p.epoch is not None and dec['hdr']['epoch'] != p.epoch: continue
            return t_av, pid, dec, i
        return None
