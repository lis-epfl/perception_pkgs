"""The receiver of one drone: relloc2's node15.Node (round 16, unchanged in core/) with both outputs carried for every peer.

  4-DoF  rule15.Maintenance (the deployed output: yaw about gravity + translation, 0.7 blend at 1 packet/s, withdraw at 20 cm / 2 deg)
  6-DoF  six.M6 (round 23 ship1_6y: full rotation, geodesic 0.7 blend, withdraw at 20 cm / 2 deg of yaw)
Both start at the same lock (the lock rule decides on the 4-DoF values, unchanged) and see the same maintenance solves. `six_primary`
picks the one whose withdraw test ends the lock (the peer is searched again); the other one is a shadow: a solve it would withdraw on is
skipped and counted (n_shadow_withdraw), and it restarts at the next lock. With six_primary=False the lock, maintenance and pair protocol
are node15.Node.feed itself; with True they are node15.Node.feed with M6 in place of Maintenance, and the pair protocol (the estimate
datagrams and the confirmation) gets the 4-DoF transform that maps the peer's current position where the 6-DoF one does (six.to4)."""
import numpy as np
from . import paths, six
import node15, rule15


class LiveNode(node15.Node):
    def __init__(self, *a, six_primary=False, **kw):
        super().__init__(*a, **kw); self.six_primary = bool(six_primary)

    def add_peer(self, pid, calib, maint=None, rig=None, codec=None):
        super().add_peer(pid, calib, maint=maint, rig=rig, codec=codec)
        p = self.peers[pid]; p.m6 = six.M6(rule=p.rule); p.n_shadow_withdraw = 0; p.n_lock = 0; p.n_withdraw = 0; p.p_peer = None

    def solve_pair(self, pid, dec, i):
        st, s = super().solve_pair(pid, dec, i)
        if s is not None:
            h = dec['hdr']; x6 = st.get('x6')
            s['w6'] = six.w6_of_solve((h['roll'], h['pitch'], h['yaw'], h['x'], h['y'], h['z']), (s['v_yaw'], s['v_x'], s['v_y'], s['v_z']), x6)
            s['x6'] = None if x6 is None else [float(v) for v in x6]
        return st, s

    def set_own_epoch(self, t, epoch):
        """the own VIO epoch is the one in the own packet headers (live.vio.VioBuffer): node15's own_epoch_reset with that value instead of
        +1, so the estimate datagrams and the headers name the same epoch (a mismatch makes the peer drop every estimate message)."""
        epoch = int(epoch)
        if epoch == self.own_epoch: return False
        self.own_epoch = epoch
        for p in self.peers.values():
            p._reset(t, 'own_epoch_reset', self.own_epoch)
            if p.link is not None: p.link.set_own_epoch(self.own_epoch, t)
        return True

    def _stale(self, p, s):
        return ('epoch' in s and p.epoch is not None and s['epoch'] != p.epoch) or s.get('own_epoch', self.own_epoch) != self.own_epoch

    def feed(self, pid, s, t):
        p = self.peers[pid]
        if s is not None and s.get('p_peer') is not None: p.p_peer = np.asarray(s['p_peer'], float)
        if not self.six_primary:
            was = p.lock.locked
            out = super().feed(pid, s, t)
            if s is None or self._stale(p, s): return out
            if not was and p.lock.locked:
                p.m6.start(six.lock6(p.lock.members, p.lock.estimate), t); p.n_lock += 1
            elif was and p.lock.locked:
                e6 = p.m6.est.copy() if p.m6.est is not None else None
                p.m6.update(dict(s, t=t), t, s.get('p_peer'))
                if p.m6.withdrawn: p.m6.est = e6; p.m6.withdrawn = False; p.n_shadow_withdraw += 1
            elif was and not p.lock.locked:
                p.m6.reset(); p.n_withdraw += 1
            return out
        return self._feed6(pid, s, t)

    def _feed6(self, pid, s, t):
        """node15.Node.feed with M6 deciding (lines as there; marked: what differs)."""
        p = self.peers[pid]
        if s is None: return p.maint.est if p.lock.locked else None
        if self._stale(p, s):
            p.n_stale += 1; return p.maint.est if p.lock.locked else None
        s = dict(s); s['t'] = t
        if not p.lock.locked:
            est = p.lock.add(s)
            if est is not None:
                e6 = six.lock6(p.lock.members, est)                                     # differs: the 6-DoF lock estimate
                p.m6.start(e6, t); p.n_lock += 1
                p.maint.start(est, t); p.t_lock = t; p.last_maint = t; p.next_maint = t + self._mper(p, t); p.buf = {}; p.events.append((t, 'lock', est.tolist()))
                if p.link is not None: p.link.own_lock(t, six.to4(e6, s.get('p_peer')))  # differs: the protocol carries the 6-DoF state
            return est
        prev = None if p.m6.est is None else p.m6.est.copy()
        out6 = p.m6.update(s, t, s.get('p_peer'))                                       # differs: M6 decides
        e4 = None if p.maint.est is None else p.maint.est.copy()
        p.maint.update(s, t, s.get('p_peer'))                                           # the 4-DoF shadow
        if p.maint.withdrawn: p.maint.est = e4; p.maint.withdrawn = False; p.n_shadow_withdraw += 1
        if p.m6.withdrawn:
            p.lock.reset(); p.next_maint = None; p.last_maint = None; p.t_lock = None; p.events.append((t, 'withdraw', None)); p.maint.reset(); p.n_withdraw += 1
            if p.link is not None: p.link.own_withdraw(t)
        elif p.link is not None and out6 is not None and (prev is None or not np.array_equal(prev, out6)): p.link.own_update(t, six.to4(out6, s.get('p_peer')))
        return p.maint.est if p.lock.locked else None

    def state(self, pid):
        """what the node publishes for one peer: status, both estimates (None when not locked), confirmation, pair-protocol source."""
        p = self.peers[pid]; o = self.output(pid); locked = p.lock.locked
        e4 = None if not locked or p.maint.est is None else np.asarray(p.maint.est, float)
        e6 = None if not locked or p.m6.est is None else np.asarray(p.m6.est, float)
        if p.link is not None: status = p.link.status if o['source'] != 'adopted' else 'adopted'
        else: status = 'provisional' if locked else 'searching'
        if not locked and o['source'] is None: status = 'searching' if status in ('none', 'withdrawn') else status
        prim = (six.to4(e6, p.p_peer) if e6 is not None else None) if self.six_primary else e4
        return {'locked': bool(locked), 'status': status, 'confirmed': o['confirmed'] is not None, 'source': o['source'], 'e4': e4, 'e6': e6,
                'adopted': (np.asarray(o['provisional'], float) if o['source'] == 'adopted' else None), 'primary': '6dof' if self.six_primary else '4dof',
                'primary4': prim, 'degraded': bool((p.m6 if self.six_primary else p.maint).degraded) if locked else False, 'p_peer': p.p_peer,
                't_lock': p.t_lock, 'n_lock': p.n_lock, 'n_withdraw': p.n_withdraw, 'n_shadow_withdraw': p.n_shadow_withdraw}
