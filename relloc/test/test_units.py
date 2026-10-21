#!/usr/bin/env python3
"""CPU unit tests of the new live code (no GPU, no ROS): python3 test_units.py -> prints one line per check, exit 1 on a failure.
  w6      six.w6_of_solve equals the direct 6-DoF composition R_vA R_bb R_vB^-1, t = p_vA + R_vA t_bb - R_w6 p_vB, and the 4-DoF one at zero tilt
  lock6   at zero tilt the 6-DoF lock estimate equals rule15.fuse of the same members
  M6      at zero tilt, M6 (chordal blend) gives rule15.Maintenance's estimate and withdrawal on a random solve sequence; the geodesic blend
          stays within 1e-6 deg / 1e-9 m of it on the same sequence after the rotation part is mapped to yaw (both are 0.7 blends)
  lens    unproject with zero lens terms = the 8-number KB4 path bit for bit; reference values with terms (relloc2 review21 md21)
  calib   calibration datagram round trip"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live import paths, six, lens, calib as CAL
import rule15
from scipy.spatial.transform import Rotation as Rot
rng = np.random.default_rng(7); FAIL = []
def check(name, ok, info=''):
    print(f'{"ok  " if ok else "FAIL"} {name} {info}'); (None if ok else FAIL.append(name))

def direct(own, peer, x6):
    RvA = Rot.from_euler('ZYX', [own[2], own[1], own[0]]); RvB = Rot.from_euler('ZYX', [peer[2], peer[1], peer[0]])
    tA = Rot.from_euler('ZYX', [0.0, own[1], own[0]]); tB = Rot.from_euler('ZYX', [0.0, peer[1], peer[0]]); R6 = Rot.from_euler('ZYX', [x6[2], x6[1], x6[0]])
    Rbb = tA.inv() * R6 * tB; tbb = tA.inv().apply(x6[3:6]); Rw = RvA * Rbb * RvB.inv()
    return Rw.as_matrix(), np.asarray(own[3:6]) + RvA.apply(tbb) - Rw.apply(peer[3:6])

e1 = e2 = e3 = 0.0
for _ in range(500):
    own = (*rng.normal(0, 0.05, 2), rng.uniform(-np.pi, np.pi), *rng.normal(0, 3, 3)); peer = (*rng.normal(0, 0.05, 2), rng.uniform(-np.pi, np.pi), *rng.normal(0, 3, 3))
    x6 = np.array([*rng.normal(0, 0.01, 2), rng.uniform(-np.pi, np.pi), *rng.normal(0, 3, 3)])
    v4 = rule15.vio_world_estimate(own, peer, x6); w = six.w6_of_solve(peer, v4, x6); R, t = direct(own, peer, x6)
    e1 = max(e1, np.abs(w[:9] - R.ravel()).max()); e2 = max(e2, np.abs(w[9:] - t).max())
    x0 = x6.copy(); x0[:2] = 0; v0 = rule15.vio_world_estimate(own, peer, x0); w0 = six.w6_of_solve(peer, v0, x0)
    e3 = max(e3, np.abs(w0[:9] - Rot.from_euler('z', v0[0], degrees=True).as_matrix().ravel()).max() + np.abs(w0[9:] - np.array(v0[1:4])).max())
check('w6 = direct 6-DoF composition', e1 < 1e-12 and e2 < 1e-9, f'(rotation {e1:.1e}, translation {e2:.1e} m)')
check('w6 at zero tilt = 4-DoF', e3 < 1e-9, f'({e3:.1e})')

ev = [];
for k in range(5):
    own = (0, 0, rng.uniform(-3, 3), *rng.normal(0, 2, 3)); peer = (0, 0, rng.uniform(-3, 3), *rng.normal(0, 2, 3)); x6 = np.array([0, 0, 0.3 + 0.01 * k, 1.0, 2.0, 0.1])
    v = rule15.vio_world_estimate(own, peer, x6); ev.append({'v_yaw': v[0], 'v_x': v[1], 'v_y': v[2], 'v_z': v[3], 'ninl': int(rng.integers(200, 900)), 'w6': six.w6_of_solve(peer, v, x6)})
f4 = rule15.fuse(ev, list(range(5)), rule15.RULES15['drones']); l6 = six.lock6(ev)
d = abs((six.to4(l6)[0] - f4[0] + 180) % 360 - 180) + np.abs(l6[9:] - f4[1:]).max()
check('lock6 at zero tilt = rule15.fuse', d < 1e-9, f'({d:.1e})')

def seq(n=400):
    out = []; yaw, t = 30.0, np.array([1.0, -2.0, 0.3])
    for k in range(n):
        jump = rng.random() < 0.03
        y = yaw + rng.normal(0, 0.3) + (rng.choice([-1, 1]) * 3.0 if jump else 0); tt = t + rng.normal(0, 0.03, 3) + (rng.normal(0, 0.3, 3) if jump else 0)
        s = {'ok': 1, 'ninl': int(rng.integers(150, 700)), 'ninl_raw': int(rng.integers(200, 900)), 'amb': float(rng.uniform(0, 0.6)), 'h_x': 0.0, 'v_yaw': y, 'v_x': tt[0], 'v_y': tt[1], 'v_z': tt[2]}
        s['w6'] = np.concatenate([Rot.from_euler('z', y, degrees=True).as_matrix().ravel(), tt]); out.append((s, rng.normal(0, 3, 3)))
    return out
for blend, tol in (('chord', 1e-9), ('geo', None)):
    m4 = rule15.Maintenance(rule=rule15.RULES15['drones']); m6 = six.M6(blend=blend, rule=rule15.RULES15['drones']); worst = 0.0; same_w = True; nw = 0; e0 = np.array([30.0, 1.0, -2.0, 0.3])
    for k, (s, p) in enumerate(seq()):
        if m4.est is None:
            m4.start(e0, k); m6.start(np.concatenate([Rot.from_euler('z', e0[0], degrees=True).as_matrix().ravel(), e0[1:]]), k)
        a = m4.update(s, k, p); b = m6.update(s, k, p); same_w &= (a is None) == (b is None); nw += a is None
        if a is not None and b is not None: worst = max(worst, abs((six.to4(b)[0] - a[0] + 180) % 360 - 180), np.abs(b[9:] - a[1:]).max())
    if tol is not None: check(f'M6 ({blend}) at zero tilt = rule15.Maintenance', same_w and worst < tol, f'({nw} withdrawals, max diff {worst:.1e})')
    else: check(f'M6 ({blend}) at zero tilt: same withdrawals as rule15.Maintenance', same_w, f'({nw} withdrawals; yaw differs by up to {worst:.2e} deg/m: 0.7 theta vs the circular-mean fraction)')

intr8 = [252.0, 252.4, 580.6, 334.9, 0.0933, -0.0303, -0.0024, 0.0008]; uv = rng.uniform([0, 0], [1280, 800], (2000, 2))
a = lens.unproject(uv, intr8); b = lens.unproject(uv, intr8 + [0.0, 0.0, 0.0])
check('lens: zero terms = 8-number path', np.array_equal(a, b))
ref_uv = np.array([[100.0, 80.0], [640.0, 400.0], [1200.0, 700.0]]); intr11 = intr8 + [1.2e-3, -8e-4, 0.0]
REF = np.array([[-0.6391871522, -0.3410147846, -0.6893103083], [0.22864238, 0.2499399599, 0.9408786736], [0.4728712024, 0.2756817006, -0.836894513]])
got = lens.unproject(ref_uv, intr11); check('lens: reference rays with terms (md21)', np.abs(got - REF).max() < 1e-9, f'(max diff {np.abs(got - REF).max():.1e})')

R = [Rot.random(random_state=k).as_matrix() for k in range(4)]; c0 = CAL.Calib(4, [intr11] * 4, R, [np.array([0.1, -0.05, 0.02])] * 4, 1280, 800)
g = c0.encode(6, session=0xDEADBEEF); r = CAL.Calib.decode(g)
check('calib datagram round trip', r is not None and r[0] == 6 and c0.same(r[1]) and r[2] == 0xDEADBEEF and len(g) == 749, f'({len(g)} B)')
check('calib datagram rejects garbage', CAL.Calib.decode(g[:-1]) is None and CAL.Calib.decode(b'RC' + bytes(30)) is None)
from live import receiver as RCV
import node15, packet_codec as PC
P_ = np.load(paths.PCA); codec = PC.Codec(P_['mu'], P_['basis'], 1280, 800)
def dummy_solve(A, B): return {'ok': 0, 'v': None}
nd = RCV.LiveNode(codec, dummy_solve, lambda i: (0.0, (0.0,) * 6, {}, None), [], own_id=3, rig='drone4')
nd.add_peer(6, node15.PeerCalib(lambda c, uv: None, [np.eye(3)] * 4, [np.zeros(3)] * 4, 1280, 800))
ch = nd.set_own_epoch(1.0, 2); nd.estimate_messages(1.0); p6 = nd.peers[6]
check('own epoch follows the header epoch', ch and nd.own_epoch == 2 and p6.link.own_ep == 2 and not nd.set_own_epoch(1.1, 2))
print('ALL PASSED' if not FAIL else f'FAILED: {FAIL}'); sys.exit(1 if FAIL else 0)
