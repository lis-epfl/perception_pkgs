#!/usr/bin/env python3
"""Pre-flight data gates (paper §6.1): static-start, timing-health, image-health.

Each gate reads a recording through bagio (which handles sqlite3 vs mcap, one file
or several, raw vs compressed images) and returns a dict verdict.
CLI: python3 gates.py <recording> [--chain chain.yaml] [--gates static,timing,image] [--json out.json]
"""
import argparse, json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagio import Recording
from chainio import parse_chain, kb4_radius

# ---------------- thresholds (calibrated on fleet data, see docs/VALIDATION_MATRIX.md) ----
STATIC_GYRO_STD = 0.03      # rad/s per-axis-norm std within a bin counted as static
STATIC_ACC_STD = 0.20       # m/s^2 (on-ground with electronics running)
STATIC_MIN_S = 1.5          # required contiguous static duration
STATIC_SEARCH_S = 12.0      # static phase must appear within this window from bag start
GRAVITY_Z_MIN = 8.0         # static accel z must exceed this (FLU: +g). Catches an inverted axis.
GRAVITY_NORM_TOL = 1.5      # |static accel| must be within this of 9.81 (catches unit errors)
TIMING_DROP_RATE = 0.03     # fraction of dropped frames above this → flag (fleet ≤1.5%, nxt1 ≈8%)
TIMING_NOISE_STD_MS = 25.0  # windowed(10 s) grid-residual std above this → flag (fleet ≤16, nxt1 ≈35)
TIMING_SYNC_MS = 1.0        # cameras' stamps must agree to this (synchronized-trigger rig)
IMAGE_PATCH = 64            # px patch size
IMAGE_GRAD_THR = 25.0       # patch is 'sharp once' if per-frame p99 |Sobel| ever exceeds this
IMAGE_DEAD_FRAC = 0.05      # camera flagged if more than this fraction of in-circle patches never sharp
IMAGE_SUB = 8               # process every Nth frame

# Lens half-FOV per camera (rad), used to turn intrinsics into an image-circle radius.
# Read from the estimator config so the tool's circle geometry and the estimator's own
# fisheye mask cannot drift apart — same keys, same semantics as VioManagerOptions.h:318-338
# (global `mask_fisheye_theta_max`, per-camera override `mask_fisheye_theta_max{i}`).
THETA_MAX_DEFAULT = 1.83


def _load_theta_max(cfg_path=None):
    """-> dict {cam_index: theta_max}. Falls back to the study rig if the config is absent."""
    root = os.environ.get('SCT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = cfg_path or os.path.join(root, 'configs', 'estimator_fleet.yaml')
    theta = {}
    try:
        txt = open(cfg_path).read()
    except OSError:
        return {0: 1.83, 1: 1.83, 2: 1.83, 3: 1.87}
    m = re.search(r'^\s*mask_fisheye_theta_max\s*:\s*([0-9.eE+-]+)', txt, re.M)
    glob = float(m.group(1)) if m else THETA_MAX_DEFAULT
    for i in range(16):
        mi = re.search(r'^\s*mask_fisheye_theta_max%d\s*:\s*([0-9.eE+-]+)' % i, txt, re.M)
        theta[i] = float(mi.group(1)) if mi else glob
    return theta


THETA_MAX = _load_theta_max()


def _rec(bag):
    """Accept a path or an already-open Recording. Which files the recording spans,
    and how they are stored, is bagio's problem rather than each gate's."""
    return bag if isinstance(bag, Recording) else Recording.open(bag)


# ---------------- gate 1: static start ----------------
def static_start_gate(bag, imu_topic=None):
    rec = _rec(bag)
    T, G, A = [], [], []
    t0 = None
    for t, g, a in rec.imu():
        if t0 is None:
            t0 = t
        if t - t0 > STATIC_SEARCH_S + 2:
            break
        T.append(t - t0)
        G.append(g)
        A.append(a)
    T, G, A = np.array(T), np.array(G), np.array(A)
    if len(T) < 50:
        return {'pass': False, 'reason': 'no/too-few IMU messages', 'static_s': 0.0}
    # 0.5 s bins: static iff gyro std and accel std below thresholds on all axes
    bins = np.arange(0, min(T[-1], STATIC_SEARCH_S), 0.5)
    static = []
    for b in bins:
        m = (T >= b) & (T < b + 0.5)
        if m.sum() < 10:
            static.append(False); continue
        static.append(bool(G[m].std(0).max() < STATIC_GYRO_STD and A[m].std(0).max() < STATIC_ACC_STD))
    # longest contiguous static run starting in the search window
    best = run = 0.0
    for s in static:
        run = run + 0.5 if s else 0.0
        best = max(best, run)
    g0 = float(G[T < min(3.0, T[-1])].std(0).max())

    # Gravity direction and magnitude at rest. Variance alone cannot see an inverted
    # axis: an IMU published in PX4's FRD body frame sits perfectly still, passes every
    # variance test, and then diverges the estimator by kilometres with no earlier
    # warning -- the calibration states still converge self-consistently, so the run
    # ends in a confident wrong verdict. This is the cheapest place to catch it.
    sm = T < min(STATIC_MIN_S, T[-1])
    a_mean = A[sm].mean(0) if sm.sum() >= 10 else A[:min(len(A), 200)].mean(0)
    a_norm = float(np.linalg.norm(a_mean))
    grav_ok = bool(a_mean[2] > GRAVITY_Z_MIN and abs(a_norm - 9.81) < GRAVITY_NORM_TOL)

    if grav_ok:
        grav_reason = ''
    elif a_mean[2] < -GRAVITY_Z_MIN:
        grav_reason = ('gravity points the WRONG WAY on IMU z (a_z=%+.2f, expected about +9.81): '
                       'the IMU axis convention is inverted. ROS/REP-103 expects FLU; PX4 '
                       'publishes body-frame FRD -- convert (x,y,z)->(x,-y,-z).' % a_mean[2])
    elif abs(a_norm - 9.81) >= GRAVITY_NORM_TOL:
        grav_reason = ('static accelerometer magnitude %.2f m/s^2, expected ~9.81 '
                       '(units or scaling wrong).' % a_norm)
    else:
        grav_reason = ('gravity not aligned with +z at rest (a=%+.2f,%+.2f,%+.2f); check the '
                       'IMU axis convention.' % tuple(a_mean))

    static_ok = best >= STATIC_MIN_S
    # The missing-static-phase reason takes priority; gravity is only meaningful once
    # we know the vehicle actually held still.
    reason = ('no >=%.1fs static phase in first %.0fs (longest %.1fs)'
              % (STATIC_MIN_S, STATIC_SEARCH_S, best)) if not static_ok else grav_reason
    return {'pass': bool(static_ok and grav_ok), 'static_s': best,
            'gyro_std_first3s': round(g0, 4), 'reason': reason,
            'static_accel_mean': [round(float(v), 4) for v in a_mean],
            'static_accel_norm': round(a_norm, 4), 'gravity_ok': grav_ok}


# ---------------- gate 2: timing health ----------------
def _fast_stamp(data):
    """Header stamp from CDR bytes (Image/CompressedImage: header is the first field)."""
    sec, nsec = struct.unpack_from('<iI', data, 4)
    return sec + nsec * 1e-9


def timing_gate(bag, cams=(0, 1, 2, 3)):
    """Flag timestamp pathologies that act as a time-varying t_d. Benign behavior passes:
    isolated frame drops and slow clock drift shared with the IMU. Flagged: dense drops
    (rate > TIMING_DROP_RATE), large short-term stamp noise (windowed grid-residual std >
    TIMING_NOISE_STD_MS), or cross-camera desynchronization (> TIMING_SYNC_MS)."""
    rec = _rec(bag)
    stamps = {c: [] for c in cams}
    # Stamps only -- images stay encoded here, so the timing gate never pays a
    # JPEG decode for a recording it is only measuring the clock of.
    for c, ts, _data in rec.images(cams):
        stamps[c].append(ts)
    out, ok = {}, True
    for c in cams:
        t = np.array(stamps[c])
        if len(t) < 300:
            out['cam%d' % c] = {'n': int(len(t)), 'flag': True, 'reason': 'too few frames'}
            ok = False
            continue
        dt = np.diff(t)
        med = float(np.median(dt))
        steps = np.maximum(np.round(dt / med), 1)
        drop_rate = float((steps > 1).sum()) / len(t)
        idx = np.concatenate([[0], np.cumsum(steps)])
        res = []
        W = 10.0
        for w0 in np.arange(0, t[-1] - t[0] - W / 2, W):
            m = (t - t[0] >= w0) & (t - t[0] < w0 + W)
            if m.sum() < 60:
                continue
            A = np.vstack([idx[m], np.ones(int(m.sum()))]).T
            cf, _, _, _ = np.linalg.lstsq(A, t[m], rcond=None)
            res.append((t[m] - A @ cf) * 1000)
        noise_std = float(np.concatenate(res).std()) if res else float('nan')
        flag = drop_rate > TIMING_DROP_RATE or not (noise_std < TIMING_NOISE_STD_MS)
        out['cam%d' % c] = {'n': int(len(t)), 'dt_median_ms': round(med * 1000, 3),
                            'drop_rate': round(drop_rate, 4), 'stamp_noise_std_ms': round(noise_std, 2),
                            'flag': bool(flag)}
        ok = ok and not flag
    # cross-camera synchronization (rig uses a shared trigger: stamps must match)
    n = min(len(stamps[c]) for c in cams)
    if n:
        desync = max(float(np.abs(np.array(stamps[c][:n]) - np.array(stamps[cams[0]][:n])).max()) * 1000
                     for c in cams)
        out['cross_cam_desync_ms'] = round(desync, 3)
        if desync > TIMING_SYNC_MS:
            ok = False
    out['pass'] = ok
    return out


# ---------------- gate 3: per-camera image health ----------------
def image_health_gate(bag, chain_yaml, cams=(0, 1, 2, 3), sub=IMAGE_SUB, max_frames=None, inject=None):
    """Track per-patch max-over-flight of the per-frame p99 |Sobel| gradient; a patch that never
    gets sharp marks a persistent optical defect. `inject(img, cam)->img` is a test hook."""
    import cv2
    cal, _ = parse_chain(chain_yaml)
    rec = _rec(bag)
    best, meta, cnt, done = {}, {}, {c: 0 for c in cams}, {c: 0 for c in cams}
    for c, _ts, data in rec.images(cams):
        cnt[c] += 1
        if cnt[c] % sub:
            continue                      # subsampled away -- never decoded
        if max_frames and done[c] >= max_frames:
            if all(done[k] >= max_frames for k in cams):
                break
            continue
        img = rec.decode(data)
        if inject is not None:
            img = inject(img, c)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        g = np.abs(gx) + np.abs(gy)
        H, W = g.shape
        ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
        blocks = g[:ph * IMAGE_PATCH, :pw * IMAGE_PATCH].reshape(ph, IMAGE_PATCH, pw, IMAGE_PATCH)
        p99 = np.percentile(blocks, 99, axis=(1, 3))
        if c not in best:
            best[c] = np.zeros((ph, pw), np.float32)
            meta[c] = (H, W, ph, pw)
        best[c] = np.maximum(best[c], p99.astype(np.float32))
        done[c] += 1
    out, ok = {}, True
    for c in cams:
        if c not in best:
            out['cam%d' % c] = {'flag': True, 'reason': 'no frames'}; ok = False; continue
        H, W, ph, pw = meta[c]
        f, k = cal[c]['f'], cal[c]['k']
        radius = kb4_radius(f, k, THETA_MAX.get(c, 1.83))
        cx, cy = f[2], f[3]
        yy, xx = np.mgrid[0:ph, 0:pw]
        pcx = (xx + 0.5) * IMAGE_PATCH
        pcy = (yy + 0.5) * IMAGE_PATCH
        incirc = ((pcx - cx) ** 2 + (pcy - cy) ** 2) < (0.92 * radius) ** 2
        dead = incirc & (best[c] < IMAGE_GRAD_THR)
        frac = float(dead.sum()) / max(int(incirc.sum()), 1)
        flag = frac > IMAGE_DEAD_FRAC
        out['cam%d' % c] = {'frames': done[c], 'patches_in_circle': int(incirc.sum()),
                            'dead_patches': int(dead.sum()), 'dead_frac': round(frac, 4), 'flag': bool(flag),
                            'dead_map': [[int(x), int(y)] for y, x in zip(*np.where(dead))]}
        ok = ok and not flag
    out['pass'] = ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--chain', default=None, help='chain yaml (needed for image gate circle geometry)')
    ap.add_argument('--gates', default='static,timing')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = {'bag': a.bag}
    if 'static' in a.gates:
        res['static'] = static_start_gate(a.bag)
    if 'timing' in a.gates:
        res['timing'] = timing_gate(a.bag)
    if 'image' in a.gates:
        assert a.chain, '--chain required for image gate'
        res['image'] = image_health_gate(a.bag, a.chain)
    res['pass'] = all(res[g]['pass'] for g in ('static', 'timing', 'image') if g in res)
    print(json.dumps({k: v for k, v in res.items() if k != 'image'} |
                     ({'image': {kk: {m: n for m, n in vv.items() if m != 'dead_map'} if isinstance(vv, dict) else vv
                                 for kk, vv in res['image'].items()}} if 'image' in res else {}), indent=1))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
