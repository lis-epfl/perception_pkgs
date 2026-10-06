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
# Static start = what the estimator's forced-static initialiser needs (ov_init StaticInitializer): a window of
# init_window_time seconds whose two halves both have accelerometer excitation sqrt(sum|a_i - mean a|^2/(n-1)) below
# init_imu_thresh -- read from the estimator config the loop will run (defaults = selfcalib/configs/estimator_calib.yaml).
# That window must END within STATIC_START_MAX_S of the recording start: forced-static initialisation fires at the
# first window that passes, so a later one means it would initialise in flight. The accelerometer test alone is
# fooled by smooth flight (a takeoff-trimmed fleet recording reads 1.50 m/s^2 while rotating at 0.38 rad/s), and the
# initialiser takes the gyro bias from the window's mean rate, so the gyro SPREAD (norm of per-axis std; the mean
# holds the bias) must also stay below STATIC_GYRO_SPREAD. Measured on real still starts: fleet 0.007-0.011, UZH-FPV
# 0.004, EuRoC V1_01 (rotors idling) 0.055, TUM-VI room1 (hand-held) 0.126 rad/s; smooth flight 0.380.
INIT_WINDOW_S_DEFAULT = 3.0   # s, estimator init_window_time
INIT_IMU_THRESH_DEFAULT = 1.5 # m/s^2, estimator init_imu_thresh
STATIC_GYRO_SPREAD = 0.2      # rad/s
STATIC_START_MAX_S = 5.0      # s from the recording start
# Starting is not enough: the estimator must start BEFORE the vehicle moves, with time to settle, not as it lifts off.
# The gate until 2026-10-05 asked only for the start window and let through a recording with 2.5 s on the ground, whose
# window ended half a second after takeoff. With the data cut to 3 s before takeoff the initialiser fires at the instant
# of takeoff and the filter gets 6-8 zero-velocity updates instead of 35 or more: the calibration run's trajectory then
# starts 5 cm off on average (15 cm worst) against 2.7 cm, reproducibly, while the published calibration stays the
# same. From 4 s before takeoff on, every result matched the full recording (Orin NX, seven fleet recordings,
# 2026-10-06, docs/VALIDATION_MATRIX.md section 1). The gate measures the standstill with the initialiser's own test,
# from the start of the first window that passes; that reads about 0.45 s longer than the time to the PX4 takeoff flag,
# so the measurements support 4.5 s. 6 s is a deliberate margin on top: the initialiser's window plus the longest arming
# lead seen on the fleet (3.0 s), so the window that starts the filter holds no motor vibration. It rejects recordings
# that calibrate normally with less, among them the nxt10 flight of 4 June (4.6 s) used in the paper.
STATIC_MIN_STILL_S = 6.0      # s of standstill in total: the initialiser's window + settling (run_tool --min-still)
GRAVITY_Z_MIN = 8.0         # static accel z below -this = gravity down the z axis (PX4 FRD published as FLU): inverted axis.
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
    cfg_path = cfg_path or os.path.join(root, 'configs', 'estimator_calib.yaml')
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
def _init_params(config):
    """init_window_time and init_imu_thresh from an estimator yaml (defaults if absent)."""
    W, tau = INIT_WINDOW_S_DEFAULT, INIT_IMU_THRESH_DEFAULT
    if config and os.path.exists(config):
        txt = open(config).read()
        m = re.search(r'^init_window_time:\s*([0-9.eE+-]+)', txt, re.M); W = float(m.group(1)) if m else W
        m = re.search(r'^init_imu_thresh:\s*([0-9.eE+-]+)', txt, re.M); tau = float(m.group(1)) if m else tau
    return W, tau


def _excitation(A):
    """the initialiser's accelerometer excitation: sqrt(sum |a_i - mean a|^2 / (n - 1)), m/s^2"""
    return float(np.sqrt(((A - A.mean(0)) ** 2).sum() / max(len(A) - 1, 1))) if len(A) > 5 else float('inf')


def static_start_gate(bag, imu_topic=None, config=None, init_window=None, init_thresh=None, min_still=None):
    rec = _rec(bag)
    W, tau = _init_params(config)
    W = init_window if init_window else W; tau = init_thresh if init_thresh else tau
    need = STATIC_MIN_STILL_S if min_still is None else float(min_still)
    settle = max(0.0, need - W)     # standstill required after the initialiser's window
    T, G, A = [], [], []
    t0 = None
    for t, g, a in rec.imu():
        if t0 is None:
            t0 = t
        if t - t0 > STATIC_START_MAX_S + settle + 0.5:
            break
        T.append(t - t0)
        G.append(g)
        A.append(a)
    T, G, A = np.array(T), np.array(G), np.array(A)
    if len(T) < 50:
        return {'pass': False, 'reason': 'no/too-few IMU messages', 'init_window_end_s': None}
    # first window [t - W, t] (t <= STATIC_START_MAX_S) that the initialiser would accept AND in which the gyro is still
    found = None; best = None
    for t in np.arange(W, min(T[-1], STATIC_START_MAX_S) + 1e-9, 0.05):
        h1 = (T > t - W) & (T <= t - W / 2); h2 = (T > t - W / 2) & (T <= t)
        e = max(_excitation(A[h1]), _excitation(A[h2]))
        m = h1 | h2
        spread = float(np.linalg.norm(G[m].std(0))) if m.sum() > 5 else float('inf')
        if best is None or max(e / tau, spread / STATIC_GYRO_SPREAD) < max(best[1] / tau, best[2] / STATIC_GYRO_SPREAD):
            best = (float(t), e, spread)
        if e < tau and spread < STATIC_GYRO_SPREAD:
            found = (float(t), e, spread); break
    ref = found or best

    # Gravity direction and magnitude at rest. Variance alone cannot see an inverted
    # axis: an IMU published in PX4's FRD body frame sits perfectly still, passes every
    # variance test, and then diverges the estimator by kilometres with no earlier
    # warning -- the calibration states still converge self-consistently, so the run
    # ends in a confident wrong verdict. This is the cheapest place to catch it.
    sm = (T > ref[0] - W) & (T <= ref[0])
    a_mean = A[sm].mean(0) if sm.sum() >= 10 else A[:min(len(A), 200)].mean(0)
    a_norm = float(np.linalg.norm(a_mean))
    # Fail only on what an IMU mounting cannot legitimately produce: gravity pointing DOWN the z axis (the PX4 FRD
    # vs FLU inversion) or a wrong magnitude (units). Any other orientation is a valid mounting -- the initialiser
    # estimates the gravity direction itself (EuRoC's IMU reads gravity along +x).
    grav_ok = bool(a_mean[2] > -GRAVITY_Z_MIN and abs(a_norm - 9.81) < GRAVITY_NORM_TOL)

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
        grav_reason = 'gravity check failed (a=%+.2f,%+.2f,%+.2f)' % tuple(a_mean)

    static_ok = found is not None
    # settling: the same test must keep passing until `settle` after the first window that passed
    still_until = None
    if static_ok:
        still_until = found[0]
        for t in np.arange(found[0] + 0.05, found[0] + settle + 1e-9, 0.05):
            h1 = (T > t - W) & (T <= t - W / 2); h2 = (T > t - W / 2) & (T <= t)
            m = h1 | h2
            if t > T[-1] or max(_excitation(A[h1]), _excitation(A[h2])) >= tau or \
                    (float(np.linalg.norm(G[m].std(0))) if m.sum() > 5 else float('inf')) >= STATIC_GYRO_SPREAD:
                break
            still_until = float(t)
    standstill = (still_until - (found[0] - W)) if static_ok else 0.0
    settle_ok = bool(static_ok and standstill >= need - 0.05)
    still_reason = ('the vehicle stays still for only %.1f s (%.1f s needed: the estimator takes %.1f s of standstill to '
                    'start and needs %.1f s more to settle before the vehicle moves; its start window ends %.2f s into the '
                    'data and the stillness ends at %.2f s)'
                    % (standstill, need, W, settle, found[0], still_until)) if static_ok and not settle_ok else ''
    reason = ('no %.1f s still window ending within the first %.0f s (initialiser test: accelerometer excitation < %.2f '
              'm/s^2 in both halves, gyro spread < %.2f rad/s; best window ending %.2f s: %.2f m/s^2, %.3f rad/s)'
              % (W, STATIC_START_MAX_S, tau, STATIC_GYRO_SPREAD, best[0], best[1], best[2])) if not static_ok else (still_reason or grav_reason)
    advice = ('re-record, leaving the vehicle untouched for at least %g s before takeoff' % need) if static_ok and not settle_ok \
        else 're-record starting on the ground'
    return {'pass': bool(static_ok and settle_ok and grav_ok), 'init_window_s': W, 'init_imu_thresh': tau,
            'min_still_s': need, 'standstill_s': round(standstill, 2), 'advice': advice,   # standstill_s is counted up to min_still only
            'init_window_end_s': round(ref[0], 2), 'accel_excitation': round(ref[1], 3), 'gyro_spread': round(ref[2], 4),
            'reason': reason, 'static_accel_mean': [round(float(v), 4) for v in a_mean],
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
    ap.add_argument('--config', default=None, help='estimator yaml the loop will run (init_window_time, init_imu_thresh)')
    ap.add_argument('--min-still', type=float, default=None,
                    help='seconds of standstill the recording must begin with (default %g)' % STATIC_MIN_STILL_S)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = {'bag': a.bag}
    if 'static' in a.gates:
        res['static'] = static_start_gate(a.bag, config=a.config, min_still=a.min_still)
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
