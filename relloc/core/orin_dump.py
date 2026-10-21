"""Device solve dump for offline lock-rule replay (round 14). Runs ON THE ORIN the receiver of orin_grid.py on every keyframe pair (i, j) that
the offset grid would request (own keyframes within +-WIN of each received keyframe, first HMAX s of each cell), each unique pair once, and
stores per solve: the pose hypothesis in the mocap chain (truth = identity), inlier count, sigma_yaw / sigma_t / ray spread, the RANSAC
ambiguity (best score of a hypothesis > 3 deg or > 0.3 m away from the winner, divided by the winner's score), the number of camera pairs,
the prefilter counts, and the MEASURED device compute time. replay.py rebuilds the grid (FIFO queue, conditional second solve, any lock rule)
from this table, so rules can be compared without re-solving.
Rig-agnostic camera-pair selection: PAIRSEL=top (CAMPAIRS best pairs with >= MIN_MNN mutual NN, rounds 10-13) | rel (pairs with
>= max(MIN_MNN, REL x best count), at most CAMPAIRS). Works for any camera count on either side (NA x NB pairs).
Env (round 15: the deployed values are the defaults): DATA_DIR, TAGX, TOPK(500), ITERS(150), WIN(0.25), OFFSTEP(4), HMAX(5), MINCONF(0.1),
THR_PX(the rig's rig15 thr_px, pixels at the pair's mean focal; THR in deg if THR_PX=0), MININL(30 = the lowest rule15 floor), WQ(0.3), CAMPAIRS(6), MIN_MNN(15),
PAIRSEL(rel), REL(0.5), LG_LAYERS(5), REFINE6(1), LO6(1), MATCHER(graph|trt), REFINE_Q(0 = deployed since round 15; 1: the 4-iteration weighted refine),
DIRSEL(B: correspondences with 3-D on both sides are refined through the sender's depth | A: through the receiver's | auto: A when the
  sender keyframe's 3-D share < DIRSEL_R(0.5) x the receiver's | health: A when the sender's header health < H_REL(0.7) x the receiver's own
  or < H_ABS(0.40) while the receiver's >= H_ABS; bytes from set_health() (node15) or HEALTH_NPZ (dumps)), RIG(drone4: rig15 rig whose
  THR_PX / FX set the test threshold at import, before and without setup(); round 16), WARMUP(8 untimed solves
before timing, so graph captures are not charged to the first grid cells). New columns: sy6/st6 (sigma from the 6-DoF normal matrix, scaled by
the weighted residual RMS), lsr/lsr_mad/nboth (median / MAD of the log ratio sender-depth : receiver-depth over inliers with 3-D on both sides)."""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import solver as S, lggraph as LGG
from modules.xfeat import XFeat
from scipy.spatial.transform import Rotation as Rot
E = os.environ.get; PAIR = None   # round 15: importable (node15.py); the grid dump runs in main()
os.environ.setdefault('LG_LAYERS', '5'); ITERS = int(E('ITERS', '150')); WIN = float(E('WIN', '0.25')); OFFSTEP = int(E('OFFSTEP', '4')); HMAX = float(E('HMAX', '5'))
MINCONF = float(E('MINCONF', '0.1')); THR = float(E('THR', '0.6')); MININL = int(E('MININL', '30')); WQ = float(E('WQ', '0.3'))   # round 16: MININL 30 as every round-15 dump (min over rule15.RULES15 floors)
CAMPAIRS = int(E('CAMPAIRS', '6')); MIN_MNN = int(E('MIN_MNN', '15')); PAIRSEL = E('PAIRSEL', 'rel'); REL = float(E('REL', '0.5')); TAGX = E('TAGX', '')
MNN_THR = float(E('MNN_THR', '0.82')); MATCHER = E('MATCHER', 'graph')
dev = 'cuda'
RIG = None; THR_PX = 0.0; FX = 0.0
def setup_rig(rig=None, fx=None):
    """round 16 (REVIEW15 S1 / item 1): the test threshold of a rig without an export: THR_PX (rig15.RIGS[rig]['thr_px'], env THR_PX
    overrides) at focal FX (fx = mean focal of both drones' cameras as export_cells writes fx_mean, else the rig's rig15 fx; env FX
    overrides). Called at import with env RIG (default drone4), so a flight node that never calls setup() runs the evaluated threshold;
    node15 calls it per peer. THR_PX=0 keeps THR (deg) from env THR."""
    global RIG, THR, THR_PX, FX
    import rig15 as _R15
    RIG = rig or E('RIG', 'drone4'); r = _R15.RIGS[RIG]
    THR_PX = float(E('THR_PX', str(r['thr_px'])))
    if THR_PX > 0: FX = float(E('FX', str(fx if fx else r['fx']))); THR = THR_PX * float(np.degrees(1.0 / FX))
    else: THR = float(E('THR', '0.6'))
    return THR
setup_rig()
def setup(pair, dd=None):
    """load one pair's exported cells (round 15: separated from import)."""
    global PAIR, DD, Z, ZF, M, NCA, NCB, W, H, tauA, tauB, yawA, yawB, pA, pB, readyA, readyB, t_air, THR, THR_PX, FX, RIG
    PAIR = pair
    DD = dd or E('DATA_DIR', 'data_grid_q'); Z = np.load(f'{DD}/{PAIR}.npz'); ZF = set(Z.files); M = json.load(open(f'{DD}/{PAIR}_meta.json'))
    NCA = M.get('ncamA', M['ncam']); NCB = M.get('ncamB', M['ncam']); W, H = M['image_size']
    tauA = Z['tauA']; tauB = Z['tauB']; yawA = Z['yawA']; yawB = Z['yawB']; pA = Z['pA']; pB = Z['pB']; readyA = M['readyA']; readyB = M['readyB']; t_air = M['t_air']
    import rig15 as _R15   # round 15: per-rig test threshold (rig15.RIGS[*]['thr_px']) for the rig whose focal length matches the export, else rig15.THR_PX
    _fx = float(M.get('fx_mean') or 0.0); _rig = next((r for r, v in _R15.RIGS.items() if _fx > 0 and abs(v['fx'] - _fx) / v['fx'] < 0.05), None); RIG = _rig or RIG
    THR_PX = float(E('THR_PX', str(_R15.RIGS[_rig]['thr_px'] if _rig else _R15.THR_PX)))
    if THR_PX > 0 and not M.get('fx_mean') and not E('FX'): THR = float(E('THR', '0.6')); print(f'[{PAIR}] no fx_mean in the export: fixed THR {THR} deg', flush=True); THR_PX = 0.0   # round 16: THR as before setup_rig() existed
    if THR_PX > 0:
        FX = float(E('FX', str(M.get('fx_mean', 0.0)))); assert FX > 0, 'THR_PX needs fx_mean in the pair meta (export_cells.py) or FX'
        THR = THR_PX * float(np.degrees(1.0 / FX)); print(f'[{PAIR}] THR_PX {THR_PX} px at f {FX:.1f} px -> {THR:.3f} deg', flush=True)
xf = XFeat(top_k=int(E('TOPK', '500'))); xf.dev = dev; os.environ['LG_HALF'] = '1'; G = LGG.LGGraph(xf, half=True)
if MATCHER == 'trt':
    import lg_trt; G = lg_trt.LGTRT(G)
def sync(): torch.cuda.synchronize()
_cache = {}
RAY_NORM = E('RAY_NORM', '1') == '1'   # round 15: unit rays (fp16 ray storage perturbed the arccos inlier test by up to ~1.3 deg per ray)
def _unit(d):
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12) if RAY_NORM else d
def load(side, k):
    key = (side, k)
    if key not in _cache:
        out = {}
        for c in range(NCA if side == 'A' else NCB):
            d = torch.from_numpy(Z[f'{side}_{k}_{c}_desc'].astype(np.float32)).to(dev); kp = torch.from_numpy(Z[f'{side}_{k}_{c}_kpts'].astype(np.float32)).to(dev)
            q = Z[f'{side}_{k}_{c}_q'].astype(np.float32) if f'{side}_{k}_{c}_q' in Z.files else np.ones(len(kp), np.float32)
            out[c] = {'dic': {'keypoints': kp, 'descriptors': d, 'scores': None, 'image_size': (W, H)}, 'gd': torch.nn.functional.normalize(d, dim=1),
                      'c': Z[f'{side}_{k}_{c}_c'].astype(np.float64), 'd': _unit(Z[f'{side}_{k}_{c}_rays'].astype(np.float64)), 'X': Z[f'{side}_{k}_{c}_X'].astype(np.float64), 'q': q}
            if E('DROP3D', '') == side: out[c]['X'] = np.full_like(out[c]['X'], np.nan)      # diagnostics: ignore this side's 3-D points
        _cache[key] = out
    return _cache[key]
def prefilter_counts(A, B):
    """mutual-NN counts for every camera pair (NA x NB) in one product when each side has equal-size packets."""
    KA = {len(A[c]['gd']) for c in A}; KB = {len(B[c]['gd']) for c in B}; na, nb = len(A), len(B)
    if len(KA) == 1 and len(KB) == 1 and min(KA) >= 16 and min(KB) >= 16:
        ka = KA.pop(); kb = KB.pop(); GA = torch.cat([A[c]['gd'] for c in range(na)]); GB = torch.cat([B[c]['gd'] for c in range(nb)])
        sim = (GA @ GB.T).view(na, ka, nb, kb); m12 = sim.argmax(3); m21 = sim.argmax(1); ia = torch.arange(ka, device=dev)
        best = torch.gather(m21, 2, m12.permute(0, 2, 1)); val = torch.gather(sim.permute(0, 2, 1, 3), 3, m12.permute(0, 2, 1)[..., None])[..., 0]
        c = ((best == ia[None, None, :]) & (val >= MNN_THR)).sum(2).cpu().numpy(); return {(ca, cb): int(c[ca, cb]) for ca in range(na) for cb in range(nb)}
    cnt = {}
    for ca in range(na):
        for cb in range(nb):
            if len(A[ca]['gd']) < 16 or len(B[cb]['gd']) < 16: cnt[(ca, cb)] = 0; continue
            sim = A[ca]['gd'] @ B[cb]['gd'].T; m12 = sim.argmax(1); m21 = sim.argmax(0); idx = torch.arange(len(m12), device=dev)
            cnt[(ca, cb)] = int(((m21[m12] == idx) & (sim[idx, m12] >= MNN_THR)).sum())
    return cnt
def select_pairs(cnt):
    order = sorted(cnt.items(), key=lambda kv: -kv[1]); best = order[0][1] if order else 0
    if PAIRSEL == 'rel': thr = max(MIN_MNN, REL * best)
    else: thr = MIN_MNN
    return [cp for cp, v in order[:CAMPAIRS] if v >= thr]
DIRSEL = E('DIRSEL', 'B')   # round 15: 'A' refines correspondences with 3-D on both sides through the receiver's own depth
# 'auto' (round 15, GT-free, no bytes): per solve, use the receiver's depth when the sender keyframe's share of keypoints with depth is
# below DIRSEL_R x the receiver keyframe's (a sender whose VIO/triangulation is weak ships few 3-D points: nxt1 ~0.31 vs 0.95-1.0)
# The share is a rolling median over the side's last DIRSEL_WIN keyframes (default 10 = 2 s at 5 Hz, past only), what a receiver sees in
# the packets it has heard; per keyframe (DIRSEL_WIN=1) it is too noisy: ~30 % of nxt1 keyframes pass 0.5 x and carry its wrong locks.
DIRSEL_R = float(E('DIRSEL_R', '0.5')); DIRSEL_WIN = int(E('DIRSEL_WIN', '10')); _DIR = ['A' if DIRSEL == 'A' else 'B']; _CUR = [None]; _SH = {}
# round 16 (REVIEW15 item 10, review15/exp/dirsel_health): DIRSEL='health' uses the receiver's depth when the sender's header health byte
# (share of fresh 3-D points seen in >= 3 views, median of its last 10 keyframes, 0 before its first triangulation) is < H_REL x the
# receiver's own, or < H_ABS while the receiver's is >= H_ABS. Health values are in [0, 1] (byte / 255); unknown -> the sender's depth.
H_REL = float(E('H_REL', '0.7')); H_ABS = float(E('H_ABS', '0.40')); _HL = {}; _HLT = [None, None]
def set_health(own, peer):
    """node15: health of the receiver's own keyframe and of the sender's (header byte / 255) for the next solve; None = unknown."""
    _HLT[0] = own; _HLT[1] = peer
def _health_setup():
    if DIRSEL != 'health' or not E('HEALTH_NPZ'): return
    Hz = np.load(E('HEALTH_NPZ')); v = E('HEALTH_VAR', ''); _HL['A'] = Hz[f'{PAIR}|hA{v}']; _HL['B'] = Hz[f'{PAIR}|hB{v}']
    assert len(_HL['A']) == len(tauA) and len(_HL['B']) == len(tauB)
def use_receiver_depth(ha, hb):
    return ha is not None and hb is not None and np.isfinite(ha) and np.isfinite(hb) and (hb < H_REL * ha or (hb < H_ABS and ha >= H_ABS))
def _share3d(P):
    n = sum(int(np.isfinite(P[c]['X']).all(1).sum()) for c in P); m = sum(len(P[c]['X']) for c in P); return n / max(m, 1)
def _share_kf(side, k):
    if (side, k) not in _SH:
        n = m = 0
        for c in range(NCA if side == 'A' else NCB):
            key = f'{side}_{k}_{c}_X'
            if key in ZF:
                X = np.asarray(Z[key], np.float64)
                if len(X): n += int(np.isfinite(X).all(1).sum()); m += len(X)
        _SH[(side, k)] = n / m if m else np.nan
    return _SH[(side, k)]
def _share_roll(side, k):
    v = [x for x in (_share_kf(side, kk) for kk in range(max(0, k - DIRSEL_WIN + 1), k + 1)) if np.isfinite(x)]
    return float(np.median(v)) if v else np.nan
def _useB(corr):
    return corr.hasB & ~corr.hasA if _DIR[0] == 'A' else corr.hasB
def _useA(corr):
    return corr.hasA & ~_useB(corr)
def refine_gn_q(corr, v, inl, thr, qA, qB, iters=4):
    x = np.array([np.arctan2(v[1], v[0]), *v[2:5]]); h = np.array([1e-4, 1e-3, 1e-3, 1e-3]); sc = np.radians(thr)
    mB = inl & _useB(corr); mA = inl & _useA(corr); wq = np.repeat(np.concatenate([np.where(qB[mB] > 0, 1.0, WQ), np.where(qA[mA] > 0, 1.0, WQ)]), 3)
    if WMODE == 'par':
        c, s_ = v[0], v[1]; wq = par_weights(corr, np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]), np.asarray(v[2:5], float), inl, qA, qB)
    def tov(X): return np.stack([np.cos(X[:, 0]), np.sin(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]], 1)
    for _ in range(iters):
        if JAC6 == 'analytic': r, J = _res4_jac(corr, x, inl)
        else:
            Xs = np.repeat(x[None], 9, 0)
            for k in range(4): Xs[1 + 2 * k, k] += h[k]; Xs[2 + 2 * k, k] -= h[k]
            Rm = S._signed_res_many(corr, tov(Xs), inl); r = Rm[0]; J = None
        if len(r) < 8 or len(r) != len(wq): break
        if J is None: J = np.stack([(Rm[1 + 2 * k] - Rm[2 + 2 * k]) / (2 * h[k]) for k in range(4)], 1)
        rn = np.linalg.norm(r.reshape(-1, 3), axis=1); wgt = np.repeat(1.0 / (1.0 + (rn / sc) ** 2), 3) * wq
        Aa = J.T @ (wgt[:, None] * J) + 1e-9 * np.eye(4); g = J.T @ (wgt * r)
        try: dx = np.linalg.solve(Aa, -g)
        except np.linalg.LinAlgError: break
        x = x + dx
        if np.abs(dx).max() < 1e-7: break
    return np.array([np.cos(x[0]), np.sin(x[0]), *x[1:4]])
SCALE8 = E('SCALE8', '0') == '1'; SCALE_SIG = float(E('SCALE_SIG', '0.05'))   # round 14: per-drone depth-scale DoF in the refinement (log scale, prior SCALE_SIG)
REFINE6 = E('REFINE6', '1') == '1'; SIG_PRIOR = np.radians(float(E('TILT_PRIOR_DEG', '0.5'))); SIG_MEAS = np.radians(float(E('SIG_MEAS_DEG', '0.2')))
def _rot6(X):
    r, p, y = X[:, 0], X[:, 1], X[:, 2]; cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(X), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp; R[:, 2, 1] = cp * sr; R[:, 2, 2] = cp * cr
    return R
def _res6(corr, X, inl):
    """signed residuals (K, n*3) of 6-DoF hypotheses X = (roll, pitch, yaw, tx, ty, tz[, log scale A, log scale B]): X_A = R X_B + t (same blocks as
    solver._signed_res_many). With 8 columns each drone's 3-D points are scaled about its rig origin by exp(s) (sender depth-scale error)."""
    R = _rot6(X); t = X[:, 3:6]; out = []
    sA = np.exp(X[:, 6]) if X.shape[1] > 6 else np.ones(len(X)); sB = np.exp(X[:, 7]) if X.shape[1] > 7 else np.ones(len(X))
    m = inl & _useB(corr)
    if m.any():
        w = sB[:, None, None] * np.einsum('kij,nj->kni', R, corr.XB[m]) + t[:, None, :] - corr.cA[m][None]; D = corr.dA[m]
        u = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-9); out.append((u - np.einsum('knj,nj->kn', u, D)[:, :, None] * D[None]).reshape(len(X), -1))
    m = inl & _useA(corr)
    if m.any():
        w = np.einsum('kji,knj->kni', R, sA[:, None, None] * corr.XA[m][None] - t[:, None, :]) - corr.cB[m][None]; D = corr.dB[m]
        u = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-9); out.append((u - np.einsum('knj,nj->kn', u, D)[:, :, None] * D[None]).reshape(len(X), -1))
    return np.concatenate(out, 1) if out else np.zeros((len(X), 0))
def refine6(corr, v, inl, thr, qA, qB, iters=4, x0=None):
    """6-DoF Gauss-Newton from the 4-DoF solution: adds the relative roll and pitch of the two gravity estimates (VIO tilt errors), with a
    zero-mean prior of TILT_PRIOR_DEG on each; Cauchy weights and the 3-D quality weights of refine_gn_q. Returns (roll, pitch, yaw, t)."""
    x = np.array([0.0, 0.0, np.arctan2(v[1], v[0]), *v[2:5]] + ([0.0, 0.0] if SCALE8 else [])) if x0 is None else np.array(x0, float); h = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3] + ([1e-4, 1e-4] if SCALE8 else [])); sc = np.radians(thr)
    nx = len(x)
    mB = inl & _useB(corr); mA = inl & _useA(corr); wq = np.repeat(np.concatenate([np.where(qB[mB] > 0, 1.0, WQ), np.where(qA[mA] > 0, 1.0, WQ)]), 3)
    if WMODE == 'par': wq = par_weights(corr, _rot6(x[None])[0], x[3:6], inl, qA, qB)
    Pp = np.diag([1 / SIG_PRIOR ** 2, 1 / SIG_PRIOR ** 2, 0, 0, 0, 0] + ([1 / SCALE_SIG ** 2] * 2 if nx > 6 else []))
    for _ in range(iters):
        if JAC6 == 'analytic' and nx == 6: r, J = _res6_jac(corr, x, inl)
        else:
            Xs = np.repeat(x[None], 1 + 2 * nx, 0)
            for k in range(nx): Xs[1 + 2 * k, k] += h[k]; Xs[2 + 2 * k, k] -= h[k]
            Rm = _res6(corr, Xs, inl); r = Rm[0]; J = None
        if len(r) < 8 or len(r) != len(wq): break
        if J is None: J = np.stack([(Rm[1 + 2 * k] - Rm[2 + 2 * k]) / (2 * h[k]) for k in range(nx)], 1)
        rn = np.linalg.norm(r.reshape(-1, 3), axis=1); wgt = np.repeat(1.0 / (1.0 + (rn / sc) ** 2), 3) * wq
        A = J.T @ (wgt[:, None] * J) / SIG_MEAS ** 2 + Pp; g = J.T @ (wgt * r) / SIG_MEAS ** 2 + Pp @ x
        LAST6['A'] = A; LAST6['rss'] = float((wgt * r * r).sum()); LAST6['nres'] = len(r)
        try: dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError: break
        x = x + dx
        if np.abs(dx).max() < 1e-7: break
    return x
LAST6 = {}
def sigma6():
    """round 15: yaw and translation sigma of the final 6-DoF pose from its Gauss-Newton normal matrix (priors included), with the
    measurement sigma replaced by the weighted residual RMS (the numeric 4-DoF sigma stage measures bearing noise only)."""
    if 'A' not in LAST6: return np.nan, np.nan
    try: C = np.linalg.inv(LAST6['A'])
    except np.linalg.LinAlgError: return np.nan, np.nan
    s2 = LAST6['rss'] / max(LAST6['nres'] - 6, 1) / SIG_MEAS ** 2
    return float(np.degrees(np.sqrt(max(C[2, 2] * s2, 0.0)))), float(np.sqrt(max(np.trace(C[3:6, 3:6]) * s2, 0.0)))
def depth_ratio(corr, x6, inl):
    """round 15: log(sender depth / receiver depth) over inliers with 3-D on both sides, measured along the receiver's rays after the
    final pose (a sender depth-scale error or a biased VIO shows as a non-zero median)."""
    m = inl & corr.hasA & corr.hasB
    if m.sum() < 3: return np.nan, np.nan, int(m.sum())
    R = _rot6(np.asarray(x6, float)[None])[0]; P = corr.XB[m] @ R.T + np.asarray(x6[3:6], float)
    l = np.log(np.maximum(np.linalg.norm(P - corr.cA[m], axis=1), 1e-6) / np.maximum(np.linalg.norm(corr.XA[m] - corr.cA[m], axis=1), 1e-6))
    med = float(np.median(l)); return med, float(np.median(np.abs(l - med))), int(m.sum())
LO6 = int(E('LO6', '1')); REF6_IT = int(E('REF6_IT', '4')); LO6_IT = int(E('LO6_IT', '4'))
JAC6 = E('JAC6', 'analytic')   # round 14: analytic Jacobian of the 6-DoF residuals (numeric: 13 batched residual evaluations per iteration)
_EX = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float); _EY = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float); _EZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
def _res6_jac(corr, x, inl):
    """residuals r (3m,) and Jacobian J (3m, 6) of _res6 at one pose x = (roll, pitch, yaw, t) (same block order)."""
    r_, p_, y_ = x[0], x[1], x[2]; Rx = Rot.from_euler('x', r_).as_matrix(); Ry = Rot.from_euler('y', p_).as_matrix(); Rz = Rot.from_euler('z', y_).as_matrix()
    R = Rz @ Ry @ Rx; dR = (Rz @ Ry @ Rx @ _EX, Rz @ Ry @ _EY @ Rx, _EZ @ Rz @ Ry @ Rx); t = x[3:6]; rs = []; Js = []
    def blk(w, dw, D):
        nw = np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9); u = w / nw; e = u - (u * D).sum(1, keepdims=True) * D
        # de/dw = (I - D D^T)(I - u u^T) / |w|, applied to each column of dw (n, 3, 6)
        g = dw - u[:, :, None] * np.einsum('ni,nik->nk', u, dw)[:, None, :]; g = g / nw[:, :, None]; g = g - D[:, :, None] * np.einsum('ni,nik->nk', D, g)[:, None, :]
        rs.append(e.reshape(-1)); Js.append(g.reshape(-1, 6))
    m = inl & _useB(corr)
    if m.any():
        X = corr.XB[m]; w = X @ R.T + t - corr.cA[m]; dw = np.empty((len(X), 3, 6)); dw[:, :, 3:] = np.eye(3)[None]
        for k in range(3): dw[:, :, k] = X @ dR[k].T
        blk(w, dw, corr.dA[m])
    m = inl & _useA(corr)
    if m.any():
        Y = corr.XA[m] - t; w = Y @ R - corr.cB[m]; dw = np.empty((len(Y), 3, 6)); dw[:, :, 3:] = -R.T[None]
        for k in range(3): dw[:, :, k] = Y @ dR[k]
        blk(w, dw, corr.dB[m])
    if not rs: return np.zeros(0), np.zeros((0, 6))
    return np.concatenate(rs), np.concatenate(Js)
def _res4_jac(corr, x4, inl):
    """4-DoF (yaw, t) residuals and Jacobian: the 6-DoF ones at zero roll and pitch (same blocks as solver._signed_res_many)."""
    r, J = _res6_jac(corr, np.array([0.0, 0.0, x4[0], x4[1], x4[2], x4[3]]), inl); return r, J[:, 2:]
def refine_gn_fast_a(corr, v, inl, thr, iters=3):
    """solver.refine_gn_fast with the analytic Jacobian (same Cauchy-weighted Gauss-Newton)."""
    if JAC6 != 'analytic': return S.refine_gn_fast(corr, v, inl, thr, iters)
    x = np.array([np.arctan2(v[1], v[0]), *v[2:5]]); sc = np.radians(thr)
    for _ in range(iters):
        r, J = _res4_jac(corr, x, inl)
        if len(r) < 8: break
        rn = np.linalg.norm(r.reshape(-1, 3), axis=1); wgt = np.repeat(1.0 / (1.0 + (rn / sc) ** 2), 3)
        A = J.T @ (wgt[:, None] * J) + 1e-9 * np.eye(4); g = J.T @ (wgt * r)
        try: dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError: break
        x = x + dx
        if np.abs(dx).max() < 1e-7: break
    return np.array([np.cos(x[0]), np.sin(x[0]), *x[1:4]])

def res6_all(corr, x):
    """angular residual (deg) of one 6-DoF pose for every correspondence (inf where neither side has a 3-D point)."""
    allm = np.ones(corr.n, bool); r = _res6(corr, np.asarray(x, float)[None], allm)[0].reshape(-1, 3); a = np.degrees(np.arcsin(np.clip(np.linalg.norm(r, axis=1), 0, 1)))
    out = np.full(corr.n, np.inf); order = np.concatenate([np.where(_useB(corr))[0], np.where(_useA(corr))[0]]); out[order] = a; return out
WMODE = E('WMODE', 'q1'); SIG_A = np.radians(float(E('SIG_A_DEG', '0.3'))); PAR_REP = np.radians(np.array([2.0, 4.2, 8.5, 17.0]))
def par_weights(corr, R, t, inl, qA, qB):
    """round 14: per-correspondence weight from the depth uncertainty of the 3-D point (2-bit parallax code q -> sigma_rel = SIG_A / parallax),
    projected into the other drone's view (sigma_ang = sigma_rel * depth * sin(angle between the two lines of sight) / range); returns
    sigma_meas^2 / (sigma_meas^2 + sigma_ang^2) per residual (x3), in the block order of _res6 / solver._signed_res_many."""
    out = []
    m = inl & corr.hasB
    if m.any():
        XB = corr.XB[m]; cB = corr.cB[m]; dB = np.linalg.norm(XB - cB, axis=1); P = XB @ R.T + t; v = P - corr.cA[m]; rA = np.linalg.norm(v, axis=1)
        uB = ((XB - cB) / np.maximum(dB[:, None], 1e-9)) @ R.T; uA = v / np.maximum(rA[:, None], 1e-9); sg = np.linalg.norm(np.cross(uA, uB), axis=1)
        srel = SIG_A / PAR_REP[np.clip(qB[m].astype(int), 0, 3)]; sa = srel * dB * sg / np.maximum(rA, 1e-3); out.append(SIG_MEAS ** 2 / (SIG_MEAS ** 2 + sa ** 2))
    m = inl & corr.hasA & ~corr.hasB
    if m.any():
        XA = corr.XA[m]; cA = corr.cA[m]; dA = np.linalg.norm(XA - cA, axis=1); P = (XA - t) @ R; v = P - corr.cB[m]; rB = np.linalg.norm(v, axis=1)
        uA = ((XA - cA) / np.maximum(dA[:, None], 1e-9)) @ R; uB = v / np.maximum(rB[:, None], 1e-9); sg = np.linalg.norm(np.cross(uA, uB), axis=1)
        srel = SIG_A / PAR_REP[np.clip(qA[m].astype(int), 0, 3)]; sa = srel * dA * sg / np.maximum(rB, 1e-3); out.append(SIG_MEAS ** 2 / (SIG_MEAS ** 2 + sa ** 2))
    return np.repeat(np.concatenate(out), 3) if out else np.zeros(0)
AMB_EXTRA = E('AMB_EXTRA', '0') == '1'; AMB_RADII = {'amb_3_06': (3.0, 0.6), 'amb_3_10': (3.0, 1.0), 'amb_5_10': (5.0, 1.0), 'amb_3_15': (3.0, 1.5)}
AMB_COLS = list(AMB_RADII) + ['amb_u', 'amb_uf']
ZSET_PX = [float(v) for v in E('ZSET', '').split(',') if v]; RANSAC_SEED = int(E('RANSAC_SEED', '0')); HYP_XP = E('HYP_XP', 'np')   # round 14: seed of the RANSAC samples (0 in every reported run; other seeds measure the sampling noise of the lock statistics)
SCORE = E('SCORE', 'jit')   # round 14: fused TorchScript scoring (score_jit.py; identical counts) | 'plain' = solver.score_many_t
import score_jit as SJ
def ransac_amb(corr, w):
    """ransac3 with the hypothesis table kept: returns the refined solution and the ambiguity ratio of the scores."""
    if HYP_XP == 'torch':   # round 14 test: closed-form hypotheses on the GPU (same samples)
        Ht = S.hyps_cf(corr, ITERS, RANSAC_SEED, None, torch)
        if Ht is None or len(Ht) == 0: return None
        Ht = Ht.float().contiguous(); Hh = Ht.double().cpu().numpy()
    else:
        Hh = S.hyps_cf(corr, ITERS, RANSAC_SEED, None, np)
        if Hh is None or len(Hh) == 0: return None
        Ht = torch.from_numpy(np.ascontiguousarray(Hh, dtype=np.float32)).cuda()
    sc = (SJ.count(corr, Ht, THR) if SCORE == 'jit' else S.score_many_t(corr, Ht, THR)).cpu().numpy()
    bi = int(np.argmax(sc)); v = Hh[bi]; best = float(sc[bi])
    dyaw = np.abs(np.degrees(np.arctan2(Hh[:, 1] * v[0] - Hh[:, 0] * v[1], Hh[:, 0] * v[0] + Hh[:, 1] * v[1]))); dt = np.linalg.norm(Hh[:, 2:5] - v[2:5], axis=1)
    far = (dyaw > 3.0) | (dt > 0.3); second = float(sc[far].max()) if far.any() else 0.0
    amb_x = {}
    for px_ in ZSET_PX:   # round 15: margin z of the RANSAC table at other test thresholds (online GT-free threshold selection, rig15.THR_SET_PX)
        sc_ = SJ.count(corr, Ht, px_ * THR / THR_PX if THR_PX > 0 else px_ * 0.6 / 2.64).cpu().numpy(); b_ = int(np.argmax(sc_)); v_ = Hh[b_]; bb_ = float(sc_[b_])
        dy_ = np.abs(np.degrees(np.arctan2(Hh[:, 1] * v_[0] - Hh[:, 0] * v_[1], Hh[:, 0] * v_[0] + Hh[:, 1] * v_[1]))); dt_ = np.linalg.norm(Hh[:, 2:5] - v_[2:5], axis=1)
        f_ = (dy_ > 3.0) | (dt_ > 0.3); s_ = float(sc_[f_].max()) if f_.any() else 0.0; amb_x[f'z_{px_:g}'] = (bb_ - s_) / np.sqrt(max(bb_ + s_, 1.0))
    if AMB_EXTRA:   # round 14 test: ambiguity with wider 'distinct mode' radii (yaw deg, translation m)
        for nm_, (ya_, ta_) in AMB_RADII.items():
            f_ = (dyaw > ya_) | (dt > ta_); amb_x[nm_] = (float(sc[f_].max()) if f_.any() else 0.0) / max(best, 1.0)
        # competitor support outside the best hypothesis's inliers (ridge hypotheses share the best's inliers; aliasing explains other ones)
        top = np.argsort(-sc)[:31]; top = top[top != bi][:30]; Mk = SJ.masks(corr, Ht[torch.from_numpy(np.concatenate([[bi], top])).cuda()], THR)
        uniq = (Mk[1:] & ~Mk[0:1]).sum(1).float(); amb_x['amb_u'] = float(uniq.max()) / max(best, 1.0) if len(top) else 0.0
        # far competitors only (> 3 deg or 30 cm, as the pose-radius margin), counted on the support the best does not explain
        fi = np.where(far)[0]; topf = fi[np.argsort(-sc[fi])[:30]] if len(fi) else fi
        if len(topf):
            Mf = SJ.masks(corr, Ht[torch.from_numpy(np.concatenate([[bi], topf])).cuda()], THR); amb_x['amb_uf'] = float((Mf[1:] & ~Mf[0:1]).sum(1).max()) / max(best, 1.0)
        else: amb_x['amb_uf'] = 0.0
    inl = corr.residuals(v) < THR
    if inl.sum() < 8: return None
    v = refine_gn_fast_a(corr, v, inl, THR); inl = corr.residuals(v) < THR
    return {'v': v, 'inl': inl, 'ninl': int(inl.sum()), 'ninl_raw': int(best), 'amb': second / max(best, 1.0), 'second': second, 'amb_x': amb_x}
PROF = E('PROFILE_SOLVE', '0') == '1'; PT = {}; REFINE_Q = E('REFINE_Q', '0') == '1'   # round 14: per-stage solve time on the device (synchronised; opt-in). REFINE_Q default 0 since round 15 (var15 V1: same accuracy, fewer wrong locks, -3.6 ms)
def _pt(name, t0):
    if PROF: torch.cuda.synchronize(); t1 = time.perf_counter(); PT[name] = PT.get(name, 0.0) + t1 - t0; return t1
    return t0
def solve(A, B):
    t_ = time.perf_counter() if PROF else 0.0
    st = {'pf_max': 0, 'pf_sum4': 0, 'npairs': 0, 'corr_n': 0, 'ninl': 0, 'ninl_raw': 0, 'amb': np.nan, 'sy': np.nan, 'st': np.nan, 'spread': np.nan, 'ok': 0, 'v': None}
    if DIRSEL == 'auto':
        if DIRSEL_WIN > 1 and _CUR[0] is not None: sa, sb = _share_roll('A', _CUR[0][0]), _share_roll('B', _CUR[0][1])
        else: sa, sb = _share3d(A), _share3d(B)
        _DIR[0] = 'A' if sb < DIRSEL_R * sa else 'B'; st['dir'] = _DIR[0]; st['s3A'] = sa; st['s3B'] = sb
    if DIRSEL == 'health':
        if _HLT[0] is not None or _HLT[1] is not None: ha, hb = _HLT
        elif _HL and _CUR[0] is not None: ha = float(_HL['A'][_CUR[0][0]]); hb = float(_HL['B'][_CUR[0][1]])
        else: ha = hb = np.nan
        _DIR[0] = 'A' if use_receiver_depth(ha, hb) else 'B'; st['dir'] = _DIR[0]; st['s3A'] = np.nan if ha is None else ha; st['s3B'] = np.nan if hb is None else hb
    if not A or not B: return st   # round 16: a keyframe without cameras (ncam 0) or an empty own side
    cnt = prefilter_counts(A, B); top = sorted(cnt.values(), reverse=True); st['pf_max'] = int(top[0]); st['pf_sum4'] = int(sum(top[:4]))
    sel = select_pairs(cnt); st['npairs'] = len(sel); t_ = _pt('prefilter', t_)
    if not sel: return st
    IDX, CONF = G.match_many([(A[ca]['dic'], B[cb]['dic']) for ca, cb in sel], min_conf=MINCONF, return_conf=True); t_ = _pt('matcher', t_)
    L = [[] for _ in range(8)]; Wc = []; QA = []; QB = []
    for (ca, cb), idx, cf in zip(sel, IDX, CONF):
        if len(idx) == 0: continue
        ia, ib = idx[:, 0], idx[:, 1]
        okr = np.isfinite(A[ca]['d'][ia]).all(1) & np.isfinite(B[cb]['d'][ib]).all(1)   # round 15: rays beyond a KB4 fold are NaN (rl.kb4_unproject)
        if not okr.all(): ia, ib, cf = ia[okr], ib[okr], np.asarray(cf)[okr]
        if len(ia) == 0: continue
        Wc.append(cf); QA.append(A[ca]['q'][ia]); QB.append(B[cb]['q'][ib])
        for l, x in zip(L, (np.repeat(A[ca]['c'][None], len(ia), 0), A[ca]['d'][ia], A[ca]['X'][ia], np.repeat(B[cb]['c'][None], len(ib), 0), B[cb]['d'][ib], B[cb]['X'][ib], np.full(len(ia), ca), np.full(len(ib), cb))): l.append(x)
    if not L[0]: return st
    corr = S.Corr(*[np.concatenate(x) for x in L]); st['corr_n'] = int(corr.n); t_ = _pt('correspondences', t_)
    r = ransac_amb(corr, np.concatenate(Wc)); t_ = _pt('ransac + 4-DoF refine', t_)
    if r is None: return st
    st['ninl_raw'] = r['ninl_raw']; st['amb'] = r['amb']; st.update(r.get('amb_x', {}))
    if WQ < 1.0 and r['ninl'] >= 8 and REFINE_Q:
        qA = np.concatenate(QA); qB = np.concatenate(QB); r['v'] = refine_gn_q(corr, r['v'], r['inl'], THR, qA, qB); r['inl'] = corr.residuals(r['v']) < THR; r['ninl'] = int(r['inl'].sum())
    st['ninl'] = r['ninl']; st['v'] = r['v']; st['x6'] = None; t_ = _pt('weighted refine', t_)
    if r['ninl'] < MININL: return st
    sy, s_t, sp = S.solution_sigma_fast(corr, r['v'], r['inl']); st.update(sy=float(sy), st=float(s_t), spread=float(sp), ok=1); t_ = _pt('sigma', t_)
    if REFINE6:
        qA = np.concatenate(QA); qB = np.concatenate(QB); st['x6'] = refine6(corr, r['v'], r['inl'], THR, qA, qB, iters=REF6_IT)
        for _ in range(LO6):                       # round 14: re-select inliers with the 6-DoF pose and refine again
            inl6 = res6_all(corr, st['x6']) < THR
            if inl6.sum() < 8: break
            st['x6'] = refine6(corr, r['v'], inl6, THR, qA, qB, x0=st['x6'], iters=LO6_IT); st['ninl6'] = int(inl6.sum())
        t_ = _pt('6-DoF refine + reselection', t_)
        st['sy6'], st['st6'] = sigma6(); inlf = res6_all(corr, st['x6']) < THR; st['lsr'], st['lsr_mad'], st['nboth'] = depth_ratio(corr, st['x6'], inlf)
    return st
# ---- round 14 doorway emulation: keep only keypoints whose ray lies within CONE_DEG of the direction to the other drone (CONE_SIDE both|A|B) ----
CONE = float(E('CONE_DEG', '0')); CONE_SIDE = E('CONE_SIDE', 'both'); PKT_K = int(E('PKT_K', '0'))
def _cone(P, yaw, p_self, p_other, cone):
    u = Rot.from_euler('z', yaw).inv().apply(np.asarray(p_other) - np.asarray(p_self)); u = u / max(np.linalg.norm(u), 1e-9); cth = np.cos(np.radians(cone)) if cone > 0 else -2.0; out = {}
    for c, v in P.items():
        keep = np.where(v['d'] @ u >= cth)[0]
        if PKT_K > 0: keep = keep[:PKT_K]   # the sender fills its packet from what it sees (export order: 3-D first, then score)
        kt = torch.from_numpy(keep).to(dev); dic = dict(v['dic']); dic['keypoints'] = v['dic']['keypoints'][kt]; dic['descriptors'] = v['dic']['descriptors'][kt]
        out[c] = {'dic': dic, 'gd': v['gd'][kt], 'c': v['c'], 'd': v['d'][keep], 'X': v['X'][keep], 'q': v['q'][keep]}
    return out
def cone_pair(A, B, i, j):
    A = _cone(A, yawA[i], pA[i], pB[j], CONE if CONE_SIDE in ('both', 'A') else 0.0)
    B = _cone(B, yawB[j], pB[j], pA[i], CONE if CONE_SIDE in ('both', 'B') else 0.0)
    return A, B
def main():
    global offs, req
    # ---- the grid's requests: for every cell, B keyframes in the first HMAX s after ry, own keyframes within +-WIN (play time) ----
    offs = np.arange(0, 61, OFFSTEP); req = set()
    for x in offs:
        for y in offs:
            rx = max(t_air + x, readyA); ry = max(t_air + y, readyB)
            for j in range(len(tauB)):
                tB = tauB[j] - ry
                if not (0 <= tB <= HMAX): continue
                for i in range(len(tauA)):
                    if 0 <= tauA[i] - rx and abs((tauA[i] - rx) - tB) <= WIN: req.add((i, j))
    req = sorted(req)
    if int(E('MAXSOLVE', '0')) > 0: req = [req[k] for k in sorted(np.random.default_rng(0).choice(len(req), min(len(req), int(E('MAXSOLVE'))), replace=False))]   # quick tests only
    print(f'[{PAIR}] {len(req)} unique solves', flush=True)
    if 'PRIO_SYNC' in globals():   # round 14 GPU-priority test (prio_test.py): capture every matcher graph before the sender thread runs (a capture fails
        K_ = int(E('PRECAP_K', '500'))   # when another thread synchronises the device), run a few solves, then wait until the sender has captured its own graph
        L_ = next((b for b in G.BUCKETS if b >= K_), ((K_ + 255) // 256) * 256)   # padded length of mixed-size batches
        for Bn in range(1, CAMPAIRS + 1): G._get(Bn, K_, K_, False); G._get(Bn, L_, L_, True)
        for (i_, j_) in req[:8]: solve(load('A', i_), load('B', j_))
        sync(); PRIO_SYNC['rx_ready'].set(); PRIO_SYNC['tx_ready'].wait(); print('[prio] receiver graphs captured, sender ready', flush=True)
    elif int(E('WARMUP', '8')) > 0:   # round 15: capture every matcher graph shape (batch 1..CAMPAIRS, the export's K) and warm up before timing
        K_ = len(Z['A_0_0_kpts']); L_ = next((b for b in G.BUCKETS if b >= K_), ((K_ + 255) // 256) * 256)   # (first-use captures took 0.12-0.76 s)
        for Bn in range(1, CAMPAIRS + 1): G._get(Bn, K_, K_, False); G._get(Bn, L_, L_, True)
        for (i_, j_) in req[:int(E('WARMUP', '8'))]: solve(load('A', i_), load('B', j_))
        sync()
    rows = []; t_start = time.time()
    for n, (i, j) in enumerate(req):
        A = load('A', i); B = load('B', j)
        if CONE > 0 or PKT_K > 0: A, B = cone_pair(A, B, i, j)
        _CUR[0] = (i, j); sync(); t0 = time.perf_counter(); st = solve(A, B); sync(); dt = time.perf_counter() - t0
        h = [np.nan] * 4; tilt = [np.nan, np.nan]
        if st['v'] is not None:
            v = st['v']; Tij = (Rot.from_euler('z', np.arctan2(v[1], v[0])), np.array(v[2:5]))
            if st.get('x6') is not None: x6 = st['x6']; Tij = (Rot.from_euler('z', x6[2]), np.array(x6[3:6])); tilt = [float(np.degrees(x6[0])), float(np.degrees(x6[1]))]
            TA = (Rot.from_euler('z', yawA[i]), pA[i]); TB = (Rot.from_euler('z', yawB[j]), pB[j])
            R = TA[0] * Tij[0] * TB[0].inv(); t = TA[1] + TA[0].apply(Tij[1]) - R.apply(TB[1]); h = [R.as_euler('ZYX', degrees=True)[0], *t]
        rows.append([i, j, dt, st['ok'], st['ninl'], st['ninl_raw'], st['amb'], st['sy'], st['st'], st['spread'], st['npairs'], st['pf_max'], st['pf_sum4'], st['corr_n'], *h, *tilt, st.get('ninl6', st['ninl']), st.get('sy6', np.nan), st.get('st6', np.nan), st.get('lsr', np.nan), st.get('lsr_mad', np.nan), st.get('nboth', 0)] + ([st.get(k_, np.nan) for k_ in AMB_COLS] if AMB_EXTRA else []) + [st.get(f'z_{p_:g}', np.nan) for p_ in ZSET_PX] + ([float(st.get('dir', 'B') == 'A'), st.get('s3A', np.nan), st.get('s3B', np.nan)] if DIRSEL in ('auto', 'health') else []))
        if n % 2000 == 0: print(f'[{PAIR}] {n}/{len(req)} {np.median([r[2] for r in rows])*1000:.1f} ms median, {time.time()-t_start:.0f} s', flush=True)
    cols = ['i', 'j', 'dt', 'ok', 'ninl', 'ninl_raw', 'amb', 'sy', 'st', 'spread', 'npairs', 'pf_max', 'pf_sum4', 'corr_n', 'h_yaw', 'h_x', 'h_y', 'h_z', 'rel_roll', 'rel_pitch', 'ninl6', 'sy6', 'st6', 'lsr', 'lsr_mad', 'nboth'] + (AMB_COLS if AMB_EXTRA else []) + [f'z_{p_:g}' for p_ in ZSET_PX] + (['dirA', 's3A', 's3B'] if DIRSEL in ('auto', 'health') else [])
    R_ = np.array(rows, float)
    np.savez(f'dump_{PAIR}{TAGX}.npz', rows=R_, cols=np.array(cols), tauA=tauA, tauB=tauB, t_air=t_air, readyA=readyA, readyB=readyB, offs=offs,
             meta=json.dumps({'pair': PAIR, 'data': DD, 'iters': ITERS, 'win': WIN, 'hmax': HMAX, 'minconf': MINCONF, 'thr': THR, 'mininl': MININL, 'wq': WQ, 'campairs': CAMPAIRS,
                              'min_mnn': MIN_MNN, 'pairsel': PAIRSEL, 'rel': REL, 'lg_layers': E('LG_LAYERS', ''), 'matcher': MATCHER, 'ncamA': NCA, 'ncamB': NCB, 'refine6': REFINE6, 'tilt_prior_deg': float(np.degrees(SIG_PRIOR)), 'wmode': WMODE, 'sig_a_deg': float(np.degrees(SIG_A)), 'lo6': LO6, 'cone_deg': CONE, 'cone_side': CONE_SIDE, 'scale8': SCALE8, 'scale_sig': SCALE_SIG,
                              'refine_q': REFINE_Q, 'dirsel': DIRSEL, 'dirsel_r': DIRSEL_R, 'dirsel_win': DIRSEL_WIN, **({'h_rel': H_REL, 'h_abs': H_ABS, 'health_var': E('HEALTH_VAR', '')} if DIRSEL == 'health' else {}), 'thr_px': THR_PX, 'seed': RANSAC_SEED, 'warmup': int(E('WARMUP', '8')), 'K': int(len(Z['A_0_0_kpts'])), 'ray_norm': RAY_NORM}))
    ok = R_[:, 3] > 0; err = np.linalg.norm(R_[:, 15:18], axis=1)
    if PROF: print('PROFILE', json.dumps({k: round(1000 * v / max(len(rows), 1), 2) for k, v in PT.items()}), flush=True)
    print(json.dumps({'pair': PAIR, 'solves': len(R_), 'ms_med': float(np.median(R_[:, 2]) * 1000), 'accepted_pct': float(ok.mean() * 100), 'err_med_cm': float(np.nanmedian(err[ok]) * 100) if ok.any() else None,
                      'wrong_pct_accepted': float(np.mean(err[ok] > 0.2) * 100) if ok.any() else None, 'wall_s': time.time() - t_start}))
if __name__ == '__main__' or 'PRIO_SYNC' in globals():
    setup(sys.argv[1] if len(sys.argv) > 1 else os.environ['PAIR']); _health_setup(); main()
