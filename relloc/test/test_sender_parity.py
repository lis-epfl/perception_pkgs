#!/usr/bin/env python3
"""live.sender.LiveSender against relloc2's orin_pkg/sender_lean.py on one of its sequence files: both build every packet of the sequence;
the test prints the SHA-256 of all datagrams (sorted per keyframe, as sender_lean's pkt_sha) and the per-keyframe datagram equality when
a reference packet list is given. The sender runs at sender_lean's settings (floor 3 deg, no memory gate, a memory depth may replace a
fresh one), not at the node's defaults (PT8F6G8). Needs the GPU.
usage: python3 test_sender_parity.py <senderseq.npz> [ref_sha] [--rules]   (ref_sha: the 'pkt_sha' sender_lean.py prints for the same file)
--rules adds checks of the parallax rule on the same keyframes (one line per check, exit 1 on a failure):
  switch     floor 8 -> 6 deg at the middle keyframe: every keyframe's fresh 3-D points (camera, keypoint, coordinates) equal those of a
             fixed 8-deg floor before the switch and of a fixed 6-deg floor from it on
  subset     the fresh 3-D points at 8 deg are a subset of those at 3 deg, with the same coordinates
  memory     the sequence is in flight and never uses the depth memory, so these runs force it on (MEM_MIN 400 instead of 100: cameras
             with fewer fresh points take memory depths): every store holds exactly the keyframe's fresh points of that camera at
             >= 8 deg, and a keyframe is stored exactly when the rule says so; with mem_keep_fresh no keypoint has two 3-D points, without
             it some do (so the check sees the case)"""
import os, sys, gc, hashlib, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live import sender
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]; RULES = '--rules' in sys.argv
Z = np.load(ARGS[0]); k0 = int(Z['k0']); n = int(Z['n']); dev = 'cuda'
NC = 1 + max(int(f.split('_')[1]) for f in Z.files if f.startswith(f'{k0}_') and f.endswith('_desc'))
W, H = tuple(int(v) for v in Z['image_size']) if 'image_size' in Z.files else (1280, 800)
NMAX = max(Z[f'{k}_{c}_desc'].shape[0] for k in range(k0 - 10, k0 + n) for c in range(NC))
LEAN = dict(parmin=3.0, parmin_early=3.0, mem_gate=0.0, mem_keep_fresh=False)   # sender_lean's settings
def feats(k): return [(torch.from_numpy(np.asarray(Z[f'{k}_{c}_desc'])).to(dev), torch.from_numpy(np.asarray(Z[f'{k}_{c}_score'])).to(dev), torch.from_numpy(np.asarray(Z[f'{k}_{c}_bear'])).to(dev),
                       np.asarray(Z[f'{k}_{c}_kpts']) if f'{k}_{c}_kpts' in Z.files else np.zeros((0, 2))) for c in range(NC)]
def pose(k): return (np.stack([Z[f'{k}_{c}_R'] for c in range(NC)]), np.stack([Z[f'{k}_{c}_p'] for c in range(NC)]), np.asarray(Z[f'{k}_pb'], float), float(Z[f'{k}_yaw']))
def hdrx(k): return (np.asarray(Z[f'{k}_rp'], float) if f'{k}_rp' in Z.files else np.zeros(2), float(Z[f'{k}_t']) if f'{k}_t' in Z.files else 0.2 * k,
                     float(Z[f'{k}_age']) if f'{k}_age' in Z.files else 0.2 * (k - k0 + 10))

def run(switch_k=None, mem_min=None, **kw):
    """the whole sequence through one sender -> (pkt_sha, packet kB per keyframe, per-keyframe record of the depth memory's inputs and outputs)"""
    S = sender.LiveSender(NC, W, H, int(os.environ.get('DRONE_ID', '0')), np_max=NMAX, reset_on_epoch=False, **kw)
    if mem_min: S.MEM_MIN = mem_min
    if switch_k is not None: S.set_floor_switch(hdrx(switch_k)[1])
    REC = []; dm0 = S.depth_memory
    def dm(k, s, out, idx):
        before = {c: id(v) for c, v in S.MEM.items()}; fresh = (out[1].copy(), out[0].copy(), out[3].copy()); par = S.OUT['par'].cpu().numpy()[idx]
        r = dm0(k, s, out, idx)
        REC.append({'k': k, 'floor': S.floor, 'fresh': fresh, 'par': par, 'cams': r[1], 'rq': r[5], 'n_mem': len(r[5]) - len(idx),
                    'stores': {c: int(len(v[1])) for c, v in S.MEM.items() if before.get(c) != id(v)}})
        return r
    S.depth_memory = dm
    ep = [int(os.environ.get('VIO_EPOCH', '0')), None]
    def epoch(k):   # sender_lean.vio_epoch
        if f'{k}_epoch' in Z.files: return int(Z[f'{k}_epoch'])
        age = hdrx(k)[2]; e = ep[0] + int(ep[1] is not None and age < ep[1] - 1e-6); ep[1] = age; ep[0] = e; return e
    for k in range(k0 - 10, k0): S.ingest(k, feats(k), *pose(k))
    sha = hashlib.sha256(); nb = []
    for k in range(k0, k0 + n):
        rp, tk, age = hdrx(k); b, info = S.keyframe(k, feats(k), *pose(k), rp, tk, age, epoch(k)); sha.update(b''.join(sorted(b))); nb.append(sum(len(g) for g in b))
    S.GR[0] = None; del S; gc.collect(); torch.cuda.empty_cache()
    return sha.hexdigest()[:16], nb, REC

sha, nb, _ = run(**LEAN)
res = {'keyframes': n, 'pkt_sha': sha, 'packet_kB_med': float(np.median(nb)) / 1000}
if len(ARGS) > 1: res['same_as_sender_lean'] = res['pkt_sha'] == ARGS[1]
print(json.dumps(res), flush=True)
FAIL = [] if len(ARGS) < 2 or res['same_as_sender_lean'] else ['parity']
if RULES:
    def check(name, ok, info=''):
        print(f'{"ok  " if ok else "FAIL"} {name} {info}', flush=True); (None if ok else FAIL.append(name))
    def pts(r):   # fresh 3-D points of one keyframe: (camera, keypoint) -> coordinates
        cams, ki, X = r['fresh']; return {(int(c), int(i)): X[j] for j, (c, i) in enumerate(zip(cams, ki))}
    def same(a, b): return a.keys() == b.keys() and all(np.array_equal(a[q], b[q]) for q in a)
    ks = k0 + n // 2
    _, _, R8 = run(parmin=8.0, parmin_early=8.0); _, _, R6 = run(parmin=6.0, parmin_early=6.0); _, _, R3 = run(parmin=3.0, parmin_early=3.0)
    _, _, RS = run(switch_k=ks, parmin=6.0, parmin_early=8.0)
    eq = [same(pts(r), pts(R8[j] if r['k'] < ks else R6[j])) for j, r in enumerate(RS)]
    fl = sorted({(r['k'] >= ks, r['floor']) for r in RS})
    check('switch: 8 deg before, 6 deg after = the fixed floors', all(eq) and fl == [(False, 8.0), (True, 6.0)],
          f'({sum(eq)} of {len(eq)} keyframes equal; switch at keyframe {ks}; fresh points per keyframe, median 8 / 6 / 3 deg: '
          f'{np.median([len(r["par"]) for r in R8]):.0f} / {np.median([len(r["par"]) for r in R6]):.0f} / {np.median([len(r["par"]) for r in R3]):.0f})')
    sub = [all(q in pts(b) and np.array_equal(x, pts(b)[q]) for q, x in pts(a).items()) for a, b in zip(R8, R3)]
    check('subset: 8-deg fresh points within the 3-deg ones', all(sub), f'({sum(sub)} of {len(sub)} keyframes)')
    for keep in (True, False):
        _, _, RM = run(switch_k=ks, mem_min=400, parmin=6.0, parmin_early=8.0, mem_gate=8.0, mem_keep_fresh=keep)
        cg = float(np.cos(np.radians(8.0))); have = set(); bad = 0; nst = 0; dup = 0; cut = 0
        for r in RM:   # the store rule, recomputed from the recorded fresh points
            cams = r['fresh'][0]; exp = {}
            for c in range(NC):
                m = cams == c; ng = int((m & (r['par'] <= cg)).sum())
                if m.sum() >= 400 and (r['k'] % 5 == 0 or c not in have) and ng >= 400: exp[c] = ng; cut += ng < m.sum()
            bad += exp != r['stores']; nst += len(r['stores']); have |= set(r['stores'])
            dup += len(r['rq']) - len(set(zip(r['cams'].tolist(), r['rq'].tolist())))
        nmem = sum(r['n_mem'] for r in RM)
        if keep:
            check('memory: gated stores as the rule says', bad == 0 and nst > 0 and cut > 0,
                  f'({nst} stores, {cut} of them without the fresh points below 8 deg, {bad} keyframes differ; {nmem} memory depths sent)')
            check('memory: no keypoint with two 3-D points (mem_keep_fresh)', dup == 0 and nmem > 0, f'({dup} duplicates, {nmem} memory depths)')
        else: check('memory: without mem_keep_fresh the check sees duplicates', dup > 0, f'({dup} keypoints with a fresh and a memory depth)')
    print('ALL PASSED' if not FAIL else f'FAILED: {FAIL}')
sys.exit(1 if FAIL else 0)
