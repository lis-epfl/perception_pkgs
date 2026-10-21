#!/usr/bin/env python3
"""live.sender.LiveSender against relloc2's orin_pkg/sender_lean.py on one of its sequence files: both build every packet of the sequence;
the test prints the SHA-256 of all datagrams (sorted per keyframe, as sender_lean's pkt_sha) and the per-keyframe datagram equality when
a reference packet list is given. Needs the GPU.
usage: python3 test_sender_parity.py <senderseq.npz> [ref_sha]   (ref_sha: the 'pkt_sha' sender_lean.py prints for the same file)"""
import os, sys, hashlib, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live import sender
Z = np.load(sys.argv[1]); k0 = int(Z['k0']); n = int(Z['n']); dev = 'cuda'
NC = 1 + max(int(f.split('_')[1]) for f in Z.files if f.startswith(f'{k0}_') and f.endswith('_desc'))
W, H = tuple(int(v) for v in Z['image_size']) if 'image_size' in Z.files else (1280, 800)
NMAX = max(Z[f'{k}_{c}_desc'].shape[0] for k in range(k0 - 10, k0 + n) for c in range(NC))
S = sender.LiveSender(NC, W, H, int(os.environ.get('DRONE_ID', '0')), np_max=NMAX, reset_on_epoch=False)
def feats(k): return [(torch.from_numpy(np.asarray(Z[f'{k}_{c}_desc'])).to(dev), torch.from_numpy(np.asarray(Z[f'{k}_{c}_score'])).to(dev), torch.from_numpy(np.asarray(Z[f'{k}_{c}_bear'])).to(dev),
                       np.asarray(Z[f'{k}_{c}_kpts']) if f'{k}_{c}_kpts' in Z.files else np.zeros((0, 2))) for c in range(NC)]
def pose(k): return (np.stack([Z[f'{k}_{c}_R'] for c in range(NC)]), np.stack([Z[f'{k}_{c}_p'] for c in range(NC)]), np.asarray(Z[f'{k}_pb'], float), float(Z[f'{k}_yaw']))
def hdrx(k): return (np.asarray(Z[f'{k}_rp'], float) if f'{k}_rp' in Z.files else np.zeros(2), float(Z[f'{k}_t']) if f'{k}_t' in Z.files else 0.2 * k,
                     float(Z[f'{k}_age']) if f'{k}_age' in Z.files else 0.2 * (k - k0 + 10))
ep = [int(os.environ.get('VIO_EPOCH', '0')), None]
def epoch(k):   # sender_lean.vio_epoch
    if f'{k}_epoch' in Z.files: return int(Z[f'{k}_epoch'])
    age = hdrx(k)[2]; e = ep[0] + int(ep[1] is not None and age < ep[1] - 1e-6); ep[1] = age; ep[0] = e; return e
for k in range(k0 - 10, k0): S.ingest(k, feats(k), *pose(k))
sha = hashlib.sha256(); nb = []
for k in range(k0, k0 + n):
    rp, tk, age = hdrx(k); b, info = S.keyframe(k, feats(k), *pose(k), rp, tk, age, epoch(k)); sha.update(b''.join(sorted(b))); nb.append(sum(len(g) for g in b))
res = {'keyframes': n, 'pkt_sha': sha.hexdigest()[:16], 'packet_kB_med': float(np.median(nb)) / 1000}
if len(sys.argv) > 2: res['same_as_sender_lean'] = res['pkt_sha'] == sys.argv[2]
print(json.dumps(res))
if len(sys.argv) > 2: sys.exit(0 if res['same_as_sender_lean'] else 1)
