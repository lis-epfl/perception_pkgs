"""Round 15: one rig description -> every rig-dependent setting of the General Route (sender_lean.py, orin_dump.py, packet_codec.py).
The review (REVIEW14.md S4) found the rig-agnostic rules living in per-run environment variables while the code defaults were older drone
values. Here they are derived from the camera setup, so a new rig needs only its description:
  K = 2000 / cameras keypoints per camera (about 2000 per keyframe), RANSAC test thr_px (per rig; 2.64 px without one) at the rig's focal length, pixel and depth bits of
  the packet word, triangulation partners 0.2-2 s back in keyframes of the rig's keyframe rate, XFeat engine shape, and the matcher graph
  shapes a receiver must capture before its sender thread runs (for every peer rig it expects).
python3 rig15.py [rig] prints the derived settings; derive(rig) / env(rig) / graph_shapes(own, peers) for code."""
import json, sys, numpy as np

RIGS = {   # measured from the exports (fx_mean = mean focal length of the rig's cameras, px)
    'drone4': {'n_cams': 4, 'model': 'kb4', 'image': (1280, 800), 'fx': 252.3, 'kf_hz': 5.0, 'thr_px': 7.5},
    'tumvi': {'n_cams': 2, 'model': 'kb4', 'image': (512, 512), 'fx': 191.8, 'kf_hz': 5.0, 'thr_px': 1.5},
    'euroc': {'n_cams': 2, 'model': 'radtan', 'image': (752, 480), 'fx': 458.9, 'kf_hz': 5.0, 'thr_px': 1.5},
    # UZH-FPV (sealed test, lock rule = TUM-VI's): the pre-registered GT-free rule on the 2000-solve subset of if3->if7 gives 1.0 px with
    # 33 accepted solves (min_n 30 decides against 0.75 px); on full dumps the same statistic gives 1.5 px (argmax) / 2.0 px (2 % tie).
    # 1.0 px is the lowest-locking candidate (5.0 % by 5 s vs 6.7 % / 7.5 %; 0 wrong at all three), cache_md/uzh15*.json
    'uzh': {'n_cams': 2, 'model': 'equi', 'image': (640, 480), 'fx': 276.8, 'kf_hz': 5.0, 'thr_px': 1.0},
}
PACKET_KP = 2000          # keypoints per keyframe over all cameras
THR_PX = 2.64             # default RANSAC / refinement test (px) for a rig without its own value
# round 15: the test threshold is a per-rig parameter (with exact unit rays the optimum spans 10x in angle: drones 1.7 deg, TUM-VI 0.45,
# EuRoC 0.19, UZH-FPV 0.21). Rule to set it for a new rig WITHOUT ground truth: run the receiver on the rig's own data at a threshold set
# THR_SET_PX and take the LARGEST value whose median RANSAC margin z of accepted solves is within 2 % of the best (ROUND15_PLAN.md,
# cache_md/thr15.json): drones 7.5 px, TUM-VI 1.5 px, EuRoC 1.5 px (1.0 px ties within 1 %), UZH-FPV 1.0 px
# (knife edge: needs >= min_n = 30 accepted solves; with more data the rule moves to 1.5-2.0 px).
THR_SET_PX = (0.75, 1.0, 1.5, 2.0, 2.64, 3.5, 4.4, 5.3, 6.2, 7.5, 9.0)
PARTNERS_S = (-2.0, -1.4, -1.0, -0.8, -0.6, -0.4, -0.2)   # triangulation partners, seconds back
MAX_CAMPAIRS = 6          # camera pairs matched per solve (relative selection, REL 0.5)
BUCKETS = (256, 512, 1024, 2048)


def _bits(n): return int(np.ceil(np.log2(max(int(n), 2))))


def derive(rig):
    r = RIGS[rig] if isinstance(rig, str) else rig
    K = PACKET_KP // r['n_cams']; W, H = r['image']; bu, bv = _bits(W), _bits(H)
    tp = r.get('thr_px', THR_PX)
    return {'K': K, 'thr_px': tp, 'thr_deg': float(np.degrees(tp / r['fx'])), 'pix_bits': bu + bv, 'dep_bits': 32 - bu - bv,
            'ds_keyframes': sorted({int(round(s * r['kf_hz'])) for s in PARTNERS_S}), 'xfeat_engine': (r['n_cams'], H, W),
            'bytes_per_3d_kp_v3': 4 + 16 + 1, 'bytes_per_2d_kp_v3': (bu + bv + 7) // 8 + 16 + 1}


def env(rig):
    """environment for sender_lean.py / orin_dump.py (their defaults already equal the drone values)."""
    d = derive(rig); r = RIGS[rig] if isinstance(rig, str) else rig
    return {'K': str(d['K']), 'DS': ','.join(map(str, d['ds_keyframes'])), 'IMG_W': str(r['image'][0]), 'IMG_H': str(r['image'][1]),
            'THR_PX': str(d['thr_px']), 'FX': f"{r['fx']:.3f}", 'TOPK': str(d['K']), **({'RIG': rig} if isinstance(rig, str) else {})}   # round 16: RIG -> orin_dump.setup_rig at import


def graph_shapes(own, peers):
    """matcher CUDA-graph shapes (batch of camera pairs, own K, peer K, masked) a receiver captures before its sender thread overlaps.
    Equal K on both sides uses the unmasked flash path; unequal K needs the padded, masked path at the bucket above the larger K."""
    ko = derive(own)['K']; out = set()
    for p in peers:
        kp = derive(p)['K']; npairs = min(MAX_CAMPAIRS, (RIGS[own] if isinstance(own, str) else own)['n_cams'] * (RIGS[p] if isinstance(p, str) else p)['n_cams'])
        for b in range(1, npairs + 1):
            if kp == ko: out.add((b, ko, kp, False))
            else: L = next(x for x in BUCKETS if x >= max(ko, kp)); out.add((b, L, L, True))
    return sorted(out)


if __name__ == '__main__':
    rigs = sys.argv[1:] or list(RIGS)
    for r in rigs:
        print(r, json.dumps(derive(r)), json.dumps(env(r)))
    print('graph shapes of a drone receiver with drone + TUM-VI + EuRoC peers:', len(graph_shapes('drone4', ['drone4', 'tumvi', 'euroc'])))
