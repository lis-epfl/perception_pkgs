#!/usr/bin/env python3
"""Robust-mean calibration publisher (paper §3.4/§6): aggregate K converged harvests into the
published calibration. Component-wise mean for intrinsics/translation/toff; chordal (SVD) mean
for rotations. Precision refinement, not an accuracy lever (§3.4).

CLI: python3 publish.py --template chain.yaml --out published_chain.yaml src1 src2 ...
     (each src = a harvest estimate_tum.txt.calib.json or a chain yaml)
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chainio import parse_chain, write_chain, cams_from_caljson


def load_source(path):
    """-> (cams, toff) from a calib.json harvest or a chain yaml."""
    if path.endswith('.json'):
        cj = json.load(open(path))
        return cams_from_caljson(cj), float(cj['toff'])
    return parse_chain(path)


def chordal_mean(Rs):
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def robust_mean(sources):
    """-> (cams, toff): component-wise mean over the settled passes; chordal mean rotations."""
    loaded = [load_source(s) for s in sources]
    cam_ids = sorted(loaded[0][0].keys())
    cams = {}
    for c in cam_ids:
        f = np.mean([L[0][c]['f'] for L in loaded], axis=0).tolist()
        k = np.mean([L[0][c]['k'] for L in loaded], axis=0).tolist()
        p = np.mean([np.asarray(L[0][c]['p']) for L in loaded], axis=0)
        R = chordal_mean([np.asarray(L[0][c]['R']) for L in loaded])
        cams[c] = {'f': f, 'k': k, 'R': R, 'p': p}
    toff = float(np.mean([L[1] for L in loaded]))
    return cams, toff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True, help='chain yaml used as write template')
    ap.add_argument('--out', required=True)
    ap.add_argument('sources', nargs='+')
    a = ap.parse_args()
    cams, toff = robust_mean(a.sources)
    write_chain(a.template, cams, toff, a.out)
    print('published %s from %d harvests (toff=%.6f s)' % (a.out, len(a.sources), toff))


if __name__ == '__main__':
    main()
