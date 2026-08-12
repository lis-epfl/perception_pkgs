#!/usr/bin/env python3
"""Post-calibration diagnosis (paper §6.3): three independent certificates -> automated verdict.

Certificates:
  1. self-consistency  — converged calibration agrees across passes/independent runs and,
                         when available, across sessions (residual in the study's criterion metric).
  2. in-distribution   — parameters lie where the fleet/lens-batch/circle-fit say they should.
  3. ATE health        — post-takeoff trajectory error at fleet-typical levels.

Verdict table (§6.3):
  sc ✓  dist ✓  ate ✓ -> deploy
  sc ✗   —      —    -> fly-again (insufficient data/excitation)
  sc ✓  dist ✗   —   -> hardware-changed (inspect, then accept new fixed point)
  sc ✓  dist ✓  ate ✗ -> platform-defect (calibration fine; timing/IMU defect — repair, not recalibrate)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chainio import parse_chain, cams_from_caljson, calib_residual
from evalate import ate_metrics

OVR = os.environ.get('SCT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# thresholds
SC_THR = 0.10            # self-consistency residual (well inside the τ=0.30 criterion)
# Settling threshold, used INSTEAD of self-consistency when the recording carries no
# ground truth. Same metric and weights as SC_THR, so the two are directly comparable
# -- this is 5x stricter. Measured across window lengths 10/15/20/30/45/full on fleet
# hardware, the last-5 settling residual of a HEALTHY run was 0.0001-0.0089; 0.02 is
# just over 2x that worst case. Note the worst case was the 15 s window, a 6x outlier
# against its neighbours, so a threshold set from any single window would be wrong.
SETTLE_THR = 0.02
SETTLE_K = 5             # snapshots at the tail of the series that must agree
DIST_FOCAL_PCT = 2.0     # |focal - fleet lens batch| — batch agrees to ~0.9% of tolerance (~0.5% abs)
DIST_K1 = 0.06           # |k1 - fleet| — batch spread ~0.02
DIST_CXCY_PX = 120.0     # |c - own circle fit| — healthy fit error is 1-19 px fleet-wide (thr ≈6× worst healthy; docs/VALIDATION_MATRIX.md)
DIST_ROT_DEG = 8.0       # extrinsic rotation vs fleet mount-position mean (fleet agrees 1-3.4°)
DIST_TRANS_CM = 15.0     # extrinsic translation vs fleet mean (fleet agrees 1-9 cm)
DIST_TD_MS = 20.0        # toff vs fleet (inside the equal-split guarantee ±20 ms)
ATE_THR_M = 0.20         # POSTG healthy bound (fleet certified 0.038-0.119 m; nxt1 0.78-3.4 m)


def fleet_members(root=None):
    """Vehicles usable as fleet reference: a theta_star.yaml AND a td_center.json entry.

    Enumerated from disk rather than hardcoded, so swapping fleet_reference/ for your own
    fleet needs no code edit. Sorted, so logs and error messages are stable run to run.
    """
    root = root or f'{OVR}/fleet_reference'
    tdc_path = os.path.join(root, 'td_center.json')
    if not os.path.isfile(tdc_path):
        raise FileNotFoundError(
            '%s is missing. It must map each reference vehicle to {"center": <t_d seconds>}.'
            % tdc_path)
    tdc = json.load(open(tdc_path))
    members = [d for d in sorted(os.listdir(root))
               if d in tdc and os.path.isfile(os.path.join(root, d, 'theta_star.yaml'))]
    return members, tdc


def fleet_stats(exclude=None):
    """Per-mount-position fleet means over the reference vehicles' fixed points."""
    members, tdc = fleet_members()
    drones = [d for d in members if d != exclude]
    if not drones:
        raise ValueError(
            'no usable fleet-reference vehicles in %s/fleet_reference (usable: %s; excluding %r). '
            'Each needs <name>/theta_star.yaml and a "<name>" key in td_center.json.'
            % (OVR, members or 'none', exclude))
    cals = [parse_chain(f'{OVR}/fleet_reference/{d}/theta_star.yaml')[0] for d in drones]
    toffs = [tdc[d]['center'] for d in drones]
    out = {'toff': float(np.mean(toffs)), 'cams': {}}
    for c in cals[0]:
        f = np.mean([cal[c]['f'] for cal in cals], axis=0)
        k = np.mean([cal[c]['k'] for cal in cals], axis=0)
        p = np.mean([cal[c]['p'] for cal in cals], axis=0)
        M = np.mean([cal[c]['R'] for cal in cals], axis=0)
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        out['cams'][c] = {'f': f, 'k': k, 'p': p, 'R': R}
    return out


def _load(src):
    if src.endswith('.json'):
        cj = json.load(open(src))
        return cams_from_caljson(cj), float(cj['toff'])
    return parse_chain(src)


def cert_self_consistency(sessions):
    """sessions: list of lists of harvest sources (>=2 sources in the session used).
    Within-session only: consecutive harvests of a session must agree."""
    details = {}
    for si, srcs in enumerate(sessions):
        cals = [_load(s) for s in srcs]
        rs = [calib_residual(cals[i][0], cals[i + 1][0]) for i in range(len(cals) - 1)]
        if rs:
            details[f'session{si}_pass_resid'] = round(max(rs), 4)
    worst = max([v for v in (details.get(f'session{si}_pass_resid') for si in range(len(sessions)))
                 if v is not None] or [float('inf')])
    return {'pass': bool(worst < SC_THR), 'worst_resid': worst, **details}


def cert_settled(series_path, k=SETTLE_K):
    """Did the online calibration stop moving WITHIN one pass?

    Reads <estimate>.calib_series.jsonl (written when calib_series_dt > 0) and measures
    the residual of each of the last k snapshots against their own median. Runs on every
    pass. Without ground truth it REPLACES self-consistency and ends the loop; with it,
    it decides which pass the ATE may be measured on and gates the verdict alongside it.
    Either way it is the stronger evidence -- self-consistency compares two END-of-run
    values and so cannot tell
    "converged" from "never moved". This fleet produced exactly that failure: a run
    pinned at its seed reported resid 0.0187 while its trajectory was 14 km out.

    A pinned run also settles trivially, so this is NOT sufficient alone -- it is
    paired with the in-distribution certificate, which a pinned run fails by sitting
    exactly on the fleet mean it was seeded from.
    """
    if not series_path or not os.path.isfile(series_path):
        return {'pass': False, 'reason': 'no calibration series (set calib_series_dt > 0)'}
    rows, bad = [], 0
    for line in open(series_path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1      # a partial final line if the run was killed mid-write
    # Counted, not swallowed: one truncated tail line is expected, but a systematically
    # malformed series would otherwise reduce to a handful of rows and read as a clean
    # settle. A silent skip here already hid a writer bug for an entire debug session.
    if bad > 1:
        return {'pass': False, 'n': len(rows), 'bad_lines': bad,
                'reason': '%d unparseable lines in the calibration series — the writer is '
                          'broken, not the calibration' % bad}
    if len(rows) < k:
        return {'pass': False, 'n': len(rows),
                'reason': 'only %d snapshots, need %d — lower calib_series_dt or use a '
                          'longer window' % (len(rows), k)}
    last = [r['calib'] for r in rows[-k:]]
    cams_list = [cams_from_caljson(c) for c in last]
    ref = {}
    for c in cams_list[0]:
        M = np.mean([np.asarray(x[c]['R']) for x in cams_list], axis=0)
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        ref[c] = {'f': np.median([np.asarray(x[c]['f']) for x in cams_list], axis=0),
                  'k': np.median([np.asarray(x[c]['k']) for x in cams_list], axis=0),
                  'p': np.median([np.asarray(x[c]['p']) for x in cams_list], axis=0),
                  'R': R}
    resid = [calib_residual(x, ref) for x in cams_list]
    toffs = np.array([c['toff'] for c in last])
    worst = float(max(resid))
    return {'pass': bool(worst < SETTLE_THR), 'worst_resid': round(worst, 5),
            'mean_resid': round(float(np.mean(resid)), 5), 'n': len(rows), 'k': k,
            'toff_spread_ms': round(float(toffs.max() - toffs.min()) * 1000, 4),
            'span_s': round(rows[-1]['t'] - rows[-k]['t'], 2)}


def cert_in_distribution(cams, toff, circle_fit=None, exclude=None):
    fs = fleet_stats(exclude)
    fails, checks = [], {}
    foc_dev = max(abs((np.mean(cams[c]['f'][:2]) / np.mean(fs['cams'][c]['f'][:2]) - 1) * 100)
                  for c in cams)
    checks['focal_vs_fleet_pct'] = round(foc_dev, 3)
    if foc_dev > DIST_FOCAL_PCT:
        fails.append('focal')
    k1_dev = max(abs(cams[c]['k'][0] - fs['cams'][c]['k'][0]) for c in cams)
    checks['k1_vs_fleet'] = round(k1_dev, 4)
    if k1_dev > DIST_K1:
        fails.append('distortion')
    if circle_fit is not None:
        cc = json.load(open(circle_fit)) if isinstance(circle_fit, str) else circle_fit
        cc_dev = max(np.hypot(cams[c]['f'][2] - cc[str(c)]['mask_cx'], cams[c]['f'][3] - cc[str(c)]['mask_cy'])
                     for c in cams if str(c) in cc)
        checks['cxcy_vs_circlefit_px'] = round(cc_dev, 1)
        if cc_dev > DIST_CXCY_PX:
            fails.append('principal_point')
    rot_dev = max(np.degrees(np.arccos(np.clip((np.trace(np.asarray(cams[c]['R']) @ fs['cams'][c]['R'].T) - 1) / 2,
                                               -1, 1))) for c in cams)
    trans_dev = max(np.linalg.norm(np.asarray(cams[c]['p']) - fs['cams'][c]['p']) * 100 for c in cams)
    checks['extr_rot_vs_fleet_deg'] = round(rot_dev, 2)
    checks['extr_trans_vs_fleet_cm'] = round(trans_dev, 2)
    if rot_dev > DIST_ROT_DEG or trans_dev > DIST_TRANS_CM:
        fails.append('extrinsics')
    td_dev = abs(toff - fs['toff']) * 1000
    checks['toff_vs_fleet_ms'] = round(td_dev, 2)
    if td_dev > DIST_TD_MS:
        fails.append('toff')
    return {'pass': not fails, 'fails': fails, **checks}


def cert_ate(est_path, gt_path, toff, cam_end=None):
    m = ate_metrics(est_path, gt_path, toff=toff, cam_end=cam_end)
    ate = m['postg_m'] if np.isfinite(m['postg_m']) else m['global_m']
    return {'pass': bool(ate < ATE_THR_M), 'postg_m': round(m['postg_m'], 4),
            'global_m': round(m['global_m'], 4), 'noum_m': round(m['noum_m'], 4)}


def verdict(sc, dist, ate, settled=None):
    # Checked before self-consistency because it subsumes it. Self-consistency compares
    # two END-of-pass values and is blind to motion in between: on this fleet the 15 s
    # window produced the TIGHTEST self-consistency of any window (0.0138) on a pass that
    # was still visibly moving (settling 0.0246). No threshold on an endpoint comparison
    # can catch that, so the within-pass check has to be its own gate.
    if settled is not None and not settled['pass']:
        return 'FLY-AGAIN: the calibration was still moving at the end of the final pass ' \
               '(settling %.4f > %.2f) — ATE and the harvested values are not on a fixed ' \
               'point; use a longer window' % (settled.get('worst_resid', float('nan')), SETTLE_THR)
    if not sc['pass']:
        return 'FLY-AGAIN: not self-consistent — insufficient data/excitation (>=15 s flight, static start)'
    if not dist['pass']:
        return 'HARDWARE-CHANGED: self-consistent but out of distribution (%s) — inspect, then accept new fixed point' \
               % ','.join(dist['fails'])
    if not ate['pass']:
        return 'PLATFORM-DEFECT: calibration is fine, the platform is not (timing/IMU) — repair, not recalibrate'
    return 'HEALTHY: deploy'


def diagnose(sessions, circle_fit=None, est_path=None, gt_path=None, exclude=None, cam_end=None,
             settle_path=None):
    # With ground truth the run does multiple passes anyway (the ATE certificate needs a
    # trajectory), so self-consistency is free. Without it, a second pass exists ONLY to
    # compare harvests -- so measure settling inside the one pass instead.
    # Measured on the final pass either way: with ground truth it says whether the
    # trajectory ATE is scored on was produced under a static calibration, and without
    # it it replaces the second pass entirely.
    settled = cert_settled(settle_path) if settle_path is not None else None
    if gt_path is None and settled is not None:
        sc = settled
        sc['source'] = 'within-pass settling'
    else:
        sc = cert_self_consistency(sessions)
        sc['source'] = 'between-pass self-consistency'
    cams, toff = _load(sessions[-1][-1])
    dist = cert_in_distribution(cams, toff, circle_fit, exclude)
    ate = cert_ate(est_path, gt_path, toff, cam_end) if est_path and gt_path else {'pass': True, 'skipped': True}
    return {'self_consistency': sc, 'settled': settled, 'in_distribution': dist, 'ate': ate,
            'verdict': verdict(sc, dist, ate, settled)}


if __name__ == '__main__':
    print('use as a library or via run_tool.py; see docs/VALIDATION_MATRIX.md for the validation cases')
