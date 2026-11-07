#!/usr/bin/env python3
"""The fisheye mask of the calibration passes when they run the bearing model ("W" mask).

The estimator's own mask is a disk around the CURRENT principal point (cx, cy) of each camera, radius
f * theta_max - mask_fisheye_inset_px (f = (fx + fy) / 2; VioManagerOptions.h, refreshed whenever the calibration
moves). The configs ship theta_max = 1.83 (cameras 0-2) and 1.87 (camera 3, mask_fisheye_theta_max3). On the fleet's
lenses that disk ends at or beyond the edge of the bright image circle: its last pixels look at the dark ring around
the image. Without the bearing model this does not matter, because the estimator cannot use observations beyond
about 86 deg anyway (OpenCV's undistortion clips them and the tracker's RANSAC drops most of them). With the bearing
model it does: the rim observations are used, up to the edge of the disk.

The W rule (the configuration the rerun of 29 Oct 2026 tested as E1W): the largest disk that stays 4 px (the inset)
inside the bright circle on every camera. Per camera, on the mean image of the tool's image pass:
  - a radial profile every 0.5 deg of azimuth around (cx, cy), 380-500 px in 0.25 px steps; only directions whose
    whole profile lies inside the sensor (2 px margin);
  - per direction the edge: the steepest fall of the profile (smoothed over 1.5 px) between 420 and 495 px; the
    level inside (median of 8-16 px before it) and of the dark ring (minimum over 30 px after it); r_half = where the
    profile first stays below the half-way level, r_end = where it first stays below 5 % above the dark ring (each
    for 3 px); no edge when the two levels differ by 10 grey levels or less;
  - a circle fitted to the r_half points (algebraic fit; points more than 2 px off are dropped and the fit repeated
    until nothing changes): directions off the circle are scene or airframe, not the lens;
  - r_bind = the smaller of the smallest r_end of the directions on the circle and the smallest distance from (cx, cy)
    to the fitted circle plus the edge width (median r_end - r_half).
Then theta_max = r_bind / f, the smallest over the cameras that share the global key mask_fisheye_theta_max, and its
own value for a camera with a per-camera key mask_fisheye_theta_max<i> in the config (camera 3 in the shipped config),
rounded to 6 decimals: the disk f * theta_max - inset then ends exactly `inset` px inside the circle end on the
binding camera and at least that far inside on the others.

The ring (380-500 px) is the fleet's lens (f about 252 px, image circle ending about 458 px from the principal point);
on another rig the profiles do not reach the circle and the rule finds no edge: the config's mask is then kept. The
rule never widens the config's disk (a value above the config's is lowered to it).

Flight (since 2026-11-07) runs the bearing model as well, with the mask of the pass a calibration published
(report.json w_mask): flight_mask() reads it and fits it to the flight config, for deploy_vio.py and the tool's flight
check alike.
"""
import math
import re
import numpy as np
import cv2

RING_PX = (380.0, 500.0)      # radial profiles, px from (cx, cy)
RING_STEP_PX = 0.25
EDGE_PX = (420.0, 495.0)      # where the steepest fall is looked for
SECTOR_DEG = 0.5
OUTLIER_PX = 2.0              # circle fit: points farther than this from the circle are dropped
MIN_CONTRAST = 10.0           # grey levels between the inside and the dark ring, else no edge in that direction
MIN_INLIERS = 100             # directions on the fitted circle needed to trust it (fleet recordings: 388-429)
DECIMALS = 6


def edge_of(R, p):
    """Edge of one radial profile p (grey levels at radii R) -> dict or None (see the module doc)."""
    ps = np.convolve(p, np.ones(7) / 7.0, mode='same')
    g = np.gradient(ps, R)
    m = (R >= EDGE_PX[0]) & (R <= EDGE_PX[1])
    ig = np.where(m)[0][np.argmin(g[m])]
    rg = R[ig]
    I_in = float(np.median(p[(R >= rg - 16) & (R <= rg - 8)]))
    B = float(p[(R >= rg) & (R <= min(rg + 30, R[-1]))].min())
    C = I_in - B
    if C <= MIN_CONTRAST:
        return None

    def first_below(level):
        # the first radius from rg - 20 px on at which the profile is below `level` and stays below for 12 samples (3 px; fewer
        # at the end of the profile): counts of samples not below, as prefix sums, instead of a loop over the candidates
        below = p < level
        bad = np.concatenate(([0], np.cumsum(~below)))
        idx = np.where((R >= rg - 20) & below)[0]
        ok = bad[np.minimum(idx + 12, len(p))] - bad[idx] == 0
        return float(R[idx[np.argmax(ok)]]) if ok.any() else None
    rh, re_ = first_below(0.5 * (I_in + B)), first_below(B + 0.05 * C)
    if rh is None or re_ is None:
        return None
    return {'r_grad': float(rg), 'r_half': rh, 'r_end': re_, 'I_in': I_in, 'B': B}


def fit_circle(x, y):
    """Algebraic circle fit -> (centre x, centre y, radius)."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return a, bb, np.sqrt(c + a * a + bb * bb)


def bright_circle(img, cx, cy):
    """End of the bright image circle as seen from (cx, cy) on a mean image (float32, grey levels).
    -> {'r_bind': px, ...} or {'error': why} when no circle can be fitted."""
    H, W = img.shape
    R = np.arange(RING_PX[0], RING_PX[1] + 1e-4, RING_STEP_PX)
    PH = np.radians(np.arange(0.0, 360.0, SECTOR_DEG))
    xs = (cx + np.outer(np.cos(PH), R)).astype(np.float32)
    ys = (cy + np.outer(np.sin(PH), R)).astype(np.float32)
    P = cv2.remap(img.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    ins = (xs.min(1) >= 2) & (xs.max(1) <= W - 3) & (ys.min(1) >= 2) & (ys.max(1) <= H - 3)
    sec = []
    for j in np.where(ins)[0]:
        e = edge_of(R, P[j])
        if e:
            e['phi'] = float(np.degrees(PH[j]))
            sec.append(e)
    if len(sec) < MIN_INLIERS:
        return {'error': 'edge found in %d directions (%d inside the sensor), fewer than %d' % (len(sec), int(ins.sum()), MIN_INLIERS),
                'n_sectors': len(sec)}
    ph = np.radians([s['phi'] for s in sec])
    rh = np.array([s['r_half'] for s in sec])
    re_ = np.array([s['r_end'] for s in sec])
    x, y = cx + rh * np.cos(ph), cy + rh * np.sin(ph)
    keep = np.ones(len(x), bool)
    for _ in range(50):
        a, b, rr = fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - a, y - b) - rr)
        new = res < OUTLIER_PX
        if (new == keep).all():
            break
        keep = new
        if keep.sum() < 3:
            break
    if keep.sum() < MIN_INLIERS:
        return {'error': '%d directions on the fitted circle, fewer than %d' % (int(keep.sum()), MIN_INLIERS), 'n_sectors': len(sec)}
    wdt = float(np.median((re_ - rh)[keep]))
    u = np.c_[np.cos(ph), np.sin(ph)]
    d0 = np.array([cx - a, cy - b])
    bq = u @ d0
    disc = bq ** 2 - (d0 @ d0 - rr ** 2)
    if not np.all(disc[keep] >= 0):
        return {'error': 'the principal point (%.1f, %.1f) lies outside the fitted circle' % (cx, cy), 'n_sectors': len(sec)}
    tq = -bq + np.sqrt(np.maximum(disc, 0.0))
    fit_end = tq + wdt
    r_raw, r_fit = float(re_[keep].min()), float(fit_end[keep].min())
    return {'r_bind': min(r_raw, r_fit), 'r_bind_from': 'raw' if r_raw <= r_fit else 'fit',
            'r_end_raw_min': r_raw, 'r_end_fit_min': r_fit, 'n_sectors': int(len(sec)), 'n_inliers': int(keep.sum()),
            'centre_offset_px': [float(a - cx), float(b - cy)], 'R_half_fit': float(rr), 'width_px': wdt,
            'rms_resid_px': float(np.sqrt(np.mean(res[keep] ** 2)))}


def config_mask(cfg_txt, cams):
    """What the estimator config says about its fisheye mask -> (auto, {cam: theta_max}, set of cameras with their own key, inset).
    auto is False when the config does not draw the disk itself (use_mask false or mask_fisheye_auto false)."""
    def val(key, default=None):
        m = re.search(r'^\s*%s\s*:\s*([^#\n]+)' % re.escape(key), cfg_txt, re.M)
        return m.group(1).strip().strip('"\'') if m else default
    auto = (val('use_mask', 'false').lower() == 'true') and (val('mask_fisheye_auto', 'false').lower() == 'true')
    glob = float(val('mask_fisheye_theta_max', '1.83'))
    own = {c for c in cams if val('mask_fisheye_theta_max%d' % c) is not None}
    theta = {c: (float(val('mask_fisheye_theta_max%d' % c)) if c in own else glob) for c in cams}
    return auto, theta, own, float(val('mask_fisheye_inset_px', '4.0'))


def w_mask(means, intr, cfg_txt):
    """The W mask for a calibration pass.
    means   {cam: mean image} of the tool's image pass
    intr    {cam: [fx, fy, cx, cy, ...]} of the calibration the pass starts from
    cfg_txt the estimator config the pass runs
    -> {'lines': {key: value} to write into the config, or None when the config's mask is kept, 'reason': why not,
        'binding_camera': ..., 'cams': {cam: diagnostics}}"""
    cams = sorted(intr)
    auto, theta_cfg, own, inset = config_mask(cfg_txt, cams)
    out = {'lines': None, 'cams': {}, 'config_theta_max': {str(c): theta_cfg[c] for c in cams}, 'inset_px': inset}
    if not auto:
        out['reason'] = 'the config does not draw the fisheye disk itself (use_mask / mask_fisheye_auto)'
        return out
    missing = [c for c in cams if c not in means]
    if missing:
        out['reason'] = 'no mean image of camera(s) %s' % ', '.join(map(str, missing))
        return out
    th = {}
    for c in cams:
        fx, fy, cx, cy = (float(v) for v in intr[c][:4])
        f = 0.5 * (fx + fy)
        r = bright_circle(means[c], cx, cy)
        r.update({'f': f, 'cx': cx, 'cy': cy})
        out['cams'][str(c)] = r
        if 'error' in r:
            out['reason'] = 'camera %d: %s' % (c, r['error'])
            return out
        th[c] = r['r_bind'] / f
    groups = [[c for c in cams if c not in own]] + [[c] for c in cams if c in own]
    lines, capped = {}, []
    for g in groups:
        if not g:
            continue
        c_bind = min(g, key=lambda c: th[c])
        v = round(th[c_bind], DECIMALS)
        key = 'mask_fisheye_theta_max' if g[0] not in own else 'mask_fisheye_theta_max%d' % g[0]
        cfgv = theta_cfg[g[0]]
        if v > cfgv:            # never wider than the config's disk
            capped.append(key)
            v = cfgv
        lines[key] = v
        for c in g:
            out['cams'][str(c)]['theta_bind'] = th[c]
            out['cams'][str(c)]['disk_px'] = out['cams'][str(c)]['f'] * v - inset
            out['cams'][str(c)]['inside_circle_end_px'] = out['cams'][str(c)]['r_bind'] - out['cams'][str(c)]['disk_px']
        if key == 'mask_fisheye_theta_max':
            out['binding_camera'] = c_bind
    out['lines'] = lines
    out['capped_at_config'] = capped
    return out


def apply_lines(cfg_txt, lines):
    """Replace the values of existing keys in an estimator config; a key the config does not have is appended."""
    for k, v in sorted(lines.items()):
        txt = ('%.*f' % (DECIMALS, v)).rstrip('0').rstrip('.')
        new, n = re.subn(r'(?m)^%s:[ \t]*[^#\n]*' % re.escape(k), '%s: %s   ' % (k, txt), cfg_txt)
        if n == 0:
            new = cfg_txt.rstrip('\n') + '\n%s: %s\n' % (k, txt)
        cfg_txt = new
    return cfg_txt


def published_lines(report):
    """The fitted mask of the pass a calibration published, from its report (report.json of run_tool.py).
    -> (lines, why): lines = {key: theta_max} that pass ran with, or None and why (a sentence) when there is none: the
    calibration ran without the bearing model (w_mask null, or a report written before 2026-11-07 without the field), or its
    published pass kept the config's own mask (the reason the fit gave). Raises ValueError when the record is malformed."""
    if 'w_mask' not in report:
        return None, 'report.json has no w_mask: a calibration by the tool before 2026-11-07, which ran without the bearing model'
    wm = report['w_mask']
    if wm is None:
        return None, 'report.json w_mask is null: the calibration ran without the bearing model'
    p = report.get('published_from_pass')
    rec = [e for e in (wm.get('passes') or []) if isinstance(e, dict) and e.get('pass') == p] if isinstance(wm, dict) else []
    if len(rec) != 1:
        raise ValueError('report.json w_mask holds %d records of the published pass (%r), not one' % (len(rec), p))
    lines = rec[0].get('lines')
    if lines is None:
        return None, 'the published pass %s kept the config\'s own mask (%s)' % (p, rec[0].get('reason') or 'no reason recorded')
    if not isinstance(lines, dict) or not lines:
        raise ValueError('report.json w_mask of the published pass: lines %r are not a mask' % (lines,))
    out = {}
    for k, v in lines.items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float('nan')
        if not re.fullmatch(r'mask_fisheye_theta_max\d*', str(k)) or isinstance(v, bool) or not 0.0 < x < math.pi:
            raise ValueError('report.json w_mask of the published pass: %r: %r is not a mask angle' % (k, v))
        out[str(k)] = x
    return out, None


def for_config(lines, cfg_txt, cams):
    """The lines of a fitted mask for another config of the rig (the flight config) -> (lines, note).
    lines None and note = why, when they cannot be used there: the config does not draw the disk itself (use_mask /
    mask_fisheye_auto false), or its mask has other keys (every camera must get its fitted value: the global key for the
    cameras without one of their own, mask_fisheye_theta_max<i> for each camera with one). Otherwise a value wider than
    this config's own is lowered to it, as in w_mask (note then names the keys, else None)."""
    auto, theta, own, _ = config_mask(cfg_txt, cams)
    if not auto:
        return None, 'the flight config does not draw the fisheye disk itself (use_mask / mask_fisheye_auto)'
    keys = {'mask_fisheye_theta_max%d' % c if c in own else 'mask_fisheye_theta_max': c for c in cams}
    if set(lines) != set(keys):
        return None, ('the fitted mask sets %s, the flight config\'s mask has %s' % (', '.join(sorted(lines)), ', '.join(sorted(keys))))
    out, capped = {}, []
    for k, v in lines.items():
        cfgv = theta[keys[k]]
        if v > cfgv:
            capped.append(k)
            v = cfgv
        out[k] = v
    return out, ('wider than the flight config, lowered to it: %s' % ', '.join(sorted(capped))) if capped else None


def flight_mask(report, cfg_txt, cams):
    """The fisheye mask a calibration is flown with when flight runs the bearing model: the fitted mask of the pass it
    published, for the flight config cfg_txt. -> (lines, note): lines None = the flight config's own mask stays, note why.
    Raises ValueError when report.json's record is malformed."""
    lines, why = published_lines(report)
    if lines is None:
        return None, why
    return for_config(lines, cfg_txt, cams)
