#!/usr/bin/env python3
"""Single temporal accumulation pass over a bag serving BOTH §5.1 (circle-fit principal point)
and §6.1 (per-patch image-health): per camera, accumulate the mean image of the frames that
changed (the image-center prior), the activity map (Σ|Δframe|), and the per-patch
max-over-flight p99 |Sobel| gradient.

Image-center prior, as in the paper (Sec. II-B): every 4th frame of each camera is read, and it
enters the per-pixel mean only if its pixels changed by more than FRAME_DIFF_GREY grey levels
(2.4 % of the brightness range) on average since the camera's previous read frame, measured on
every 4th pixel in each direction -- so a still camera cannot imprint a single view on the mean.
The mean image is thresholded at 15 % of its brightest pixel, the binary map is correlated with
a disk of the fleet-shared radius (FFT), and the argmax is the principal point."""
import os, sys
import numpy as np
import cv2
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates import _rec, IMAGE_PATCH, IMAGE_GRAD_THR, IMAGE_DEAD_FRAC, THETA_MAX
from chainio import kb4_radius

FRAME_SUB = 4            # read every 4th frame of each camera
FRAME_DIFF_GREY = 6.0    # mean |frame - previous read frame| (grey levels of 255) a frame needs to enter the mean
FRAME_DIFF_PIX = 4       # ... measured on every 4th pixel in each direction
HEALTH_EVERY = 2         # image-health gradients on every 2nd read frame (= every 8th frame, as before)
MEAN_THR = 0.15          # binary map: mean-image pixels brighter than 15 % of the brightest pixel


def accumulate(bag, cams=(0, 1, 2, 3), sub=FRAME_SUB, max_frames=None, inject=None):
    rec = _rec(bag)
    acc, prev, cnt = {}, {}, {c: 0 for c in cams}
    for c, _ts, data in rec.images(cams):
        cnt[c] += 1
        if cnt[c] % sub:
            continue                      # subsampled away -- never decoded
        if max_frames and c in acc and acc[c]['n'] >= max_frames:
            if all(k in acc and acc[k]['n'] >= max_frames for k in cams):
                break
            continue
        g = rec.decode(data)
        if inject is not None:
            g = inject(g, c)
        gf = g.astype(np.float32)
        if c not in acc:
            H, W = g.shape
            ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
            acc[c] = {'act': np.zeros_like(gf), 'sum': np.zeros_like(gf), 'n': 0, 'n_read': 0,
                      'gradbest': np.zeros((ph, pw), np.float32), 'shape': (H, W)}
        small = gf[::FRAME_DIFF_PIX, ::FRAME_DIFF_PIX]
        if c in prev:
            acc[c]['act'] += np.abs(gf - prev[c][0])
            if float(np.abs(small - prev[c][1]).mean()) > FRAME_DIFF_GREY:
                acc[c]['sum'] += gf          # the frame changed: it enters the mean image
                acc[c]['n'] += 1
        if acc[c]['n_read'] % HEALTH_EVERY == 0:
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            gr = np.abs(gx) + np.abs(gy)
            H, W = gr.shape
            ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
            blocks = gr[:ph * IMAGE_PATCH, :pw * IMAGE_PATCH].reshape(ph, IMAGE_PATCH, pw, IMAGE_PATCH)
            acc[c]['gradbest'] = np.maximum(acc[c]['gradbest'], np.percentile(blocks, 99, axis=(1, 3)).astype(np.float32))
        acc[c]['n_read'] += 1
        prev[c] = (gf, small)
    return acc


def fit_centers(acc, radii):
    """radii: {cam: px}. -> {cam: {'mask_cx','mask_cy','activity_cx','activity_cy','frames_in_mean','frames_read'}}
    (mask = the deployed variant: binary map of the mean image correlated with the disk)."""
    out = {}
    for c, a in acc.items():
        r = radii[c]
        yy, xx = np.ogrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
        disk = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
        M = (a['sum'] / max(a['n'], 1))
        Sm = fftconvolve((M > M.max() * MEAN_THR).astype(np.float32), disk, mode='same')
        jm = np.unravel_index(np.argmax(Sm), Sm.shape)
        Sa = fftconvolve(a['act'], disk, mode='same')
        ja = np.unravel_index(np.argmax(Sa), Sa.shape)
        out[c] = {'mask_cx': float(jm[1]), 'mask_cy': float(jm[0]),
                  'activity_cx': float(ja[1]), 'activity_cy': float(ja[0]),
                  'frames_in_mean': int(a['n']), 'frames_read': int(a['n_read'])}
    return out


def health_from_acc(acc, centers, radii):
    out, ok = {}, True
    for c, a in acc.items():
        ph, pw = a['gradbest'].shape
        cx, cy, r = centers[c]['mask_cx'], centers[c]['mask_cy'], radii[c]
        yy, xx = np.mgrid[0:ph, 0:pw]
        pcx, pcy = (xx + 0.5) * IMAGE_PATCH, (yy + 0.5) * IMAGE_PATCH
        incirc = ((pcx - cx) ** 2 + (pcy - cy) ** 2) < (0.92 * r) ** 2
        dead = incirc & (a['gradbest'] < IMAGE_GRAD_THR)
        frac = float(dead.sum()) / max(int(incirc.sum()), 1)
        flag = frac > IMAGE_DEAD_FRAC
        out['cam%d' % c] = {'frames': a['n_read'], 'patches_in_circle': int(incirc.sum()),
                            'dead_patches': int(dead.sum()), 'dead_frac': round(frac, 4), 'flag': bool(flag)}
        ok = ok and not flag
    out['pass'] = ok
    return out


def radii_from_chain(chain_cams):
    return {c: kb4_radius(chain_cams[c]['f'], chain_cams[c]['k'], THETA_MAX.get(c, 1.83))
            for c in chain_cams}


def radii_from_fleet(fleet_chains):
    """The fleet-shared disk radius per camera position: the mean of the image-circle radii of the
    reference vehicles' calibrations (identical lenses, so the radius is a property of the lens design)."""
    rr = [radii_from_chain(ch) for ch in fleet_chains]
    return {c: float(np.mean([r[c] for r in rr])) for c in rr[0]}
