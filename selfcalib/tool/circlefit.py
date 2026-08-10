#!/usr/bin/env python3
"""Single temporal accumulation pass over a bag serving BOTH §5.1 (circle-fit principal point)
and §6.1 (per-patch image-health): per camera, accumulate the activity map (Σ|Δframe|), the mean
image, and the per-patch max-over-flight p99 |Sobel| gradient."""
import os, sys
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gates import _reader, IMAGE_PATCH, IMAGE_GRAD_THR, IMAGE_DEAD_FRAC, THETA_MAX
from chainio import kb4_radius


def accumulate(bag, cams=(0, 1, 2, 3), sub=8, max_frames=None, inject=None):
    r = _reader(bag, ['/cam%d/image_raw' % c for c in cams])
    acc, prev, cnt = {}, {}, {c: 0 for c in cams}
    while r.has_next():
        topic, data, _ = r.read_next()
        c = int(topic[4])
        cnt[c] += 1
        if cnt[c] % sub:
            continue
        if max_frames and c in acc and acc[c]['n'] >= max_frames:
            if all(k in acc and acc[k]['n'] >= max_frames for k in cams):
                break
            continue
        m = deserialize_message(data, Image)
        g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
        if inject is not None:
            g = inject(g, c)
        gf = g.astype(np.float32)
        if c not in acc:
            H, W = g.shape
            ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
            acc[c] = {'act': np.zeros_like(gf), 'sum': np.zeros_like(gf), 'n': 0,
                      'gradbest': np.zeros((ph, pw), np.float32), 'shape': (H, W)}
        if c in prev:
            acc[c]['act'] += np.abs(gf - prev[c])
        acc[c]['sum'] += gf
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gr = np.abs(gx) + np.abs(gy)
        H, W = gr.shape
        ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
        blocks = gr[:ph * IMAGE_PATCH, :pw * IMAGE_PATCH].reshape(ph, IMAGE_PATCH, pw, IMAGE_PATCH)
        acc[c]['gradbest'] = np.maximum(acc[c]['gradbest'], np.percentile(blocks, 99, axis=(1, 3)).astype(np.float32))
        acc[c]['n'] += 1
        prev[c] = gf
    return acc


def fit_centers(acc, radii):
    """radii: {cam: px}. -> {cam: {'mask_cx','mask_cy','activity_cx','activity_cy'}} (mask = deployed variant)."""
    out = {}
    for c, a in acc.items():
        r = radii[c]
        yy, xx = np.ogrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
        disk = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
        M = (a['sum'] / max(a['n'], 1))
        Sm = fftconvolve((M > M.max() * 0.15).astype(np.float32), disk, mode='same')
        jm = np.unravel_index(np.argmax(Sm), Sm.shape)
        Sa = fftconvolve(a['act'], disk, mode='same')
        ja = np.unravel_index(np.argmax(Sa), Sa.shape)
        out[c] = {'mask_cx': float(jm[1]), 'mask_cy': float(jm[0]),
                  'activity_cx': float(ja[1]), 'activity_cy': float(ja[0])}
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
        out['cam%d' % c] = {'frames': a['n'], 'patches_in_circle': int(incirc.sum()),
                            'dead_patches': int(dead.sum()), 'dead_frac': round(frac, 4), 'flag': bool(flag)}
        ok = ok and not flag
    out['pass'] = ok
    return out


def radii_from_chain(chain_cams):
    return {c: kb4_radius(chain_cams[c]['f'], chain_cams[c]['k'], THETA_MAX.get(c, 1.83))
            for c in chain_cams}
