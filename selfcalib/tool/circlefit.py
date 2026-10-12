#!/usr/bin/env python3
"""Single temporal accumulation pass over a bag serving BOTH §5.1 (circle-fit principal point)
and §6.1 (per-patch image-health): per camera, accumulate the mean image of the frames that
changed (the image-center prior), the activity map (Σ|Δframe|), and which patches were sharp
in at least one frame (99th-percentile |Sobel| gradient at or above IMAGE_GRAD_THR).

Image-center prior, as in the paper (Sec. II-B): every 4th frame of each camera is read, and it
enters the per-pixel mean only if its pixels changed by more than FRAME_DIFF_GREY grey levels
(2.4 % of the brightness range) on average since the camera's previous read frame, measured on
every 4th pixel in each direction -- so a still camera cannot imprint a single view on the mean.
The mean image is thresholded at 15 % of its brightest pixel, the binary map is correlated with
a disk of the fleet-shared radius (FFT), and the argmax is the principal point.

image_pass() is what the tool calls: the same pass with one process per camera. The cameras do
not interact in any step, so the result is the one of accumulate() + fit_centers() +
health_from_acc() in a single process; on the flight computer it takes about 8 s instead of
about 18 s for 30 s of flight."""
import json, os, subprocess, sys, tempfile
import numpy as np
import cv2
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagio import Recording
from gates import _rec, IMAGE_PATCH, IMAGE_GRAD_THR, IMAGE_DEAD_FRAC, THETA_MAX
from chainio import kb4_radius

FRAME_SUB = 4            # read every 4th frame of each camera
FRAME_DIFF_GREY = 6.0    # mean |frame - previous read frame| (grey levels of 255) a frame needs to enter the mean
FRAME_DIFF_PIX = 4       # ... measured on every 4th pixel in each direction
HEALTH_EVERY = 2         # image-health gradients on every 2nd read frame (= every 8th frame, as before)
MEAN_THR = 0.15          # binary map: mean-image pixels brighter than 15 % of the brightest pixel


def _sharp_patches(gr):
    """Per IMAGE_PATCH-px patch of one gradient image: is its 99th percentile >= IMAGE_GRAD_THR?
    Same answer as np.percentile(blocks, 99, axis=(1, 3)) >= IMAGE_GRAD_THR, which was most of the
    image pass's time, from a count: the percentile lies between the sorted values at lo and
    lo + 1, so it is at or above the threshold when the value at lo is, below it when the value
    at lo + 1 is not, and only when exactly that one value separates the two does the
    interpolation decide -- then the percentile of that patch is computed."""
    H, W = gr.shape
    ph, pw = H // IMAGE_PATCH, W // IMAGE_PATCH
    blocks = gr[:ph * IMAGE_PATCH, :pw * IMAGE_PATCH].reshape(ph, IMAGE_PATCH, pw, IMAGE_PATCH)
    npx = IMAGE_PATCH * IMAGE_PATCH
    lo = int(np.floor(0.99 * (npx - 1)))
    above = (blocks >= IMAGE_GRAD_THR).sum(axis=(1, 3))      # pixels at or above the threshold, per patch
    sharp = above >= npx - lo
    for i, j in zip(*np.where(above == npx - lo - 1)):
        sharp[i, j] = (np.percentile(blocks[i:i + 1, :, j:j + 1, :], 99, axis=(1, 3)).astype(np.float32) >= IMAGE_GRAD_THR)[0, 0]
    return sharp


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
                      'sharp': np.zeros((ph, pw), bool), 'shape': (H, W)}
        small = gf[::FRAME_DIFF_PIX, ::FRAME_DIFF_PIX]
        if c in prev:
            acc[c]['act'] += np.abs(gf - prev[c][0])
            if float(np.abs(small - prev[c][1]).mean()) > FRAME_DIFF_GREY:
                acc[c]['sum'] += gf          # the frame changed: it enters the mean image
                acc[c]['n'] += 1
        if acc[c]['n_read'] % HEALTH_EVERY == 0:
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            acc[c]['sharp'] |= _sharp_patches(np.abs(gx) + np.abs(gy))
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
        ph, pw = a['sharp'].shape
        cx, cy, r = centers[c]['mask_cx'], centers[c]['mask_cy'], radii[c]
        yy, xx = np.mgrid[0:ph, 0:pw]
        pcx, pcy = (xx + 0.5) * IMAGE_PATCH, (yy + 0.5) * IMAGE_PATCH
        incirc = ((pcx - cx) ** 2 + (pcy - cy) ** 2) < (0.92 * r) ** 2
        dead = incirc & ~a['sharp']
        frac = float(dead.sum()) / max(int(incirc.sum()), 1)
        flag = frac > IMAGE_DEAD_FRAC
        out['cam%d' % c] = {'frames': a['n_read'], 'patches_in_circle': int(incirc.sum()),
                            'dead_patches': int(dead.sum()), 'dead_frac': round(frac, 4), 'flag': bool(flag)}
        ok = ok and not flag
    out['pass'] = ok
    return out


def usable_cpus():
    """CPUs this process may run on (a pinned run has fewer than the machine)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def image_pass(bag, radii, cams=(0, 1, 2, 3), processes=None, scratch=None):
    """The tool's image pass -> (centers, health): fit_centers() and health_from_acc() of
    accumulate(), with one process per camera (at most `processes` at a time; default one per
    camera, up to the CPUs this process may use). Each process opens the recording itself and
    reads its own camera, window included. processes=1, or a single camera, runs here in this
    process. Both results are keyed in the order of `cams`. `scratch`: where the processes
    leave their small result files while the pass runs (default: the system's temp folder).
    An interrupt ends the processes; a caller that is killed outright leaves them to finish."""
    rec = _rec(bag)
    cams = tuple(int(c) for c in cams)
    processes = min(usable_cpus() if processes is None else processes, len(cams))
    centers, health = {}, {}
    if processes <= 1:
        acc = accumulate(rec, cams=cams)
        centers = fit_centers(acc, radii)
        health = health_from_acc(acc, centers, radii)
    else:
        running = []
        with tempfile.TemporaryDirectory(prefix='circlefit_', dir=scratch) as tmp:
            with open(os.path.join(tmp, 'rec.json'), 'w') as f:
                json.dump(rec.spec(), f)
            todo = list(cams)
            try:
                while todo or running:
                    while todo and len(running) < processes:
                        c = todo.pop(0)
                        err = open(os.path.join(tmp, 'cam%d.err' % c), 'w')
                        running.append((c, err, subprocess.Popen(
                            [sys.executable] + (['-B'] if sys.dont_write_bytecode else []) +
                            [os.path.abspath(__file__), '--camera', str(c), '--radius', repr(float(radii[c])),
                             '--rec', os.path.join(tmp, 'rec.json'), '--out', os.path.join(tmp, 'cam%d.json' % c)],
                            stdout=err, stderr=subprocess.STDOUT)))
                    c, err, proc = running[0]
                    rc = proc.wait()
                    running.pop(0)
                    err.close()
                    said = open(err.name, errors='replace').read()
                    try:
                        res = json.load(open(os.path.join(tmp, 'cam%d.json' % c))) if rc == 0 else None
                    except (OSError, ValueError):
                        res = None                  # exit code 0 but no readable result (e.g. the scratch folder is full)
                    if res is None:
                        raise RuntimeError('image pass: the process of camera %d failed (exit code %d):\n%s'
                                           % (c, rc, said[-2000:]))
                    if said.strip():                # what the libraries printed there, as a single process would show it
                        sys.stderr.write(''.join('[camera %d] %s\n' % (c, ln) for ln in said.splitlines()))
                    if res:
                        centers[c], health['cam%d' % c] = res['center'], res['health']
            finally:
                for _c, err, proc in running:       # only after a failure or an interrupt: do not leave the others running
                    proc.kill()
                    proc.wait()
                    err.close()
    centers = {c: centers[c] for c in cams if c in centers}
    out = {'cam%d' % c: health['cam%d' % c] for c in centers}
    out['pass'] = not any(h['flag'] for h in out.values())
    return centers, out


def _camera_main():
    """One camera of image_pass(), in its own process."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, required=True)
    ap.add_argument('--radius', type=float, required=True)
    ap.add_argument('--rec', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    c, radii = a.camera, {a.camera: a.radius}
    acc = accumulate(Recording.from_spec(json.load(open(a.rec))), cams=(c,))
    res = {}
    if c in acc:                         # no frame of this camera in the window: the tool gets no entry for it, as before
        centers = fit_centers(acc, radii)
        res = {'center': centers[c], 'health': health_from_acc(acc, centers, radii)['cam%d' % c]}
    with open(a.out, 'w') as f:
        json.dump(res, f)


def radii_from_chain(chain_cams):
    return {c: kb4_radius(chain_cams[c]['f'], chain_cams[c]['k'], THETA_MAX.get(c, 1.83))
            for c in chain_cams}


def radii_from_fleet(fleet_chains):
    """The fleet-shared disk radius per camera position: the mean of the image-circle radii of the
    reference vehicles' calibrations (identical lenses, so the radius is a property of the lens design)."""
    rr = [radii_from_chain(ch) for ch in fleet_chains]
    return {c: float(np.mean([r[c] for r in rr])) for c in rr[0]}


if __name__ == '__main__':
    _camera_main()
