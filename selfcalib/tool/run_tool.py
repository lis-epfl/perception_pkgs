#!/usr/bin/env python3
"""End-to-end flight-data calibration tool (paper §6): gates → seed (fleet + circle fit) →
warm-start self-calibration to self-consistency → robust-mean publish → diagnosis verdict.

Usage:
  python3 run_tool.py --drone nxt3 --bag bags/nxt3_raw_4cam_basin \
      --template cal/nxt3/theta_star.yaml --imu-chain cal/nxt3/kalibr_imu_chain.yaml \
      --out tool/e2e_nxt3 [--gt bags/nxt3_gt.tum] [--fleet-exclude nxt3] [--max-pass 8] [--domain 70]
The template chain provides the yaml skeleton + mount-position extrinsics source when
--fleet-exclude derives fleet means (leave-one-out); the seed never uses the template's own
intrinsics beyond the file structure.

Ground truth is optional and OPPORTUNISTIC. If --gt is not given, the tool looks for an
OptiTrack/mocap pose stream inside the bag itself (see gtdetect.py) and uses it if one is
there. With ground truth from either source the run additionally produces the ATE-health
certificate (§6.3) and the mount solve (§6.2, <drone>_mount.json). Without it, both are
skipped and everything else is unchanged. --no-gt-autodetect turns the search off.
"""
import argparse, json, os, re, shutil, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chainio import parse_chain, write_chain, cams_from_caljson, calib_residual
from gates import static_start_gate, timing_gate
from circlefit import accumulate, fit_centers, health_from_acc, radii_from_chain
from publish import robust_mean, chordal_mean
from diagnose import fleet_stats, diagnose
import gtdetect
import mount

OVR = os.environ.get('SCT_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC_STOP = 0.10       # warm-start loop stops when consecutive harvests agree to this
TD_STOP_MS = 10.0


def log(s):
    print('[tool] ' + s, flush=True)


def _json_safe(o):
    """Replace non-finite floats with None so the output is VALID JSON.

    Python's json writes bare NaN/Infinity, which RFC 8259 does not allow: strict parsers
    reject such a file (verified — node's JSON.parse fails with "Unexpected token 'N'").
    report.json is the machine-readable record of a calibration, so it must be readable by
    things that are not Python.

    NaN legitimately occurs here: evalate.ate_metrics returns postg_m = NaN when the recording
    has fewer than 20 post-takeoff poses. null is the right JSON spelling of "not computed",
    and cert_ate already falls back to global_m in that case, so the verdict is unaffected.
    """
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(obj, path):
    """Write valid JSON (no bare NaN/Infinity)."""
    with open(path, 'w') as f:
        json.dump(_json_safe(obj), f, indent=1)


def resolve_vio_root(explicit=None):
    """Locate the VIO package, which owns the estimator and the only supported runner.

    This package deliberately does NOT bundle an estimator: calibration and flight VIO must
    run the same binary, so both consume one shared vio/ package. Resolution order:
      1. --vio-root
      2. $VIO_ROOT
      3. the sibling default  <selfcalib>/../vio
    """
    RUNNER = os.path.join('vio_deploy', 'scripts', 'run_serial.sh')

    def ok(root):
        return os.path.isfile(os.path.join(root, RUNNER))

    # An EXPLICIT request that cannot be honoured is an error, never a silent substitution:
    # falling back would run a different estimator than the one that was asked for.
    for value, origin in ((explicit, '--vio-root'), (os.environ.get('VIO_ROOT'), '$VIO_ROOT')):
        if value:
            root = os.path.abspath(value)
            if ok(root):
                return root
            raise SystemExit(
                '[tool] ERROR: %s=%s does not contain %s.\n'
                '  Point it at the vio/ directory (the one holding ov_msckf/ and vio_deploy/).'
                % (origin, value, RUNNER))

    sibling = os.path.abspath(os.path.join(os.path.dirname(OVR), 'vio'))
    if ok(sibling):
        return sibling
    raise SystemExit(
        '[tool] ERROR: cannot find the VIO package.\n'
        '  This package ships no estimator — it drives the sibling vio/ package so that\n'
        '  calibration and flight VIO always run the same binary.\n'
        '  Looked for <root>/%s under the sibling default:\n'
        '    %s\n'
        '  Pass --vio-root or set $VIO_ROOT to the vio/ directory.' % (RUNNER, sibling))


def bag_seconds(bag):
    """Recording duration in seconds; 0.0 if the metadata cannot be read."""
    try:
        import rosbag2_py
        m = rosbag2_py.Info().read_metadata(bag, 'sqlite3')
        d = m.duration
        return (d.nanoseconds if hasattr(d, 'nanoseconds') else float(d)) * 1e-9
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drone', required=True)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--imu-chain', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gt', default=None,
                    help='ground-truth TUM file; omit to auto-detect a mocap stream in the bag')
    ap.add_argument('--gt-topic', default=None,
                    help='force this bag topic as ground truth instead of auto-picking')
    ap.add_argument('--no-gt-autodetect', action='store_true',
                    help='never look inside the bag for ground truth')
    ap.add_argument('--fleet-exclude', default=None)
    ap.add_argument('--max-pass', type=int, default=8)
    ap.add_argument('--domain', type=int, default=70)
    ap.add_argument('--cam-end', type=float, default=None)
    ap.add_argument('--pass-timeout', type=float, default=None,
                    help='seconds per estimator pass (default: 20x bag duration, min 1800)')
    ap.add_argument('--vio-root', default=None,
                    help='path to the vio/ package (default: $VIO_ROOT, else the sibling ../vio)')
    a = ap.parse_args()

    # Validate before doing any work — each of these otherwise fails deep in the run.
    if not (0 <= a.domain <= 232):
        ap.error('--domain must be 0-232 (higher silently fails in DDS); got %d' % a.domain)
    if a.max_pass < 1:
        ap.error('--max-pass must be >= 1; got %d' % a.max_pass)
    if a.pass_timeout is not None and a.pass_timeout <= 0:
        ap.error('--pass-timeout must be > 0 seconds; got %g' % a.pass_timeout)
    for label, path in (('--bag', a.bag), ('--template', a.template), ('--imu-chain', a.imu_chain)):
        if not os.path.exists(path):
            ap.error('%s does not exist: %s' % (label, path))
    vio_root = resolve_vio_root(a.vio_root)
    run_serial = os.path.join(vio_root, 'vio_deploy', 'scripts', 'run_serial.sh')
    os.makedirs(a.out, exist_ok=True)
    report = {'drone': a.drone, 'bag': a.bag, 'vio_root': vio_root}
    log('estimator: %s' % run_serial)

    # ---- 0. ground truth: explicit --gt wins, else look for a mocap stream in the bag ----
    if a.gt:
        # Fail fast: a missing GT file would otherwise only surface in the diagnosis step,
        # after every estimator pass has already run.
        if not os.path.isfile(a.gt):
            log('ERROR: --gt file does not exist: %s' % a.gt)
            return 3
        report['ground_truth'] = {'found': True, 'path': a.gt, 'source': 'command line'}
    elif a.gt_topic or not a.no_gt_autodetect:
        # An explicit --gt-topic is an explicit request — honour it even with --no-gt-autodetect.
        det = gtdetect.autodetect(a.bag, os.path.join(a.out, 'ground_truth.tum'),
                                  prefer_topic=a.gt_topic, hint=a.drone)
        det['source'] = 'bag autodetect'
        report['ground_truth'] = det
        if det.get('found'):
            a.gt = det['path']
            log('ground truth: %s (%s) — %d poses @ %.1f Hz, %.0f%% coverage, stamps=%s'
                % (det['topic'], det['type'], det['poses'], det['hz'],
                   100 * det['coverage'], det['stamp_source']))
            if det.get('warning'):
                log('  WARNING: ' + det['warning'])
        else:
            log('ground truth: none found in bag (%s) — ATE certificate and mount solve skipped'
                % det.get('reason'))
    else:
        report['ground_truth'] = {'found': False, 'reason': 'autodetect disabled, no --gt'}

    # The rig's camera set comes from the template chain and is threaded through every stage
    # that reads per-camera topics. Defaulting to (0,1,2,3) made a 2-camera rig fail the
    # timing gate on absent topics and a 6-camera rig silently ignore its last two.
    tmpl_cams, _ = parse_chain(a.template)
    cam_ids = tuple(sorted(tmpl_cams))
    report['cameras'] = list(cam_ids)

    # ---- 1. cheap gates ----
    log('gate: static start')
    report['gate_static'] = static_start_gate(a.bag)
    if not report['gate_static']['pass']:
        report['verdict'] = 'GATE-FAIL: ' + report['gate_static']['reason'] + ' — re-record starting on the ground'
        write_json(report, os.path.join(a.out, 'report.json'))
        log(report['verdict'])
        return 1
    log('gate: timing')
    report['gate_timing'] = timing_gate(a.bag, cams=cam_ids)
    if not report['gate_timing']['pass']:
        log('WARNING: timing gate flagged (proceeding — predicts a platform-defect verdict, §6.3)')

    # ---- 2. one accumulation pass: circle fit + image health ----
    log('image pass: activity/mask/gradient accumulation')
    radii = radii_from_chain(tmpl_cams)
    acc = accumulate(a.bag, cams=cam_ids)
    centers = fit_centers(acc, radii)
    report['circle_fit'] = centers
    report['gate_image'] = health_from_acc(acc, centers, radii)
    if not report['gate_image']['pass']:
        log('WARNING: image-health gate flagged camera(s) — clean lenses and re-record for best results')

    # ---- 3. seed: fleet means + own circle fit ----
    log('seed: fleet means (exclude=%s) + own circle-fit c_x,c_y' % a.fleet_exclude)
    fs = fleet_stats(exclude=a.fleet_exclude)
    seed = {}
    for c in tmpl_cams:
        seed[c] = {'f': [float(fs['cams'][c]['f'][0]), float(fs['cams'][c]['f'][1]),
                         centers[c]['mask_cx'], centers[c]['mask_cy']],
                   'k': [float(x) for x in fs['cams'][c]['k']],
                   'R': fs['cams'][c]['R'], 'p': fs['cams'][c]['p']}
    seed_toff = fs['toff']
    rd = os.path.join(a.out, 'calib')
    os.makedirs(rd, exist_ok=True)
    # Copy the calibration config, redirecting the estimator's four absolute /tmp scratch
    # paths into this run's own directory. They are inert by default (record_timing_information
    # and save_total_state are false) but are fixed paths, so two concurrent calibrations on
    # one machine would overwrite each other's the moment either flag is enabled. --domain
    # separates the DDS traffic; nothing separated these.
    cfg_txt = open(f'{OVR}/configs/estimator_fleet.yaml').read()
    for key, fname in (('record_timing_filepath', 'traj_timing.txt'),
                       ('filepath_est', 'openvins_est.txt'),
                       ('filepath_std', 'ov_estimate_std.txt'),
                       ('filepath_gt', 'ov_groundtruth.txt')):
        # absolute: the estimator resolves these itself, so they must not depend on cwd
        cfg_txt = re.sub(r'^(%s:\s*)"[^"]*"' % key,
                         r'\1"%s"' % os.path.join(os.path.abspath(rd), fname),
                         cfg_txt, flags=re.M)
    open(f'{rd}/estimator_config.yaml', 'w').write(cfg_txt)
    shutil.copy(a.imu_chain, f'{rd}/kalibr_imu_chain.yaml')
    write_chain(a.template, seed, seed_toff, f'{rd}/kalibr_imucam_chain.yaml')

    # ---- 4. warm-start loop to self-consistency ----
    # Camera count comes from the template chain, not a constant: a rig with a different
    # number of cameras would otherwise silently be run as if it had four.
    ncam = len(tmpl_cams)
    # The estimator runs at roughly 1-2x real time, so a fixed cap silently truncates long
    # recordings. Scale with the bag, keep the study's 1800 s as the floor.
    timeout_s = a.pass_timeout if a.pass_timeout else max(1800.0, 20.0 * bag_seconds(a.bag))
    harvests = []
    prev = None
    for p in range(1, a.max_pass + 1):
        log('pass %d/%d' % (p, a.max_pass))
        cjp = f'{rd}/out/estimate_tum.txt.calib.json'
        if os.path.exists(cjp):
            os.remove(cjp)          # never mistake a previous pass's harvest for this one's
        try:
            cp = subprocess.run(['bash', run_serial, a.bag,
                                 f'{rd}/estimator_config.yaml', f'{rd}/out', str(ncam), 'true',
                                 '42', str(a.domain)], capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log('ERROR: estimator pass %d exceeded %.0f s and was killed. Raise --pass-timeout, '
                'or shorten the recording.' % (p, timeout_s))
            report['verdict'] = 'ERROR: estimator timed out on pass %d' % p
            write_json(report, os.path.join(a.out, 'report.json'))
            return 4
        if cp.returncode != 0 or not os.path.exists(cjp):
            log('ERROR: estimator pass %d produced no calibration harvest (exit %d).' % (p, cp.returncode))
            log('  Read %s/out/run.log — the usual causes are bag topics not matching the chain\'s' % rd)
            log('  rostopic fields, an IMU chain missing its Tw/Ta/Tg blocks, or --domain outside 0-232.')
            tail = (cp.stderr or b'').decode('utf-8', 'replace').strip().splitlines()[-3:]
            for t in tail:
                log('  | ' + t)
            report['verdict'] = 'ERROR: estimator produced no output on pass %d (see calib/out/run.log)' % p
            write_json(report, os.path.join(a.out, 'report.json'))
            return 5
        cj = json.load(open(cjp))
        hp = os.path.join(a.out, 'harvest_pass%d.calib.json' % p)
        shutil.copy(f'{rd}/out/estimate_tum.txt.calib.json', hp)
        shutil.copy(f'{rd}/out/estimate_tum.txt', os.path.join(a.out, 'estimate_pass%d.tum' % p))
        harvests.append(hp)
        cams = cams_from_caljson(cj)
        if prev is not None:
            r = calib_residual(cams, prev[0])
            dtd = abs(cj['toff'] - prev[1]) * 1000
            log('  self-consistency vs previous pass: resid=%.4f dtoff=%.2f ms' % (r, dtd))
            if r < SC_STOP and dtd < TD_STOP_MS:
                report['converged_at_pass'] = p
                break
        prev = (cams, cj['toff'])
        write_chain(f'{rd}/kalibr_imucam_chain.yaml', cams, cj['toff'], f'{rd}/kalibr_imucam_chain.yaml')
    report['passes_run'] = len(harvests)
    if 'converged_at_pass' not in report:
        report['verdict'] = 'FLY-AGAIN: never self-consistent within %d passes' % a.max_pass
        write_json(report, os.path.join(a.out, 'report.json'))
        log(report['verdict'])
        return 2

    # ---- 5. publish robust mean of the settled harvests ----
    srcs = harvests[-min(3, len(harvests)):]
    cams_pub, toff_pub = robust_mean(srcs)
    pub = os.path.join(a.out, '%s_published_chain.yaml' % a.drone)
    write_chain(a.template, cams_pub, toff_pub, pub)
    report['published'] = pub
    report['published_toff'] = toff_pub
    log('published %s (robust mean of %d harvests, toff=%.6f)' % (pub, len(srcs), toff_pub))

    # ---- 6. diagnosis ----
    est = os.path.join(a.out, 'estimate_pass%d.tum' % report['converged_at_pass'])
    diag = diagnose(sessions=[harvests], circle_fit={str(c): centers[c] for c in centers},
                    est_path=est if a.gt else None, gt_path=a.gt,
                    exclude=a.fleet_exclude, cam_end=a.cam_end)
    if not report['gate_timing']['pass'] and diag['verdict'].startswith('PLATFORM-DEFECT'):
        diag['verdict'] += ' [timing gate had flagged this recording — consistent]'
    report['diagnosis'] = diag
    report['verdict'] = diag['verdict']

    # ---- 7. mount solve (§6.2) — needs ground truth, so it is opportunistic ----
    # M is the constant body/mount rotation separating the start-anchored alignment from the
    # best-fit one. Solve it once per vehicle and reuse it via mount.apply() on later flights
    # to get an accurate UNALIGNED trajectory without fitting against ground truth each time.
    if a.gt:
        try:
            ms = mount.solve(est, a.gt, toff=toff_pub, cam_end=a.cam_end)
            mp = os.path.join(a.out, '%s_mount.json' % a.drone)
            write_json(ms, mp)
            report['mount'] = {k: v for k, v in ms.items() if k != 'M'}
            report['mount']['path'] = mp
            log('mount: yaw=%.2f° tilt=%.2f° | anchored %.3f m -> %.3f m (best-fit floor %.3f m)'
                % (ms['mount_yaw_deg'], ms['mount_tilt_deg'], ms['A_anchored_m'],
                   ms['A_mounted_m'], ms['A_bestfit_m']))
        except Exception as e:
            report['mount'] = {'error': '%s: %s' % (type(e).__name__, e)}
            log('mount: solve failed (%s: %s) — not fatal, calibration is unaffected'
                % (type(e).__name__, e))

    write_json(report, os.path.join(a.out, 'report.json'))
    log('ATE: %s' % json.dumps(diag['ate']))
    log('VERDICT: ' + report['verdict'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
