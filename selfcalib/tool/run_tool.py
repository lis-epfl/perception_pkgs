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
from bagio import Recording
from chainio import parse_chain, write_chain, cams_from_caljson, calib_residual
from gates import static_start_gate, timing_gate
from circlefit import accumulate, fit_centers, health_from_acc, radii_from_chain
from publish import robust_mean, chordal_mean
from diagnose import fleet_stats, diagnose, cert_settled
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


def _override_track_frequency(cfg_txt, hz):
    """Rewrite track_frequency in a config copy. The warm-start loop and flight want
    different rates -- flight tracks every frame for low-latency state estimation,
    calibration wants parallax -- and they share one file, so the loop overrides it here
    rather than the file carrying a value that is wrong for one of its two consumers."""
    if not hz:
        return cfg_txt
    out, seen = [], False
    for line in cfg_txt.splitlines():
        if line.startswith('track_frequency:'):
            out.append('track_frequency: %.1f   # overridden by run_tool --track-frequency' % hz)
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append('track_frequency: %.1f' % hz)
    return '\n'.join(out) + '\n'


def _inject_window(cfg_txt, window):
    """Append the read-time trim. Must go into every config the estimator is given, or
    the calibration and its certificate would be measured over different data."""
    if not window:
        return cfg_txt
    return (cfg_txt.rstrip('\n') +
            '\n\n# --- read-time window (written by run_tool.py --window) ---\n'
            'bag_t_start: 0.0\nbag_t_end: %.3f\n' % float(window))


def _inject_layout(cfg_txt, rec):
    """Append this recording's topic/type layout to an estimator config.

    BOTH configs need it. The calibration config and the flight config are separate
    files, and neither ships swarm-nxt topic names -- their defaults are the study
    layout (/imu0 + /cam<N>, sensor_msgs). Omitting it does not error: the estimator
    opens the bag, matches no topics, and exits 0 having written nothing
    (imu=0 img=0 frames=0), which surfaces as a bare "no trajectory" failure.
    """
    if rec.imu_topic == '/imu0':
        return cfg_txt      # already the study layout; nothing to say
    lines = ['', '# --- recording layout (written by run_tool.py from the detected bag) ---',
             'imu_topic: "%s"' % rec.imu_topic,
             'imu_msg_type: "%s"' % rec.imu_type,
             'cam_msg_type: "%s"' % rec.cam_type]
    for ci in sorted(rec.cam_topics):
        lines.append('cam_topic%d: "%s"' % (ci, rec.cam_topics[ci]))
    return cfg_txt.rstrip('\n') + '\n' + '\n'.join(lines) + '\n'


def _redirect_scratch(cfg_txt, dirpath):
    """Point the estimator's four fixed scratch paths into this run's own directory.

    They are inert while record_timing_information and save_total_state are false, but
    they are absolute and fixed, so two concurrent runs on one machine would overwrite
    each other's the moment either flag is enabled. --domain separates the DDS traffic;
    nothing separated these.
    """
    for key, fname in (('record_timing_filepath', 'traj_timing.txt'),
                       ('filepath_est', 'openvins_est.txt'),
                       ('filepath_std', 'ov_estimate_std.txt'),
                       ('filepath_gt', 'ov_groundtruth.txt')):
        cfg_txt = re.sub(r'^(%s:\s*)"[^"]*"' % key,
                         r'\1"%s"' % os.path.join(os.path.abspath(dirpath), fname),
                         cfg_txt, flags=re.M)
    return cfg_txt


def frozen_pass(vio_root, run_serial, a, published_chain, outdir, ncam, timeout_s):
    """Run the published calibration under the FLIGHT config, exactly as it will be deployed.

    This is what the ATE certificate must measure. The warm-start passes run the
    CALIBRATION operating point -- loose priors, a still-moving calibration, and a
    different set of estimator knobs (max_slam 0 vs 250, num_pts 2000 vs 3000,
    track_frequency 29 vs 40, init_window_time 1.0 vs 3.0, chi2 1.0 vs 0.7) -- so
    certifying one of them applies ATE_THR_M to a different quantity than the one it
    was derived from. Measured across seven healthy runs the two differ by ~1.4x
    (median 4.05 cm on the last calibration pass vs 2.57 cm flying the same chain).

    NOTE ON "FROZEN": online calibration is NOT disabled here. The flight config keeps
    calib_cam_extrinsics/intrinsics true; what makes it frozen in effect is
    flight_stiffness.env, whose OV_PRIOR_* values tighten the calibration priors x0.10
    so the states stay tethered to the published chain instead of random-walking.
    Measured drift over a full flight is sub-pixel and sub-tenth-degree. Argument 5 of
    run_serial.sh is use_stereo, not an online-calibration switch; 'false' here matches
    the documented flight invocation.

    Returns the trajectory path. Raises on any failure -- the caller degrades honestly.
    """
    cfgdir = os.path.join(vio_root, 'vio_deploy', 'config')
    fcfg = os.path.join(cfgdir, 'estimator_flight.yaml')
    fenv = os.path.join(cfgdir, 'flight_stiffness.env')
    for p in (fcfg, fenv):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    os.makedirs(outdir, exist_ok=True)
    # The flight config resolves its chains by RELATIVE name, so they must sit beside the copy.
    rec = Recording.open(a.bag)
    open(f'{outdir}/estimator_flight.yaml', 'w').write(
        _inject_window(_inject_layout(_redirect_scratch(open(fcfg).read(), outdir), rec), a.window))
    shutil.copy(a.imu_chain, f'{outdir}/kalibr_imu_chain.yaml')
    shutil.copy(published_chain, f'{outdir}/kalibr_imucam_chain.yaml')

    env = dict(os.environ)
    for line in open(fenv):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    bag_arg = ','.join(rec.uris)
    cp = subprocess.run(['bash', run_serial, bag_arg, f'{outdir}/estimator_flight.yaml',
                         f'{outdir}/out', str(ncam), 'false', '42', str(a.domain)],
                        capture_output=True, timeout=timeout_s, env=env)
    est = f'{outdir}/out/estimate_tum.txt'
    if cp.returncode != 0 or not os.path.exists(est):
        tail = (cp.stderr or b'').decode('utf-8', 'replace').strip().splitlines()[-2:]
        raise RuntimeError('flight pass exit %d%s' % (cp.returncode, (': ' + ' | '.join(tail)) if tail else ''))
    return est


def bag_seconds(bag):
    """Recording duration in seconds; 0.0 if the metadata cannot be read."""
    try:
        from bagio import Recording
        return Recording.open(bag).duration()
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
    ap.add_argument('--calib-config', default=None,
                    help="estimator config driving the WARM-START loop. Default 'calib': "
                         'configs/estimator_calib.yaml with OV_PRIOR_* UNSET. Pass '
                         "'flight' for the untouched flight operating point, or a path. The ATE "
                         'certificate is unaffected -- it always runs flight config + priors.')
    ap.add_argument('--fleet-exclude', default=None)
    ap.add_argument('--max-pass', type=int, default=8)
    ap.add_argument('--domain', type=int, default=70)
    ap.add_argument('--window', type=float, default=None,
                    help='seconds of the recording to use, measured from the instant every '
                         'stream is live. The input contract needs only a static start plus '
                         '~10-15 s of motion, so a long recording costs passes it does not '
                         'need. Applied to BOTH the warm-start passes and the frozen '
                         'deployment pass, so the certificate measures the same data.')
    ap.add_argument('--track-frequency', type=float, default=None,
                    help='KLT tracking rate for the WARM-START passes only, overriding the '
                         'loop config. Default 15: the cameras run at 30 Hz and the throttle '
                         'skips a frame whose stamp is < last + 1/freq, so this tracks every '
                         '3rd frame and doubles the parallax baseline -- which the study notes '
                         'is "the dominant accuracy lever". Measured on an Orin NX it halves a '
                         'pass (54.7 -> 29.1 s) with ATE unchanged (0.0166 -> 0.0168 m) and the '
                         'best self-consistency of any setting tried (0.0138). NOT applied to '
                         'the --frozen-check pass, which must run true flight settings. Only '
                         'divisors of the frame rate do anything: 30/15/10/7.5 -- 20 and 29 both '
                         'round to the same every-other-frame pattern.')
    ap.add_argument('--frozen-check', action=argparse.BooleanOptionalAction, default=None,
                    help='certify the ATE on a dedicated extra pass that runs the published '
                         'chain under the FLIGHT config plus flight_stiffness.env, instead of '
                         'on the last warm-start pass. Costs one more estimator pass (~a third '
                         'of total runtime). OFF by default: with --gt the loop already runs one '
                         'pass warm-started from a settled harvest and verifies that pass settled '
                         'too, so its whole trajectory came from a static calibration -- the one '
                         'property freezing was buying. Measured on one recording: 0.0208 m for '
                         'that pass against 0.0153 m frozen, and a better noumenal 0.0656 vs '
                         '0.1136, both far inside the 0.20 m gate. Turn it on to certify under '
                         'the flight config exactly rather than the calibration config, which '
                         'differs in track_frequency.')
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
    # One windowed Recording for every python stage. --window must trim the gates and
    # the circle fit too, not just the estimator: the circle fit produces the seed's
    # principal point, so if it read the whole recording while the estimator read 15 s,
    # the seed would describe data the calibration never saw. It is also where the time
    # goes -- these stages stream every frame, and on an Orin they cost ~54 s of a
    # 3 min run regardless of the window.
    _wrec = Recording.open(a.bag, window=a.window)
    report['gate_static'] = static_start_gate(_wrec)
    if not report['gate_static']['pass']:
        report['verdict'] = 'GATE-FAIL: ' + report['gate_static']['reason'] + ' — re-record starting on the ground'
        write_json(report, os.path.join(a.out, 'report.json'))
        log(report['verdict'])
        return 1
    log('gate: timing')
    report['gate_timing'] = timing_gate(_wrec, cams=cam_ids)
    if not report['gate_timing']['pass']:
        log('WARNING: timing gate flagged (proceeding — predicts a platform-defect verdict, §6.3)')

    # ---- 2. one accumulation pass: circle fit + image health ----
    log('image pass: activity/mask/gradient accumulation')
    radii = radii_from_chain(tmpl_cams)
    acc = accumulate(_wrec, cams=cam_ids)
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
    # DEFAULT: drive the warm-start loop with the FLIGHT config knobs, priors UNSET.
    # The two operating points differ in the estimator internals (max_slam 250 vs 0,
    # num_pts 3000 vs 2000, track_frequency 40 vs 29, chi2 0.7 vs 1.0,
    # init_window_time 3.0 vs 1.0), and the calibration states stay mobile either way
    # because OV_PRIOR_* is never exported here -- only frozen_pass() exports it.
    # Pass --calib-config flight to run the untouched flight operating point.
    _calib_cfg = f'{OVR}/configs/estimator_calib.yaml'
    if a.calib_config == 'flight':
        _calib_cfg = os.path.join(vio_root, 'vio_deploy', 'config', 'estimator_flight.yaml')
    elif a.calib_config and a.calib_config != 'calib':
        _calib_cfg = a.calib_config
    log('warm-start loop config: %s (OV_PRIOR_* unset)' % os.path.basename(_calib_cfg))
    # Guard the documented footgun: flight priors leaking in from the caller's shell
    # would pin the calibration at its seed, and it fails by reporting a suspiciously
    # LOW self-consistency residual rather than by erroring.
    _leaked = sorted(k for k in os.environ if k.startswith('OV_PRIOR'))
    if _leaked:
        log('WARNING: %s set in the environment — these tether the calibration to its seed. '
            'Unsetting for the warm-start passes.' % ','.join(_leaked))
    report['calib_config'] = _calib_cfg
    cfg_txt = open(_calib_cfg).read()
    for key, fname in (('record_timing_filepath', 'traj_timing.txt'),
                       ('filepath_est', 'openvins_est.txt'),
                       ('filepath_std', 'ov_estimate_std.txt'),
                       ('filepath_gt', 'ov_groundtruth.txt')):
        # absolute: the estimator resolves these itself, so they must not depend on cwd
        cfg_txt = re.sub(r'^(%s:\s*)"[^"]*"' % key,
                         r'\1"%s"' % os.path.join(os.path.abspath(rd), fname),
                         cfg_txt, flags=re.M)
    # Tell the estimator how THIS recording is laid out. The shared config stays on
    # the study defaults (one sqlite3 bag, /imu0 + /cam<N>); only the per-run copy
    # names the fleet's topics and message types, and only when bagio detected them.
    # Getting this wrong is silent -- the estimator would find no topics and produce
    # no harvest -- so the values come from the same object the gates just read.
    _rec = _wrec
    cfg_txt = _override_track_frequency(
        _inject_window(_inject_layout(cfg_txt, _rec), a.window), a.track_frequency)
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
    settled_at = None       # first pass whose calibration settled; ATE uses the one after it
    for p in range(1, a.max_pass + 1):
        log('pass %d/%d' % (p, a.max_pass))
        cjp = f'{rd}/out/estimate_tum.txt.calib.json'
        if os.path.exists(cjp):
            os.remove(cjp)          # never mistake a previous pass's harvest for this one's
        try:
            # A fleet recording spans two files (IMU and cameras are recorded
            # separately), so hand the estimator every URI bagio resolved.
            _env = {k: v for k, v in os.environ.items() if not k.startswith('OV_PRIOR')}
            cp = subprocess.run(['bash', run_serial, ','.join(_rec.uris),
                                 f'{rd}/estimator_config.yaml', f'{rd}/out', str(ncam), 'true',
                                 '42', str(a.domain)], capture_output=True, timeout=timeout_s, env=_env)
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
        sp = f'{rd}/out/estimate_tum.txt.calib_series.jsonl'
        series_path = os.path.join(a.out, 'calib_series_pass%d.jsonl' % p)
        if os.path.exists(sp):
            shutil.copy(sp, series_path)
        else:
            series_path = None

        st = cert_settled(series_path) if series_path else None
        if st is not None:
            log('  settling in pass %d: %s' % (p, 'worst resid %.5f%s' % (
                st.get('worst_resid', float('nan')), '' if st.get('pass') else ' — NOT settled')
                if st.get('worst_resid') is not None else (st.get('reason') or '?')))
        # Without ground truth there is no ATE certificate, so a further pass would exist
        # only to compare harvests -- settling inside THIS pass answers the same question
        # and is the stronger test. With ground truth ATE has to be measured on a pass that
        # was settled THROUGHOUT, and the pass in which settling is first reached was still
        # moving early on. So run exactly one more, warm-started from the settled harvest:
        # that pass is a valid ATE trajectory AND gives self-consistency against its parent,
        # which is why no separate frozen pass is needed.
        if st is not None and st.get('pass'):
            if a.gt is None:
                log('  settled within pass %d (no ground truth, so no second pass is needed)' % p)
                report['converged_at_pass'] = p
                report['settle'] = st
                break
            if settled_at is None:
                settled_at = p
                log('  settled within pass %d — running one more from it to certify the ATE' % p)
            elif p > settled_at:
                report['converged_at_pass'] = p
                report['settle'] = st
                break
        cams = cams_from_caljson(cj)
        if prev is not None:
            r = calib_residual(cams, prev[0])
            dtd = abs(cj['toff'] - prev[1]) * 1000
            log('  self-consistency vs previous pass: resid=%.4f dtoff=%.2f ms' % (r, dtd))
            # Legacy stop, and only a fallback for runs with no calibration series: two
            # agreeing endpoints do not imply either pass was settled, so where settling
            # IS measurable it decides when to stop and this must not preempt it.
            if r < SC_STOP and dtd < TD_STOP_MS and st is None:
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
    # Which trajectory the ATE certificate measures. Default is the last warm-start
    # pass, which under the FLIGHT-config loop tracks a dedicated frozen run closely:
    # measured over 12 fleet recordings, median 1.17x, identical verdicts on all 12,
    # and better than frozen on 3 of them. --frozen-check buys the dedicated run at the
    # cost of a third of the runtime.
    #
    # That equivalence is a property of the LOOP CONFIG, not a general fact. Under the
    # calibration config (max_slam 0, num_pts 2000, track_frequency 29) the same
    # comparison gave 21.49 cm on the last pass against 1.94 cm frozen -- across the
    # 0.20 m gate. If the loop stops using the flight config, turn --frozen-check back on.
    #
    # ate_source records which one was used either way, so a stored report can never
    # imply a deployment measurement that did not happen.
    est_cert, cert_kind = est, 'calibration-pass'
    series_final = os.path.join(a.out, 'calib_series_pass%d.jsonl' % report['converged_at_pass'])
    series_final = series_final if os.path.isfile(series_final) else None
    settled_final = cert_settled(series_final) if series_final else None
    # Settle FIRST, then spend the extra pass. A frozen pass replays a calibration that
    # is still moving just as faithfully as one that has settled, so running it before
    # this check would buy an authoritative measurement of the wrong chain -- and the
    # verdict is FLY-AGAIN either way.
    # OFF by default: the ATE pass is already warm-started from a settled harvest and is
    # itself verified settled, so its whole trajectory came from a static calibration --
    # which is the only property the frozen pass was buying. Measured on the same
    # recording, that pass gives ATE 0.0208 m against 0.0153 m frozen (and a BETTER
    # noumenal 0.0656 vs 0.1136), both an order of magnitude inside the 0.20 m gate,
    # for ~130 s less. Turn it on to certify under the flight config exactly.
    frozen = bool(a.frozen_check)
    if frozen and settled_final is not None and not settled_final['pass']:
        log('skipping the deployment check: the calibration had not settled by the end of '
            'pass %d (worst resid %.5f) — nothing stable to freeze'
            % (report['converged_at_pass'], settled_final.get('worst_resid', float('nan'))))
        frozen = False
    if a.gt and frozen:
        fd = os.path.join(a.out, 'deploy_check')
        try:
            est_cert = frozen_pass(vio_root, run_serial, a, pub, fd, ncam, timeout_s)
            cert_kind = 'frozen-deployment'
            log('deployment check: flight config + flight_stiffness.env priors over the same recording')
        except Exception as e:
            rl = os.path.join(fd, 'out', 'run.log')
            log('deployment check FAILED (%s: %s) — falling back to the calibration-pass '
                'trajectory, which reads WORSE than deployment.%s'
                % (type(e).__name__, e, (' See ' + rl) if os.path.exists(rl) else ''))
            report['deploy_check_error'] = '%s: %s' % (type(e).__name__, e)
    report['ate_source'] = cert_kind
    diag = diagnose(sessions=[harvests], circle_fit={str(c): centers[c] for c in centers},
                    est_path=est_cert if a.gt else None, gt_path=a.gt,
                    exclude=a.fleet_exclude, cam_end=a.cam_end, settle_path=series_final)
    if not report['gate_timing']['pass'] and diag['verdict'].startswith('PLATFORM-DEFECT'):
        diag['verdict'] += ' [timing gate had flagged this recording — consistent]'
    report['diagnosis'] = diag
    report['verdict'] = diag['verdict']

    # ---- 7. mount solve (§6.2) — needs ground truth, so it is opportunistic ----
    # M is the constant body/mount rotation separating the start-anchored alignment from the
    # best-fit one. Solve it once per vehicle and reuse it via mount.apply() on later flights
    # to get an accurate UNALIGNED trajectory without fitting against ground truth each time.
    if a.gt and not diag['ate'].get('pass', False):
        # A diverged trajectory still yields a confident rotation fit. Measured on a
        # defective vehicle: yaw 51.23 deg, tilt 102.85 deg, residual 8446 m, written
        # to mount.json and silently wrong for anything later calling mount.apply().
        report['mount'] = {'skipped': 'ATE certificate failed — trajectory diverged, '
                                      'a mount fitted to it would be meaningless'}
        log('mount: skipped — the trajectory failed the ATE certificate, so there is no '
            'trustworthy attitude to fit a mount to')
    elif a.gt:
        try:
            # Solve on the certified (flight-config) trajectory: M is applied to future
            # flights, which run that config, so it must be measured under it.
            ms = mount.solve(est_cert, a.gt, toff=toff_pub, cam_end=a.cam_end)
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
