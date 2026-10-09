#!/usr/bin/env python3
"""Turn a finished calibration into a ready-to-fly VIO configuration folder.

This is step 3 of the workflow: calibrate -> read the values -> deploy them.

It reads a `run_tool.py` output directory, extracts the calibrated values (camera
intrinsics, camera-IMU extrinsics, camera-IMU time offset, IMU noise model, and the mount
rotation if ground truth was available), prints them for inspection, and writes the exact
folder layout the VIO estimator expects:

    <out>/estimator_flight.yaml     flight config, copied from the vio package
    <out>/kalibr_imucam_chain.yaml  the published calibration (intrinsics + extrinsics + t_d)
    <out>/kalibr_imu_chain.yaml     the IMU chain the calibration ran with
    <out>/flight_stiffness.env      the flight settings (stiff calibration priors, lens terms as states,
                                    matching between cameras), copied from the vio package, plus the
                                    lens terms of this calibration (OV_NONRADIAL; empty if it has none)
    <out>/campaign.env              the vehicle's tracker and scheduler switches, copied from the vio package
    <out>/mount.json                mount rotation, if solved   (NOT an estimator input)
    <out>/lens_terms.env            OV_NONRADIAL alone, if the calibration estimated lens terms
    <out>/MANIFEST.txt              what each file is and the command to fly it

The estimator resolves `relative_config_imu` / `relative_config_imucam` RELATIVE TO THE
CONFIG FILE, which is why all three must sit in one directory. That is the whole reason this
step exists: the calibration writes `<drone>_published_chain.yaml`, but the estimator will
only look for a file literally named `kalibr_imucam_chain.yaml` next to its config.

IMPORTANT — the mount rotation is NOT read by the estimator. Intrinsics, extrinsics and the
time offset go INTO the filter through the chain; the mount rotation M is applied AFTERWARDS,
to the estimator's output trajectory (see mount.apply()), to make the unaligned/anchored
trajectory accurate without fitting against ground truth. It is copied here so the value
travels with the calibration that produced it.

Usage:
  python3 deploy_vio.py --calib-out out_myvehicle --out flight_myvehicle
"""
import argparse, json, os, re, shlex, shutil, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chainio import parse_chain
from run_tool import resolve_vio_root, _env_file, _number


def log(s):
    print('[deploy] ' + s, flush=True)


def rpy_deg(R):
    """Roll/pitch/yaw in degrees, for human inspection only (the chain carries the matrix)."""
    R = np.asarray(R)
    pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def terms_problem(terms, ncam):
    """Why `terms` is not a value of OV_NONRADIAL for ncam cameras ('p1,p2,s' per camera, joined by ';'), or None."""
    ents = terms.split(';')
    if len(ents) != ncam:
        return '%d entries for %d cameras' % (len(ents), ncam)
    for i, e in enumerate(ents):
        v = e.split(',')
        try:
            ok = len(v) == 3 and all(abs(float(x)) < 1.0 for x in v)     # also refuses nan and inf
        except ValueError:
            ok = False
        if not ok or not re.fullmatch(r'[0-9eE+\-., ]+', e):
            return 'entry %d (%r) is not three numbers' % (i, e[:60])
    return None


def put(path, text):
    """Write a file of the flight folder, replacing whatever an earlier deployment left there (even read-only)."""
    if os.path.lexists(path):
        os.remove(path)
    open(path, 'w').write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib-out', required=True, help='the --out directory of a run_tool.py run')
    ap.add_argument('--out', required=True, help='flight configuration folder to create')
    ap.add_argument('--vio-root', default=None,
                    help='path to the vio/ package (default: $VIO_ROOT, else the sibling ../vio)')
    ap.add_argument('--force', action='store_true',
                    help='deploy even if the calibration verdict was not HEALTHY')
    a = ap.parse_args()

    cal = os.path.abspath(a.calib_out)
    rep_path = os.path.join(cal, 'report.json')
    if not os.path.isfile(rep_path):
        raise SystemExit('[deploy] ERROR: %s is not a calibration output directory '
                         '(no report.json).' % cal)
    rep = json.load(open(rep_path))
    drone = rep.get('drone', 'vehicle')
    verdict = rep.get('verdict', '(none)')

    # ---- 1. gate on the verdict: never deploy a calibration the tool did not accept ----
    log('calibration: %s   verdict: %s' % (cal, verdict))
    if not verdict.startswith('HEALTHY'):
        msg = ('[deploy] ERROR: verdict is not HEALTHY — refusing to deploy.\n'
               '  %s\n'
               '  FLY-AGAIN means the recording was inadequate: record another flight and\n'
               '  recalibrate. HARDWARE-CHANGED means the calibration is valid but the vehicle\n'
               '  left the fleet distribution — inspect, then re-run with --force to accept it.\n'
               '  PLATFORM-DEFECT means a defect independent of calibration; repair first.'
               % verdict)
        if not a.force:
            raise SystemExit(msg)
        log('WARNING: verdict is %s but --force was given; deploying anyway.' % verdict.split(':')[0])

    # ---- 2. locate the calibrated inputs ----
    chain = os.path.join(cal, '%s_published_chain.yaml' % drone)
    if not os.path.isfile(chain):
        raise SystemExit('[deploy] ERROR: published chain not found: %s' % chain)
    imu = os.path.join(cal, 'calib', 'kalibr_imu_chain.yaml')
    if not os.path.isfile(imu):
        raise SystemExit('[deploy] ERROR: IMU chain not found: %s\n'
                         '  (run_tool.py copies the --imu-chain it used to calib/)' % imu)
    mount_json = os.path.join(cal, '%s_mount.json' % drone)
    # Present only when the calibration ran with OV_PRIOR_NONRAD_SIG: two lens terms per camera,
    # which the chain format cannot hold. The estimator reads them from OV_NONRADIAL.
    terms_txt = os.path.join(cal, '%s_published_terms.txt' % drone)
    terms = (open(terms_txt).read().strip() or None) if os.path.isfile(terms_txt) else None
    # They become the values the flight estimator starts from, so they must be the ones this run published: a file left
    # by an earlier run in the same folder, or an emptied one, is refused.
    rep_terms = (rep.get('published_nonradial') or '').strip() or None
    if terms != rep_terms:
        raise SystemExit('[deploy] ERROR: %s %s.\n'
                         '  The lens terms are part of the calibration. If the file was left out when the calibration\n'
                         '  output was copied, copy it; otherwise run the calibration again into an empty --out.'
                         % (terms_txt, 'holds lens terms, but report.json of this run published none' if not rep_terms else
                            'is missing or empty, but report.json of this run published lens terms' if not terms else
                            'holds lens terms that differ from the ones report.json of this run published'))

    vio = resolve_vio_root(a.vio_root)
    flight_cfg = os.path.join(vio, 'vio_deploy', 'config', 'estimator_flight.yaml')
    if not os.path.isfile(flight_cfg):
        raise SystemExit('[deploy] ERROR: flight config not found: %s' % flight_cfg)
    stiffness = os.path.join(vio, 'vio_deploy', 'config', 'flight_stiffness.env')
    if not os.path.isfile(stiffness):
        raise SystemExit('[deploy] ERROR: flight settings not found: %s' % stiffness)
    fl_src = _env_file(stiffness)
    PRIORS = ('OV_PRIOR_DT_SIG', 'OV_PRIOR_EXTR_Q_SIG', 'OV_PRIOR_EXTR_T_SIG', 'OV_PRIOR_INTR_F_SIG', 'OV_PRIOR_INTR_D_SIG')
    if not all(k in fl_src for k in PRIORS):
        raise SystemExit('[deploy] ERROR: %s lacks one of the five flight priors (%s)' % (stiffness, ', '.join(PRIORS)))

    # ---- 3. read the values back and show them ----
    cams, toff = parse_chain(chain)
    if terms and terms_problem(terms, len(cams)):
        raise SystemExit('[deploy] ERROR: lens terms in %s: %s' % (terms_txt, terms_problem(terms, len(cams))))
    log('camera-IMU time offset t_d: %+.6f s' % toff)
    log('per-camera calibrated values (from %s):' % os.path.basename(chain))
    print('        %-4s %10s %10s %10s %10s   %-28s %-24s'
          % ('cam', 'fx', 'fy', 'cx', 'cy', 'distortion k1..k4', 'extrinsic T_imu_cam'))
    for c in sorted(cams):
        f, k = cams[c]['f'], cams[c]['k']
        p = np.asarray(cams[c]['p'])
        r, pi, y = rpy_deg(cams[c]['R'])
        print('        %-4d %10.3f %10.3f %10.3f %10.3f   [%6.3f %6.3f %6.3f %6.3f]  '
              't=[%+.3f %+.3f %+.3f] m' % (c, f[0], f[1], f[2], f[3], k[0], k[1], k[2], k[3],
                                           p[0], p[1], p[2]))
        print('        %-4s %10s rpy = %+7.2f %+7.2f %+7.2f deg' % ('', '', r, pi, y))

    mount = None
    if os.path.isfile(mount_json):
        mount = json.load(open(mount_json))
        log('mount rotation: yaw %+.2f deg, tilt %.2f deg  (anchored %.3f m -> %.3f m)'
            % (mount['mount_yaw_deg'], mount['mount_tilt_deg'],
               mount['A_anchored_m'], mount['A_mounted_m']))
        log('  NOTE: the estimator does NOT read this. Apply it to the OUTPUT trajectory '
            '(mount.apply).')
    else:
        log('mount rotation: not solved (no ground truth in that calibration run) — '
            'the deployed folder simply omits it')

    # ---- 4. write the folder the estimator expects ----
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(flight_cfg, os.path.join(a.out, 'estimator_flight.yaml'))
    campaign_env = os.path.join(vio, 'vio_deploy', 'config', 'campaign.env')
    if os.path.isfile(campaign_env):
        shutil.copy(campaign_env, os.path.join(a.out, 'campaign.env'))
    else:
        log('WARNING: %s not found -- the deployed folder will not carry the campaign switches' % campaign_env)
    shutil.copy(chain, os.path.join(a.out, 'kalibr_imucam_chain.yaml'))
    shutil.copy(imu, os.path.join(a.out, 'kalibr_imu_chain.yaml'))
    if mount is not None:
        shutil.copy(mount_json, os.path.join(a.out, 'mount.json'))
    elif os.path.lexists(os.path.join(a.out, 'mount.json')):
        os.remove(os.path.join(a.out, 'mount.json'))            # of an earlier calibration deployed into --out
    # The flight settings travel with the calibration: the folder gets its own copy of flight_stiffness.env, ending with
    # the lens terms of THIS calibration, so that one sourced file gives a flight everything it needs on every platform.
    # The line is written even when there are no terms (empty = zero), so that sourcing this folder after another one
    # cannot leave the other calibration's terms in the shell. Quoted: the value holds ';'.
    put(os.path.join(a.out, 'flight_stiffness.env'),
        '# COPY written by deploy_vio.py on %s for %s, from %s\n'
        '# (calibration: %s). Source THIS file to fly: its last line holds the lens terms of that calibration.\n#\n'
        % (time.strftime('%Y-%m-%d'), drone, stiffness, cal)
        + open(stiffness).read().rstrip('\n') + '\n\n'
        '# ADDED BY deploy_vio.py: the two lens terms per camera of THIS calibration, the values the flight estimator\n'
        '# starts from (empty: the calibration has none, they start at zero).\n'
        "OV_NONRADIAL='%s'\n" % (terms or ''))
    if terms:
        put(os.path.join(a.out, 'lens_terms.env'), "OV_NONRADIAL='%s'\n" % terms)
        log('lens terms: %d cameras -> last line of flight_stiffness.env (and lens_terms.env); part of this calibration'
            % len(terms.split(';')))
    else:
        if os.path.lexists(os.path.join(a.out, 'lens_terms.env')):
            os.remove(os.path.join(a.out, 'lens_terms.env'))    # of an earlier calibration deployed into --out
        log('lens terms: none in this calibration; in flight they start at zero')

    # ---- 5. verify the deployed folder is self-consistent before declaring success ----
    problems = []
    for f in ('estimator_flight.yaml', 'kalibr_imucam_chain.yaml', 'kalibr_imu_chain.yaml'):
        p = os.path.join(a.out, f)
        if not os.path.isfile(p):
            problems.append('missing %s' % f); continue
        first = open(p).readline().strip()
        if not first.startswith('%YAML:1.0'):
            problems.append('%s: %%YAML:1.0 must be line 1 (cv::FileStorage rejects it '
                            'otherwise), found %r' % (f, first[:40]))
    dep_cams, dep_toff = parse_chain(os.path.join(a.out, 'kalibr_imucam_chain.yaml'))
    if sorted(dep_cams) != sorted(cams):
        problems.append('camera set changed in transit: %s -> %s' % (sorted(cams), sorted(dep_cams)))
    cfg = open(os.path.join(a.out, 'estimator_flight.yaml')).read()
    for key, want in (('relative_config_imu', 'kalibr_imu_chain.yaml'),
                      ('relative_config_imucam', 'kalibr_imucam_chain.yaml')):
        if '%s: "%s"' % (key, want) not in cfg:
            problems.append('%s in the flight config does not point at %s' % (key, want))
    fl = _env_file(os.path.join(a.out, 'flight_stiffness.env'))
    if fl.get('OV_NONRADIAL', None) != (terms or ''):
        problems.append('flight_stiffness.env does not end with the lens terms of this calibration')
    if {k: v for k, v in fl.items() if k != 'OV_NONRADIAL'} != {k: v for k, v in fl_src.items() if k != 'OV_NONRADIAL'}:
        problems.append('flight_stiffness.env of the folder does not set what %s sets' % stiffness)
    if problems:
        raise SystemExit('[deploy] ERROR: deployed folder failed verification:\n  - '
                         + '\n  - '.join(problems))

    ncam = len(dep_cams)
    out_abs = os.path.abspath(a.out); q = shlex.quote(out_abs)
    # what the copied flight settings hold (the vio package may be older or newer than this tool)
    states = (_number(fl.get('OV_PRIOR_NONRAD_SIG')) or 0.0) > 0.0
    matching = fl.get('OV_XCAM', '')[:1] == '1'
    holds = ['calibration priors x0.10']
    holds.append('the two lens terms per camera as calibration states' if states else 'the lens terms as fixed values')
    holds.append('matching of features between neighbouring cameras' if matching else 'no matching between cameras')
    # the topics of the recording the calibration was made on = the vehicle's own
    layout = []
    cfg_cal = os.path.join(cal, 'calib', 'estimator_config.yaml')
    if os.path.isfile(cfg_cal):
        layout = [l.rstrip() for l in open(cfg_cal) if re.match(r'(imu_topic|imu_msg_type|cam_msg_type|cam_topic\d+):', l)]
    manifest = """VIO flight configuration for %s
generated by selfcalib/tool/deploy_vio.py from %s
calibration verdict: %s

FILES
  estimator_flight.yaml      flight-mode estimator config (from the vio package).
  kalibr_imucam_chain.yaml   THE CALIBRATION: per-camera intrinsics [fx fy cx cy],
                             distortion coefficients, T_imu_cam extrinsics, and the
                             camera-IMU time offset timeshift_cam_imu = %+.9f s.
  kalibr_imu_chain.yaml      IMU noise densities + IMU-intrinsics blocks.
  flight_stiffness.env       the flight settings (from the vio package):
                             %s.
                             Its last line holds the lens terms of THIS calibration
                             (%s).
  campaign.env               the OV_* environment of the campaign numbers (aarch64 only).
  mount.json                 mount rotation M. NOT read by the estimator -- apply it to the
                             OUTPUT trajectory (selfcalib/tool/mount.py, apply()).%s%s

All three YAML files must stay in THIS directory together: the estimator resolves
relative_config_imu / relative_config_imucam relative to the config file's own location.

FLY IT
  # the flight settings of this folder, on every platform. REQUIRED.
  set -a; source %s/flight_stiffness.env; set +a
  # on the vehicle (aarch64) ALSO source the campaign switches -- the GPU tracker, scheduler and
  # algorithm gates every quoted fleet number was measured with. Never on a desktop.
  [ "$(uname -m)" = aarch64 ] && { set -a; source %s/campaign.env; set +a; }
  # check: this must print %d lines. If it prints none, the file was not found and the estimator
  # would fly with loose priors.
  env | grep OV_PRIOR

  # offline replay of a recording (on a desktop first: export OV_GROUP_CAMS=1, which hands the
  # estimator the four images of an instant as one set, as on the vehicle)
  bash <vio>/vio_deploy/scripts/run_serial.sh BAG %s/estimator_flight.yaml OUT %d false 42 DOMAIN

  # live on the vehicle
  ros2 run ov_msckf run_online_msckf %s/estimator_flight.yaml

TOPICS: both programs read the topics from estimator_flight.yaml, which names none as
deployed. They then read /imu0 (sensor_msgs/Imu) and /cam<N>/image_raw (sensor_msgs/Image).
Any other vehicle, the fleet's included, needs its topics added to that file first.
%s
Without them the live node starts and receives nothing, and a replay writes no trajectory.

flight_stiffness.env is REQUIRED: it tightens the calibration priors x0.10 so the online
calibration stays tethered to this chain instead of random-walking during flight. Without
it the estimator runs with loose calibration priors, which is the CALIBRATION operating
point, not the flight one. Source the copy in THIS folder: it also hands the estimator the
lens terms of this calibration, which the chain file cannot hold.

The matching between cameras runs where the four images of an instant reach the estimator
as one set: in run_online_msckf, and in a replay with OV_GROUP_CAMS=1 (campaign.env sets it
on the vehicle). run_subscribe_msckf feeds one camera per message and reads neither the PX4
IMU nor compressed images: it is not the flight node.
""" % (drone, cal, verdict, dep_toff, (';\n' + 29 * ' ').join(holds),
       'empty: it has none' if not terms else 'two numbers and a held third per camera',
       '' if mount is not None else '  (mount.json absent: that calibration had no ground truth)',
       '' if not terms else
       '\n  lens_terms.env             the same lens terms alone (OV_NONRADIAL), for whatever else\n'
       '                             uses this chain: without them it is not the calibrated lens model.',
       q, q, sum(1 for k in fl if k.startswith('OV_PRIOR')), q, ncam, q,
       ('The recording this calibration was made on used these (add them as they are):\n' + '\n'.join('  ' + l for l in layout)) if layout else
       'Add imu_topic, imu_msg_type, cam_msg_type and cam_topic0..%d (the recording this calibration\nwas made on used the defaults).' % (ncam - 1))
    open(os.path.join(a.out, 'MANIFEST.txt'), 'w').write(manifest)

    log('deployed -> %s' % os.path.abspath(a.out))
    for f in sorted(os.listdir(a.out)):
        log('   %s' % f)
    log('verified: %d cameras, t_d %+.6f s, chain/config cross-references consistent'
        % (ncam, dep_toff))
    log('fly it with the command in %s/MANIFEST.txt' % os.path.abspath(a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
