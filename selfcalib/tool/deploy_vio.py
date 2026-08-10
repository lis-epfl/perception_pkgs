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
    <out>/mount.json                mount rotation, if solved   (NOT an estimator input)
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
import argparse, json, os, shutil, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chainio import parse_chain
from run_tool import resolve_vio_root


def log(s):
    print('[deploy] ' + s, flush=True)


def rpy_deg(R):
    """Roll/pitch/yaw in degrees, for human inspection only (the chain carries the matrix)."""
    R = np.asarray(R)
    pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


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

    vio = resolve_vio_root(a.vio_root)
    flight_cfg = os.path.join(vio, 'vio_deploy', 'config', 'estimator_flight.yaml')
    if not os.path.isfile(flight_cfg):
        raise SystemExit('[deploy] ERROR: flight config not found: %s' % flight_cfg)

    # ---- 3. read the values back and show them ----
    cams, toff = parse_chain(chain)
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
    shutil.copy(chain, os.path.join(a.out, 'kalibr_imucam_chain.yaml'))
    shutil.copy(imu, os.path.join(a.out, 'kalibr_imu_chain.yaml'))
    if mount is not None:
        shutil.copy(mount_json, os.path.join(a.out, 'mount.json'))

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
    if problems:
        raise SystemExit('[deploy] ERROR: deployed folder failed verification:\n  - '
                         + '\n  - '.join(problems))

    ncam = len(dep_cams)
    manifest = """VIO flight configuration for %s
generated by selfcalib/tool/deploy_vio.py from %s
calibration verdict: %s

FILES
  estimator_flight.yaml      flight-mode estimator config (from the vio package).
  kalibr_imucam_chain.yaml   THE CALIBRATION: per-camera intrinsics [fx fy cx cy],
                             distortion coefficients, T_imu_cam extrinsics, and the
                             camera-IMU time offset timeshift_cam_imu = %+.9f s.
  kalibr_imu_chain.yaml      IMU noise densities + IMU-intrinsics blocks.
  mount.json                 mount rotation M. NOT read by the estimator -- apply it to the
                             OUTPUT trajectory (selfcalib/tool/mount.py, apply()).%s

All three YAML files must stay in THIS directory together: the estimator resolves
relative_config_imu / relative_config_imucam relative to the config file's own location.

FLY IT
  # offline replay of a recording
  set -a; source <vio>/vio_deploy/config/flight_stiffness.env; set +a
  bash <vio>/vio_deploy/scripts/run_serial.sh BAG %s/estimator_flight.yaml OUT %d false 42 DOMAIN

  # live on the vehicle
  set -a; source <vio>/vio_deploy/config/flight_stiffness.env; set +a
  ros2 run ov_msckf run_subscribe_msckf %s/estimator_flight.yaml

flight_stiffness.env is REQUIRED: it tightens the calibration priors x0.10 so the online
calibration stays tethered to this chain instead of random-walking during flight. Without
it the estimator runs with loose calibration priors, which is the CALIBRATION operating
point, not the flight one.
""" % (drone, cal, verdict, dep_toff,
       '' if mount is not None else '  (mount.json absent: that calibration had no ground truth)',
       os.path.abspath(a.out), ncam, os.path.abspath(a.out))
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
