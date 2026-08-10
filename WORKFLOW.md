# The workflow, end to end

Three steps: **calibrate** a vehicle from an ordinary flight recording, **read** the
calibrated values, **deploy** them into the folder the VIO estimator flies with.

```
  recording ──▶ [1] calibrate ──▶ published chain ──▶ [2+3] deploy ──▶ flight config folder ──▶ fly
                  selfcalib          + mount.json          deploy_vio.py         vio
```

---

## Step 0 — build the estimator (once per machine)

```bash
bash vio/vio_deploy/scripts/build_openvins.sh          # ~6 min → ~/ov_ws_vio
```

Both operating points use this one build. Needs Ubuntu 22.04, ROS 2 Humble, and
`libeigen3-dev libboost-all-dev libceres-dev`.

---

## Step 1 — calibrate

```bash
python3 selfcalib/tool/run_tool.py \
    --drone myvehicle \
    --bag /path/to/recording_bag \
    --template selfcalib/fleet_reference/nxt3/theta_star.yaml \
    --imu-chain /path/to/kalibr_imu_chain.yaml \
    --out out_myvehicle \
    --fleet-exclude myvehicle
```

The recording is an ordinary flight — no target, no manual session. It must start with the
vehicle **still on the ground** and contain real motion afterwards (see `selfcalib/README.md`
§3 for the measured limits). Ground truth is optional: if the bag happens to carry an
OptiTrack/mocap topic the tool finds it automatically and you additionally get the ATE
certificate and the mount solve.

**Check the verdict before going further:**

| verdict | meaning | next |
|---|---|---|
| `HEALTHY: deploy` | all certificates passed | continue to step 2 |
| `FLY-AGAIN` | the *recording* was inadequate, not the vehicle | record another flight, recalibrate |
| `HARDWARE-CHANGED` | converged, but off the fleet distribution — a real physical change | inspect the vehicle, then deploy with `--force` |
| `PLATFORM-DEFECT` | calibration is fine, the platform is not (timing/IMU) | repair; recalibrating will not help |

Exit codes: `0` ok · `1` gate failure · `2` never converged (`FLY-AGAIN`) · `3` `--gt` file
missing · `4` estimator pass timed out · `5` estimator produced no harvest (read
`out_myvehicle/calib/out/run.log`).

If you script this: `2` is also what argparse returns for a bad argument, so distinguish the
two by whether `report.json` exists — the tool writes it on every outcome it reaches, and a
usage error exits before that.

### What step 1 leaves behind

```
out_myvehicle/
  myvehicle_published_chain.yaml   ← THE CALIBRATION (intrinsics, extrinsics, time offset)
  myvehicle_mount.json             ← mount rotation M, only if ground truth was available
  report.json                      verdict + all three certificates + what was detected.
                                   Valid JSON (jq-parseable). Note `diagnosis.ate.postg_m`
                                   is `null` when the recording has fewer than 20
                                   post-takeoff poses — the ATE certificate then falls back
                                   to `global_m`, so the verdict is still meaningful.
  calib/kalibr_imu_chain.yaml      the IMU chain the run used (copied for you)
  estimate_pass*.tum               per-pass trajectories
  harvest_pass*.calib.json         per-pass raw harvests
  ground_truth.tum                 extracted mocap stream, if one was found in the bag
```

---

## Steps 2 + 3 — read the values and deploy them

One command does both. It prints every calibrated value for inspection, then writes the
folder the estimator expects:

```bash
python3 selfcalib/tool/deploy_vio.py --calib-out out_myvehicle --out flight_myvehicle
```

```
[deploy] calibration: out_myvehicle   verdict: HEALTHY: deploy
[deploy] camera-IMU time offset t_d: -0.039256 s
[deploy] per-camera calibrated values (from myvehicle_published_chain.yaml):
        cam          fx         fy         cx         cy   distortion k1..k4     extrinsic T_imu_cam
        0       252.247    252.611    580.597    335.025   [0.093 -0.032 ...]  t=[-0.090 +0.068 +0.032] m
                          rpy =  -91.03   -1.20  +45.19 deg
        ...
[deploy] mount rotation: yaw -3.24 deg, tilt 3.25 deg  (anchored 0.245 m -> 0.059 m)
[deploy]   NOTE: the estimator does NOT read this. Apply it to the OUTPUT trajectory.
[deploy] verified: 4 cameras, t_d -0.039256 s, chain/config cross-references consistent
```

It **refuses to deploy a calibration whose verdict was not `HEALTHY`** unless you pass
`--force`, and it verifies the result before declaring success: all three YAML files present,
`%YAML:1.0` on line 1 of each, camera set unchanged in transit, and the config's
`relative_config_*` keys actually pointing at the two chain files.

### What you get

```
flight_myvehicle/
  estimator_flight.yaml       flight config (from the vio package)
  kalibr_imucam_chain.yaml    ← the published chain, renamed
  kalibr_imu_chain.yaml       ← the IMU chain
  mount.json                  mount rotation (if solved)
  MANIFEST.txt                what each file is, and the exact command to fly it
```

**Why the rename matters.** The estimator resolves `relative_config_imu` and
`relative_config_imucam` *relative to the config file's own directory*, and it looks for
those literal filenames. A published chain called `myvehicle_published_chain.yaml` sitting
somewhere else will never be found. Getting this wrong is the single most common way to
"fly with the wrong calibration" — the estimator falls back to whatever chain it does find,
and nothing flags it. That is the entire reason this step is a tool and not a note.

---

## Step 4 — fly

```bash
set -a; source vio/vio_deploy/config/flight_stiffness.env; set +a   # REQUIRED

# offline replay of a recording
bash vio/vio_deploy/scripts/run_serial.sh BAG flight_myvehicle/estimator_flight.yaml OUT 4 false 42 70

# live on the vehicle
ros2 run ov_msckf run_subscribe_msckf flight_myvehicle/estimator_flight.yaml
```

`flight_stiffness.env` is **not optional**. It tightens the calibration priors ×0.10 so the
online calibration stays tethered to the chain you just deployed. Without it you are running
the *calibration* operating point in flight, with loose priors that let the calibration
random-walk. Check with `env | grep OV_PRIOR` — flying should print five variables;
calibrating should print none.

---

## Where the mount rotation goes

Intrinsics, extrinsics and the time offset are **estimator inputs** — they enter the filter
through `kalibr_imucam_chain.yaml`. The mount rotation `M` is **not**: the estimator never
reads it. It is applied afterwards, to the estimator's output trajectory:

```python
import sys; sys.path.insert(0, 'selfcalib/tool')
import mount, json
M = json.load(open('flight_myvehicle/mount.json'))['M']
print(mount.apply('estimate_tum.txt', 'gt.tum', M, toff=<published_toff>))
```

It makes the **unaligned** (start-anchored) trajectory accurate without fitting against
ground truth on every flight — which is the point, since you have no ground truth in the
field.

**`M` transfers between flights only if each flight is anchored at the same initial pose.**
`M = R_fit · R_anchor⁻¹`, so it inherits the attitude the vehicle held at its anchor instant:
take off from the same spot on the same heading and `M` is reproduced; start rotated by θ and
`M` is wrong by θ. Measured on the reference flight, the discrepancy between `M` solved at
two different anchor instants is 3.446° and the discrepancy between the two `R_anchor` values
is 3.446° — identical to machine precision. Every bit of the variation is the anchor.

So: **fix a takeoff pose per vehicle and keep to it**, and one solved `M` serves all its
later flights.

---

## Recalibrating

Re-run step 1 on a new recording and step 2+3 again. Deploy into a *new* folder rather than
overwriting, so a bad calibration cannot silently replace a good one — the folders are a few
KB each. `report.json` in the calibration output is the record of which recording produced
which chain.
