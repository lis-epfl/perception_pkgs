# Flight-Data Self-Calibration Tool

Calibrates a multi-camera + IMU rig **from an ordinary flight recording** — no target, no
manual session, no ground truth. One command runs the full pipeline: recording health gates →
zero-cost seed (fleet transfer + image-circle principal point) → iterated warm-start
self-calibration to self-consistency → robust-mean publish → diagnosis verdict
(HEALTHY / FLY-AGAIN / HARDWARE-CHANGED / PLATFORM-DEFECT).

Validated on a 3-vehicle fisheye fleet (certified convergence basin, ~7,000 measured chains),
a held-out vehicle it had never seen (calibrated in one pass; correctly attributed the
residual error to a hardware fault), and three public datasets (TUM-VI, EuRoC, UZH-FPV).
The estimator is **not** in this package: it lives in the sibling `vio/` package, which
flight VIO uses too. One estimator, two operating points — see `../README.md`.

## Contents

```
README.md                  <- you are here: setup + usage + input contracts
tool/                      run_tool.py   the calibration pipeline (entry point)
                           deploy_vio.py the deployment step: read a finished calibration
                                         and write the folder the VIO estimator flies with
configs/
  estimator_calib.yaml       the CALIBRATION config (priors loose; 4-camera fisheye KB4 rigs)
fleet_reference/           per-vehicle published calibrations — the seed source for
                           --fleet-exclude and the in-distribution certificate. Replace with
                           YOUR fleet. Each theta_star.yaml doubles as a working example of
                           the seed chain format.
docs/
  PROTOCOL.md                warm-start pass, self-consistency, envelope, verdicts
  TROUBLESHOOTING.md         failure signatures and fixes
  TOOL_INTERNALS.md          per-module internals
  VALIDATION_MATRIX.md       what was tested and how
  DEPLOYMENT_STRIP.md        how this package and vio/ were reduced, and the evidence
```

The FLIGHT config and the estimator runner live in `vio/` — this package owns only the
calibration config and calls `vio/vio_deploy/scripts/run_serial.sh` to drive the estimator.

This is the **deployment package**: the study package reduced to what a fleet vehicle
actually needs (404 MB -> 2.3 MB), with the bundled OpenVINS fork pruned to the code that
is actually reachable in these two operating points (-1,907 lines; 41 experiment toggles
down to 2, both live). It reproduces the study's recorded reference run **byte-for-byte**
through the whole pipeline. The only functional additions are ground-truth auto-detection
(`tool/gtdetect.py`) and the wired-in mount solve. `docs/DEPLOYMENT_STRIP.md` has the
complete change list and the equivalence evidence.

## 1. Prerequisites

- Ubuntu 22.04, **ROS 2 Humble** (`/opt/ros/humble` must exist), `colcon`.
- OpenCV ≥ 4.5 (system), Eigen3, Boost, Ceres (OpenVINS deps — all apt-installable:
  `libeigen3-dev libboost-all-dev libceres-dev`).
- Python 3.10+: `numpy scipy opencv-python` (plus `rosbag2_py`/`rclpy`, which come with ROS).

## 2. Build the estimator (once)

The estimator belongs to the sibling package:

```bash
bash ../vio/vio_deploy/scripts/build_openvins.sh     # ~6 min → ~/ov_ws_vio
```

This package locates `vio/` via `--vio-root`, then `$VIO_ROOT`, then the sibling `../vio`.
Side-by-side needs no configuration; `run_tool.py` prints the runner it resolved on startup
and fails with an explicit message listing all three candidates if it finds none.

Everything downstream invokes the estimator ONLY through `vio/vio_deploy/scripts/run_serial.sh`,
which sources the workspace, pins `ROS_LOCALHOST_ONLY=1`, sets the RNG seed, and runs the
serial node. Never run the estimator any other way; parallel invocations must use distinct
`DOMAIN` arguments (valid range 0–232 — higher silently fails).

> **Reproducibility caveat.** Short recordings reproduce exactly (8/8 identical runs on a
> 30 s bag), but on long ones repeated identical runs sometimes diverge — around pose 419 of
> 2806 on the 185 s reference recording, reaching ~0.12 m. The effect on the published
> calibration is far inside the acceptance thresholds (0.13 px principal point against a
> 2.5 px limit; 0.035° rotation against 0.35°), ~3 orders of magnitude below the warm-start
> convergence criterion, so it changes no verdict. Flight mode shows ~0.2 cm ATE spread. It
> is a property of the estimator, not of any config here. Compare published chains against
> thresholds, not byte-for-byte.

## 3. Input contract (read carefully — the gates enforce this)

**Recording**: a ROS 2 sqlite3 bag with topics `/cam0/image_raw … /camN/image_raw` (`mono8`)
and `/imu0` (`sensor_msgs/Imu`), on a shared clock with nanosecond stamps. (The study
package's ASL/EuRoC and image-list bag converters are not shipped here — fleet vehicles
record ROS 2 bags natively. Take them from the study package if you need to replay a
public dataset.)

Hard requirements (measured limits, not preferences):
1. **The recording must begin with a static segment ≥ 4 s** (vehicle on the ground / rig held
   still). Without it, initialization fails at every tested flight length and pass budget.
   That 4 s is the *recommendation*; the gate that actually runs (`gates.STATIC_MIN_S`)
   enforces ≥ 1.5 s of contiguous stillness within the first 12 s, so a recording between
   1.5 s and 4 s passes the gate but is below what the study validated.
2. **≥ ~3.2 s of actual motion after the static start** (10–15 s recommended; more only helps).
3. Time offset between camera and IMU clocks **within ≈ ±0.1 s** of the seed's
   `timeshift_cam_imu`. This is the one parameter self-calibration cannot recover from far
   away (its basin is a freeze cliff, not a graded edge — see docs/PROTOCOL.md). Supply it:
   hardware timestamps or a sibling vehicle's value are both fine.

**Seed calibration files** (kalibr chain format — copy a working pair from
`fleet_reference/<vehicle>/` rather than writing one from scratch):
- `kalibr_imucam_chain.yaml`: per camera `cam0..N`: **`T_imu_cam`** (4×4), `intrinsics:
  [fx, fy, cx, cy]`, `distortion_coeffs` (4 values, KB4/equidistant or radtan),
  `distortion_model`, `resolution`, `rostopic` (must match the bag topics exactly),
  `timeshift_cam_imu`. **All lists single-line** (kalibr's own YAML dumper wraps lines —
  re-emit through a YAML loader if needed).
  **Use the key name `T_imu_cam`, not `T_cam_imu`.** The estimator accepts either spelling
  (`ov_core/src/utils/opencv_yaml_parse.h`), but `tool/chainio.py` reads only `T_imu_cam` and
  **refuses a chain that uses the other spelling**, naming it in the error — the two are
  inverses, so reinterpreting one as the other would publish a silently wrong calibration.
  Invert the 4×4 matrices and rename the key. Every `fleet_reference/*/theta_star.yaml`
  already uses `T_imu_cam`.
- `kalibr_imu_chain.yaml`: noise densities + `update_rate` + `rostopic: /imu0`, and it MUST
  contain the full IMU-intrinsics block (`Tw`, `Ta`, `R_IMUtoGYRO`, `R_IMUtoACC`, `Tg`,
  identity-valued) — the estimator refuses to parse without them. Copy
  `fleet_reference/nxt3/kalibr_imu_chain.yaml` and edit the noise values.

**Every YAML the estimator reads must start with `%YAML:1.0` on line 1.** OpenCV's
`cv::FileStorage` rejects the file if even a comment precedes it, and the estimator aborts
with `(-5:Bad argument) Input file is invalid in function 'open'`.

Seed quality: the basin is wide. Datasheet focal (±tens of %), image-center principal point,
zero distortion, mounting-sketch extrinsics (±few degrees / ±tens of cm) all converge. The
tool builds its own seed from the fleet + the recording when you use `--fleet-exclude`.

## 4. Configuration policy (do not tune)

Pick ONE config and edit ONLY hardware descriptors:

| you have | config | edit these fields only |
|---|---|---|
| 4-camera fisheye rig (like the study fleet) | `configs/estimator_calib.yaml` | `gravity_mag`, IMU noise (via imu chain), masks |

(The study package's `estimator_standard_template.yaml` for stereo / other standard rigs is
not shipped here — this package targets the fisheye fleet. `tool/run_tool.py` always uses
`configs/estimator_calib.yaml` for calibration.) The rig's camera count comes from the
template chain and the per-camera lens half-FOV from this config's `mask_fisheye_theta_max{i}`
keys, so a different fisheye rig is a config-and-chain change rather than a code change; a
non-fisheye rig still needs the standard-rig config from the study package.

`init_imu_thresh` is the accelerometer-jerk threshold that ends static initialization —
0.3–0.5 for gentle platforms, 1.5 for vehicles that take off hard. Estimator internals
(clone counts, feature counts, sigmas, chi2) are deliberately not per-platform knobs: the
study's results replicate across two very different settings of them, and tuning them
invites overfitting your one recording. Calibration states
(`calib_cam_intrinsics/extrinsics/timeoffset: true`) must stay on.

## 5. Run

```bash
# SCT_ROOT is OPTIONAL: the tool defaults to its own package root (the directory
# containing tool/). Set it only when running a tool/ copied out of the package.
python3 tool/run_tool.py \
    --drone myvehicle \
    --bag /path/to/recording_bag \
    --template fleet_reference/nxt3/theta_star.yaml \   # yaml SKELETON + mount extrinsics source
    --imu-chain /path/to/kalibr_imu_chain.yaml \
    --out out_myvehicle \
    [--fleet-exclude myvehicle]   # derive seed from fleet means, leave-one-out
    [--gt gt.tum]                 # optional; omit and the tool looks in the bag itself
    [--max-pass 8] [--domain 70]
```

Stages (all logged to `--out`): ground-truth lookup → gates (static start, timing health,
image health) → one temporal-accumulation pass over the bag (image-circle principal-point
fit) → seed assembly → warm-start loop: run estimator, harvest calibration, re-seed, repeat
**until two consecutive harvests agree to 0.10 in the residual metric and <10 ms in t_d**
(typically 1–2 passes from a fleet seed, ≤8 from a blind seed) → robust-mean publish (median
intrinsics, chordal-mean rotations over harvests) → diagnosis → mount solve.

Runtime: roughly (bag length) × (passes) × 1–2× real time, single machine, no GPU.

**Outputs**: `--out/<drone>_published_chain.yaml` (the calibration), `--out/report.json`
(diagnosis + certificates — valid JSON; `diagnosis.ate.postg_m` is `null` when the recording
has under 20 post-takeoff poses, and the certificate falls back to `global_m`), per-pass
harvests and logs. With ground truth, also
`--out/ground_truth.tum` and `--out/<drone>_mount.json`.

### Next step: deploy it

Do **not** hand-copy the published chain into a flight folder. The estimator looks for the
literal filenames `kalibr_imucam_chain.yaml` / `kalibr_imu_chain.yaml` *beside its config*,
so a chain named `<drone>_published_chain.yaml` is silently never loaded — the estimator
picks up whatever chain it does find instead. Use the deployment step, which does the rename,
refuses a calibration whose verdict was not `HEALTHY`, and verifies the result:

```bash
python3 tool/deploy_vio.py --calib-out out_myvehicle --out flight_myvehicle
```

It also prints every calibrated value (intrinsics, extrinsics, t_d, mount) for inspection.
See `../WORKFLOW.md` for the full four-step walkthrough.

**Exit codes**: `0` success · `1` gate failure (re-record) · `2` never reached
self-consistency (`FLY-AGAIN`) · `3` `--gt` file missing · `4` an estimator pass timed out
(raise `--pass-timeout`) · `5` an estimator pass produced no harvest (read
`--out/calib/out/run.log`). Anything non-zero also writes `report.json` with the reason.

### Ground truth is optional and found automatically

Calibration itself never needs ground truth. When it *is* available, two extra things come
out of the same run: the **ATE-health certificate** (§6.3, the difference between a
`HEALTHY` and a `PLATFORM-DEFECT` verdict) and the **mount solve** (§6.2).

You do not have to export it yourself. If `--gt` is not given, `tool/gtdetect.py` scans the
bag for a mocap pose stream and extracts it to `--out/ground_truth.tum`. It handles
`geometry_msgs/PoseStamped`, `PoseWithCovarianceStamped`, `TransformStamped`,
`nav_msgs/Odometry` and `tf2_msgs/TFMessage` (picking the vehicle's `child_frame_id`), and
ranks candidates by topic name — `optitrack`, `vrpn`, `mocap`, `vicon`, `natnet`,
`qualisys`, `ground_truth` score up; estimator outputs like `/ov_msckf/...` score down; the
`--drone` name is used as a tiebreak. What it chose and what it rejected is recorded under
`ground_truth` in `report.json`.

If nothing suitable is in the bag, the run proceeds exactly as before and the ATE
certificate is skipped. Override the search with `--gt-topic /your/topic`, or turn it off
with `--no-gt-autodetect`.

```bash
python3 tool/gtdetect.py BAG                    # just list what it can see
python3 tool/gtdetect.py BAG --out gt.tum       # extract the best candidate
```

Two things to check when it does find something: `stamp_source` must be `header` (a
`bag_receive_time` fallback means the publisher left `header.stamp` empty, so the mocap
stream is not on the sensor clock and the alignment is biased), and `coverage` should be
close to 1.0.

### The mount solve

With ground truth, the run also solves `M`, the constant body/mount rotation that separates
the start-anchored alignment from the best-fit one, and writes `--out/<drone>_mount.json`.
Solve it once per vehicle and reuse it with `mount.apply()` on later flights: it makes the
**unaligned** trajectory accurate without fitting against ground truth every flight, which
is the point — you cannot fit against ground truth you do not have in the field.

On the nxt3 reference recording it recovers essentially the whole gap: anchored error
0.230 m → 0.048 m, against a best-fit floor of 0.048 m (yaw −3.5°, tilt 3.7°). The `noum_m`
field in `report.json` is the error *before* this correction.

**`M` transfers across flights only if each flight is anchored at the same initial pose.**
This is a hard precondition, not a detail. `M = R_fit · R_anchor⁻¹` with
`R_anchor = R_gt(t₀) · R_est(t₀)⁻¹`, so `M` inherits whatever attitude the vehicle held at
its anchor instant. Take off from the same spot on the same heading and `R_anchor` — and
therefore `M` — is reproduced; start rotated by θ and `M` is wrong by θ. Measured on the
reference flight: the discrepancy between `M` solved at two different anchor instants is
**3.446°**, and the discrepancy between the two `R_anchor` values is **3.446°** — equal to
machine precision. Every bit of `M`'s variation is the anchor; none of it is `M` itself.

So: fix a takeoff pose per vehicle and keep to it, and one solved `M` serves all its later
flights. The residual wobble if you do is VIO yaw drift on the anchor (≤3.3° over 170 s
here), which is why the anchor should be the pre-takeoff static phase — the point of least
accumulated drift.

## 6. Reading the verdict

- `HEALTHY` — calibration self-consistent, in fleet distribution, (ATE healthy if GT given).
  Deploy `<drone>_published_chain.yaml`.
- `FLY-AGAIN` — the recording, not the vehicle, was inadequate (gate details say which
  gate); record another ordinary flight and rerun.
- `HARDWARE-CHANGED` — converged cleanly but off the fleet distribution: the vehicle really
  is different (re-mounted camera, new lens). The calibration is still valid; the flag is
  informational.
- `PLATFORM-DEFECT` — self-calibration converged but a defect independent of calibration
  remains (e.g., a timing fault; the timing gate usually pre-flags this). Repair, don't
  recalibrate.

`fleet_reference/` ships the study fleet's published calibrations so the in-distribution
certificate has a reference population; **replace its contents with your own fleet's
published chains** (same layout: `fleet_reference/<vehicle>/theta_star.yaml`) once you have
≥2 vehicles. With no fleet, run without `--fleet-exclude` (seed from your own files) and
ignore the in-distribution certificate.

## 7. When something fails

`docs/TROUBLESHOOTING.md` lists every failure signature this pipeline has ever produced,
with causes and fixes — read it before debugging. The three most common: estimator produces
no output at all (topic/domain/imu-chain-format mismatch — check `--out/calib/out/run.log`), t_d frozen
at its seed value pass after pass (seed offset beyond ±0.1 s — supply a better t_d), and
near-limit oscillation without convergence (envelope fitted on too few settled passes —
`docs/PROTOCOL.md` §envelope).

## Calibration mode vs flight mode

The estimator is one binary with two operating points, and this package only ever drives the
**calibration** one: `configs/estimator_calib.yaml` with `OV_PRIOR_*` **unset**. Loose stock
priors leave the calibration states mobile so the warm-start iterations can converge away
from the seed.

Flight mode — the tightened priors, `estimator_flight.yaml` and `flight_stiffness.env` —
lives in the `vio/` package and is documented in `../vio/README.md`.

**This matters operationally:** `run_serial.sh` passes your shell environment through, so
flight variables leaking into a calibration run silently pin the calibration at its seed.
The convergence gates will flag the non-convergence, but the root cause is the environment.
Check with `env | grep OV_PRIOR` before a calibration run — it should print nothing.
