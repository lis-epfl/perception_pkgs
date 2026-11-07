# Troubleshooting — every failure mode this pipeline has produced, with signatures

Ordered by how often each actually happened across ~10,000 chains on 7 platforms/recordings.

## 1. Estimator produces no output at all (`no_init`, missing `estimate_tum.txt.calib.json`)

Check `out/run.log` first; the cause is almost always one of:

| signature in run.log | cause | fix |
|---|---|---|
| `the node Ta ... was not found` (also Tw/Tg/R_IMUto*) | IMU chain missing the IMU-intrinsics blocks | copy `fleet_reference/nxt3/kalibr_imu_chain.yaml`, edit noise values only |
| `port number is too high` / `domainId is over` | DDS domain > 232 | keep `--domain` in 0–232; parallel runs need distinct domains |
| runs but zero features / no messages | bag topics ≠ chain `rostopic` fields | topics must match EXACTLY (`/cam0/image_raw`, `/imu0`) |
| initializes never (waits forever in static) | `init_imu_thresh` too high for a gentle platform | lower it (0.3); or recording lacks a takeoff jerk |
| initializes then instantly diverges | `init_imu_thresh` too low (initialized on noise mid-static) | raise it; verify the recording's static start |
| yaml parse error | multiline lists (kalibr's dumper wraps) or `T_imu_cam` key | single-line lists; convert T_imu_cam → T_cam_imu by inversion |

## 2. t_d frozen: `t̂_d` identical to the seed value, digit-for-digit, pass after pass

The seed's time offset is beyond the recovery cliff (≈±0.1 s on well-excited platforms;
narrower under joint error). This is physics, not a bug: beyond the visual–inertial
correlation window the update has no gradient in t_d. Supply a better t_d (hardware
timestamps, sibling vehicle). Do NOT add passes — a frozen t_d never moves.

## 3. Near-limit oscillation: residuals hover at 1–3× the convergence limit, wandering, never entering

Signature: non-monotone worst-residual around 1–2× for many passes; across many seeds the
failure rate is INDEPENDENT of seed distance. This is a criterion artifact, not divergence:
the convergence envelope was fitted on too few settled passes and sits inside the
oscillation's own tail. Fix: re-derive the envelope from ≥16 settled passes (see
docs/PROTOCOL.md §envelope). Contrast with genuine divergence: residuals 10–300× the limit,
or frozen identical digits (that's mode 2).

## 4. Premature "flat" verdicts on noisy platforms

If a convergence loop stops chains whose residual descent stalls, noisy platforms
(motion blur, cheap IMU) descend non-monotonically and get killed while still converging.
Any flatness early-stop must only fire when the residual is clearly outside (>3× the limit);
chains hovering below that must get their full pass budget.

## 5. Silent warm-start no-ops

Two ways the loop can silently re-run the same seed forever:
- Chain file uses `T_imu_cam` while the writer expects `T_cam_imu`: extrinsics reset to the
  template every pass (intrinsics still update — partial progress masks the bug). Verify by
  diffing the chain file between passes: extrinsics must change.
- t_d in a cams-only loop: the offset is re-seeded each pass by design; if you are testing
  t_d recovery specifically, the loop must carry `t̂_d` forward explicitly.

## 6. Cross-talk between parallel runs

Two estimator processes on the same DDS domain read each other's topics and corrupt both
runs. `run_serial.sh` pins `ROS_LOCALHOST_ONLY=1`; you must still give every concurrent run
a distinct `DOMAIN` (last argument). Disjoint domain RANGES per campaign is the safe pattern.
Stale/zombie ROS 2 processes poison whole domains — if runs behave impossibly, check for
orphaned processes before anything else.

## 7. Operational traps (cost us real hours)

- `nohup python3 x.py > some/new/dir/log` fails silently if the directory doesn't exist —
  the redirect target is created by the SHELL before python runs. `mkdir -p` first.
- `pkill -f <pattern>` / `pgrep -f <pattern>` match the invoking command line itself when it
  contains the pattern (including in OTHER arguments of the same compound command, e.g. a
  `sed` on the same filename). Kill in an isolated command with a split-string pattern:
  `kill $(pgrep -f "prefix_"suffix)`.
- Truncated-window runs must be judged against that window's OWN envelope, never the
  full-recording envelope: short windows cannot settle as tightly, and the full-recording
  box rejects fully-settled short-window chains.
- Timestamps: converters must emit one shared clock for images and IMU; check the first IMU
  sample reads ≈ pure gravity if the recording claims a static start.

## 8. Interpreting borderline diagnosis outcomes

- Converged + off-distribution + you DID change hardware: expected (`HARDWARE-CHANGED`).
- Converged + timing gate flagged: trust the gate (`PLATFORM-DEFECT` — fix timestamps; the
  calibration itself is usually fine but the vehicle will fly badly regardless).
- Gates pass + convergence oscillates between two states across seeds: both are valid
  fixed-point-adjacent states (multi-basin structure); the robust-mean publish handles it —
  report the published mean, never a single harvest.

## 9. Bearing model (calibration passes and flight, on by default since 2026-11-07)

- `WARNING: the bearing model was asked for, but the estimator did not report it` — the
  estimator was built before the commit that added the model and ignored `OV_KB_BEARING`:
  rebuild it (`vio/vio_deploy/scripts/build_openvins.sh`). The calibration is then the one
  without the model, with the fitted mask; `report.json` `calib_passes.bearing_model_ran` is
  false, and `deploy_vio.py` and `--frozen-check` fly it with the original model (a WARNING).
- `fisheye mask of the config kept: ...` — the tool found no image circle to fit the mask to
  (another lens, whose circle does not end 380-500 px from the principal point, or the circle
  outside the sensor), the config does not draw the disk itself (`use_mask` /
  `mask_fisheye_auto` false), a camera had no mean image, or the fit itself failed (the reason
  then names the error). The passes then run the bearing model with the config's own mask.
  `report.json` `w_mask.passes[k].reason` says which. When the published pass is one of them,
  the calibration is flown with the original model (`deploy_vio.py` and `--frozen-check`, a
  WARNING): the bearing model flies only with a fitted mask.
- The estimator ends at start-up with `[bearing] OV_KB_BEARING=1 does not support OV_PREJAC=1`:
  unset `OV_PREJAC` for calibration, or pass `--no-bearing`; in flight, unset it or deploy the
  original model.
- To calibrate as before 2026-11-07: `--no-bearing` (or export `OV_KB_BEARING=0`).
- `WARNING: OV_KB_BEARING=0 in the environment together with flight priors` — the shell sourced
  the `flight_stiffness.env` of a folder deployed for the original model. The calibration then
  runs without the bearing model; unset `OV_KB_BEARING` (and the flight settings) for the default.
- deploy_vio.py: `measurement model: the ORIGINAL, without the bearing model (...)` — expected
  with `--no-bearing` or `OV_KB_BEARING` other than 1 in the shell; otherwise the calibration
  ran without the model (`report.json` `w_mask` null, or no `w_mask` at all: a calibration by the
  tool before 2026-11-07), or the vio package's `flight_stiffness.env` does not switch it on (a
  package older than the tool). Recalibrate with the tool's default, or update the package, to
  fly the bearing model.
- deploy_vio.py: `WARNING: measurement model: the ORIGINAL, without the bearing model (...)` —
  the calibration ran the bearing model, but its fitted mask cannot fly as it is, and the model
  flies only with it (the combination measured in flight): the estimator of the calibration did
  not run the model (the first item of this section), the pass it published kept the config's
  mask (the item about `fisheye mask of the config kept`), or the flight config cannot take the
  fitted mask unchanged (it does not draw the disk itself, keys its cameras under other keys than
  the calibration config, or its disk is narrower than the fitted one). The folder flies the
  original model with the config's mask, the flight of before 2026-11-07; the end of the WARNING
  and the folder's MANIFEST say what would let the calibration fly the bearing model.
  `--frozen-check` makes the same choice (`WARNING: deployment check: the ORIGINAL model ...`,
  `report.json` `deploy_check.bearing_model_off`).
- deploy_vio.py: `ERROR: report.json w_mask ...` — the record of the fitted mask is damaged (not
  a mask angle, or no record of the published pass): run the calibration again into an empty
  `--out`, or deploy the original model (`--no-bearing`).
- A flight log without `[bearing-cfg] on=1` although the folder was deployed with the model: the
  shell did not source the folder's `flight_stiffness.env`, something exported `OV_KB_BEARING`
  other than 1 after it, or the estimator was built before commit 250f9f2 (it then flies the
  original model with the fitted mask). `env | grep BEARING` and the log's
  `auto fisheye mask` lines tell which.
- A flight log WITH `[bearing-cfg] on=1` and the config's mask (`auto fisheye mask` lines at
  1.83 / 1.87): the bearing model with a mask it was never measured with. The folder was deployed
  before 2026-11-07 and the shell kept the switches of a folder of the bearing model sourced
  before it: such a folder does not set `OV_KB_BEARING` itself (and the MANIFEST of one deployed
  before 2026-10-09 sources the vio package's `flight_stiffness.env`, which now switches the model
  on). Fly it from a fresh shell, or deploy it again: every folder deployed since sets
  `OV_KB_BEARING` itself.
