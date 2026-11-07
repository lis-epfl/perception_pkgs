# Calibration-Tool Component Tests (2026-07-07)

Implements and validates every §6 component of the paper. All tests on real study data unless
marked synthetic. Thresholds cited here are the ones shipped in the code.

## 1. Static-start gate (`gates.static_start_gate`)
*Rewritten 2026-09-23 to test what the estimator's static initialiser needs; standstill
requirement added 2026-10-06. The 2026-07-07 rule (1.5 s of stillness within the first 12 s) and
its table no longer apply.*

Counted from the instant every stream is live, the gate passes when
(a) a window of `init_window_time` (3 s) that the initialiser would accept ends within the first
5 s — accelerometer excitation below `init_imu_thresh` (1.5 m/s²) in both halves, gyro spread
below 0.2 rad/s — and
(b) the same test keeps passing until 6 s after that window began (`STATIC_MIN_STILL_S`,
`run_tool.py --min-still`). Idling motors do not break it (0.2–1.1 m/s² on the fleet; arming is
1.8–3.0 s before takeoff).

**What the measurements support, and the margin.** Orin NX, released tool, data cut to start N
seconds before the PX4 takeoff flag (2026-10-06). Seven fleet recordings (nxt3, nxt6, nxt10); two
runs where two values are given. Calibration distance = from the full-recording run, in the
stopping rule's tolerance units (median, max; the rule allows 2).

| data starts before takeoff | zero-velocity updates before motion | first pose off, cm | error after mount correction, cm | ATE, cm | calibration distance |
|---|---|---|---|---|---|
| 3 s | 6–8 | 4.9, 5.6 | 5.8, 6.4 | 2.67, 2.68 | 0.52, 0.91 / 0.55, 0.74 |
| 4 s | 35–38 | 3.0 | 4.1 | 2.66 | 0.57, 0.89 |
| 5 s | 60–67 | 2.8 | 4.0 | 2.77 | 0.49, 1.03 |
| 6 s | 91–96 | 2.7 | 3.9 | 2.71 | 0.47, 0.59 |
| 8 s | 144–156 | 2.7 | 3.9 | 2.61 | 0.39, 0.62 |
| full recording (4.1–11.1 s) | 40–244 | 2.7 | 3.8, 3.9 | 2.58, 2.65 | second run: 0.41, 0.73 |

With 3 s the initialiser fires at the instant of takeoff. Every run stays HEALTHY in two passes
and the published calibration is unchanged within run-to-run spread, but the calibration run's
trajectory starts worse, reproducibly per recording. (The error after mount correction pins the
trajectory to its first pose; over 53 runs it follows the first-pose error with a correlation of
0.98.) From 4 s before takeoff nothing differs from the full recording. The standstill the gate
measures reads about 0.45 s longer than the time to the takeoff flag, so the measurements support
4.5 s. **6 s is a margin on top**: the initialiser's window plus the longest arming lead seen on
the fleet (3.0 s), so that the window that starts the filter holds no motor vibration.

Two things the number does not rest on:
- **Flight configuration.** With the calibration fixed and the flight priors, the standstill
  length made no difference on the same recordings: ATE 3.37 / 3.18 / 3.68 cm at 3 / 4 / 6 s
  against 3.29 / 3.46 / 3.46 cm for the full recording (deterministic runs).
- **nxt1.** The vehicle with the camera-timing defect (§2) is erratic at every standstill length.
  On 23 June b its calibration runs gave ATE 7–11 cm at 6 s and more, 8–43 cm at 4–5 s and
  15–17 cm at 3 s; its flight-configuration runs went the other way (34 cm with the full
  standstill, 9 cm at 3 s).

**The gate on the available recordings.**

| recording | standstill, s | gate until 2026-10-05 | gate at 6 s |
|---|---|---|---|
| nxt3 / nxt6, 4 June | 11.6 / 9.2 | PASS | PASS |
| nxt10, 4 June | 4.6 | PASS | **FAIL** — a paper study flight; calibrates normally with `--min-still 4.5` (HEALTHY, two passes, ATE 1.8 and 2.2 cm in two runs) |
| nxt3 / nxt6, 23 June a | 10.3 / 8.6 | PASS | PASS |
| nxt3 / nxt6, 23 June b | 9.3 / 7.5 | PASS | PASS |
| nxt1, 23 June a / b | 5.7 / 7.5 | PASS | **FAIL** / PASS |
| nxt3 / nxt6 / nxt10, 3 June (15 Hz) | 6.8 / 4.7 / 3.0 | PASS | PASS / **FAIL** / **FAIL** |
| same data cut to 3, 4, 5 s before takeoff | 3.4–5.6 | PASS | **FAIL** (all) |
| same data cut to 6, 8 s before takeoff | ≥ 6 | PASS | PASS (all) |

The 3 June nxt10 recording shows what the earlier gate let through: 2.5 s on the ground, and a
start window that ended half a second after takeoff.

## 2. Timing-health gate (`gates.timing_gate`)
Naive inter-frame jitter flags healthy vehicles (dropped frames look like jitter). Shipped
metrics: **drop rate** (missing-frame fraction from drop-aware reindexing; flag > 3 %),
**windowed (10 s) grid-residual std** (short-term stamp noise; flag > 25 ms), and
**cross-camera desync** (shared-trigger rig; flag > 1 ms). Slow clock drift shared with the IMU
is deliberately not flagged.

| vehicle | drop rate | stamp noise std | desync | verdict (expected) |
|---|---|---|---|---|
| nxt3 | 0.25 % | 7.4 ms | 0.000 ms | PASS (pass) |
| nxt6 | 1.48 % | 16.2 ms | 0.000 ms | PASS (pass) |
| nxt10 | 0.19 % | 6.0 ms | 0.000 ms | PASS (pass) |
| nxt1 s2 | **8.07 %** | **37.1 ms** | 0.000 ms | **FLAG** (flag — the §5.5 defect vehicle) |
| nxt1 s3 | **7.62 %** | **34.9 ms** | 0.000 ms | **FLAG** (flag) |

Separation ≥2× on both metrics between the certified-healthy fleet and the known-defective
vehicle.

## 3. Per-camera image-health gate (`gates.image_health_gate` / `circlefit.health_from_acc`)
64 px patches inside 0.92× the fisheye circle; a patch is healthy if its per-frame p99 |Sobel|
reaches 25 at least once in the frames the image pass reads (the run's window, at most 30 s after
takeoff by default); camera flagged if >5 % of in-circle patches stay dead.
- **Specificity:** all 12 fleet cameras (3 bags, full flights): 0 dead patches, no flags.
- **Sensitivity (synthetic smudge):** blur σ=20 + 65 % local-contrast loss inside a 130 px disk
  injected into nxt3 cam2 only → cam2 flagged with 7 dead patches, all inside the disk
  (+1-patch margin); cams 0/1/3 remain 0-dead. A pure σ=14 blur without contrast loss is *not*
  flagged (measured) — the gate detects contrast-killing defects (smudge/oil/occlusion), not
  mild defocus.

## 4. Robust-mean publisher (`publish.py`)
Component-wise median (intrinsics, translations, toff) + chordal/SVD mean (rotations) over
converged harvests; writes a deployable chain yaml (all four timeshift slots = rig toff).
- **Centering:** published toff within 0.001/0.024/0.009 ms of the independently computed
  rig-global population centers (td_center.json); published calibration at residual
  0.0043/0.0084/0.0050 from stored θ* (i.e., at the centroid).
- **Fixed-point (functional):** seeding OV with each published chain, one full pass returns to
  the published state: residual 0.0036/0.0147/0.0050 (τ = 0.30) with |Δtoff| ≤ 0.002 ms —
  self-consistent on all three vehicles.

## 5. Diagnosis (`diagnose.py`) — 3 certificates → §6.3 verdict table
(2026-10: self-consistency is now the paper's between-pass stopping rule, and the
in-distribution check covers every parameter type: f_x and f_y separately, and the mean of
k1..k4 < 0.008 besides k1 -- each fleet vehicle passes against the others, a synthetic 4 %
focal or 0.02 mean-distortion change is caught. The records below predate this.)
Thresholds: self-consistency < 0.10; in-distribution: focal < 2 % vs fleet, k1 < 0.06,
c_x,c_y < 120 px vs own circle fit (healthy fit error is 7–83 px fleet-wide — see note below),
extrinsics < 8°/15 cm vs fleet mount means, toff < 20 ms vs fleet; ATE POSTG < 0.20 m.

| case | inputs | expected | got |
|---|---|---|---|
| nxt3 | 2 independent converged runs + circle fit + selftest ATE (POSTG 0.058 m) | HEALTHY | HEALTHY ✓ |
| nxt6 | same (POSTG 0.031 m) | HEALTHY | HEALTHY ✓ |
| nxt10 | same (POSTG 0.091 m) | HEALTHY | HEALTHY ✓ |
| nxt1 (held-out) | s2+s3 sessions (cross-session resid 0.024), circle fit, s2 ATE POSTG 0.937 m | PLATFORM-DEFECT | PLATFORM-DEFECT ✓ |
| blind seed, 5–6 s flight, 2 passes (toff −30 ms → +75 ms) | pass1+pass2 harvests | FLY-AGAIN | FLY-AGAIN ✓ (sc 0.327) |
| synthetic: published nxt3 chain, c_x +150 px, stable | same chain twice | HARDWARE-CHANGED | HARDWARE-CHANGED ✓ (cc 175 px) |

**Finding fed back to the paper:** the first threshold draft (40 px, from the paper's "7–26 px"
circle-fit claim) false-alarmed on healthy nxt6/nxt10. Root cause: §5.1's 7–26 px was the
nxt3-only range; the true fit-vs-fixed-point accuracy is 7–26 / 10–41 / 25–83 px
(nxt3/nxt10/nxt6), consuming 0.9/1.8/12.3 % of the unit-ball squared budget. Paper §2 and §5.1
corrected; threshold set to 120 px (1.4× worst healthy).

**Re-verification (2026-07-17, circle_verify2.py + paper/figs_ral/circle_fit_verify.png):** the
7–26 / 10–41 / 25–83 px figures above do not reproduce under a controlled re-measurement (same
radius source, same subsample, same threshold, re-anchored θ* references; the earlier era mixed
radius sources and, on nxt3, a silently cut bag). Verified accuracy of the mask fit, leave-one-out
fleet radius, vs θ*: nxt3 1–19 px, nxt6 2–15 px, nxt10 4–9 px (held-out nxt1new 2–14 px), and the
fit is window-insensitive (first 15 s vs full recording agree to ≤2 px per fleet camera) — so
`accumulate()`'s full-recording pass is fine as the seed source; no tool change needed. The largest
errors are radius-transfer artifacts (own-radius worst case 12 px). Threshold 120 px now ≈6× the
worst healthy error (still valid, more conservative than designed).

**2026-10-12:** the tool's pass now reads the window only up to 30 s after takeoff
(`--image-after-takeoff`). On the ten fleet recordings with 90 s of flight (40 cameras) the
centre from 30 s is as far from the calibrated principal point as the one from 90 s (median
7 px, largest 24 px, against the calibrations of 8 October; single centres differ by up to 6 px
between the two), and on all eleven every patch of the image check is sharp within those 30 s.
The 120 px threshold is about 5 times this largest error. On each of the eleven recordings, for
windows of 20, 30 and 90 s after takeoff, the new pass (counting test, one process per camera,
read stopped after the window) returns exactly the centres, image check and timing check of the
pass it replaces (workstation; on the flight computer, ten runs on three vehicles against the
reports of the released tool). The values are those of the release; their keys in report.json
now always come in camera order.

## 6. Mount calibration (`mount.py`) — cross-flight transfer test
Solve M = R_fit·R_anchor⁻¹ on flight A only; apply to flight B with **no fitting on B**
(post-takeoff RMSE, toff-corrected, cam-end windows).

| drone | solve on A (anchored → mounted / own best-fit) | apply to B (anchored → mounted / B's own best-fit) |
|---|---|---|
| nxt3 → nxt3b | 31.4 → 14.7 cm (best-fit 14.7) | 23.7 → **9.2 cm** (own best-fit 9.5) |
| nxt10 → nxt10b | 17.1 → 14.5 cm (best-fit 14.5) | 11.3 → **9.0 cm** (own best-fit 10.4) |

A's constant correction (yaw −3.46°/−1.10°, tilt 3.03°/1.27°) recovers essentially all
recoverable constant error on a flight it never saw — mounted-from-A matches or beats B's own
best-fit rotation. Confirms §6.2: the offset is constant, not drift.

## 7. End-to-end driver (`run_tool.py`)
(Validated before 2026-10-14, when the calibration had neither the matching between cameras nor
the two lens terms, nor since 2026-11-07 the bearing model; `--no-matching --no-lens-terms
--no-bearing` gives that run. Section 8 has the tests of the bearing model.)
gates → one accumulation pass (circle fit + image health) → leave-one-out fleet seed + own
circle fit (scenario D, zero own calibration) → warm-start to self-consistency → robust-mean
publish → diagnosis.

| run | result |
|---|---|
| nxt3, full bag, --fleet-exclude nxt3, GT attached | all gates PASS; converged at pass 2 (resid 0.0064, Δtoff 0.00 ms); published toff −0.039256 (population center −0.039251); ATE POSTG **3.61 cm**; verdict **HEALTHY: deploy** |
| nxt1 s2 (defect path) | static PASS, **timing FLAG** (8.07 % drops / 37.1 ms noise, run proceeds with warning), image PASS; zero-own-calibration seed converged at pass 2; published toff **−0.054584** (independently re-derives the study's triple-confirmed −0.054 from a fleet seed 15 ms away); diagnosis: self-consistent 0.0103, in-distribution, ATE POSTG 0.884 m unhealthy → verdict **PLATFORM-DEFECT — repair, not recalibrate** [+ timing-gate consistency note] |

## 8. Bearing model and fitted fisheye mask (2026-11-07)
The configuration the calibration passes run since 2026-11-07 (README, "Bearing model and fitted
fisheye mask"), tested on the fleet's recordings before it was made the default.

**Estimator (`vio/`), every switch unset.** One calibration pass (nxt3 4 June, 90 s, the
calibration's own settings, `OV_DETERMINISTIC=1`) and a flight-configuration replay (the whole
recording, every line of `flight_stiffness.env`, matching on 5,375 sets of images): trajectory,
calibration and calibration series byte-identical to the estimator of 3a96bc0, on the desktop.
`OV_BEARING_ALL=1` alone: the same. The build adds no compiler warning (desktop, CPU tracker; Orin
NX, GPU path). On the Orin NX, the flight configuration as flown (GPU tracker, `campaign.env`,
`flight_stiffness.env`, the lens terms of a published calibration, `OV_DETERMINISTIC=1`) replayed
over the whole nxt10 4 June recording gives the trajectory and calibration of the estimator of
3a96bc0 byte for byte (1,958 poses; the estimator of 3a96bc0 run twice gives the same bytes too).

**Estimator, the calibration's switches** (`OV_KB_BEARING=1 OV_BEARING_ALL=1 OV_BEARING_RANSAC=1`).
The same pass gives the calibration of the build the rerun of 29 Oct 2026 tested, byte for byte;
the trajectory differs only by the standstill poses that main writes since commit dc8d058. Its
unit tests (projection and its inverse, Jacobians against finite differences, triangulation,
measurement rows with and without first-estimate Jacobians, chi2 of a synthetic update) give the
same output.

**Calibration chains** (the paper's protocol: 6 starting values at r = 0.1 on each of the three
test recordings, 90 s of flight, two passes, the mask fitted at each recording's final
calibration, as tested). Spread over the starting values after pass 2: 0.85 mm / 0.0144 deg
(tested build, same starting values: 0.80 mm / 0.0134 deg; the estimator without the model, two
stored runs: 0.78-0.95 mm / 0.0140-0.0153 deg). Same starting value, this build minus the tested
one: 1.21 mm / 0.0195 deg RMS; two runs of the estimator without the model differ by 1.18 mm /
0.0220 deg. Lens model at 105-110 deg off the axis, spread over the starting values: 0.43-0.54 px
(tested: 0.59-0.77 px; without the model: 3.07-4.10 px). Largest step from pass 1 to pass 2:
0.12-0.29 envelopes (the stopping rule's limit is 2).

**Fitted mask** (`wmask.py`). On the eleven fleet recordings, at the same calibration, the fit on
the tool's own mean images (30 s of flight) gives the disk of the fit on 300 pictures over the
whole recording to within 0.25 px of radius (cameras 0-2) and 0.28 px (camera 3), with the rule's
first implementation giving the same end of the circle on all 44 cameras. Around the principal
points of the seed (pass 1) the disk is 23 px narrower to 4 px wider; around those of a pass-1
calibration (every later pass) 0.4 px narrower to 1.2 px wider, and in pass 2 of the tool's runs
below (eleven recordings on the desktop, three on the flight computer) 0.7 px narrower to 1.7 px
wider. The fit takes 0.6 s per pass on the flight computer.

**End to end** (leave-one-out fleet seed, ground truth from the bag, desktop unless noted).

| run | result |
|---|---|
| `--no-bearing`, nxt3 4 June, 90 s, `OV_DETERMINISTIC=1` | published chain, lens terms, both harvests and the mount byte-identical to the tool and estimator of 3a96bc0; `report.json` only adds `calib_passes.bearing_model` and `w_mask` (null) |
| default, the eleven fleet recordings, 90 s | 11 of 11 **HEALTHY: deploy** at pass 2 (largest step 0.13-0.62 envelopes); the model ran in every pass; tool ATE 2.47-4.16 cm (mean 3.09; the tested configuration 3.17, without the model 3.30). Against the tested configuration's runs (the mask fitted at each recording's final calibration), median over the recordings of the largest projection difference: 0.36 px below 60 deg, 0.53 px at 100-105 deg, 0.83 px at 105-110 deg; two runs without the model differ by 0.29 / 2.66 / 4.24 px, the tested configuration from a run without the model by 0.49 / 6.95 / 12.05 px |
| default, the three test recordings, 20 s (the default flight time) | 3 of 3 HEALTHY at pass 2 (largest step 0.73-1.16 envelopes; without the model 0.80-1.55), tool ATE 1.16-4.04 cm (without the model 1.22-2.60; single runs) |
| default + `--frozen-check`, nxt10 4 June, 20 s, `OV_KB_BEARING=1 OV_BEARING_ALL=1` exported in the shell | calibration passes ran the model; the flight check did not (no `[bearing-cfg]` line, mask 1.83 / 1.87, matching on 711 sets); HEALTHY |
| default with a mask fit that raises (test copy of `wmask.py`), nxt3 4 June, 20 s | WARNING, the config's mask in both passes, HEALTHY at pass 2 |
| default on the Orin NX (GPU 612 MHz), the three test recordings, 90 s, two runs each, alternating with the tool and estimator of 3a96bc0 (one run at a time) | 6 of 6 **HEALTHY: deploy** at pass 2 (3a96bc0: 6 of 6); the model ran in every pass. An estimator pass 64.6-68.4 s against 62.0-64.5 s (paired: +4.6 % median, +2.0 to +4.2 s, of which 0.5 s is the mask fit), a whole run 151.6-157.8 s against 144.1-148.9 s (+3.6 % median). Two runs of the same recording differ by 0.28-0.50 px below 60 deg and 0.71-1.30 px at 105-110 deg (3a96bc0: 0.34-0.43 / 5.48-8.23 px); the runs differ from the desktop's run by 0.31-0.47 / 0.80-1.59 px |

## 9. Bearing model in flight (2026-11-07)
Since 2026-11-07 flight runs the bearing model as well (`vio/vio_deploy/config/flight_stiffness.env`,
point 5), with the fisheye mask of the pass the calibration published, which `deploy_vio.py` writes
into the flight folder (README, "Bearing model and fitted fisheye mask"). The real-time test on the
Orin NX that decided it is in `../../vio/README.md`, "Bearing measurement model": the same tracking
error (3.69 against 3.73 cm over 11 flights x 3 runs per model, no run above 6 cm in either), about
10 % more time per set of images. The `--frozen-check` row of section 8 describes the tool before
this change. Tests of the change (workstation; the estimator's code is that of 5301181, unchanged):

**Deployment** (no estimator; the three 90 s calibrations the real-time test flew, nxt3 4 June,
nxt6 23 June, nxt10 4 June; `deploy_vio.py` of 5301181 as the reference; 216 checks, all pass).

| case | result |
|---|---|
| default | the fitted mask of the published pass in `estimator_flight.yaml` (nxt3 1.808444 / 1.818816, nxt6 1.817016 / 1.819301, nxt10 1.800899 / 1.83665); with each recording's topic layout it is the real-time test's config of the bearing model for all 11 recordings, byte for byte, and the sourced `flight_stiffness.env` is that test's environment (campaign.env and the real-time replay switches aside); every other file as deployed before |
| `OV_KB_BEARING=0` exported, `--no-bearing`, or both with `OV_KB_BEARING=1` | the folder of 5301181 file for file (`estimator_flight.yaml`, chains, mount, lens terms, `campaign.env`); the sourced flight settings differ only by `OV_KB_BEARING=0`; it is the real-time test's config and environment of the original model |
| `report.json` without `w_mask` (a tool before 2026-11-07), or `w_mask` null (`--no-bearing` calibration) | the original model, the config's mask |
| the published pass kept the config's mask; a flight config without the automatic disk, or without the camera-3 key | the bearing model with the config's own mask, WARNING |
| a flight disk narrower than the fitted one | the fitted mask lowered to it, noted |
| `w_mask` record damaged (not an angle, a key that is not a mask key, no record of the published pass) | refused |
| a vio package whose `flight_stiffness.env` does not set the model | the original model, no fitted mask |
| deployed again into the same folder: model, original, model | each time the folder of that deployment |

**Replays** (CPU tracker, `OV_DETERMINISTIC=1`, `OV_GROUP_CAMS=1`, 2 threads, whole recordings; each
recording with its own calibration):

| | nxt3 4 June | nxt6 23 June | nxt10 4 June |
|---|---|---|---|
| default folder = the real-time test's config and environment of the bearing model | identical | identical | identical |
| override folder = folder of 5301181 = the real-time test's original model | identical | identical | identical |
| default folder sourced, then the override folder, in one shell = folder of 5301181 | identical | | |
| `OV_KB_BEARING=0` exported over the default folder (the fitted mask stays) | differs from the original (as documented) | | |

**Tool** (`--frozen-check`, nxt3 4 June, 20 s, `OV_DETERMINISTIC=1`, against the tool of 5301181): all
four runs HEALTHY at pass 2. Default: published chain, lens terms and both harvests byte-identical;
the flight check flies the bearing model with the mask of the published pass (1.807651 / 1.815172),
and its trajectory and calibration equal those of the folder `deploy_vio.py` writes, replayed over
the same window. The ATE certificate and the mount, which the tool computes on the flight check,
follow it (POSTG 2.09 against 1.65 cm, mount yaw -3.73 against -3.54 deg). `--no-bearing`:
everything byte-identical to the tool of 5301181, the flight check included (the original model,
which the deployed folder of that calibration flies as well).
