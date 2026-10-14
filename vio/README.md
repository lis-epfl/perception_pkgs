# vio — the VIO estimator package

A pruned OpenVINS fork plus the flight/deployment configuration. This is the **only**
estimator in the repository: `selfcalib/` calls it rather than bundling its own, so
calibration and flight always run the same binary.

## Contents — four ament packages, side by side

```
ov_core/      ov_init/      ov_msckf/      pruned OpenVINS
vio_deploy/                                deployment assets (this package's own)
  config/estimator_flight.yaml               FLIGHT config
  config/flight_stiffness.env                flight settings: x0.10 calibration-prior stiffness,
                                             lens terms as states, matching between cameras,
                                             limits for slow flight
  config/campaign.env                        the vehicle's GPU tracker and scheduler switches
  scripts/run_serial.sh                      the ONLY supported estimator invocation
  scripts/build_openvins.sh                  builds this workspace
config/                                    placeholder; ov_msckf's cmake installs ../config/
BASELINE.txt  LICENSE  PRUNED.txt
```

They must stay **siblings**: colcon does not descend into a directory that already has a
`package.xml`, so nesting them under a parent package would hide them from the build.

## Build

```bash
bash vio_deploy/scripts/build_openvins.sh        # ~6 min → ~/ov_ws_vio
# or: OV_WS=/custom/path bash vio_deploy/scripts/build_openvins.sh
```

The programs to know:

| binary | use |
|---|---|
| `run_serial_msckf` | deterministic offline replay — what `selfcalib` drives, and what flight-mode replay uses |
| `run_online_msckf` | the live node of the vehicle: it reads the PX4 IMU and the compressed images and hands the estimator the four images of an instant as one set |
| `run_subscribe_msckf` | the stock subscriber (`sensor_msgs` topics, one camera per message). It cannot read the vehicle's topics and does not match between cameras: not the flight node |

Requires Ubuntu 22.04 + **ROS 2 Humble**, OpenCV ≥ 4.5, Eigen3, Boost, Ceres
(`libeigen3-dev libboost-all-dev libceres-dev`). ROS 1 is not supported; the CMake files
fail with a clear message rather than silently falling back.

## Two operating points, one binary

The difference is the config plus the variables of `flight_stiffness.env`, read once at
startup. The first five `OV_PRIOR_*` variables set the initial standard deviations of the
online-calibration priors (patched `State.cpp`); unset, the stock values apply. The sixth, when
set, makes the two lens terms per camera calibration states with that standard deviation.

| env var | state | stock (unset) | flight (`flight_stiffness.env`) |
|---|---|---|---|
| `OV_PRIOR_DT_SIG` | cam-IMU time offset (s) | 0.01 | 0.001 |
| `OV_PRIOR_EXTR_Q_SIG` | cam-IMU rotation (rad) | 0.005 | 0.0005 |
| `OV_PRIOR_EXTR_T_SIG` | cam-IMU translation (m) | 0.015 | 0.0015 |
| `OV_PRIOR_INTR_F_SIG` | intrinsics fx/fy/cx/cy (px) | 1.0 | 0.1 |
| `OV_PRIOR_INTR_D_SIG` | distortion coefficients | 0.005 | 0.0005 |
| `OV_PRIOR_NONRAD_SIG` | two lens terms per camera | not states (a calibration by `selfcalib` sets 0.002) | 0.0002 |

The same file switches on what flight has had since 2026-10-09 (next section): the lens terms
as states (the sixth row, with `OV_NONRAD_SKEW_FIXED=1`) and the matching of features between
neighbouring cameras (`OV_XCAM=1`, `OV_XCAM_PRESET=rt`). Since 2026-10-14 it also sets four limits
for slow flight (the section after that).

**Flight** — `config/estimator_flight.yaml` **with** `config/flight_stiffness.env` sourced.
The tightened priors tether the online calibration to its published chain. With the five
priors of the first rows, the flight settings until 2026-10-08, this pair (uniform, no
per-vehicle tuning) produced 2.4–4.5 cm SE(3) ATE across an 11-flight, 3-vehicle benchmark.

**Calibration** — `selfcalib/configs/estimator_calib.yaml` with the flight priors of
`flight_stiffness.env` **unset**. Loose priors keep the calibration states mobile so the
warm-start iterations can converge away from the seed. Since 2026-10-14 `selfcalib` sets the
matching and the lens terms itself, at calibration values: `OV_XCAM=1`, `OV_XCAM_PRESET=rt`,
`OV_NONRAD_SKEW_FIXED=1`, `OV_PRIOR_NONRAD_SIG=0.002` (ten times the flight value) and, on a
desktop, `OV_GROUP_CAMS=1`; `--no-matching` and `--no-lens-terms` switch them off
(`../selfcalib/README.md`).

The per-vehicle flight folder is produced by the self-calibration package, not written by
hand — `selfcalib/tool/deploy_vio.py` copies this config next to the calibrated chains under
the filenames the estimator resolves (`kalibr_imucam_chain.yaml`, `kalibr_imu_chain.yaml`),
adds a copy of `flight_stiffness.env` that ends with the lens terms of that calibration, and
verifies the result. See `../WORKFLOW.md`.

```bash
# the flight settings, from a folder produced by deploy_vio.py (its copy also carries the
# lens terms of that calibration); on the vehicle also the tracker and scheduler switches
set -a; source flight_myvehicle/flight_stiffness.env; set +a
[ "$(uname -m)" = aarch64 ] && { set -a; source flight_myvehicle/campaign.env; set +a; }

# flight replay (on a desktop first: export OV_GROUP_CAMS=1, which hands the estimator the
# four images of an instant as one set, as on the vehicle)
bash vio_deploy/scripts/run_serial.sh BAG flight_myvehicle/estimator_flight.yaml OUT 4 false 42 DOMAIN

# live on the vehicle
ros2 run ov_msckf run_online_msckf flight_myvehicle/estimator_flight.yaml
```

Both programs read the topics from `estimator_flight.yaml`, which names none as deployed:
they then read `/imu0` (`sensor_msgs/Imu`) and `/cam<N>/image_raw` (`sensor_msgs/Image`). For
any other vehicle, the fleet's included, first add `imu_topic`, `imu_msg_type`, `cam_msg_type`
and `cam_topic0..N` to that file; the folder's `MANIFEST.txt` lists the ones of the recording
the calibration was made on. Without them the live node starts and receives nothing, and a
replay writes no trajectory. `selfcalib` adds them for its own runs.

A folder deployed before 2026-10-09 has no `flight_stiffness.env` of its own: deploy it again
(`deploy_vio.py` on the same calibration output), or follow that folder's own MANIFEST.

Rule of thumb: **calibrating → flight settings unset; flying → source them.**
`run_serial.sh` passes your shell environment through, so check
`env | grep -E 'OV_PRIOR|OV_XCAM|OV_NONRAD'` if a run behaves oddly — flight priors leaking
into a calibration run pin the calibration at its seed. `run_tool.py` takes the flight
priors out of its own calibration passes, puts the lens-term prior back to the calibration
value, and says so; `run_serial.sh` used directly does not.

`run_serial.sh` exit codes: `64` usage/bad DOMAIN · `66` missing bag or config · `69` missing
ROS or workspace · `70` ran but produced no trajectory · otherwise the estimator's own code.

## Matching and lens terms in flight

Since 2026-10-09 `flight_stiffness.env` switches on two additions of the estimator in flight.
Until then only a calibration that asked for them used them (next section).

| lines of `flight_stiffness.env` | what flight does |
|---|---|
| `OV_XCAM=1`, `OV_XCAM_PRESET=rt` | A feature tracked in one camera is searched in each neighbouring camera and tracked there under the same id. At most 30 features are searched per set of images, and each match is checked by the search back. |
| `OV_PRIOR_NONRAD_SIG=0.0002`, `OV_NONRAD_SKEW_FIXED=1` | The two lens terms per camera stay calibration states, with a prior ten times narrower than in a calibration. The third term is held. |

The lens terms start at the values of the calibration (`OV_NONRADIAL`, the last line of the
flight folder's copy of `flight_stiffness.env`). They start at zero when the calibration has
none: one made with `--no-lens-terms`, or by `selfcalib` before 2026-10-14.

The matching runs where the four images of an instant reach the tracker as one set:
`run_online_msckf`, and a replay with `OV_GROUP_CAMS=1` (`campaign.env` sets it on the
vehicle; on a desktop export it yourself). Where the images arrive one camera at a time the
estimator prints one `[xcam-warn]` line and its output is byte-identical to a run without
`OV_XCAM`. (The same line can appear once in a healthy flight, for an incomplete set of
images.) The log of a flight shows both additions: `[nonradial]: camN p1, p2, skew are
CALIBRATION STATES, start ..., prior sigma 2.000e-04` at the start, and `[xcam-total]
framesets=N ... matched=M ... runs=R` at the end, with R close to the number of image sets.
(Before the first of these the log prints one line per camera that ends with `(fixed, from
OV_NONRADIAL)` or `(all zero: unmodified fisheye code path)`. It is printed when the cameras
are loaded and does not mean that the terms are fixed: the `CALIBRATION STATES` lines decide.)

Measured on the Orin NX (estimator of main cab3e44, GPU clock at the fleet's 612 MHz). Every
flight is flown with a calibration made on another flight of the same vehicle. Tracking error
is the RMS position error after a rigid fit to motion capture: from takeoff on for the live
runs, and from 3 s after the first pose, which comes at takeoff, for the replays. The four
optical centres are the corners of a square (side 149.6 mm on the fleet); "longest minus
shortest side" is taken from the camera positions the estimator holds as calibration states
at the end of the flight, 0 mm being a perfect square.

| | flight settings until 2026-10-08 | with both on |
|---|---|---|
| replay, tracking error, calibrations with lens terms, 6 flights | 2.8–4.2 cm, median 3.8 | 2.4–5.3 cm, median 3.6 |
| replay, tracking error, calibrations without lens terms (the tool's default until 2026-10-13), 3 flights | 4.3, 3.0, 3.0 cm | 2.9, 2.8, 3.9 cm |
| replay, longest minus shortest side, calibrations with lens terms | 3.2–6.0 mm, median 4.1 | 0.6–5.3 mm, median 2.0 |
| replay, longest minus shortest side, calibrations without lens terms | 31, 15, 35 mm | 5.2, 2.0, 6.4 mm |
| live node, tracking error, calibrations with lens terms, 3 flights | 3.3, 2.9, 3.8 cm | 4.2, 2.7, 4.5 cm |
| live node, tracking error, calibrations without lens terms, the same 3 flights | 3.0, 2.6, 3.4 cm | 5.7, 2.6, 3.3 cm |
| live node, the first of these runs once more | 2.6 cm | 2.9 cm |
| live node, time of the feed thread per set of four images (decoding, tracking, matching; the filter update runs on a second thread): mean (99th percentile) | 14.0–14.2 ms (20–25) | 16.7–17.4 ms (28–31) |

A replay is a deterministic run over the whole flight. A live run is the live node on the
recording played back in real time: one run, which does not repeat exactly. Rows with three
values list the same three flights in the same order.

In the replays tracking does not change beyond the scatter between flights: per flight, both
on minus the earlier settings, it changes by −1.5 to +1.0 cm, with a median of −0.03 cm over
the six flights and −0.2 cm over the three with calibrations without lens terms.

In the live node the matching costs accuracy. Each addition was also run alone on the three
flights with lens-term calibrations: the lens terms as states alone gave 4.0, 2.7, 3.7 cm
(3.3, 2.9, 3.8 cm with the earlier settings) and add no time; the matching alone gave 2.6,
3.8, 6.7 cm. Over all live runs, the ten with the matching (alone or with the lens terms)
average 3.9 cm and two of them reach 5.7 and 6.7 cm; the ten without it average 3.2 cm and
the largest is 4.0 cm. The large errors are single runs: the one at 5.7 cm, made once more,
gave 2.9 cm. The replays do not show this difference, and its cause is not known. To keep the lens
terms as states and fly without the matching, `unset OV_XCAM OV_XCAM_PRESET` after sourcing
the file.

What changes for the better is the camera geometry that the estimator carries during the
flight. With a calibration that has lens terms the sides of the square are closer to equal at
the end of five of the six flights. With a calibration without lens terms, whose square has sides 15–35 mm
apart, they come to within 2–6 mm during the flight.

A set of images costs the feed thread about 3 ms more, all of it from the matching, whose own
step takes 1.6–1.9 ms (99th percentile 4.3–4.9 ms). The pose of a filter step leaves the node
3–5 ms later: its median delay after the images are complete goes from 49, 79, 63 ms to 54,
82, 68 ms. No set of images was lost in any of the twenty live runs.

Each addition alone, in the replays of three of the flights (same order): tracking 2.9, 3.8,
4.0 cm with the matching only and 2.4, 3.0, 4.5 cm with the lens terms as states only, against
2.8, 3.9, 4.2 cm with neither and 3.6, 2.4, 5.3 cm with both. Neither addition moves tracking
in one direction. On the first two of these flights the sides of the square come closer,
mainly through the matching (longest minus shortest side 3.8 and 6.0 mm with neither, 1.7 and
1.1 mm with the matching only, 2.9 and 4.2 mm with the lens terms only, 1.5 and 0.6 mm with
both). On the third they move apart with either addition (3.4 mm with neither; 5.9, 4.7 and
5.3 mm).

To fly as before 2026-10-09, unset the four variables after sourcing the file
(`unset OV_XCAM OV_XCAM_PRESET OV_PRIOR_NONRAD_SIG OV_NONRAD_SKEW_FIXED`); the lens terms of
`OV_NONRADIAL` are then fixed values again. An estimator built before commit 6bbaf8c has
neither addition and ignores the four variables.

## Slow flight

Since 2026-10-14 `flight_stiffness.env` sets four more variables. They keep the estimator on
time while the vehicle hovers or flies slowly, which the settings before did not.

The filter makes a clone every 0.133 s and keeps 30 of them (4 s). An MSCKF update uses a
feature when it is lost or when it reaches the end of that window. In cruise flight features
are lost after 1-2 s and an update has 400-450 measurement rows. In a hover nearly every
feature lives through the whole window, and all of them reach its end together with the
features born with them: one update, the sub-update after the lead of a clone tick, then takes
2500-6300 rows, 57 per feature. On the Orin NX that costs 135-250 ms against 33 ms between two
sets of images (7.5 ms per 1000 rows to build the system, 15.9 ms per 1000 rows to compress
it, about 17 ms for the EKF update and 13 ms for the one landmark it initialises). In two
shadow flights of nxt3 with every set of images complete (13 Oct 2026) the poses left the node
up to 400 and 533 ms after their images, from 4 s after the takeoff, when the window had
filled, and again in the slow descent, against 53-63 ms in cruise flight.

| line of `flight_stiffness.env` | what it does |
|---|---|
| `OV_MSCKF_MAX_OBS=12` | Of one feature, at most 12 observations per camera go into an MSCKF update, spread evenly over its track, the first and the last always among them. A track through the whole window brings 21 rows instead of 57, as much as a track of cruise flight. |
| `OV_MSCKF_ROWS=900` | A clone tick takes about 900 rows of window-end features at most, and an update about 900 together with its lost features. A window-end feature that does not fit is alive: it waits and is offered again at the next ticks, each time with one old observation less and one new one more. Lost features are always taken. Needs the line above: a hover brings about 650 rows per tick with it and 1770 without. |
| `OV_SPREAD_ONE_GATE=1` | One update gate for all cameras instead of one per camera. With one per camera, a set of images that lacks a camera at a clone tick leaves that camera ticking at instants of its own for the rest of the flight: 10.7 ticks per second instead of 7.0 on a recording with 6 % incomplete sets, 14.6 in a live flight with 23 %. |
| `OV_ZUPT_MAX_DZ=0.3` | No zero-velocity update more than 0.3 m above or below the height of the last accepted one. The update is accepted on image disparity alone (`zupt_max_disparity`), which a vehicle that hovers passes below about 0.1 m/s: in each of the six flights below it was accepted at 1.5 m height, on 1 to 28 sets of images. On the ground, before the takeoff and again after the landing, nothing changes. |

Unset, each of them is off and the estimator's output is byte-identical to that of the build
without them (deterministic replay of a flight on a desktop).

Measured in replays on a desktop with the vehicle's GPU tracker (`run_serial_msckf` with the
environment of the flight folder), on six flights of nxt3 of 13 Oct 2026: 21-47 s of flight
with a 3 s hover after the takeoff and a slow landing; two of the recordings have 3 % and 6 %
incomplete sets of images, one ends in an upset. Three to six runs per flight and setting.
Tracking error is the RMS position error after a rigid fit to motion capture, from takeoff on.
"Pinned" is the error of the trajectory held at its pose at arming and turned by the mount,
which is what a vehicle that flies on the estimator sees.

| | before | with the four lines |
|---|---|---|
| rows per clone tick, mean of the six flights | 779 | 495 |
| largest update of a flight | 4800-10500 rows | 880-2650 rows (above 900: lost features, in the upset and on the first fast leg) |
| tracking error, per flight | 2.9, 3.0, 3.0, 3.4, 3.6, 5.2 cm | 2.6, 2.7, 2.6, 3.3, 4.1, 5.2 cm |
| tracking error, mean of the six | 3.54 cm | 3.41 cm |
| the same flight run three times: range of the tracking error | 1.1 cm on average, up to 3.4 | 0.5 cm on average, up to 1.0 |
| pinned error, mean of the six: RMS / largest | 4.95 / 13.3 cm | 4.59 / 11.1 cm |
| zero-velocity updates between takeoff and landing | 1 to 28 sets of images per flight | none |

(Flights in the order 20:58, 20:31, 17:30, 15:43, 19:56, 16:46. One flight, 19:56, is 0.45 cm
worse with the four lines, inside the range of its own runs before.)

On the Orin NX of nxt3, the vehicle these flights were flown with, the replay runner in real
time (`OV_REALTIME=1`) on three of them, 20:31, 20:58 and 19:56, before and with the four
lines: the largest sub-update 160-275 ms before and 81-87 ms with them; sub-updates over
100 ms 2-3 per flight before, none with them; sets of images handed to the filter more than
100 ms late 0.4-0.7 % of those in motion before, none with them, the latest 73-77 ms. The
runner holds sub-updates back while sets of images wait, which the live node does not, so its
delays before are smaller than those the flights measured; the durations of the sub-updates
are what compares. One run of 19:56 with the four lines, made while the Orin was busy with the
system's update check after a boot, fell behind for 1.5 s, up to 403 ms; the same run on the
idle Orin did not. No flight hovers longer than 3 s, and none was flown with the four lines.

## Calibration switches

Three additions, read from the environment. All are off when unset, and the estimator's
output is then byte-identical to the build without them (checked on a desktop and on the
Orin NX). Flight sets the first three rows (section above); since 2026-10-14 a calibration by
`selfcalib` sets them too, unless switched off (`--no-matching`, `--no-lens-terms`).

| env var | what it does | tested value in a calibration |
|---|---|---|
| `OV_XCAM` | A feature found in one camera is searched in each neighbouring camera and tracked there under the same id, so the filter's ordinary multi-camera update ties the cameras together. Needs `OV_GROUP_CAMS=1` and `use_stereo` false. `OV_XCAM_PRESET=rt` selects the cheaper settings: about 2 ms per set of four images on the Orin NX. | `1` |
| `OV_PRIOR_NONRAD_SIG` | Two more lens terms per camera become calibration states with this initial standard deviation (Kannala-Brandt cameras only). `OV_NONRAD_SKEW_FIXED=1` holds the third term, which flight data cannot determine. | `0.002` |
| `OV_NONRADIAL` | The lens terms, `p1,p2,s` per camera joined by `;` (quote it): fixed values while `OV_PRIOR_NONRAD_SIG` is unset, otherwise the values the states start from. The tool hands them from pass to pass and publishes them; set it wherever a calibration made with them is used. | from the tool |
| `OV_RIG_DIST`, `OV_RIG_PLANAR`, `OV_RIG_SIG_MM`, ... | The known shape of the rig as a prior on the camera positions (`ov_msckf/src/update/UpdaterRigShape.h`). | see below |

Matching and lens terms do different jobs and both are needed: the lens terms put the cameras
in the right place, the matching fixes the rotations between them. They need about 50 s of
flight. A calibration made with lens terms must be flown with `OV_NONRADIAL` set, which the
flight folder's `flight_stiffness.env` does.

The rig prior is for calibration windows that must stay at 20-30 s, where the pictures cannot
fix the distances between the cameras. Tested for the fleet's 149.6 mm square, as a prior of
1 mm with a 1 degree prior on the viewing direction:

```bash
export OV_RIG_DIST=0-1:149.6,1-2:149.6,2-3:149.6,3-0:149.6,0-2:211.566,1-3:211.566
export OV_RIG_PLANAR=0,1,2,3 OV_RIG_SIG_MM=1 OV_RIG_EVERY=0
export OV_RIG_RADIAL_DEG=0 OV_RIG_RADIAL_SIG_DEG=1 OV_RIG_TILT_SIG_DEG=1
```

It makes the four sides of a 20 s calibration more equal; it does not improve the angles
between the cameras or the stereo depth, and from 60 s on it should be left out.

## What was pruned

`PRUNED.txt` records it: the fork's experiment harness (alternative KLT trackers,
cross-camera stereo, tangent-image KLT, robust-gate variants, disparity marginalization,
debug instrumentation) plus ROS 1 support and the test/simulation executables — 25% of the
fork's source lines, 41 `OV_*` toggles down to 2, both live. No upstream OpenVINS code was
removed; verification is in `../selfcalib/docs/DEPLOYMENT_STRIP.md`.

> **Reproducibility.** The estimator is not bit-deterministic on long recordings: repeated
> identical runs occasionally diverge (~0.2 cm on flight ATE, far inside every acceptance
> threshold, changing no verdict). `OV_DET_LIVE=1` makes the live node deterministic
> (single-threaded OpenCV + stamp-deterministic IMU release). Compare results against
> thresholds, not byte-for-byte.
