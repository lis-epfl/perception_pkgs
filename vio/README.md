# vio — the VIO estimator package

A pruned OpenVINS fork plus the flight/deployment configuration. This is the **only**
estimator in the repository: `selfcalib/` calls it rather than bundling its own, so
calibration and flight always run the same binary.

## Contents — four ament packages, side by side

```
ov_core/      ov_init/      ov_msckf/      pruned OpenVINS
vio_deploy/                                deployment assets (this package's own)
  config/estimator_flight.yaml               FLIGHT config
  config/flight_stiffness.env                x0.10 calibration-prior stiffness
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

Produces exactly two binaries:

| binary | use |
|---|---|
| `run_serial_msckf` | deterministic offline replay — what `selfcalib` drives, and what flight-mode replay uses |
| `run_subscribe_msckf` | the live ROS 2 node — flight VIO on the vehicle |

Requires Ubuntu 22.04 + **ROS 2 Humble**, OpenCV ≥ 4.5, Eigen3, Boost, Ceres
(`libeigen3-dev libboost-all-dev libceres-dev`). ROS 1 is not supported; the CMake files
fail with a clear message rather than silently falling back.

## Two operating points, one binary

The difference is the config plus five `OV_PRIOR_*` variables read once at startup by the
patched `State.cpp`. They set the initial standard deviations of the online-calibration
priors; unset, the stock values apply.

| env var | state | stock (unset) | flight (`flight_stiffness.env`) |
|---|---|---|---|
| `OV_PRIOR_DT_SIG` | cam-IMU time offset (s) | 0.01 | 0.001 |
| `OV_PRIOR_EXTR_Q_SIG` | cam-IMU rotation (rad) | 0.005 | 0.0005 |
| `OV_PRIOR_EXTR_T_SIG` | cam-IMU translation (m) | 0.015 | 0.0015 |
| `OV_PRIOR_INTR_F_SIG` | intrinsics fx/fy/cx/cy (px) | 1.0 | 0.1 |
| `OV_PRIOR_INTR_D_SIG` | distortion coefficients | 0.005 | 0.0005 |

**Flight** — `config/estimator_flight.yaml` **with** `config/flight_stiffness.env` sourced.
The tightened priors tether the online calibration to its published chain. This exact pair
(uniform, no per-vehicle tuning) produced 2.4–4.5 cm SE(3) ATE across an 11-flight,
3-vehicle benchmark.

**Calibration** — `selfcalib/configs/estimator_calib.yaml` with `OV_PRIOR_*` **unset**.
`selfcalib` never sets them. Loose priors keep the calibration states mobile so the
warm-start iterations can converge away from the seed.

The per-vehicle flight folder is produced by the self-calibration package, not written by
hand — `selfcalib/tool/deploy_vio.py` copies this config next to the calibrated chains under
the filenames the estimator resolves (`kalibr_imucam_chain.yaml`, `kalibr_imu_chain.yaml`)
and verifies the result. See `../WORKFLOW.md`.

```bash
# flight replay, using a folder produced by deploy_vio.py
set -a; source vio_deploy/config/flight_stiffness.env; set +a
bash vio_deploy/scripts/run_serial.sh BAG flight_myvehicle/estimator_flight.yaml OUT 4 false 42 DOMAIN

# live on the vehicle
ros2 run ov_msckf run_subscribe_msckf flight_myvehicle/estimator_flight.yaml
```

Rule of thumb: **calibrating → priors loose (env unset); flying → priors tight (source the
env).** `run_serial.sh` passes your shell environment through, so check `env | grep OV_PRIOR`
if a run behaves oddly — flight vars leaking into a calibration run pin the calibration at
its seed.

`run_serial.sh` exit codes: `64` usage/bad DOMAIN · `66` missing bag or config · `69` missing
ROS or workspace · `70` ran but produced no trajectory · otherwise the estimator's own code.

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
