# perception_pkgs

Two packages that share **one** estimator.

```
vio/          the VIO package — the estimator, and the FLIGHT/deployment config
selfcalib/    the self-calibration package — the tool, and its OWN calibration config
relloc/       relative localization between drones, beside the VIO (off unless RELLOC=1; relloc/README.md)
```

## Why they are split

The estimator is one binary with two operating points. What separates them is the config
and the environment variables of `flight_stiffness.env`, not the code:

| | config | calibration priors |
|---|---|---|
| **flight** (in `vio/`) | `vio_deploy/config/estimator_flight.yaml` + `flight_stiffness.env` | tethered (x0.10) — the online calibration can absorb a small per-flight offset but cannot random-walk. Since 2026-10-09 the same file also keeps two lens terms per camera as states and matches features between neighbouring cameras, and since 2026-11-07 it runs the bearing measurement model, with the fisheye mask the calibration fitted (`deploy_vio.py` writes it into the flight folder; `vio/README.md`) |
| **calibration** (in `selfcalib/`) | `configs/estimator_calib.yaml`, flight priors unset | loose (stock) — the states must be mobile enough for the warm-start iterations to converge. Since 2026-10-14 the calibration also matches between cameras and keeps the two lens terms as states (prior 0.002; `selfcalib/README.md`). Since 2026-11-07 it runs the bearing measurement model with a fisheye mask fitted to each lens's image circle, and flight runs it too (`vio/README.md`) |

Keeping the estimator in one place is the point. If each package carried its own copy they
would drift, and the day they drift is the day a calibration produced by one version gets
consumed by a flight stack running another — silently, because the published chain carries
no version stamp. `selfcalib/` deliberately ships **no estimator**; it calls `vio/`'s.

## Layout

```
vio/                        drop into a colcon workspace src/ — four ament packages
  ov_core/  ov_init/  ov_msckf/    pruned OpenVINS (see vio/PRUNED.txt)
  vio_deploy/                      flight config + flight_stiffness.env + the run scripts
  config/                          placeholder — ov_msckf's cmake installs ../config/
  BASELINE.txt  LICENSE  PRUNED.txt
relloc/                     python + ROS 2 node, no colcon build: scripts/run_relloc.sh, config/relloc.env
selfcalib/
  tool/run_tool.py          the calibration pipeline
  tool/deploy_vio.py        read a finished calibration -> write the flight folder
  configs/estimator_calib.yaml     the CALIBRATION config
  fleet_reference/          per-vehicle published chains for seeding + in-distribution checks
  docs/
```

## Quick start

**Read `WORKFLOW.md` for the full walkthrough.** The short version:

```bash
# 0. build the estimator once — both operating points use this one build
#    (rebuild after updating: the calibration's default matching and lens terms need commit 6bbaf8c or later,
#     the bearing model of calibration and flight commit 250f9f2 or later)
bash vio/vio_deploy/scripts/build_openvins.sh              # ~6 min

# 1. calibrate from an ordinary flight recording (no target, no manual session)
python3 selfcalib/tool/run_tool.py \
    --drone myvehicle --bag /path/to/bag \
    --template selfcalib/fleet_reference/nxt3/theta_star.yaml \
    --imu-chain /path/to/kalibr_imu_chain.yaml \
    --out out_myvehicle --fleet-exclude myvehicle
#    → check the verdict; HEALTHY means go on

# 2+3. read the calibrated values and deploy them into a flight-ready folder
python3 selfcalib/tool/deploy_vio.py --calib-out out_myvehicle --out flight_myvehicle
#    → prints intrinsics / extrinsics / t_d / mount, writes the folder the estimator wants
#      (the bearing model with the calibration's fitted fisheye mask; add --no-bearing for the original model,
#       which a calibration whose fitted mask cannot fly as it is also gets, with a WARNING)

# 4. fly (the folder's own flight settings; its MANIFEST.txt gives every line)
set -a; source flight_myvehicle/flight_stiffness.env; set +a           # REQUIRED
[ "$(uname -m)" = aarch64 ] && { set -a; source flight_myvehicle/campaign.env; set +a; }   # vehicle only
# replay of a recording (on a desktop first: export OV_GROUP_CAMS=1)
bash vio/vio_deploy/scripts/run_serial.sh BAG flight_myvehicle/estimator_flight.yaml OUT 4 false 42 70
# live on the vehicle (its topics must be in estimator_flight.yaml: MANIFEST.txt lists them)
ros2 run ov_msckf run_online_msckf flight_myvehicle/estimator_flight.yaml
```

A folder deployed before 2026-10-09 has no `flight_stiffness.env` of its own: deploy it again
(`deploy_vio.py` on the same calibration output). Its own MANIFEST no longer applies: it sources
the vio package's `flight_stiffness.env`, which since 2026-11-07 switches on the bearing model,
never measured with the config's mask such a folder holds. A folder deployed before 2026-11-07
does not set `OV_KB_BEARING` itself: fly it from a fresh shell, not after sourcing another
folder, or deploy it again.

Step 2+3 exists because the estimator resolves its chain files **relative to its config** and
by literal filename: a chain called `myvehicle_published_chain.yaml` sitting elsewhere is
never found, and nothing flags the mismatch. `deploy_vio.py` does the rename, gates on the
calibration verdict, and verifies the result.

`selfcalib` finds `vio` via `--vio-root`, then `$VIO_ROOT`, then the sibling `../vio`. Keeping
the two side by side needs no configuration.

`WORKFLOW.md` is the end-to-end walkthrough. `vio/README.md` and
`selfcalib/README.md` cover each package in detail.
