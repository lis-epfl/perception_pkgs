# perception_pkgs

Two packages that share **one** estimator.

```
vio/          the VIO package — the estimator, and the FLIGHT/deployment config
selfcalib/    the self-calibration package — the tool, and its OWN calibration config
```

## Why they are split

The estimator is one binary with two operating points. What separates them is the config
and five environment variables, not the code:

| | config | calibration priors |
|---|---|---|
| **flight** (in `vio/`) | `vio_deploy/config/estimator_flight.yaml` + `flight_stiffness.env` | tethered (x0.10) — the online calibration can absorb a small per-flight offset but cannot random-walk |
| **calibration** (in `selfcalib/`) | `configs/estimator_fleet.yaml`, `OV_PRIOR_*` unset | loose (stock) — the states must be mobile enough for the warm-start iterations to converge |

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
selfcalib/
  tool/run_tool.py          the calibration pipeline
  tool/deploy_vio.py        read a finished calibration -> write the flight folder
  configs/estimator_fleet.yaml     the CALIBRATION config
  fleet_reference/          per-vehicle published chains for seeding + in-distribution checks
  docs/
```

## Quick start

**Read `WORKFLOW.md` for the full walkthrough.** The short version:

```bash
# 0. build the estimator once — both operating points use this one build
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

# 4. fly
set -a; source vio/vio_deploy/config/flight_stiffness.env; set +a      # REQUIRED
bash vio/vio_deploy/scripts/run_serial.sh BAG flight_myvehicle/estimator_flight.yaml OUT 4 false 42 70
# live on the vehicle:
ros2 run ov_msckf run_subscribe_msckf flight_myvehicle/estimator_flight.yaml
```

Step 2+3 exists because the estimator resolves its chain files **relative to its config** and
by literal filename: a chain called `myvehicle_published_chain.yaml` sitting elsewhere is
never found, and nothing flags the mismatch. `deploy_vio.py` does the rename, gates on the
calibration verdict, and verifies the result.

`selfcalib` finds `vio` via `--vio-root`, then `$VIO_ROOT`, then the sibling `../vio`. Keeping
the two side by side needs no configuration.

`WORKFLOW.md` is the end-to-end walkthrough. `vio/README.md` and
`selfcalib/README.md` cover each package in detail.
