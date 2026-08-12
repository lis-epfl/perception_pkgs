# Deployment package: what was removed, what changed, how it was verified

> **Note on layout.** This document was written while the tool and the estimator lived in one
> directory. They are now two packages — `vio/` (estimator + flight config) and `selfcalib/`
> (this package + the calibration config) — so paths like `openvins/...` now mean
> `vio/ov_core`, `vio/ov_init`, `vio/ov_msckf`; `configs/estimator_flight.yaml` is
> `vio/vio_deploy/config/`; and `scripts/` is `vio/vio_deploy/scripts/`. The workspace default has
> since moved from `~/ov_ws_selfcalib` to `~/ov_ws_vio`. The prune record and
> equivalence evidence below are unchanged and still describe the shipped estimator.

This package is the study package (`vio_dataset/selfcalib_tool`) reduced to what a fleet
vehicle needs to run **calibration mode** and **flight mode**. 404 MB -> 2.3 MB.

No source file that the estimator compiles was touched. The changes below are build
plumbing, one YAML syntax fix, documentation, and the ground-truth/mount additions to
`run_tool.py`.

## Removed

| Path | Size | Why it is safe to drop |
|---|---|---|
| `openvins/ov_data/` | 377 MB | Ground-truth trajectories for public datasets. Declared only as a soft dependency — it is commented out in `ov_msckf/package.xml` (lines 69-71) |
| `openvins/config/` *contents* | 8.4 MB | Stock estimator configs for EuRoC / TUM-VI / KAIST / UZH-FPV / RealSense. Both operating points here use `configs/*.yaml`, passed to `run_serial.sh` by absolute path |
| `openvins/ov_eval/` | 8.2 MB | Upstream trajectory-evaluation package. `ov_msckf` does not link it (soft dep, same package.xml block); the tool computes ATE itself in `tool/evalate.py` |
| `openvins/docs/`, `Doxyfile*` | 4.1 MB | Doxygen sources and generated docs |
| `openvins/Dockerfile_*` (6), `.github/`, `run_format.sh`, `run_copyright.sh`, `run_size.sh`, `ReadMe.md` | ~40 KB | Upstream development and CI scaffolding |
| `e2e_test_nxt3/`, `e2e_test.sh`, `e2e_test.log` | 4.3 MB | A recorded reference run plus a driver hardcoded to `/tmp/swarm30_runs` and `~/vio_dataset/ov_reverify` |
| ~~`tool/mount.py`~~ | — | **Kept.** Was dropped as dead code, then restored and wired into `run_tool.py` — see "Ground truth and the mount solve" below |
| `scripts/converters/`, `scripts/truncate_bag.py` | 3 files | ASL/EuRoC and image-list bag converters (fleet vehicles record ROS 2 bags natively) plus a debug helper with hardcoded paths |
| `configs/estimator_standard_template.yaml`, `configs/examples/` | 3 files | Stereo/standard-rig template and documentation examples. `run_tool.py` hardcodes `estimator_calib.yaml` in its seed-assembly step; `fleet_reference/*/theta_star.yaml` serves as the chain-format example |

**`openvins/config/` the directory must keep existing** (a `README.txt` placeholder holds it
open). `ov_msckf/cmake/ROS2.cmake:123` runs `install(DIRECTORY ../config/ ...)` and colcon
fails the install step with `file INSTALL cannot find .../config` if it is absent.

## Changed

1. **`scripts/build_openvins.sh`** — builds `ov_core ov_init ov_msckf` (was: those plus
   `ov_eval`); default workspace is now `$HOME/ov_ws_selfcalib` instead of `$HOME/ov_ws`, so
   building this package cannot repoint an unrelated workspace's `src/` symlinks at this
   checkout; falls back to `/usr/bin/cmake` when the `cmake` on `PATH` is >= 4.0, which
   rejects OpenVINS's `cmake_minimum_required(VERSION 3.3)` outright.
2. **`scripts/run_serial.sh`** — `OV_WS` default follows the same change. One line.
3. **`configs/estimator_flight.yaml`** — moved `%YAML:1.0` to line 1. **This was a real bug
   in the study package**: a 10-line comment banner sat above the directive, and OpenCV's
   `cv::FileStorage` rejects any file whose first line is not the directive, so flight mode
   aborted at startup with `(-5:Bad argument) Input file is invalid in function 'open'`.
   Every non-comment line is byte-identical to the original; only the banner moved.
4. **`README.md`** — contents list, build defaults, and the seed-chain section updated to
   match what actually ships. Also corrects the `T_cam_imu` / `T_imu_cam` instruction (see
   "Known issues" below), and replaces the bit-identical-output claim with the measured
   reproducibility caveat.
5. **`tool/run_tool.py`** — ground-truth auto-detection (step 0) and the mount solve
   (step 7) wired in; new flags `--gt-topic` / `--no-gt-autodetect`; `--gt` now fails fast
   (exit 3) if the file does not exist, instead of crashing in the diagnosis step after all
   estimator passes have run. See "Ground truth and the mount solve" below. The gates, seed,
   warm-start loop, publish, and diagnosis code paths are character-identical to the study
   package.
6. **`tool/gtdetect.py`** — new module (the detector itself).
7. **`docs/TROUBLESHOOTING.md`** — one pointer updated: the IMU-intrinsics fix now references
   `fleet_reference/nxt3/kalibr_imu_chain.yaml` (the `configs/examples/` file it used to
   point at is not shipped).

Complete change surface vs the study package, verified by `diff -rq` over the whole tree —
**twelve modified files** and **four new** (`tool/gtdetect.py`, `docs/DEPLOYMENT_STRIP.md`,
`openvins/config/README.txt`, `.gitignore`):

```
configs/estimator_calib.yaml     configs/estimator_flight.yaml    README.md
docs/TOOL_INTERNALS.md           docs/TROUBLESHOOTING.md
scripts/build_openvins.sh        scripts/run_serial.sh
tool/chainio.py   tool/diagnose.py   tool/gates.py   tool/mount.py   tool/run_tool.py
```

Byte-identical to the original: `tool/circlefit.py`, `tool/evalate.py`, `tool/publish.py`,
`configs/flight_stiffness.env`, all of `fleet_reference/`, `docs/PROTOCOL.md`,
`docs/VALIDATION_MATRIX.md`, and **all 149 OpenVINS source files** — so the estimator itself
is untouched, which is what the machine-code comparison below independently confirms.

The two config edits are the `use_klt_tangent` flag plus comments; every other non-comment
line in both configs is unchanged (`diff` of comment-stripped files shows only that flag).
`tool/mount.py` and `docs/TOOL_INTERNALS.md` changed only in comments/prose.

## Verification

### Build equivalence (strongest evidence)

The stripped `openvins/` sources are byte-identical to the study package's across all 149
files in `ov_core` + `ov_init` + `ov_msckf` (`diff -r`, no differences). Building both into
separate colcon workspaces and disassembling `run_serial_msckf`:

```
address-normalised disassembly:  68,865 instructions,  IDENTICAL
```

The binaries differ only in `.rodata`, by exactly the embedded build-path strings (22
occurrences; the two workspace paths are 115 vs 29 characters). **The strip cannot change
what the estimator computes, because it produces the same machine code.**

### Full-pipeline reproduction

`run_tool.py` from this package on the nxt3 basin bag (185 s, 4 cameras), against the study
package's recorded run in `e2e_test_nxt3/`:

```
report.json          every field identical (recursive compare; only the three ATE fields
                     absent, because that run had no ground truth by design)
estimate_pass1.tum   BYTE-IDENTICAL          harvest_pass1.calib.json   BYTE-IDENTICAL
estimate_pass2.tum   BYTE-IDENTICAL          harvest_pass2.calib.json   BYTE-IDENTICAL
published chain      BYTE-IDENTICAL          calib/ seed chain + config BYTE-IDENTICAL
```

Independently re-derived from the bag and matched against the recorded report:
static gate, timing gate (all 4 cameras), circle fit (all 4 centres), image-health gate —
all identical.

### Flight mode

Reference build vs stripped build, same bag / chain / seed, `flight_stiffness.env` sourced,
ASLR disabled on both sides so the comparison is meaningful:

```
estimate_tum.txt  ->  BYTE-IDENTICAL (1841 poses each)
SE(3) ATE  postg = 2.42 cm   global = 2.42 cm
```

2.42 cm sits at the bottom of the README's stated 2.4-4.5 cm benchmark band.

### Cross-vehicle end-to-end (nxt6, leave-one-out)

nxt6 calibrated from a 60 s recording, seeded only from the nxt3 + nxt10 means — the tool
never saw nxt6's own calibration. Recovered vs the study's published nxt6 fixed point:

```
worst principal point 0.82 px | focal 0.524 % | rotation 0.143 deg | translation 1.35 cm
time offset 0.81 ms  |  criterion-metric residual 0.0130  (tau = 0.30 acceptance)
in-distribution certificate PASS (no fails)   VERDICT: HEALTHY: deploy
```

### Pipeline paths

| path | result |
|---|---|
| GATE-FAIL (synthetic bag with no static start) | correct verdict, **exit 1**, stops before any estimator work |
| FLY-AGAIN (`--max-pass 1`) | correct verdict, **exit 2** |
| `--gt` pointing at a missing file | fails fast with **exit 3** before the estimator runs |
| success | **exit 0** |
| no `--fleet-exclude` (seed from all three) | converged pass 2, HEALTHY |
| `--no-gt-autodetect` | ground-truth search suppressed (recorded in `report.json`) |
| `--gt-topic` | forces an otherwise-rejected topic — **including when combined with `--no-gt-autodetect`** (explicit wins) |
| no `SCT_ROOT`, run from an unrelated cwd | package self-locates via its own path; full run unaffected |

### Module-level, against recorded values

41 checks, all passing (`chainio` round-trip, `publish.robust_mean` reproducing the
published chain from the harvests, `diagnose.fleet_stats` for all four exclude variants,
all six in-distribution figures, all three ATE figures, `umeyama`/`q_to_R` against closed
form, `mount.solve`/`apply` algebra, `gtdetect` scoring). Two precision characteristics
worth knowing, both benign:

- `write_chain` formats `timeshift_cam_imu` with `%.9f`, so a published chain quantises the
  time offset to 1 ns (the tool's own t_d tolerances are 10 ms and 20 ms).
- Rotations round-trip through `%.10f` text, so a chain re-read from disk is orthonormal only
  to ~1e-10. `calib_residual(x, x)` is therefore 1.2e-5 rather than 0 — about 8,000x below
  the 0.10 convergence stop. `chordal_mean` re-orthonormalises via SVD, which is correct.

## Deployment trim (final leanness pass)

With deployment as the bar, everything not needed to build and run the two shipped
binaries (`run_serial_msckf` for calibration, `run_subscribe_msckf` for live flight VIO)
was removed:

| removed | detail |
|---|---|
| ROS 1 support | `ROS1Visualizer.{cpp,h}`, `ros1_serial_msckf.cpp`, three `cmake/ROS1.cmake` (84 KB, never compiled — this package builds with ament). All three `CMakeLists.txt` now fail with a clear message on non-ROS 2 builds |
| 8 test/sim executables | `run_simulation`, `test_sim_meas`, `test_sim_repeat`, `test_webcam`, `test_profile`, `test_simulation`, `test_dynamic_mle`, `test_dynamic_init` — sources and CMake targets. Build now produces exactly 2 binaries |
| dataset benchmark scripts | `ov_msckf/scripts/` (eth / kaist / tumvi / uzhfpv / sim runners) |
| doc configs | three `rosdoc.yaml` |
| ROS 1 launch/rviz files | kept: `ov_msckf/launch/subscribe.launch.py` + `display_ros2.rviz` (they target the live flight node); `ov_init/launch/` reduced to a placeholder (its cmake install needs the directory) |

Fresh from-scratch deployment build (`rm -rf ~/ov_ws_selfcalib && bash
scripts/build_openvins.sh`): **5 min 38 s**, exactly two binaries. Validated on that fresh
workspace: the full calibration pipeline reproduced the study reference **byte-identically
in both passes** (resid 0.0064, ATE {0.0361, 0.0366, 0.2377}, chain deltas 0.0000 →
E2E_MATCH) and flight mode gave 2.42 cm — in the measured set. Suites 41/41 + 18/18.

## Reproducibility: the estimator is NOT bit-deterministic on long sequences

The study package's README claimed "Same bag + same config + same seed = bit-identical
output" (this package's README now carries the corrected statement). That claim is
**not true in general**, and it was already not true in the study package — this is a
property of the estimator, not of the strip.

Measured on the pass-2 configuration of the nxt3 basin bag (185 s), repeated identical runs:

```
nxt3_T30  (30 s) : 8/8 runs identical (ASLR on), 8/8 identical (ASLR off)
nxt3 basin (185 s): runs sometimes diverge, starting around pose 419 of 2806,
                    growing to ~0.12 m by the end of the trajectory
```

Flight mode is affected too, and its practical spread is worth knowing: three sequential
ASLR-disabled replays of the reference recording with the shipped flight config gave
**2.42 / 2.22 / 2.42 cm** SE(3) ATE (two distinct trajectories). So treat ~0.2 cm as the
run-to-run noise floor when comparing flight results — a difference that size is not a
regression.

Ruled out as the cause, each tested by repeated runs:

- `num_opencv_threads: 0` (OpenCV parallelism off) — still diverges
- `init_dyn_mle_max_threads: 1` (Ceres single-threaded) — still diverges
- `OMP_NUM_THREADS=1` together with both of the above — still diverges
- the asynchronous initializer thread in `VioManagerHelper.cpp:96` — not it;
  `use_multi_threading_subs` defaults to `false`, so that thread is `join()`ed
- ASLR (`setarch -R`) — reduces the frequency but does **not** eliminate it; two ASLR-disabled
  runs of bit-identical machine code still diverged

**The root cause is not identified.** What is established is the impact, which is small: when
a divergence does occur, the published calibration still lands far inside the study's own
acceptance thresholds —

```
principal point 0.13 px (limit 2.5) | focal 0.06 % (0.5) | rotation 0.035 deg (0.35)
translation 0.25 cm (2.5)           | time offset 0.001 ms (6)
```

Practical consequences:

1. Do not use byte-equality as a regression test for calibration mode on long recordings.
   Compare published chains against thresholds, the way the study package's `e2e_test.sh`
   does (that driver is not shipped here — it is hardcoded to the study machine's paths).
2. The warm-start loop's convergence criterion (residual < 0.10) is roughly three orders of
   magnitude above this noise, so convergence is unaffected.
3. If you need a reproducible audit trail for a fleet, record the published chain and the
   harvests, not a promise of bit-equality.

## Robustness fixes applied

Each of these was a defect found while validating this package. All are behaviour-preserving
on a healthy run — they change what happens when something is *wrong*, or remove a
duplicated source of truth. The full-pipeline reproduction above was re-run after all of
them.

1. **`tool/chainio.py`** — removed the dead `OVR = '/home/toumieh/vio_dataset/ov_reverify'`
   absolute path. `parse_chain` now **raises a diagnosable error** when a chain has no
   `T_imu_cam` block, and says so explicitly when the file uses the `T_cam_imu` spelling
   instead. Previously this parsed silently into a chain with no extrinsics and died much
   later as a bare `KeyError` during seed assembly. The two keys are inverses, so the tool
   refuses rather than reinterpreting — silently guessing would publish a wrong calibration
   that nothing downstream flags.
2. **`tool/diagnose.py`** — `fleet_stats` no longer hardcodes `('nxt3','nxt6','nxt10')`. New
   `fleet_members()` enumerates `fleet_reference/` for directories that have both a
   `theta_star.yaml` and a `td_center.json` entry. Swapping in your own fleet needs no code
   edit, and an unusable fleet gives a message naming what it found instead of a
   `FileNotFoundError` traceback. Verified to reproduce the previous means exactly for all
   four exclude variants.
3. **`tool/gates.py`** — `THETA_MAX` is now read from `configs/estimator_calib.yaml`
   (`mask_fisheye_theta_max` plus per-camera `mask_fisheye_theta_max{i}`), matching
   `VioManagerOptions.h:318-338` semantics, with the study rig's values as fallback if the
   config is absent. The tool's image-circle geometry and the estimator's own fisheye mask
   can no longer drift apart, and a rig with different lenses is a config edit rather than a
   code edit. Verified to yield the previously hardcoded values from the shipped config.
4. **`tool/run_tool.py`** — the estimator invocation was fire-and-forget. It now:
   deletes the previous pass's harvest first (so a failed pass cannot be mistaken for a
   successful one), **checks the return code**, reports a missing harvest with the likely
   causes and the last lines of stderr instead of a bare `FileNotFoundError`, and derives the
   camera count from the template chain rather than hardcoding `4`. The 1800 s per-pass cap
   is now `max(1800, 20x bag duration)` with a `--pass-timeout` override, and a timeout is
   reported rather than silently truncating a long recording. New exit codes: 4 = timeout,
   5 = estimator produced no harvest.
5. **`tool/run_tool.py`** — arguments are validated before any work: `--domain` in 0-232,
   `--max-pass >= 1`, and `--bag`/`--template`/`--imu-chain` must exist.
6. **`tool/run_tool.py`** — the four absolute `/tmp` scratch paths in the estimator config are
   rewritten into each run's own `calib/` directory when the config is copied, so concurrent
   calibrations on one machine cannot overwrite each other's. (`--domain` separated the DDS
   traffic; nothing separated these.) Inert by default, since both writing flags are `false`.
7. **`scripts/run_serial.sh`** — validated the bag, config, ROS installation and workspace
   before running, with distinct exit codes (64 usage, 66 missing input, 69 missing
   prerequisite, 70 empty output). Sourcing a nonexistent workspace `setup.bash` used to be
   swallowed by `>/dev/null`, so the script could "succeed" having produced nothing — or pick
   up a *different* `ov_msckf` from the ambient environment. The estimator's exit code is now
   propagated and an empty trajectory is an error. `DOMAIN` is range-checked.
8. **`configs/*.yaml`** — `use_klt_tangent` and `klt_tangent_patch_size` **removed entirely**
   from both. They were `true`/`15` and completely inert (see below). The C++ default is
   already `false` (`VioManagerOptions.h:481`) and both keys are optional, so omitting them is
   exactly equivalent — verified by re-running the pipeline, which reproduced the previous
   result to the digit (resid 0.0243, toff -0.039297). A one-line comment marks them as
   deliberately absent so they do not get re-added.
9. **Dangling documentation references** — `docs/TOOL_INTERNALS.md` pointed at `tool_tests.md`
   and `configs/FINAL_single_config.yaml`, and `tool/gates.py` / `tool/diagnose.py` /
   `tool/mount.py` cited `tool_tests.md` in comments. Neither file exists in this package or
   in the study package. All now point at `docs/VALIDATION_MATRIX.md` and
   `configs/estimator_calib.yaml`.

### `use_klt_tangent` was inert, and the keys are now gone

`VioManager.cpp:135` does construct a `TrackKLTTangent` when the flag is true, but
`perform_matching` is not declared in `TrackBase` and carries no `virtual` in
`TrackKLT.h:139`, so the calls at `TrackKLT.cpp:148` and `:293` bind statically to the base
implementation. The tangent tracker never executes. Confirmed empirically as well: three
replicate runs at `true` and three at `false` produce **the same set of outputs**
(per-run `cmp` is meaningless here — this config/bag pair is one of the nondeterministic
ones, yielding two distinct outcomes within a single setting).

Every validated result in this package therefore comes from plain `TrackKLT`. Re-adding the
flag as `true` expecting tangent tracking would do nothing; *fixing* the missing `virtual`
would silently change the estimator and invalidate the validation matrix. Either is a
deliberate decision requiring re-validation.

**The dead C++ itself is deliberately left in `openvins/`.** `TrackKLTTangent.{cpp,h}`,
`CamDS.h` and `test_ds_and_tangent_klt.cpp` total 43 KB — 1.8% of the package — and deleting
them is not file removal but source surgery on the vendored fork: `VioManager.cpp` (include +
construction branch), `VioManagerOptions.h` (three options, three parse calls, the `"ds"`
camera-model branch, an include) and `ov_msckf/src/sim/Simulator.cpp` all reference them, and
`VioManagerOptions.h` is itself part of `UNCOMMITTED.diff`. Doing it here would break the
149-file byte-identity and the machine-code equivalence proof, and would diverge this copy
from the fork that flight VIO is meant to share — the exact drift this package is otherwise
built to avoid. If the dead code should go, remove it **in the fork**, once, and let both
consumers pick it up.

## The fork was pruned to its reachable part

`UNCOMMITTED.diff` added **1776 lines across 30 files** on top of a baseline that already
carried two local feature commits (double-sphere camera model, tangent-image KLT), and
defined **41 `OV_*` environment toggles** of which this package ever set **six**. Roughly two
thirds of that was the experiment harness used to *find* the operating point — inert in both
operating points. It has been removed.

```
source lines   34,515 -> 32,608   (-1,907)
source files      124 -> 121      (TrackKLTTangent.{cpp,h}, test_ds_and_tangent_klt.cpp,
                                   plus a stray State.cpp.bak_prior left in the fork)
OV_* toggles       41 -> 2        (OV_RNG_SEED, OV_DET_LIVE — both live)
```

### What was removed

| removed | why it was unreachable |
|---|---|
| `TrackKLT` SE(2) / tangent-LK / distort-KLT trackers (~515 lines) | `OV_SE2_KLT`, `OV_TANGENT_LK`, `OV_DISTORT_KLT` never set |
| `TrackKLT::stereo_match_epitangent` + the `TrackBase` extrinsics plumbing that fed it | `OV_XCAM_STEREO` never set |
| `TrackKLT` JSONL feature dumps, edge-survival stats, detect-theta lever | `OV_FEAT_DUMP_DIR`, `OV_TLK_SURVSTATS`, `OV_DETECT_THETA_MAX` never set |
| `TrackKLTTangent` class + its `VioManager` branch + three options | non-virtual `perform_matching`, so it never executed even when selected |
| `StateHelper::decide_clone_marginalization` + `_override_marg_time` + two `StateOptions` | `use_disparity_marg` defaults false and is in neither config |
| `OV_ROBUST_GATE` DCS down-weighting in `UpdaterMSCKF` / `UpdaterSLAM` | never set; the stock hard chi2 gate is what ran |
| `OV_ZUPT_GRADUAL` noise ramp | never set |
| per-camera `calib_cam_extrinsics_<i>` override (`StateOptions`, `State.cpp`, 4 `UpdaterHelper` call sites) | keys absent from both configs, so it always equalled the global flag |
| `OV_CALIB_FREEZE_T`, `OV_DBG_FEEDIDS` in `VioManager` | never set |
| `run_serial_msckf` harness knobs: viz dump, mid-run calib snapshots, feed timing, IMU decimation/gaps, frame decimation | never set; all study instrumentation |
| `OV_KLT_WIN`/`OV_KLT_PYR`, `OV_UPDATE_MIN_DT` env override, `OV_INIT_DEBUG`, `OV_INIT_FEAT_THRESH`, `OV_FORCE_STATIC_INIT` env, `OV_INIT_DYN_BIAS_*_SIGMA` | never set; the config keys (`update_min_dt`, `init_force_static`) are the live interface and are untouched |
| `TrackDescriptor` XFeat loader (`OV_XFEAT_DIR`) | `use_klt: true` means `TrackDescriptor` is never constructed |

Kept because they are live: the fisheye auto-masking, the per-call RNG seeding that
`OV_RNG_SEED` drives, `update_min_dt`, the `OV_PRIOR_*` prior sigmas, force-static init via
its config key, the dynamic-init DENSE_SVD covariance retry, `ROS2Visualizer`, and
`OV_DET_LIVE` — the last is *not* dead scaffolding but a flight-mode determinism switch
(single-threaded OpenCV + stamp-deterministic IMU release), directly relevant to the
reproducibility caveat below.

### Nothing of upstream OpenVINS was removed

The prune was checked against **true vanilla** — upstream commit `6948812`
("Merge pull request #530 from rpng/android"), the last upstream ancestor named in
`BASELINE.txt`, exported read-only from the fork's own git repo. Three trees were compared:
vanilla, the pre-prune fork, and the pruned result.

```
                    differing files vs vanilla    differing lines vs vanilla
  pre-prune fork              76                          1,860
  pruned                      69                            719
```

Per-file semantic diff (vanilla -> pre-prune vs vanilla -> pruned) confirms:

```
  NO vanilla content — code OR comments — was removed by the prune.
```

Every line upstream OpenVINS has that the fork still carried is still present **in the
compiled estimator code**. (A later leanness pass additionally removed whole *auxiliary*
vanilla files that are never compiled into the two shipped binaries — ROS 1 support, test
and simulation executables, dataset benchmark scripts, doc configs; the file list is in
"Deployment trim" below. That is file-level removal of non-estimator content, consistent
with the original strip of `ov_eval`/`ov_data`; the estimator sources themselves remain
vanilla-faithful.) What the
prune removed was exclusively fork-authored: the 1,776-line `UNCOMMITTED.diff` working-tree
patch, plus the fork's own two feature commits (`CamDS.h`, `TrackKLTTangent.{cpp,h}`,
`test_ds_and_tangent_klt.cpp` — 1,049 of the 1,144 lines those commits added).

The pruned tree is therefore *closer* to upstream than the fork was, while retaining every
live fork feature: fisheye auto-masking, `OV_RNG_SEED` seeding, `update_min_dt`, the
`OV_PRIOR_*` prior sigmas (visible as `sig_eq`/`sig_et` where vanilla hardcodes 0.005/0.015),
config-driven force-static init, the dynamic-init DENSE_SVD retry, the `.calib.json` harvest,
`ROS2Visualizer`, `OV_DET_LIVE`, and the `UpdaterMSCKF` anchor-missing guard that prevents a
throw during out-of-order multi-stereo processing.

### Equivalence after pruning

Machine-code comparison no longer applies, so equivalence was established behaviourally,
against the study package's recorded run:

```
pruned build, the study's own pass-2 input, ASLR off, 3 replicates:
    all three BYTE-IDENTICAL to e2e_test_nxt3/estimate_pass2.tum

full pipeline (185 s bag, with ground truth):
    estimate_pass1.tum   BYTE-IDENTICAL      estimate_pass2.tum   BYTE-IDENTICAL
    resid 0.0064 (reference 0.0064)
    ATE   {postg 0.0361, global 0.0366, noum 0.2377}  — identical to the reference report
    published chain: pp 0.0000 px | focal 0.0000 % | rot 0.0007 deg | trans 0.0000 cm
                     | td 0.0000 ms   -> E2E_MATCH

flight mode: SE(3) ATE 2.22 cm — inside the pre-prune measured set {2.42, 2.22}
cross-vehicle nxt6 leave-one-out: resid 0.0143, td 0.81 ms (tau = 0.30) -> HEALTHY
```

The pruned build reproduces the study reference **byte-for-byte through the whole pipeline**.

`openvins/BASELINE.txt` names the fork's baseline commits. The 145 KB `UNCOMMITTED.diff`
pre-prune patch record is not shipped (it no longer applied to these sources); it remains in
the study package, and the fork's git repo is the authoritative history. `openvins/PRUNED.txt`
records this in place. Also removed as zero-consumer dead weight: `ov_core/src/plot/`
(matplotlibcpp, 68 KB, included by nothing) and `ov_init/src/sim/` (SimulatorInit, 32 KB,
compiled into the lib but referenced only by the already-removed init test binaries).
`ov_msckf`'s `Simulator` stays: the live flight visualizer's constructor takes it.

## Known issues still carried over

1. **`gates.py:18`** — `STATIC_MIN_S = 1.5` is the gate's actual threshold; the README's
   "static segment >= 4 s" is the *recommendation*. Both are now stated in the README, but
   the two numbers remain intentionally different, and only the code one is enforced.
2. **The estimator is not bit-reproducible on long recordings** — see the section below.
   Not fixed because the root cause is unidentified and any fix would change the estimator.

## Ground truth and the mount solve

`tool/mount.py` is kept. It was initially dropped (nothing imported it) and has since been
restored and connected to the pipeline, together with a new module `tool/gtdetect.py`.

Ground truth is treated as **opportunistic**: calibration never needs it, but when a
recording happens to carry a mocap stream, the run should get the ATE certificate and the
mount solve for free rather than requiring a manual export.

- `tool/gtdetect.py` scans the bag for a mocap pose stream when `--gt` is not supplied, and
  writes `--out/ground_truth.tum`. Supports `geometry_msgs/PoseStamped`,
  `PoseWithCovarianceStamped`, `TransformStamped`, `nav_msgs/Odometry` and
  `tf2_msgs/TFMessage`. Candidates are ranked on topic name, message rate, and the `--drone`
  name as a tiebreak; estimator-output topics (`/ov_msckf/...`, `*vio*`, `*estimate*`) are
  scored down so they cannot be mistaken for truth. The choice and the rejected candidates
  are recorded under `ground_truth` in `report.json`.
- `run_tool.py` step 7 calls `mount.solve()` whenever ground truth is available and writes
  `--out/<drone>_mount.json` (the 3x3 `M` plus yaw/tilt and before/after errors), for reuse
  via `mount.apply()` on later flights.

  **Verified about the mount solve.** The algebra: `M` is a proper rotation, `apply()` with
  the flight's own `M` reproduces `solve()` exactly, `apply()` with identity reproduces the
  raw anchored error. Within a flight the correction reaches the best-fit floor exactly
  (0.2304 -> 0.0481 m against a 0.0481 m floor on the 45 s run; 0.2354 -> 0.0997 against
  0.0997 on the full 185 s run).

  **`M` is constant; the anchor is the whole story.** With
  `M = R_fit · R_anchor⁻¹` and `R_anchor(t) = R_gt(t) · R_est(t)⁻¹`, solving `M` at two
  different anchor instants of the reference flight gives a **3.446°** discrepancy — and the
  two `R_anchor` values differ by **3.446°**, equal to machine precision. The variation is
  entirely the anchor; `M` itself does not move. That is why the cross-flight transfer works
  **provided each flight is anchored at the same initial pose**: reproduce the takeoff
  attitude and you reproduce `R_anchor`, hence `M`. Sampling `R_anchor(t)` across the flight
  shows it wandering by at most 3.3° over 170 s — VIO yaw drift on the anchor — which is the
  reason to anchor in the pre-takeoff static phase, where drift has not accumulated.

  (The study data ships one GT recording per vehicle — `nxt3_raw_4cam` and `nxt3_basin_4cam`
  are the same recording, identical start time and duration — so a literal two-flight
  transfer cannot be replayed here. Splitting one trajectory in half is not a substitute:
  the second half anchors mid-flight and inherits the first half's drift, degrading its own
  best-fit floor to 0.104 m. The `R_anchor` identity above tests the mechanism directly and
  is not subject to that confound.)
- New flags: `--gt-topic` to force a topic, `--no-gt-autodetect` to disable the search.

**Behaviour without ground truth is unchanged.** The detector returns
`{'found': False, 'reason': ...}`, the ATE certificate and mount solve are skipped, and the
published chain is identical — verified by re-running the full pipeline on the original
nxt3 bag, which has no pose topic (see "Verification").

Note that on a bag that *does* carry mocap, a run that previously reported `HEALTHY` with
the ATE certificate skipped may now report `PLATFORM-DEFECT`, because the certificate is
actually being evaluated. That is the intended consequence of having more information, not
a regression.

### Detector test coverage

Verified against synthetic bags built from the real nxt3 ground truth, plus the real bag:

| case | result |
|---|---|
| `geometry_msgs/PoseStamped` on `/vrpn_client_node/nxt3/pose`, with an `/ov_msckf/poseimu` decoy | correct topic chosen (score 27.0 vs -45.0); extracted trajectory matches source exactly |
| `nav_msgs/Odometry` on `/optitrack/nxt3/odom` | detected; exact match |
| `geometry_msgs/TransformStamped` on `/vicon/nxt3/nxt3` | detected; exact match |
| `tf2_msgs/TFMessage`, vehicle frame plus a second rigid body | correct `child_frame_id`; exact match |
| same, with the decoy frame emitted first and equal message counts | resolved by the `--drone` hint; without a hint it falls back to alphabetical order and reports `child_frames_considered` so the ambiguity is visible |
| publisher leaves `header.stamp` empty | falls back to bag receive time and raises a warning that the stream is not on the sensor clock (on both the auto and the `--gt-topic` path) |
| dense stream covering only ~36% of the bag | detected, with a coverage warning |
| real nxt3 bag (cameras + IMU only) | `found: False`, pipeline continues unchanged |

Rate-floor nuance: a candidate must average >= 5 Hz **over the whole bag duration**, so a
stream present for only a small fraction of a long recording is rate-diluted and can be
rejected even if locally dense. Intentional — such a fragment cannot support a whole-flight
ATE — and `--gt-topic` overrides it when you know better.

Full pipeline on a 45 s bag with an injected OptiTrack topic and **no `--gt` flag**:
detected `/vrpn_client_node/nxt3/pose` (5517 poses @ 119.6 Hz, 100% coverage, header
stamps), converged at pass 2, `HEALTHY: deploy`, ATE postg 3.76 cm, mount yaw -3.53 deg /
tilt 3.66 deg taking anchored error 0.230 m -> 0.048 m against a 0.048 m best-fit floor.

**Cross-path consistency** (the decisive check on the GT plumbing): the same 30 s recording
run twice, once with `--gt <file>` and once with the ground truth injected as an in-bag
topic and auto-detected. Both converged at pass 2, `HEALTHY`, ATE evaluated and mount
solved in both `report.json`s. Deltas — chain residual 0.0062 (exactly the same-bag
repeat spread of the estimator itself), ATE postg 2.2 mm apart, mount rotations 0.25 deg
apart. The GT delivery path does not change the result; the residual difference is the
estimator's own run-to-run noise.
