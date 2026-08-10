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
