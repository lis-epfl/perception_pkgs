# Protocol definitions — the exact semantics behind the tool

These are the definitions the study's ~10,000-chain measurement campaigns used; the tool
implements the deployment subset. Precise versions matter when you extend or re-verify.

## Warm-start pass
One full run of the estimator over the recording from a seed calibration, with online
calibration of camera intrinsics, camera–IMU extrinsics, and time offset enabled. The run's
final calibration state (harvest) becomes the next pass's seed — cameras always; t_d is
carried by the tool's loop via the chain's `timeshift_cam_imu`. Deterministic per
(bag, config, RNG seed): re-running reproduces bit-identical output.

## Fixed point θ\* and the settled oscillation
Iterating passes settles into a stationary oscillation around a fixed point θ\*. The
oscillation's width is the platform's end-to-end repeatability (clean industrial rigs:
sub-pixel; noisy racing platforms: several pixels, degrees, centimeters). θ\* is
observability-equivalent to ground truth, not identical to it — validated externally by
trajectory accuracy and by landing on independently produced target-based calibrations.

## Convergence envelope (the box criterion)
Per parameter type (principal point px / focal % / distortion abs / rotation deg /
translation cm / t_d ms), a limit ≈1.5–2× the settled oscillation's maximum deviation:
"converged" = every parameter of the final harvest within its limit of θ\*.
**Sample-size rule (hard-earned): estimate the oscillation from ≥16 settled passes.**
Fewer samples undershoot the tail on noisy platforms and produce a radius-independent
~2–5% false-failure floor whose signature is residuals hovering at 1–3× the limit
(TROUBLESHOOTING §3). Focal is limited in percent because a focal error acts
proportionally on the image for any central camera model; t_d limits are gate-insensitive
in 1–10 ms because its failure mode is bimodal (recovery <1 ms or frozen at seed).

## Self-consistency stop (the tool's deployment criterion)
The loop stops when two consecutive harvests agree to 0.10 in the width-normalized residual
metric and <10 ms in t_d. This is the ground-truth-free surrogate for "reached θ\*"; from
fleet-quality seeds it fires after 1–2 passes.

## Basin, ruler, certificate (context for the numbers you may see quoted)
Per-axis tolerance half-widths W\* are measured by geometric-scan + bisection ladders
(magnitudes ×1.6, bracket to 35%, midpoint; ±15% is as fine as the graded edge permits).
The joint seed radius r is the Euclidean norm of per-axis errors each divided by its W\*.
Certified radii are randomized certificates: N fresh uniformly-drawn 15-dimensional
directions at radius r, zero failures, bounding the failing direction fraction below
~3/N at 95% confidence (Clopper–Pearson; N fixed in advance). They are per-platform,
per-protocol numbers — re-measure on new platforms; do not transfer constants.
Practical seeding targets measured across four platforms: fleet-transfer + image-circle
seeds land at r ≈ 0.02–0.07 versus certified radii of 0.25–0.40 — margins of 4–20×.

## t_d (time offset) — the exception
Beyond the visual–inertial correlation window (≈±0.1 s on well-excited platforms), the
update has no gradient in t_d: the estimate freezes at its seed exactly (identical digits,
unlimited passes). No graded zone, no recovery by iteration. t_d must be SUPPLIED within
≈±0.1 s (hardware timestamps or fleet transfer, both <1 ms in practice) — after which the
estimator polishes it to sub-millisecond online. Joint error tightens the cliff further on
noisy platforms.

## Diagnosis certificates (verdict.json)
1. **Self-consistency**: consecutive-harvest agreement (above).
2. **In-distribution**: published calibration within the fleet population's per-parameter
   spread (requires `fleet_reference/`; leave-one-out when the vehicle is itself a member).
3. **ATE health** (only with `--gt`): frozen-deployment trajectory error within the fleet's
   healthy band.
Verdict table: all pass → HEALTHY; gates failed → FLY-AGAIN; (1) pass + (2) fail →
HARDWARE-CHANGED; (1)+(2) pass + (3) fail or timing gate flagged → PLATFORM-DEFECT.
