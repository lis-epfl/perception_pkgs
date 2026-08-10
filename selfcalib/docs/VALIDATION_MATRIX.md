# Calibration-Tool Component Tests (2026-07-07)

Implements and validates every §6 component of the paper. All tests on real study data unless
marked synthetic. Thresholds cited here are the ones shipped in the code.

## 1. Static-start gate (`gates.static_start_gate`)
Requires ≥1.5 s contiguous static IMU (0.5 s bins: gyro std < 0.03 rad/s, accel std < 0.20 m/s²
per axis) within the first 12 s.

| recording | expected | result |
|---|---|---|
| bags/nxt3_raw_4cam_basin | PASS | PASS (12.0 s static) |
| bags/nxt6_raw_4cam | PASS | PASS (12.0 s) |
| bags/nxt10_raw_4cam | PASS | PASS (7.0 s) |
| nxt1 s2 / s3 | PASS | PASS (8.5 / 9.5 s) |
| bags/nxt3_raw_4cam (cut: static phase removed) | **FAIL** | **FAIL** (0.0 s static) |

The cut bag is the real negative control from the study's bag-swap incident (§7 round 5).

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
ever exceeds 25 during the flight; camera flagged if >5 % of in-circle patches stay dead.
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
gates → one accumulation pass (circle fit + image health) → leave-one-out fleet seed + own
circle fit (scenario D, zero own calibration) → warm-start to self-consistency → robust-mean
publish → diagnosis.

| run | result |
|---|---|
| nxt3, full bag, --fleet-exclude nxt3, GT attached | all gates PASS; converged at pass 2 (resid 0.0064, Δtoff 0.00 ms); published toff −0.039256 (population center −0.039251); ATE POSTG **3.61 cm**; verdict **HEALTHY: deploy** |
| nxt1 s2 (defect path) | static PASS, **timing FLAG** (8.07 % drops / 37.1 ms noise, run proceeds with warning), image PASS; zero-own-calibration seed converged at pass 2; published toff **−0.054584** (independently re-derives the study's triple-confirmed −0.054 from a fleet seed 15 ms away); diagnosis: self-consistent 0.0103, in-distribution, ATE POSTG 0.884 m unhealthy → verdict **PLATFORM-DEFECT — repair, not recalibrate** [+ timing-gate consistency note] |
