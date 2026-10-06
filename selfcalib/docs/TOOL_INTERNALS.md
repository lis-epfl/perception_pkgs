# Flight-Data Calibration Tool

Implementation of the paper's §6 tool design. Every component is tested on real study data —
see `VALIDATION_MATRIX.md` for the full validation matrix (fleet PASS / cut-bag FAIL / nxt1 FLAG /
synthetic-smudge localization / fixed-point publish / 6 diagnosis verdicts / cross-flight mount).

## Components
- `chainio.py` — kalibr-chain parse/write, criterion-metric calibration residual, KB4 circle radius.
- `gates.py` — §6.1 pre-flight gates: static-start (IMU), timing-health (drop rate, windowed
  stamp noise, cross-camera desync), per-camera image-health (per-patch max-gradient-over-flight).
- `circlefit.py` — the single temporal accumulation pass shared by the image-center prior
  (mean of the frames that changed > 2.4 %, 15 % binary map, fleet-radius disk argmax) and the
  image-health gate.
- `publish.py` — §3.4/§6 robust-mean publisher (median + chordal rotation mean over harvests);
  kept for the study, the tool now publishes the converged pass itself.
- `evalate.py` — trajectory metrics (toff-corrected; global/POSTG/NOUM), study conventions.
- `mount.py` — §6.2 one-time mount solve (M = R_fit·R_anchor⁻¹ on flight A; apply anchored on B).
- `diagnose.py` — §6.3 three certificates (self-consistency, in-distribution, ATE health) →
  verdict table {HEALTHY / FLY-AGAIN / HARDWARE-CHANGED / PLATFORM-DEFECT}.
- `run_tool.py` — end-to-end driver: window (takeoff + 20 s) → gates → accumulation pass →
  fleet + image-center seed (scenario D) → warm-start until the stopping rule → publish the last
  pass → diagnose.

## One-command usage (new vehicle, zero own calibration)
```bash
python3 tool/run_tool.py --drone <name> --bag <recording> \
    --template <any fleet chain yaml> --imu-chain <imu chain yaml> \
    --out tool/e2e_<name> [--gt <gt.tum>] [--fleet-exclude <name>] [--domain 70]
```
Recording requirements enforced by the gates: starts on the ground with ≥6 s of standstill (3 s for
the estimator to start + 3 s to settle, `--min-still`); ≥15 s of
flight (§4.5 floor); clean lenses (image gate); sane timestamps (timing gate — a flag here
predicts the PLATFORM-DEFECT diagnosis, not a calibration failure).

Outputs in `--out`: `report.json` (gates, circle fit, passes, diagnosis, verdict),
`<drone>_published_chain.yaml` (deployable robust-mean calibration), per-pass harvests and
trajectory estimates.

## Fleet reference data used
`cal/{nxt3,nxt6,nxt10}/theta_star.yaml` (fixed points), `td_center.json` (rig-global toff
centers), `circle_center_*.json` (fit references), `configs/estimator_calib.yaml` +
`run_serial.sh` (deterministic estimator).
