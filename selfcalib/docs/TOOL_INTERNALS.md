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
  image-health gate. `image_pass()` runs it with one process per camera. The tool gives it the
  run's window, cut at 30 s after takeoff when the window is longer (`--image-after-takeoff`;
  with 0, or a recording without a takeoff, it reads the whole window). It also hands back the
  mean image of each camera, for `wmask.py`.
- `wmask.py` — the fisheye mask of the calibration passes when they run the bearing model
  (since 2026-11-07): per camera, where the lens's bright image circle ends on the mean image,
  seen from the principal point of the calibration a pass starts from (radial profiles every
  0.5 deg, 380-500 px, robust circle fit), and `mask_fisheye_theta_max` = the smallest
  end radius / focal length over the cameras that share the global key (camera 3 has its own),
  so that the estimator's disk ends 4 px inside the circle. The rule and its checks are in the
  module's docstring.
- `publish.py` — §3.4/§6 robust-mean publisher (median + chordal rotation mean over harvests);
  kept for the study, the tool now publishes the converged pass itself.
- `evalate.py` — trajectory metrics (toff-corrected; global/POSTG/NOUM), study conventions.
- `mount.py` — §6.2 one-time mount solve (M = R_fit·R_anchor⁻¹ on flight A; apply anchored on B).
- `diagnose.py` — §6.3 three certificates (self-consistency, in-distribution, ATE health) →
  verdict table {HEALTHY / FLY-AGAIN / HARDWARE-CHANGED / PLATFORM-DEFECT}.
- `run_tool.py` — end-to-end driver: window (takeoff + 20 s) → gates → accumulation pass →
  fleet + image-center seed (scenario D) → warm-start until the stopping rule → publish the last
  pass → diagnose. The passes run the bearing model with the fitted mask unless `--no-bearing`
  (README, "Bearing model and fitted fisheye mask").

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
`<drone>_published_chain.yaml` (deployable robust-mean calibration), `<drone>_published_terms.txt`
(its two lens terms per camera, part of the calibration; absent with `--no-lens-terms`), per-pass
harvests and trajectory estimates.

## Fleet reference data used
`cal/{nxt3,nxt6,nxt10}/theta_star.yaml` (fixed points), `td_center.json` (rig-global toff
centers), `circle_center_*.json` (fit references), `configs/estimator_calib.yaml` +
`run_serial.sh` (deterministic estimator).
The toff centers are 0 s since 2026-10-15, when the camera driver started stamping the images
with their capture time instead of their arrival (about 45 ms later); the centers measured on
arrival-stamped recordings stay in the file as `center_arrival_stamps`.
