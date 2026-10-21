# relloc — relative localization between drones

relloc estimates, on every drone, where each other drone's VIO world lies in its own VIO world. It is relloc2's "General Route"
(2026-09/10), run live beside the VIO. Every 0.2 s a drone sends a packet of its image features: 500 keypoints per camera, most with depth
from its own motion. Each receiver matches a peer's packet against its own keyframes and solves the relative pose on its GPU. It locks on
two agreeing solves, then maintains the estimate with one solve per second.

Off by default: nothing runs unless `RELLOC=1`.

## Turning it on

On every drone, beside `run_online_msckf`:

```bash
source /opt/ros/humble/setup.bash; source ~/ros2_swarmnxt_ws/install/setup.bash   # rclpy + px4_msgs
set -a; source relloc/config/relloc.env; set +a
export RELLOC=1 RELLOC_FLIGHT_DIR=~/flight_nxt3            # the drone's deploy_vio.py folder
export RELLOC_DEST=192.168.1.106:47800,192.168.1.110:47800 # the other drones (or 255.255.255.255:47800)
export RELLOC_6DOF=1                                        # optional: the 6-DoF output drives the lock (section below)
bash relloc/scripts/run_relloc.sh --check                   # prints the resolved settings and exits
bash relloc/scripts/run_relloc.sh
```

| flag | default | what it does |
|---|---|---|
| `RELLOC` | 0 | 1 = run. Otherwise `run_relloc.sh` exits at once |
| `RELLOC_6DOF` | 0 | 1 = the 6-DoF output decides when an estimate is withdrawn and is the primary `transform`; 0 = the 4-DoF output (relloc2's deployed configuration). Both are published either way |
| `RELLOC_ID` | last number of the hostname | this drone's number, 0-255; it is in every datagram |
| `RELLOC_FLIGHT_DIR` | required | the flight folder: `kalibr_imucam_chain.yaml`, the lens terms (`OV_NONRADIAL` in its `flight_stiffness.env`), camera topics (`cam_topic0..3` of `estimator_flight.yaml`) |
| `RELLOC_DEST` | 255.255.255.255:47800 | where the datagrams go: the other drones as `host:port,...` (unicast), or the broadcast address |
| `RELLOC_PORT` | 47800 | the UDP port this drone listens on |
| `RELLOC_TAKEOFF_HOLD_S` | 10 | seconds after take-off (VIO height 0.3 m above its first pose) before the drone sends packets and solves. relloc2 found every wrong lock of a healthy pair within 8.4 s of the sender's take-off. 0 = from the first VIO pose, also on the ground |
| `RELLOC_XFEAT_ENGINE` | empty (torch) | TensorRT engine of the XFeat backbone, built on the drone: `python3 relloc/scripts/build_xfeat_engine.py` |
| `RELLOC_LOG_DIR` | ~/relloc_logs | one `relloc_<id>_<time>.jsonl` per run |
| `RELLOC_POSE_TOPIC` | /poseimu | the VIO pose (`run_online_msckf`, PoseWithCovarianceStamped; `RELLOC_POSE_MSG_TYPE=nav_msgs/msg/Odometry` for an odometry topic) |
| `RELLOC_IMU_TOPIC` | `imu_topic` of the folder's `estimator_flight.yaml`, else /fmu/out/sensor_combined | the gyro (px4_msgs SensorCombined; `RELLOC_IMU_MSG_TYPE=sensor_msgs/msg/Imu` for an Imu topic). Needed: see "What it needs" |
| `RELLOC_CAM_TOPICS` | from the flight folder | comma list of the four camera topics, if `estimator_flight.yaml` has no `cam_topic0..3` |

Less often needed: `RELLOC_TOPIC` (output prefix, `relloc`), `RELLOC_PEERS` (accept only these drone numbers), `RELLOC_MAX_QUEUE_S` (a
solve that waited longer than this is dropped, 1.0), `RELLOC_SIM_TIME=1` (replays), `RELLOC_TD_S` (camera-IMU time offset; default the
chain's `timeshift_cam_imu`).

## What it needs

- **The VIO running**, with this drone's flight folder. relloc reads the VIO pose (`/poseimu`, about 7.5 poses per second in flight) and
  sends it in every packet header.
- **The PX4 IMU.** The VIO publishes a pose after each filter update. Between two poses the drone turns by up to several degrees, so relloc
  propagates the attitude from the last VIO pose with the gyro, as relloc2 did for every evaluated number. Without the gyro the attitude
  is interpolated between poses: in a 45-s replay of the test below the 4-DoF value of a solve was then off in tilt by 1.0-1.1 deg
  (median), against 0.70-0.73 deg with the gyro on the same 45 s. The census line reports `gyro_attitudes` and `slerp_attitudes`.
- **Python** on the drone: torch with CUDA (the JetPack wheel), kornia, scipy, opencv-python, PyYAML; TensorRT for the engine. No colcon build.
- **A link between the drones.** One drone sends about 43 kB per keyframe (36 datagrams of at most 1,200 B), 214 kB/s = 1.7 Mbit/s, plus
  60-B estimate messages and a 749-B calibration message per second. relloc2's radio specification: residual loss below 5 %, loss bursts
  shorter than 8 datagrams. On Wi-Fi a broadcast frame goes at the network's basic rate, which can be too slow: prefer the unicast list.

## Outputs

Per peer `<id>`, under `relloc/peer<id>/`, while the own lock holds:

| topic | type | content |
|---|---|---|
| `transform` | geometry_msgs/TransformStamped | the primary output (`RELLOC_6DOF`): `global` <- `peer<id>/global`. It maps a point of the peer's VIO world (e.g. its `/poseimu` position) into this drone's VIO world |
| `transform_4dof` | TransformStamped | rotation about gravity + translation (relloc2's deployed output) |
| `transform_6dof` | TransformStamped | full rotation + translation |
| `position` | geometry_msgs/PointStamped | the peer's position in its last packet header (5 Hz), mapped by `transform`; stamped with the peer's keyframe time on this drone's clock |
| `relloc/status` | std_msgs/String (JSON, 1 Hz) | per peer: status, confirmed, both estimates, the relative tilt of the 6-DoF one, lock and withdraw counts |

Status: `searching`; `provisional` (own lock); `confirmed` (the peer's own, independent estimate agrees within 20 cm / 2 deg); `adopted`
(the peer locked first and its estimate is shown inverted, in the status only). relloc2's rule for consumers: act on `confirmed`, display
or seed from `provisional`. The transforms are published when they change (one maintenance solve per second). `position` lags the peer
by the packet's age (about 0.1-0.3 s); a consumer that needs the peer's current position needs its pose stream (relloc2: a 10-Hz pose +
velocity stream, or a velocity in the packet header; not part of this package).

The log has one JSON object per line: `own_kf` (every keyframe: pose, 3-D points, timing), `peer_kf` (every received header),
`solve` (every solve: inliers, margin, 4-DoF and 6-DoF values), `lock`, `withdraw`, `output`, and a `census` every 5 s. With the drones'
motion capture this is enough to grade a flight offline.

## The 6-DoF output

The 4-DoF output takes the two VIO worlds to share gravity exactly, so its tilt error is the two VIOs' tilt disagreement. The solver
already estimates the relative roll and pitch in every solve (REFINE6). The 6-DoF output keeps them: lock = chordal mean of the lock
members' rotations, maintenance = geodesic 0.7 blend, withdraw test on the yaw difference (relloc2 round 23, `review23/o6maint`, variant
ship1_6y). relloc2's replays found lower error only while the VIOs disagree by more than REFINE6's own tilt error: e_1m lower by
0.6-1.1 cm (median) in the first 30 s after take-off on the June flights; no gain later in a flight, and none on TUM-VI or EuRoC.

With `RELLOC_6DOF=0` the lock, the maintenance and the withdrawals are relloc2's 4-DoF ones, and the 6-DoF estimate follows them as a
shadow. With `RELLOC_6DOF=1` the 6-DoF maintenance decides the withdrawals and the 4-DoF estimate is the shadow. A shadow skips a solve
that its own test would withdraw on (counted as `n_shadow_withdraw`). Both outputs therefore come from the same solves in every flight.

## Layout

```
relloc_node.py        the node
live/                 new code: sender.py (the sender of relloc2's sender_lean.py as a class), receiver.py (node15.Node + the 6-DoF
                      estimate), six.py, vio.py (pose at the image instant), calib.py (flight folder, calibration datagram), lens.py,
                      transport.py (UDP)
core/                 relloc2 modules, byte-identical copies (md5 below): the evaluated receiver, lock rule, codec
third_party/          XFeat + LighterGlue (verlab/accelerated_features e92685f, Apache-2.0: modules and weights unchanged)
config/relloc.env     the flags
scripts/              run_relloc.sh, build_xfeat_engine.py
test/                 test_units.py, test_sender_parity.py, play_pair.py (replay of recordings for the node)
```

`core/` = relloc2's `orin_pkg/` (and `rule15.py`) of 2026-10-21, md5 (first 12 digits): `lg_trt.py` e28ac6ce6d5f, `lggraph.py`
0af5173cf243, `node15.py` 7a4f9339c890, `orin_dump.py` 9099bc2efefd, `packet_codec.py` 736ce588e7ef, `rig15.py` ac8fd1064250, `rule15.py`
75f995f0507a, `score_jit.py` 6440e24dae03, `solver.py` eca1e7c2ad9a, `xfeat_trt.py` 157675155eda, `data/desc_pca_tumvi.npz` 2e4d21ed009b,
`data/xfeat_b4_800x1280.onnx` 3c7e56cbba02. `data/desc_pca_tumvi.npz` is the PCA basis of the descriptors, `data/xfeat_b4_800x1280.onnx`
the XFeat backbone for TensorRT.

What the live code does differently from relloc2's evaluated replay:

- Features: XFeat runs on the four images in one batch (relloc2 ran one image per call): keypoint sets agree to Jaccard 0.998-0.999.
  The TensorRT backbone agrees with torch to Jaccard 0.986-0.991 (relloc2 measured 0.99).
- Attitude: gyro propagation with a gyro bias estimated online from consecutive VIO poses (relloc2 used the run's final bias); position
  interpolated linearly between two VIO poses (relloc2: cubic spline over the whole trajectory).
- Lens terms: applied to every ray, own and peer (relloc2 round 21 measured 0.3-0.7 cm higher error when a calibration with lens terms
  is read without them). The terms travel in the calibration datagram (new: 749 B, once per second, with a random session number; a
  receiver that sees the session change treats the peer as new, e.g. after a restart of its relloc).
- Times: a keyframe's time (packet header and own keyframes) is its image stamp + t_d, the instant on the IMU clock, which is the VIO's
  and the system's clock (relloc2 used the image log time). The gyro is integrated on the IMU payload stamps, as the VIO does (relloc2's
  replay used their log time). node15 maps a peer's header time to the own clock with the minimum of (arrival - header time) minus the
  minimum packet latency; that latency is the own sender's measured minimum (relloc2: 0.09 s fixed; live it includes the wait for the
  VIO pose, 60-75 ms in the replay).
- VIO epoch: the epoch in the packet headers and in the estimate messages is the one of the pose stream (+1 when it pauses for more than
  2 s or goes back in time); node15 counted its own resets.
- A solve that waited in the queue longer than 1 s is dropped (the replay's queue had no limit).
- The take-off hold (relloc2 recommended it in DEPLOY_CONFIG; its replays did not apply it).
- Rig: drones of the fleet only (four cameras, 1280 x 800, KB4). Peers of another rig are ignored.

## Tested

On the workstation (RTX 5070, ROS 2 Humble), 2026-10-22:

- `test/test_sender_parity.py`: the sender class builds the same packets as relloc2's `sender_lean.py`, byte for byte, on its sequence
  file (40 keyframes of nxt3, 23 June; SHA-256 216e2fed6c33b515).
- `test/test_units.py`: 6-DoF composition = the direct 6-DoF transform (1e-15); at zero tilt the 6-DoF lock and maintenance equal the
  4-DoF ones; rays with lens terms = relloc2 round 21 (`md21.py`) bit for bit on 20,000 pixels; calibration datagram round trip; the
  node's epoch follows the header epoch.
- An independent review of the new code before the final runs found two faults, both fixed: the epoch of the estimate messages could
  differ from the packet headers' after a VIO restart (the peer then drops every estimate message and nothing is ever confirmed), and a
  restarted peer process was not recognised (now: session number in the calibration datagram).
- Replay: two drones of 23 June (s3: nxt3 and nxt6) through two nodes on one workstation, UDP on localhost, ROS 2 sim time
  (`test/play_pair.py`: the recorded images at 30 Hz, the recorded PX4 IMU, the VIO trajectories of relloc2's registry as `/poseimu`;
  relloc2's calibrations of these flights, without lens terms). Graded offline against motion capture with relloc2's physical mounts
  (e_t: where the receiver places the peer's IMU in its current VIO frame against where it is):

| 23 June, s3, 160 s of sending per drone (nxt3 <- nxt6 / nxt6 <- nxt3) | 4-DoF primary, torch XFeat | 6-DoF primary, TensorRT XFeat |
|---|---|---|
| first lock after the drone's first packet | 0.33 / 0.47 s | 0.31 / 0.42 s |
| withdrawals (each followed by a re-lock) | 0 / 1 | 0 / 2 |
| share of the locked time with status `confirmed` | 100 / 99.9 % | 100 / 98.4 % |
| e_t, median / 90th percentile | 3.3 / 6.5 cm, 3.5 / 7.9 cm | 3.3 / 6.9 cm, 3.4 / 7.0 cm |
| e_1m (= e_t + 1 m x rotation error), median, 4-DoF / 6-DoF output | 4.1 / 4.1 cm, 4.3 / 4.3 cm | 4.2 / 4.1 cm, 4.2 / 4.1 cm |
| tilt error, median, 4-DoF / 6-DoF, first 30 s of sending (10-40 s after take-off) | 0.59 / 0.38 deg, 0.57 / 0.40 deg | 0.61 / 0.41 deg, 0.61 / 0.44 deg |
| e_1m, median, 4-DoF / 6-DoF, same 30 s | 4.6 / 4.2 cm, 5.5 / 5.2 cm | 4.2 / 3.8 cm, 4.7 / 4.3 cm |
| tilt error, median, 4-DoF / 6-DoF, after 60 s of sending | 0.34 / 0.39 deg, 0.32 / 0.37 deg | 0.35 / 0.41 deg, 0.35 / 0.38 deg |
| e_t > 20 cm, share of graded headers | 0.4 / 0.0 % | 0.0 / 1.7 % |
| keyframes sent (5 Hz = 800) | 712 / 732 | 714 / 730 |
| sender per keyframe, median / 90th pct (decode + XFeat + triangulation + packet) | 69-71 / 86-89 ms | 63-66 / 80-84 ms |
| wait for the VIO pose after the image, median | 74-79 ms | 73-78 ms |
| solve, median / 90th pct (two nodes share one GPU) | 72-74 / 118-121 ms | 72-73 / 112-114 ms |

Both runs replay every camera frame (30 Hz); 89-91 % of the 5-Hz keyframes went out, the others lacked one of the four images at the
node (localhost DDS dropped 6-16 % of the 0.2-0.4-MB best-effort image messages). Graded: every received peer header while locked, 708-729
per direction. The 6-DoF output is better in the first 30 s of sending (tilt 0.16-0.21 deg lower, e_1m 0.3-0.4 cm lower) and slightly
worse after 60 s (tilt 0.03-0.06 deg higher), as in relloc2 round 23. One run per setting: the differences between the two columns
(engine, primary output) are within run-to-run scatter and do not compare the two settings. Per solve (at the solve's own keyframes, 4-DoF
primary run, 112-118 solves per direction) the tilt error is 0.36-0.37 deg for the 4-DoF value and 0.26-0.29 deg for the 6-DoF one.

Not tested: on the Orin; over a radio link; with the live VIO (the replay feeds recorded VIO trajectories); with three or more drones; on
recordings made after 15 Oct (capture stamps, calibrations with lens terms); a VIO restart in flight.

## Limits

- Compute on the Orin is not measured. On the workstation (RTX 5070) the sender takes 63-71 ms per keyframe and a solve 72-74 ms
  (medians, two nodes on one GPU). relloc2's device times were measured at 918 MHz; the fleet's GPU cap is 612 MHz. A sender slower than 0.2 s skips keyframes (census:
  `kf_dropped_busy`); the TensorRT engine is the first lever.
- The packet header carries the peer's pose at its keyframe; `position` is therefore 0.1-0.3 s old.
- No time synchronisation is needed between drones (each receiver estimates the clock offset from the packet headers), but the camera
  stamps, the VIO stamps and the IMU stamps of one drone must share its clock, as the VIO requires anyway.

## Replaying recordings

One node per drone, each with its own `RELLOC_ID`, flight folder, ports and the player's topics, then the player:

```bash
export ROS_DOMAIN_ID=87 RELLOC_SIM_TIME=1 RELLOC=1
RELLOC_ID=3 RELLOC_FLIGHT_DIR=<folder nxt3> RELLOC_PORT=47803 RELLOC_DEST=127.0.0.1:47806 RELLOC_TOPIC=nxt3/relloc \
  RELLOC_CAM_TOPICS=/nxt3/cam0/compressed,/nxt3/cam1/compressed,/nxt3/cam2/compressed,/nxt3/cam3/compressed \
  RELLOC_POSE_TOPIC=/nxt3/poseimu RELLOC_IMU_TOPIC=/nxt3/fmu/out/sensor_combined bash relloc/scripts/run_relloc.sh &
# ... the same for nxt6 (RELLOC_ID=6, ports swapped, /nxt6/...)
python3 relloc/test/play_pair.py --all-frames --drone ns=nxt3,mcap=<cams.mcap>,traj=<vio.tum>,bag=<bag.mcap>,toff=<t_d> --drone ns=nxt6,...
```

`traj` is the estimator's trajectory in camera time (`toff` = its t_d shifts it to the IMU time of `/poseimu`). For recordings made
before 15 Oct (arrival stamps) pass `--imu-stamp log`, which reproduces how relloc2 aligned the gyro with those recordings.
