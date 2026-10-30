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
set -a; source relloc/config/relloc.env; set +a              # ROS 2 between the drones, in domain 100 (section "Between the drones")
export RELLOC=1 RELLOC_FLIGHT_DIR=~/flight_nxt3            # the drone's deploy_vio.py folder
export RELLOC_6DOF=1                                        # optional: the 6-DoF output drives the lock (section below)
bash relloc/scripts/run_relloc.sh --check                   # prints the resolved settings and exits
bash relloc/scripts/run_relloc.sh
```

Raw UDP instead of ROS 2: `export RELLOC_TRANSPORT=udp RELLOC_DEST=192.168.1.106:47800,192.168.1.110:47800` (the other drones, or
255.255.255.255:47800).

| flag | default | what it does |
|---|---|---|
| `RELLOC` | 0 | 1 = run. Otherwise `run_relloc.sh` exits at once |
| `RELLOC_6DOF` | 0 | 1 = the 6-DoF output decides when an estimate is withdrawn and is the primary `transform`; 0 = the 4-DoF output (relloc2's deployed configuration). Both are published either way |
| `RELLOC_ID` | last number of the hostname | this drone's number, 0-255; it is in every datagram |
| `RELLOC_FLIGHT_DIR` | required | the flight folder: `kalibr_imucam_chain.yaml`, the lens terms (`OV_NONRADIAL` in its `flight_stiffness.env`), camera topics (`cam_topic0..3` of `estimator_flight.yaml`) |
| `RELLOC_TRANSPORT` | ros2 | how the datagrams travel between the drones: `ros2` (one best-effort ROS 2 message per datagram on `/relloc/dg/d<RELLOC_ID>`) or `udp` (raw UDP to `RELLOC_DEST`). Section "Between the drones" |
| `RELLOC_ROS_DOMAIN` | 100 |  ros2: the ROS domain of the datagram topics, joined by every drone's relloc and nothing else; empty = the node's own domain, and the fleet's domain_bridge must forward the topics |
| `RELLOC_ROS_DEPTH` | 128 | ros2: history depth of the datagram topics (at least 72: two bursts of 36 datagrams) |
| `RELLOC_DEST` | 255.255.255.255:47800 | udp: where the datagrams go: the other drones as `host:port,...` (unicast), or the broadcast address |
| `RELLOC_PORT` | 47800 | udp: the port this drone listens on |
| `RELLOC_TAKEOFF_DZ` | 0.5 | the take-off latch: VIO height in m above its first pose. Every keyframe from the first VIO pose is processed (the first keyframes sent have triangulation partners from the ground and the start of the climb); packets leave, and the drone's own keyframes enter its receiver, from the latch on. -1 = no latch: the drone sends from its first VIO pose, also on the ground |
| `RELLOC_TAKEOFF_HOLD_S` | 0 | seconds after the latch before the drone sends packets and solves |
| `RELLOC_PARMIN_EARLY_DEG` | 8 | parallax floor of every triangulated point until the latch + `RELLOC_PARMIN_EARLY_S` (section "Take-off and parallax floor") |
| `RELLOC_PARMIN_EARLY_S` | 8 | seconds after the latch at which the floor changes to `RELLOC_PARMIN_DEG` |
| `RELLOC_PARMIN_DEG` | 6 | the floor after the change |
| `RELLOC_MEM_GATE_DEG` | 8 | a keyframe becomes a camera's depth memory only when at least 100 of its points reach this parallax, and only those points are stored. 0 = no gate |
| `RELLOC_XFEAT_ENGINE` | empty (torch) | TensorRT engine of the XFeat backbone, built on the drone: `python3 relloc/scripts/build_xfeat_engine.py` |
| `RELLOC_LOG_DIR` | ~/relloc_logs | one `relloc_<id>_<time>.jsonl` per run |
| `RELLOC_POSE_TOPIC` | /poseimu | the VIO pose (`run_online_msckf`, PoseWithCovarianceStamped; `RELLOC_POSE_MSG_TYPE=nav_msgs/msg/Odometry` for an odometry topic) |
| `RELLOC_IMU_TOPIC` | `imu_topic` of the folder's `estimator_flight.yaml`, else /fmu/out/sensor_combined | the gyro (px4_msgs SensorCombined; `RELLOC_IMU_MSG_TYPE=sensor_msgs/msg/Imu` for an Imu topic). Needed: see "What it needs" |
| `RELLOC_CAM_TOPICS` | from the flight folder | comma list of the four camera topics, if `estimator_flight.yaml` has no `cam_topic0..3` |

Less often needed: `RELLOC_TOPIC` (output prefix, `relloc`), `RELLOC_PEERS` (accept only these drone numbers; with ros2 also the only
peer topics subscribed, otherwise every `/relloc/dg/d<k>` seen is subscribed, looked up every 2 s), `RELLOC_MAX_QUEUE_S` (a
solve that waited longer than this is dropped, 1.0), `RELLOC_SIM_TIME=1` (replays), `RELLOC_TD_S` (camera-IMU time offset; default the
chain's `timeshift_cam_imu`).

## What it needs

- **The VIO running**, with this drone's flight folder. relloc reads the VIO pose (`/poseimu`, about 7.5 poses per second in flight) and
  sends it in every packet header. The build must publish poses on the ground (at or after dc8d058: `ros2 topic hz /poseimu` about 28 Hz
  standing): without them the node has no ground keyframes, and the first keyframes of the climb have no triangulation partners. The
  replays used a build with the slow-flight switches (a8808ea: `strings <ws>/install/ov_msckf/lib/libov_msckf_lib.so | grep OV_MSCKF_ROWS`).
- **A flight folder calibrated with the drone's current camera driver.** relloc takes every pose at image stamp + the chain's t_d. Since
  15 Oct the driver stamps the capture time (t_d about 0); the June calibrations have -39 to -47 ms. A folder calibrated on an
  arrival-stamped recording and flown with the capture-stamp driver puts every pose about 40 ms off (4 cm at 1 m/s), in relloc and in
  the VIO. `run_relloc.sh --check` prints `td_s`: a few ms with the current driver.
- **The PX4 IMU.** The VIO publishes a pose after each filter update. Between two poses the drone turns by up to several degrees, so relloc
  propagates the attitude from the last VIO pose with the gyro, as relloc2 did for every evaluated number. Without the gyro the attitude
  is interpolated between poses: in a 45-s replay of the test below the 4-DoF value of a solve was then off in tilt by 1.0-1.1 deg
  (median), against 0.70-0.73 deg with the gyro on the same 45 s. The census line reports `gyro_attitudes` and `slerp_attitudes`.
- **Python** on the drone: torch with CUDA (the JetPack wheel), kornia, tqdm (imported by XFeat), scipy, opencv-python, PyYAML; TensorRT
  for the engine. No colcon build. `--check` does not import torch, kornia, tqdm or TensorRT: check them with
  `python3 -c 'import torch, kornia, tqdm; print(torch.cuda.is_available())'`.
- **A restart before every take-off.** The latch fires once per process: after a landing the drone would keep sending ground packets
  from its descent. Check that one relloc runs per drone (`pgrep -af relloc_node.py`): a second process with the same drone number
  breaks every pair with that drone without an error.
- **A link between the drones.** One drone sends about 43 kB per keyframe (36 datagrams of at most 1,200 B), 214 kB/s = 1.7 Mbit/s, plus
  60-B estimate messages and a 749-B calibration message per second; over ROS 2 about 6-8 % more airtime (section "Between the drones").
  relloc2's radio specification: residual loss below 5 %, loss bursts shorter than 8 datagrams. With UDP: on Wi-Fi a broadcast frame goes
  at the network's basic rate, which can be too slow: prefer the unicast list.

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

The log has one JSON object per line: `own_kf` (every keyframe: pose, height above the first pose, 3-D points, floor, timing, sent or
not), `takeoff` (the latch), `floor` (each change of the floor), `peer_kf` (every received header), `solve` (every solve: inliers,
margin, 4-DoF and 6-DoF values), `lock`, `withdraw`, `output`, and a `census` every 5 s. With the drones' motion capture this is enough
to grade a flight offline.

## Between the drones

Every datagram travels on its own, best effort: per keyframe 36 of at most 1,200 B, a 60-B estimate message 5 times per second, a 749-B
calibration message once per second. A lost datagram costs the 54-56 keypoints it carries, never the keyframe. `RELLOC_TRANSPORT` picks
the carrier:

- `ros2` (relloc.env): one ROS 2 message per datagram, `std_msgs/msg/UInt8MultiArray`. Each drone publishes `/relloc/dg/d<RELLOC_ID>` and
  subscribes to `/relloc/dg/d<k>` of every peer: the drones of `RELLOC_PEERS`, or every such topic it sees (looked up every 2 s). QoS
  best effort, keep last `RELLOC_ROS_DEPTH` (128), volatile, on both sides.
- `udp`: one UDP datagram per datagram to every address of `RELLOC_DEST`, received on `RELLOC_PORT` (the previous default).

relloc2's study of ROS 2 for this traffic (review24/ros2: two network namespaces with netem loss and rate limits, Fast DDS 2.6.11 as in
the fleet) set the design:

- One best-effort message per datagram delivered the same share of keypoints as raw UDP under every loss condition: 89.1-89.8 % against
  89.9-90.4 % at 10 % independent loss; 0.40 % of the keyframes lost against 0.41 % with 10 % loss in bursts. It adds no loss, and no
  reliability either.
- One message per keyframe (43 kB, the usual ROS way) is lost with any of its 30-34 fragments: at 10 % loss 1.6-3.3 % of the keyframes
  arrived. RELIABLE with default timers does not rescue it, because its repairs arrive after the 0.19-s decode window. Keep one message
  per datagram and best effort.

Domains (`RELLOC_ROS_DOMAIN`):

- `100` (relloc.env): the datagram topics live in domain 100, in a second DDS participant inside relloc's process. Every drone's relloc
  joins it and nothing else does; the cameras, the VIO and PX4 stay in the drone's own domain (its number). One domain shared by
  everything would mix the drones' identically named topics (`/fmu/in/*`, `/poseimu`, the cameras). No bridge is needed. Why 100: it is
  no drone number and not the ground station's 0, and it lies within 0-101, the IDs whose DDS ports (for 100: 32400-32649) stay below
  Linux's ephemeral ports. Drone numbers must be unique within it: two groups flying on one network need two values.
- empty: the topics live in each node's own domain and the fleet's domain_bridge forwards them, as it does `traj_full`. In swarm-nxt's
  `ansible/templates/drone_bridge_config.yaml.j2`, beside the `traj_full` loop (needs best effort and a depth of at least 72; not tested,
  domain_bridge is not installed on the test workstation):

```yaml
{% for drone in groups['drones'] %}
{% set drone_num_id = drone | regex_replace('\.local', '') | regex_search('[0-9]+$') %}
{% if loop.index0 != current_drone_idx %}
  /relloc/dg/d{{ drone_num }}:
    type: std_msgs/msg/UInt8MultiArray
    to_domain: {{ drone_num_id }}
    qos:
      reliability: best_effort
      durability: volatile
      history: keep_last
      depth: 128
{% endif %}
{% endfor %}
```

What ROS 2 costs against raw UDP:

- **Latency:** the study measured +1.0-1.4 ms per keyframe at 24 Mbit/s and +3.6-5.6 ms at 6 Mbit/s: the time to send the extra headers.
  This version on the workstation, over loopback UDP (relloc2 `review24/ros2tx` bench): a back-to-back burst is complete after 0.79-0.82
  ms against 0.25 ms; with the datagrams spaced as at 24 Mbit/s each one arrives 0.035 ms after its send against 0.01 ms.
- **Airtime:** 1.10-1.12 times the payload against 1.036, +6-8 %: the RTPS and CDR headers add 88-91 B per datagram.
- **Receive CPU per peer:** the study's receiver, one callback per message, took 15.4 ms per keyframe on the Orin with the datagrams
  spaced at 24 Mbit/s (7.7 % of one core), against 0.91 ms (0.46 %) for UDP. The receive thread here waits on a bare wait set, drains every
  subscription after each wake-up and slices the payload out of the serialized message. On the workstation it takes 2.1 ms per keyframe
  spaced and 0.85 ms in bursts, against 0.7 and 0.3 ms for UDP; the study's receiver took 8.3 and 5.7 ms there. Not measured on the Orin;
  scaled by the study's Orin / workstation ratio (1.85, spaced), about 4 ms per keyframe, 2 % of one core per peer (an estimate).
- **Sender:** 2.7 against 0.48 ms per keyframe on the Orin (study); here 1.7 against 1.1 ms on the workstation (bursts).
- **Discovery:** Fast DDS's default, multicast on the Wi-Fi. In the study (Fast DDS and Cyclone) every one of 298 trials found the
  peer; under loss Fast DDS's first usable keyframe came after 1.06 s (median), 5.06 s at most. Without RELLOC_PEERS the node looks for
  peer topics every 0.25 s for its first 10 s, then every 2 s.
- **Not tested, to check on the drones before a flight relies on it:**
  - rclpy's publish keeps Python's interpreter lock while Fast DDS sends (synchronous publication). If the send buffer fills on a
    congested radio, every relloc thread waits; raw UDP gives up after 0.2 s. On the drones' Wi-Fi with three drones, compare with
    `RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS`.
  - A receive thread that stalls for more than ~0.7 s loses the oldest datagrams of each peer (depth 128) without a counter.
  - Outages longer than Fast DDS's 20-s lease: relloc2's outage study used UDP.
  - The study's 1,400-B cap for discovery messages over a lossy radio is not applied (Fast DDS defaults).

The wait set is rclpy's internal `_rclpy.WaitSet`, built as rclpy's own executor builds it. If an rclpy lacks it, the node falls back to
an executor (`receive` in the start log's `transport` event) with the same drain, at the higher CPU cost.

In flight, the census line shows `ros2_rx`, `ros2_tx`, `ros2_drop`, `ros2_err` and `ros2`: the subscribers matched to the own topic
(`readers`: one per peer that listens to this drone) and, per subscribed topic, the datagrams received with their DDS latency (reception
minus source time: the one-way delay when the drones' clocks agree) and the take lag of the receive thread, p50 / p90 / max in ms.
`rx_peer` gives, per peer and for both transports, the datagrams received by kind (`kf`, `est`, `cal`) and node15's drops (`late`: after
the keyframe was decoded; `dup`, `stale`, `bad`).

## The 6-DoF output

The 4-DoF output takes the two VIO worlds to share gravity exactly, so its tilt error is the two VIOs' tilt disagreement. The solver
already estimates the relative roll and pitch in every solve (REFINE6). The 6-DoF output keeps them: lock = chordal mean of the lock
members' rotations, maintenance = geodesic 0.7 blend, withdraw test on the yaw difference (relloc2 round 23, `review23/o6maint`, variant
ship1_6y). relloc2's replays found lower error only while the VIOs disagree by more than REFINE6's own tilt error: e_1m lower by
0.6-1.1 cm (median) in the first 30 s after take-off on the June flights; no gain later in a flight, and none on TUM-VI or EuRoC.

With `RELLOC_6DOF=0` the lock, the maintenance and the withdrawals are relloc2's 4-DoF ones, and the 6-DoF estimate follows them as a
shadow. With `RELLOC_6DOF=1` the 6-DoF maintenance decides the withdrawals and the 4-DoF estimate is the shadow. A shadow skips a solve
that its own test would withdraw on (counted as `n_shadow_withdraw`). Both outputs therefore come from the same solves in every flight.

## Take-off and parallax floor

The sender triangulates its keypoints against its own past keyframes; a point counts only when its views differ by at least a parallax
floor. Until this change the floor was 3 deg and the node waited 10 s after take-off. relloc2's review24 traced the early wrong solves
to the sender's hover map: in the hover after the climb no past keyframe has the 8-cm baseline, so every 3-D point the sender sends comes
from its depth memory, one stored keyframe from the end of the climb. At a 3-deg floor those points had a median parallax of 3.8-5.6 deg
(13-15 deg in flight) and depth errors of 4.3-6.9 % (2.2-2.5 % in flight; `review24/early`). The rule PT8F6G8 (`review24/fix` section 8):

- floor 8 deg for each drone's first 8 s after its take-off latch, then 6 deg;
- a keyframe becomes a camera's memory only when at least 100 of its points reach 8 deg, and only those points are stored;
- a memory depth never replaces a fresh one (before, it could);
- keyframes are processed from the first VIO pose and sent from the latch (0.5 m above the first pose), with no hold.

relloc2's replay of the seven June flights against the previous setting (floor 3 deg with the memory fix; 10 healthy drone pairs,
seeds 0-2, the same requests for both rules; wrong = more than 20 cm off; lock-eligible = accepted with a z-margin of at least 2.7, a
solve the lock rule can use; public datasets: round 16's setup):

| | floor 3 deg | PT8F6G8 |
|---|---|---|
| lock-eligible wrong solves in the sender's first 8 s, of ~28,000 accepted | 344 | 1 |
| lock-eligible wrong solves after the sender's first 20 s, of ~132,000 accepted | 189 | 105 |
| wrong locks, of 6,528 lock starts (2-s horizon) | 19 | 2 |
| locks more than 10 cm off | 357 | 114 |
| median lock time when both drones take off together (30 starts, 918-MHz compute model) | 0.32 s | 0.56 s |
| EuRoC held out (V1_01 pairs, 2,700 starts): starts that lock / wrong locks | 38.8 % / 13 | 33.9 % / 0 |
| TUM-VI held out (rooms 1, 4, 5, 4,050 starts): starts that lock / wrong locks | 94.9 % / 8 | 94.0 % / 13 |

The costs: when both drones take off together the median lock comes later (0.56 s against 0.32 s), and scenes with little parallax
lock less often (EuRoC). Ten of PT8F6G8's 11 wrong-lock events on TUM-VI are the confusion between rooms 4 and 5
(184-190 cm off) that every rule shows. The rule was chosen on the seven June flights, which share one climb-and-hover profile.

`RELLOC_PARMIN_DEG=3 RELLOC_PARMIN_EARLY_DEG=3 RELLOC_MEM_GATE_DEG=0` gives the previous floor (with the memory fix) for a comparison
flight. All drones of a flight must use the same values: a packet does not carry its sender's floor (each node logs its own: `floor`
events, `own_kf.floor`).

## Layout

```
relloc_node.py        the node
live/                 new code: sender.py (relloc2's sender_lean.py as a class, with the parallax rule), receiver.py (node15.Node + the 6-DoF
                      estimate), six.py, vio.py (pose at the image instant), calib.py (flight folder, calibration datagram), lens.py,
                      transport.py (between the drones: ROS 2 or UDP)
core/                 relloc2 modules, byte-identical copies (md5 below): the evaluated receiver, lock rule, codec
third_party/          XFeat + LighterGlue (verlab/accelerated_features e92685f, Apache-2.0: modules and weights unchanged)
config/relloc.env     the flags
scripts/              run_relloc.sh, build_xfeat_engine.py
test/                 test_units.py, test_sender_parity.py, test_ros2_transport.py, play_pair.py (replay of recordings for the node)
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
- Take-off: relloc2's replay solved from each drone's 0.5-m crossing with its ground keyframes as partners, as the latch does here; its
  floor switch came at lift-off + 8 s, here at the latch + 8 s (0.3-0.5 s later in the replays above).
- Rig: drones of the fleet only (four cameras, 1280 x 800, KB4). Peers of another rig are ignored.

## Tested

This version (ROS 2 between the drones), 2026-10-30, on the workstation (RTX 5070, ROS 2 Humble, Fast DDS 2.6.11; relloc2
`review24/ros2tx`). The sender and core/ are unchanged.

- `test/test_units.py`: all pass. `test/test_sender_parity.py ... 216e2fed6c33b515 --rules`: sender_lean's packets, every rule check passes.
- `test/test_ros2_transport.py` (new; 34 checks pass): two transports in one process, in both domain modes and with both receive loops.
  Discovery; 760 datagrams each way (20 bursts of 36 plus calibration and estimate datagrams, one side sending from two threads) arrive
  byte for byte and in order; the counters agree; no own datagram; a message with another layout is deserialized; clean close.
- Transport benchmark, two processes over loopback with Fast DDS on UDP only, 40 s per run, 2 runs per cell: every datagram arrived
  (7,998 of 7,998). Back-to-back bursts: keyframe complete after 0.79-0.82 ms against 0.25 ms for raw UDP, receive CPU 0.82-0.88 against
  0.30 ms per keyframe. Datagrams spaced as at 24 Mbit/s: 0.035 against 0.01 ms per datagram, receive CPU 2.06-2.17 against 0.66-0.71
  ms per keyframe. The first version of the receive loop (an executor) took 6.4-6.9 ms per keyframe spaced, 1.2-1.3 ms in bursts.
- Replays through two nodes of this version, as in the test below (three scenarios, every camera frame at half speed, graded against
  motion capture), raw UDP and ROS 2 (`RELLOC_ROS_DOMAIN` set) interleaved, one run each, plus one ROS 2 run with the datagram topics in
  the nodes' own domain (`RELLOC_ROS_DOMAIN` empty). Both nodes ran on one host, where Fast DDS used shared memory between them (each
  ROS 2 node mapped 4 more Fast DDS segments), so the latency and CPU below are not the radio's:

| scenario (per direction: nxt3 <- nxt6, nxt6 <- nxt3) | transport | first lock after the later lift-off: error | wrong locks / withdrawals | held error, median / 90th pct |
|---|---|---|---|---|
| s3, simultaneous take-off | UDP | 1.00 s: 2.8 cm, 1.00 s: 1.4 cm | 0 / 0 | 3.8 / 6.7, 3.4 / 7.1 cm |
| | ROS 2, domain of its own | 1.00 s: 2.8 cm, 1.00 s: 1.4 cm | 0 / 0 | 3.7 / 6.8, 3.6 / 7.0 cm |
| | ROS 2, the nodes' domain | 1.00 s: 1.3 cm, 0.77 s: 4.7 cm | 0 / 0 | 3.9 / 6.9, 3.7 / 6.9 cm |
| s3, nxt6 takes off 30 s after nxt3 | UDP | 0.40 s: 5.6 cm, 0.37 s: 3.5 cm | 0 / 0 | 3.9 / 7.4, 3.8 / 7.0 cm |
| | ROS 2, domain of its own | 0.40 s: 5.6 cm, 0.37 s: 3.5 cm | 0 / 0 | 3.8 / 7.0, 3.8 / 7.3 cm |
| s1, nxt6 takes off 25 s after nxt3 | UDP | 0.30 s: 6.6 cm, 0.41 s: 7.9 cm | 0 / 0 | 4.5 / 7.9, 5.0 / 8.8 cm |
| | ROS 2, domain of its own | 0.30 s: 5.6 cm, 0.15 s: 4.8 cm | 0 / 0 | 4.2 / 7.0, 4.9 / 8.3 cm |

- Datagrams: in every run and direction the receiver got every keyframe datagram the sender sent (6,660-22,320 per direction), all in
  time (no late, duplicate, stale or bad datagram), and every datagram but at most the last one sent at the stop. No queue drop, no
  publish or receive error over ROS 2; one UDP calibration datagram failed in the first 5 s, sent before the peer's port was bound.
- Header latency (the receiver's arrival of a keyframe's first header datagram minus the sender's send, on the shared sim clock): mean
  0.6-1.2 ms later over ROS 2 than over UDP (wall-clock equivalent; the clock steps by 5 ms, about 470 headers per direction). The DDS
  part was 0.03-0.05 ms (median); the receive thread took the datagrams 0.21-0.27 ms after their reception (median; the 90th percentile
  of the last 2,000 datagrams reached 3.7 ms once).
- CPU, at half speed: the node processes took 21-29 % of one core with either transport (ROS 2 minus UDP, same scenario and node: 0.0
  to +2.4 points); the receive thread 0.18-0.23 % (ROS 2) against 0.06-0.12 % (UDP).
- In both s3 scenarios UDP and ROS 2 (domain of its own) first locked on the same keyframe pairs with the same estimate. In s1 the
  nxt6 <- nxt3 lock came at 0.15 s over ROS 2 and 0.41 s over UDP; two earlier UDP runs of this direction locked at 0.18 and 0.31 s
  (review24/clock and the runs below), so one run per arm does not separate the transports. One accepted maintenance solve of the run
  in the nodes' domain was 20.2 cm off; it caused no withdrawal.

This version (take-off latch and PT8F6G8), 2026-10-25, on the workstation (RTX 5070, ROS 2 Humble; relloc2 `review24/branch`):

- `test/test_sender_parity.py`: at sender_lean's settings the sender still builds sender_lean's packets (216e2fed6c33b515). With
  `--rules`, on the same 40 keyframes: a switch from 8 to 6 deg at the middle keyframe gives every keyframe the fresh 3-D points of a
  fixed 8-deg floor before it and of a fixed 6-deg floor from it on (40 of 40); the 8-deg points are a subset of the 3-deg ones; with
  the depth memory forced on, each of the 16 stores holds exactly its keyframe's points at >= 8 deg (the gate removed points from 12 of
  them), and no keypoint has two 3-D points (2,466 do without the memory fix).
- `test/test_units.py`: all pass.
- Replays through two nodes of this version with their defaults, from each drone's first VIO pose on the ground (`test/play_pair.py`,
  and relloc2's copy of it that plays one drone late, `review24/deploy/e2e/scripts/play_shift.py`), as in the first test below but with relloc2's current inputs (the VIO of a4b4f19 and its
  flight folders, lens terms), every camera frame at half speed. Per drone the node processed its 26-52 keyframes before the latch (on
  the ground and below 0.5 m) without sending them; the first keyframe sent was the first above 0.5 m (at 0.55-0.70 m, 0.3-0.5 s after
  lift-off: the drones climb at 1.0-1.4 m/s, 0.20-0.28 m per keyframe); no solve used an own keyframe from before it, and the peer
  received no older header; the floor changed from 8 to 6 deg at the keyframe 8 s after the latch. Graded against motion capture (error
  = where the receiver places the peer's IMU):

| scenario (per direction) | first lock after the later drone's lift-off: error | wrong locks / withdrawals | held error, median / 90th pct |
|---|---|---|---|
| s3, simultaneous take-off | 1.1 s: 2.1 cm, 0.9 s: 3.2 cm | 0 / 0 | 3.7-3.9 / 7.2-7.4 cm |
| s3, nxt6 takes off 30 s after nxt3 | 0.4 s: 5.5 cm, 0.5 s: 5.2 cm | 0 / 0 | 3.6-3.9 / 6.7-7.0 cm |
| s1, nxt6 takes off 25 s after nxt3 (the late sender of relloc2's early wrong locks) | 0.3 s: 5.9 cm, 0.3 s: 5.8 cm | 0 / 0 | 4.1-4.9 / 7.4-9.0 cm |

The first version on the same three scenarios locked 10.1-10.3 s after the later lift-off, 0.8-14.7 cm off. Of the 436 accepted solves,
2 were more than 20 cm off, neither usable by the lock rule. One run per scenario and 7-9 solves in a sender's first 8 s: the runs show
that the node applies the rule as intended; the per-solve effect comes from relloc2's replay (section "Take-off and parallax floor").

The first version (10-s hold, floor 3 deg), 2026-10-22, on the workstation (RTX 5070, ROS 2 Humble; not rerun with this version):

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
recordings made after 15 Oct (capture stamps, calibrations with lens terms); a VIO restart in flight; a landing and a second take-off
without a restart of relloc.

## Limits

- Compute on the Orin is not measured. On the workstation (RTX 5070) the sender takes 63-71 ms per keyframe and a solve 72-74 ms
  (medians, two nodes on one GPU). relloc2's device times were measured at 918 MHz; the fleet's GPU cap is 612 MHz. A sender slower than 0.2 s skips keyframes (census:
  `kf_dropped_busy`); the TensorRT engine is the first lever. Scaled from relloc2's device times, a keyframe at 612 MHz would take
  225-260 ms with torch XFeat and 160-190 ms with the engine (an estimate): build the engine on each drone.
- relloc has not run beside the flight VIO. The VIO tracks on the GPU and pins its threads to all eight cores; relloc adds a CUDA
  context, XFeat at 5 Hz and the solves, now from the first VIO pose on the pad. Before a flight relies on it, compare the VIO alone with
  the VIO beside `taskset -c 4-7 nice -n 5 bash relloc/scripts/run_relloc.sh` (pose delay, incomplete image sets), with PX4 on motion
  capture.
- The packet header carries the peer's pose at its keyframe; `position` is therefore 0.1-0.3 s old.
- ROS 2 between the drones has run on one workstation only: not over the drones' Wi-Fi, not on the Orin (its receive CPU above is an
  estimate), not through domain_bridge, not with three or more drones. Measure it on the drones' Wi-Fi before a flight relies on it.
- No time synchronisation is needed between drones (each receiver estimates the clock offset from the packet headers), but the camera
  stamps, the VIO stamps and the IMU stamps of one drone must share its clock, as the VIO requires anyway.

## Replaying recordings

One node per drone, each with its own `RELLOC_ID`, flight folder and the player's topics, then the player:

```bash
export ROS_DOMAIN_ID=87 RELLOC_SIM_TIME=1 RELLOC=1 RELLOC_TRANSPORT=ros2 RELLOC_ROS_DOMAIN=88   # a domain nobody else uses on this host
RELLOC_ID=3 RELLOC_FLIGHT_DIR=<folder nxt3> RELLOC_TOPIC=nxt3/relloc \
  RELLOC_CAM_TOPICS=/nxt3/cam0/compressed,/nxt3/cam1/compressed,/nxt3/cam2/compressed,/nxt3/cam3/compressed \
  RELLOC_POSE_TOPIC=/nxt3/poseimu RELLOC_IMU_TOPIC=/nxt3/fmu/out/sensor_combined bash relloc/scripts/run_relloc.sh &
# ... the same for nxt6 (RELLOC_ID=6, /nxt6/...)
python3 relloc/test/play_pair.py --all-frames --drone ns=nxt3,mcap=<cams.mcap>,traj=<vio.tum>,bag=<bag.mcap>,toff=<t_d> --drone ns=nxt6,...
```

With `RELLOC_ROS_DOMAIN` empty the datagram topics share the player's domain, which both nodes are in. With `RELLOC_TRANSPORT=udp` give
each node its ports: `RELLOC_PORT=47803 RELLOC_DEST=127.0.0.1:47806` for nxt3, swapped for nxt6. On one host Fast DDS may carry the
datagrams through shared memory, so latency and CPU there are not the radio's.

`traj` is the estimator's trajectory in camera time (`toff` = its t_d shifts it to the IMU time of `/poseimu`). For recordings made
before 15 Oct (arrival stamps) pass `--imu-stamp log`, which reproduces how relloc2 aligned the gyro with those recordings.
