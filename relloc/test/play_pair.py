#!/usr/bin/env python3
"""Play two or more drones' recordings at once, in real time, as the live node would see them: per drone its four camera topics (the
recorded CompressedImage messages, original header stamps), its VIO pose (geometry_msgs/PoseWithCovarianceStamped, as run_online_msckf's
poseimu, from a TUM trajectory file of the estimator: 't x y z qx qy qz qw', IMU pose in the VIO world) and its PX4 IMU (the recorded
px4_msgs/SensorCombined of the drone's bag), all under /<ns>/..., plus /clock (start the nodes with RELLOC_SIM_TIME=1).
Nothing is written to disk. The recordings are read straight from their mcap files (no merged bag).
  /<ns>/cam<c>/compressed   only the image sets the node keeps (>= 0.199 s after the last kept set, the node's own rule), unless --all-frames
  /<ns>/poseimu             each pose published `--pose-latency` s after its stamp (the estimator's pose leaves 47-80 ms after its images);
                            stamp = trajectory time + toff (the estimator stamps IMU time = camera time + t_d)
  /<ns>/fmu/out/sensor_combined   at its log time; --imu-stamp log writes the log time into the payload timestamp (the clock relloc2's
                            md.VioTraj integrated the gyro on for the June recordings), payload keeps the recorded stamp
usage: python3 play_pair.py --drone ns=nxt3,mcap=/path/cams.mcap,traj=/path/vio.tum,bag=/path/bag.mcap[,toff=S] [--drone ...] [--start S]
       [--duration S] [--rate 1.0] [--pose-latency 0.05] [--imu-stamp payload|log] [--cams '/oak.../CAM_A/compressed,...'] [--all-frames]
--start / --duration: seconds after the first image of the earliest drone."""
import argparse, heapq, time, sys, struct
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovarianceStamped
from px4_msgs.msg import SensorCombined
from rosgraph_msgs.msg import Clock
from mcap.reader import make_reader

ap = argparse.ArgumentParser()
ap.add_argument('--drone', action='append', required=True); ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--duration', type=float, default=0.0)
ap.add_argument('--rate', type=float, default=1.0); ap.add_argument('--pose-latency', type=float, default=0.05); ap.add_argument('--all-frames', action='store_true')
ap.add_argument('--imu-stamp', default='payload', choices=['payload', 'log']); ap.add_argument('--imu-topic', default='/fmu/out/sensor_combined')
ap.add_argument('--cams', default=','.join(f'/oak_ffc_4p_driver_node/CAM_{x}/compressed' for x in 'ABCD'))
A = ap.parse_args(); CAMS = A.cams.split(',')
D = [dict(kv.split('=', 1) for kv in d.split(',')) for d in A.drone]

def images(k, d):
    f = open(d['mcap'], 'rb'); r = make_reader(f)
    for sch, ch, m in r.iter_messages(topics=CAMS, log_time_order=True):
        yield (m.log_time * 1e-9, 1, k, CAMS.index(ch.topic), m.data)

def poses(k, d):
    T = np.loadtxt(d['traj'], comments='#'); toff = float(d.get('toff', 0.0))
    for row in T: yield (row[0] + toff + A.pose_latency, 0, k, row[0] + toff, row[1:8])

def imu(k, d):
    if 'bag' not in d: return
    f = open(d['bag'], 'rb'); r = make_reader(f)
    for sch, ch, m in r.iter_messages(topics=[A.imu_topic], log_time_order=True): yield (m.log_time * 1e-9, 2, k, m.log_time, m.data)

rclpy.init(); node = rclpy.create_node('relloc_play')
q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=20)
pc = node.create_publisher(Clock, '/clock', 10)
P = [{'cam': [node.create_publisher(CompressedImage, f"/{d['ns']}/cam{c}/compressed", q) for c in range(len(CAMS))], 'pose': node.create_publisher(PoseWithCovarianceStamped, f"/{d['ns']}/poseimu", q),
      'imu': node.create_publisher(SensorCombined, f"/{d['ns']}{A.imu_topic}", QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=200))} for d in D]
streams = [images(k, d) for k, d in enumerate(D)] + [poses(k, d) for k, d in enumerate(D)] + [imu(k, d) for k, d in enumerate(D)]
first = None; w0 = None; keep = [dict() for _ in D]; last = [-1e18] * len(D); n = [[0, 0, 0] for _ in D]; t_last_clock = -1.0
for ev in heapq.merge(*streams, key=lambda e: (e[0], e[1])):
    t = ev[0]
    if ev[1] == 1 and first is None: first = t
    if first is None: continue
    if t < first + A.start: continue
    if A.duration > 0 and t > first + A.start + A.duration: break
    if w0 is None: w0 = time.perf_counter() - (t - first - A.start) / A.rate
    dt = w0 + (t - first - A.start) / A.rate - time.perf_counter()
    if dt > 0: time.sleep(dt)
    if t - t_last_clock >= 0.005:
        c = Clock(); c.clock.sec = int(t); c.clock.nanosec = int((t - int(t)) * 1e9); pc.publish(c); t_last_clock = t
    k = ev[2]
    if ev[1] == 0:
        ts, v = ev[3], ev[4]; m = PoseWithCovarianceStamped(); ns_ = int(round(ts * 1e9)); m.header.stamp.sec = ns_ // 1_000_000_000; m.header.stamp.nanosec = ns_ % 1_000_000_000
        m.header.frame_id = 'global'; m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = (float(x) for x in v[:3])
        m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w = (float(x) for x in v[3:7])
        P[k]['pose'].publish(m); n[k][1] += 1; continue
    if ev[1] == 2:
        m = deserialize_message(ev[4], SensorCombined)
        if A.imu_stamp == 'log': m.timestamp = int(ev[3] // 1000)
        P[k]['imu'].publish(m); n[k][2] += 1; continue
    c = ev[3]; raw = bytes(ev[4])   # published as recorded (serialized); the stamp read from the CDR header
    if raw[:2] == b'\x00\x01': sec, nsec = struct.unpack_from('<iI', raw, 4)
    else: h = deserialize_message(raw, CompressedImage).header.stamp; sec, nsec = h.sec, h.nanosec
    ts = sec * 1_000_000_000 + nsec
    if not A.all_frames:
        if ts not in keep[k]:
            keep[k][ts] = (ts - last[k]) * 1e-9 >= 0.2 - 1e-3
            if keep[k][ts]: last[k] = ts
            for s in [s for s in keep[k] if s < ts - 2_000_000_000]: del keep[k][s]
        if not keep[k][ts]: continue
    P[k]['cam'][c].publish(raw); n[k][0] += 1
print({d['ns']: {'images': n[k][0], 'poses': n[k][1], 'imu': n[k][2]} for k, d in enumerate(D)}, flush=True)
node.destroy_node(); rclpy.shutdown()
