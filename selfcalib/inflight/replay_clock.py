#!/usr/bin/env python3
"""replay_clock.py -- helpers of run_inflight_replay.sh: what the recording says, and when its data go live in the replay.

  replay_clock.py info --recording DIR
      Facts about a recording in the vehicle layout (DIR/bag = PX4 rosbag2, DIR/cams/recording.mcap = the cameras), printed
      as shell assignments for `eval`: the takeoff (PX4 land detector, seconds after every stream is live, as the tool finds
      it), the stamp of that instant (anchor), the files and their first log times (what `ros2 bag play` paces by), the
      topics, and whether every .mcap file is complete (ends with the MCAP magic: a file still being written does not).
  replay_clock.py first-frame --topic T --out FILE [--count 3] [--timeout 180]
      Subscribes to camera topic T of the replay and writes "stamp clock" of its first COUNT images, the clock being
      time.monotonic() (CLOCK_MONOTONIC, the clock of the estimator's pacing: OV_RT_T0_WALL), then exits.
  replay_clock.py wait --first FILE --anchor-stamp S --takeoff T --after A [--timeout 300]
      Waits for FILE, then until the replay is A seconds past the takeoff (anchor stamp + T on the recording's clock), and
      prints  T0_BAG T0_CLOCK TAKEOFF_CLOCK GO_CLOCK NOW  (stamp and arrival clock of the first image received; the clock of
      the takeoff and of the start instant in the replay; the clock when it returned).
  replay_clock.py stamp
      Copies stdin to stdout, each line prefixed with time.monotonic() ("%.3f ").
"""
import argparse
import io
import os
import shlex
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MCAP_MAGIC = b'\x89MCAP0\r\n'


def mcap_complete(path):
    """True if the .mcap file ends with the MCAP magic, i.e. its writer closed it."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() < 2 * len(MCAP_MAGIC):
                return False
            f.seek(-len(MCAP_MAGIC), os.SEEK_END)
            return f.read() == MCAP_MAGIC
    except OSError:
        return False


def cmd_info(a):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tool'))
    sys.dont_write_bytecode = True
    import rosbag2_py
    from bagio import Recording, _storage_id
    rec_dir = os.path.abspath(a.recording)
    bag_dir, cams = os.path.join(rec_dir, 'bag'), os.path.join(rec_dir, 'cams', 'recording.mcap')
    if not (os.path.isdir(bag_dir) and os.path.isfile(cams)):
        sys.exit('replay_clock.py: %s is not a recording in the vehicle layout (bag/ and cams/recording.mcap)' % rec_dir)
    mcaps = [os.path.join(bag_dir, f) for f in sorted(os.listdir(bag_dir)) if f.endswith('.mcap')] + [cams]
    incomplete = [p for p in mcaps if not mcap_complete(p)]
    out = {'REC_INCOMPLETE': ' '.join(incomplete)}
    if not incomplete:
        rec = Recording.open(rec_dir)
        takeoff = rec.px4_takeoff()
        starts = []
        for uri in rec.uris:
            info = rosbag2_py.Info().read_metadata(uri, _storage_id(uri))
            starts.append(info.starting_time.nanoseconds * 1e-9)
        span = rec.span_after_anchor()
        out.update({
            'REC_IMU_URI': rec.uris[0], 'REC_CAM_URI': rec.uris[1],
            'REC_IMU_STORAGE': _storage_id(rec.uris[0]), 'REC_CAM_STORAGE': _storage_id(rec.uris[1]),
            'REC_IMU_START': '%.9f' % starts[0], 'REC_CAM_START': '%.9f' % starts[1],
            'REC_ANCHOR_STAMP': '%.6f' % rec._anchor(),
            'REC_TAKEOFF_S': '' if takeoff is None else '%.3f' % takeoff,
            'REC_SPAN_S': '' if span is None else '%.3f' % span,
            'REC_IMU_TOPIC': rec.imu_topic, 'REC_IMU_TYPE': rec.imu_type, 'REC_CAM_TYPE': rec.cam_type,
            'REC_NCAM': str(len(rec.cam_topics)),
            'REC_CAM_TOPICS': ' '.join(rec.cam_topics[c] for c in sorted(rec.cam_topics)),
        })
    for k, v in out.items():
        print('%s=%s' % (k, shlex.quote(v)))


def cmd_first_frame(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    if a.raw:
        from sensor_msgs.msg import Image as Msg
    else:
        from sensor_msgs.msg import CompressedImage as Msg
    rclpy.init()
    node = Node('inflight_first_frame')
    got = []

    def cb(m):
        got.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, time.monotonic()))

    node.create_subscription(Msg, a.topic, cb, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                                          history=HistoryPolicy.KEEP_LAST))
    t_end = time.monotonic() + a.timeout
    while len(got) < a.count and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.2)
    tmp = a.out + '.tmp'
    with open(tmp, 'w') as f:
        f.write(''.join('%.6f %.6f\n' % g for g in got[:a.count]))
    os.replace(tmp, a.out)          # the waiting side never sees a half-written file
    node.destroy_node()
    rclpy.shutdown()
    return 0 if got else 3


def cmd_wait(a):
    t_end = time.monotonic() + a.timeout
    row = None
    while time.monotonic() < t_end:
        if os.path.exists(a.first) and os.path.getsize(a.first) > 0:
            first = open(a.first).readline().split()
            row = (float(first[0]), float(first[1]))
            break
        time.sleep(0.1)
    if row is None:
        print('NOFRAME')
        return 3
    t0_bag, t0_clock = row
    takeoff_clock = t0_clock + (a.anchor_stamp + a.takeoff - t0_bag)
    go = takeoff_clock + a.after
    while time.monotonic() < go:
        time.sleep(min(0.05, max(0.0, go - time.monotonic())))
    print('%.6f %.6f %.6f %.6f %.6f' % (t0_bag, t0_clock, takeoff_clock, go, time.monotonic()))
    return 0


def cmd_stamp(_a):
    src = io.TextIOWrapper(sys.stdin.buffer, errors='replace')
    for line in src:
        sys.stdout.write('%.3f %s' % (time.monotonic(), line))
        sys.stdout.flush()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('info')
    p.add_argument('--recording', required=True)
    p = sub.add_parser('first-frame')
    p.add_argument('--topic', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--raw', action='store_true', help='sensor_msgs/Image instead of CompressedImage')
    p = sub.add_parser('wait')
    p.add_argument('--first', required=True)
    p.add_argument('--anchor-stamp', type=float, required=True)
    p.add_argument('--takeoff', type=float, required=True)
    p.add_argument('--after', type=float, required=True)
    p.add_argument('--timeout', type=float, default=300.0)
    sub.add_parser('stamp')
    a = ap.parse_args()
    return {'info': cmd_info, 'first-frame': cmd_first_frame, 'wait': cmd_wait, 'stamp': cmd_stamp}[a.cmd](a) or 0


if __name__ == '__main__':
    sys.exit(main())
