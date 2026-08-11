#!/usr/bin/env python3
"""Ground-truth auto-detection: find an OptiTrack/mocap pose stream inside the recording
bag and export it as a TUM file the rest of the tool can consume.

Flight recordings made in a mocap volume usually already carry the tracker's pose stream
alongside /camN/image_raw and /imu0. When they do, the ATE-health certificate (paper §6.3)
and the mount solve (§6.2) come for free — no separate --gt file, no manual export. When
they don't, nothing changes: the tool runs exactly as it does without ground truth.

CLI:
  python3 gtdetect.py <bag>                       # list candidates and pick
  python3 gtdetect.py <bag> --out gt.tum          # extract the best candidate
  python3 gtdetect.py <bag> --topic /foo --out gt.tum
"""
import argparse, importlib, os, sys
import numpy as np
from rclpy.serialization import deserialize_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bagio import Recording

# ---------------------------------------------------------------- what counts as ground truth
# type string -> (module, class) for deserialization; _pq() maps each to (position, orientation)
POSE_TYPES = {
    'geometry_msgs/msg/PoseStamped': ('geometry_msgs.msg', 'PoseStamped'),
    'geometry_msgs/msg/PoseWithCovarianceStamped': ('geometry_msgs.msg', 'PoseWithCovarianceStamped'),
    'geometry_msgs/msg/TransformStamped': ('geometry_msgs.msg', 'TransformStamped'),
    'nav_msgs/msg/Odometry': ('nav_msgs.msg', 'Odometry'),
    'tf2_msgs/msg/TFMessage': ('tf2_msgs.msg', 'TFMessage'),
    # What a swarm-nxt drone actually records in a mocap volume. Unlike every type
    # above it carries no std_msgs/Header -- the time is a bare `stamp` field and
    # the pose sits under `rigid_body` -- so _stamp()/_pq() special-case it.
    'optitrack_multiplexer_ros2_msgs/msg/RigidBodyStamped':
        ('optitrack_multiplexer_ros2_msgs.msg', 'RigidBodyStamped'),
}

# Substrings that mark a topic as (not) a mocap stream. Scored, not absolute — a bag that
# names its tracker topic /pose still wins on message rate if nothing better is present.
POSITIVE = ('optitrack', 'mocap', 'motion_capture', 'vicon', 'vrpn', 'natnet', 'qualisys',
            'ground_truth', 'groundtruth', 'gt_pose', '/gt', 'mcs', 'world_pose', 'vrpn_client')
NEGATIVE = ('ov_msckf', 'openvins', 'vio', 'estimate', 'odometry_filtered', 'ekf', 'msckf',
            'setpoint', 'command', 'cmd', 'target', 'reference', 'desired')

MIN_MSGS = 50        # below this it cannot support an ATE fit (evalate needs >=50 samples)
MIN_HZ = 5.0         # a mocap stream is >=20 Hz in practice; 5 is a generous floor
MIN_COVERAGE = 0.50  # fraction of bag duration the stream must span


def _load(type_str):
    mod, cls = POSE_TYPES[type_str]
    return getattr(importlib.import_module(mod), cls)


def _rec(bag):
    return bag if isinstance(bag, Recording) else Recording.open(bag)


class _Reader:
    """Adapts bagio's generator to the has_next()/read_next() shape used below."""

    def __init__(self, rec, topic):
        self._it = iter(rec.read_topic(topic))
        self._head = next(self._it, None)

    def has_next(self):
        return self._head is not None

    def read_next(self):
        t_bag, data = self._head
        self._head = next(self._it, None)
        return None, data, t_bag


def _reader(bag, topics=None):
    return _Reader(_rec(bag), topics[0])


def _meta(bag):
    rec = _rec(bag)
    return rec.topics(), rec.duration()


def _stamp(hdr):
    return hdr.stamp.sec + hdr.stamp.nanosec * 1e-9


RIGID_BODY_TYPE = 'optitrack_multiplexer_ros2_msgs/msg/RigidBodyStamped'


def _pq(msg, type_str):
    """-> (position(3), quaternion xyzw(4)) for the single-pose types."""
    if type_str == 'geometry_msgs/msg/PoseStamped':
        p, q = msg.pose.position, msg.pose.orientation
    elif type_str == 'geometry_msgs/msg/PoseWithCovarianceStamped':
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
    elif type_str == 'nav_msgs/msg/Odometry':
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
    elif type_str == 'geometry_msgs/msg/TransformStamped':
        p, q = msg.transform.translation, msg.transform.rotation
    elif type_str == RIGID_BODY_TYPE:
        # optitrack_wrapper_ros2_msgs has its own Point/Quaternion, and its
        # quaternion fields are q_x..q_w rather than geometry_msgs' x..w.
        p, q = msg.rigid_body.pose.position, msg.rigid_body.pose.orientation
        return (p.x, p.y, p.z), (q.q_x, q.q_y, q.q_z, q.q_w)
    else:
        raise ValueError('not a single-pose type: ' + type_str)
    return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)


def _tracked(msg, type_str):
    """False for samples the tracker itself flags as invalid. An occluded rigid
    body keeps publishing at full rate with a stale or zeroed pose, which would
    otherwise enter the ATE fit as if it were a real measurement."""
    if type_str == RIGID_BODY_TYPE:
        return bool(msg.rigid_body.tracking_valid)
    return True


def _name_score(name, hint=None):
    """Score a topic name or a tf child_frame_id. `hint` is usually the vehicle name."""
    t = name.lower()
    s = 0.0
    for h in POSITIVE:
        if h in t:
            s += 10.0
    for h in NEGATIVE:
        if h in t:
            s -= 25.0
    if hint and hint.lower() in t:
        s += 15.0            # "/vrpn_client_node/nxt3/pose" or child_frame "nxt3_body"
    if t.endswith('/pose') or t.endswith('/pose_stamped'):
        s += 2.0
    return s


def candidates(bag, hint=None):
    """Rank every topic that could plausibly be a mocap ground-truth stream."""
    topics, dur = _meta(bag)
    out = []
    for name, info in topics.items():
        if info['type'] not in POSE_TYPES:
            continue
        hz = info['count'] / dur if dur > 0 else 0.0
        score = _name_score(name, hint) + min(hz / 10.0, 5.0)
        if info['count'] < MIN_MSGS:
            score -= 50.0
        if hz < MIN_HZ:
            score -= 20.0
        out.append({'topic': name, 'type': info['type'], 'count': info['count'],
                    'hz': round(hz, 2), 'score': round(score, 2)})
    # Sort by score, then by name so equal scores never resolve arbitrarily run-to-run.
    out.sort(key=lambda c: (-c['score'], c['topic']))
    return out, dur


def _tf_children(bag, topic, hint=None, limit=400):
    """TFMessage carries many frames; rank child_frame_ids so we can pick the vehicle.

    Counting alone is not enough: a mocap bridge that publishes the vehicle and a handful of
    other rigid bodies emits all of them at the same rate, so the counts tie. Name scoring
    (and the --drone hint) breaks the tie; the frame name is the only signal there is.
    """
    r = _reader(bag, [topic])
    cls = _load('tf2_msgs/msg/TFMessage')
    seen, n = {}, 0
    while r.has_next() and n < limit:
        _, data, _ = r.read_next()
        for tr in deserialize_message(data, cls).transforms:
            seen[tr.child_frame_id] = seen.get(tr.child_frame_id, 0) + 1
        n += 1
    ranked = [{'child_frame': k, 'count': v, 'score': round(_name_score(k, hint) + min(v / 100.0, 3.0), 2)}
              for k, v in seen.items()]
    ranked.sort(key=lambda c: (-c['score'], c['child_frame']))
    return ranked


def extract(bag, topic, dst, type_str=None, child_frame=None, hint=None):
    """Write topic to dst in TUM format (t x y z qx qy qz qw). -> stats dict."""
    topics, dur = _meta(bag)
    if topic not in topics:
        raise KeyError('topic not in bag: ' + topic)
    type_str = type_str or topics[topic]['type']
    if type_str not in POSE_TYPES:
        raise ValueError('unsupported ground-truth type: ' + type_str)
    cls = _load(type_str)
    is_tf = type_str == 'tf2_msgs/msg/TFMessage'
    tf_ranked = None
    if is_tf and child_frame is None:
        tf_ranked = _tf_children(bag, topic, hint)
        if not tf_ranked:
            raise ValueError('no transforms found on ' + topic)
        child_frame = tf_ranked[0]['child_frame']

    rows, n_hdr, n_bag = [], 0, 0
    r = _reader(bag, [topic])
    while r.has_next():
        _, data, t_bag = r.read_next()
        msg = deserialize_message(data, cls)
        items = []
        if is_tf:
            for tr in msg.transforms:
                if tr.child_frame_id == child_frame:
                    items.append((_stamp(tr.header), (tr.transform.translation.x, tr.transform.translation.y,
                                                      tr.transform.translation.z),
                                  (tr.transform.rotation.x, tr.transform.rotation.y,
                                   tr.transform.rotation.z, tr.transform.rotation.w)))
        elif _tracked(msg, type_str):
            p, q = _pq(msg, type_str)
            # RigidBodyStamped has no std_msgs/Header; its time is a bare `stamp`.
            items.append((_stamp(msg if type_str == RIGID_BODY_TYPE else msg.header), p, q))
        for ts, p, q in items:
            # Header stamps are what share the clock with /imu0 and the images. A zero
            # header stamp means the publisher never filled it in; fall back to the bag's
            # receive time, which is close enough to align but is NOT the sensor clock.
            if ts <= 0.0:
                ts = t_bag * 1e-9
                n_bag += 1
            else:
                n_hdr += 1
            rows.append((ts, *p, *q))

    if len(rows) < MIN_MSGS:
        raise ValueError('only %d usable poses on %s (need >=%d)' % (len(rows), topic, MIN_MSGS))
    a = np.array(rows, dtype=float)
    a = a[np.argsort(a[:, 0])]
    a = a[np.isfinite(a).all(axis=1)]
    # Drop duplicate stamps: np.interp in evalate.load_aligned needs a strictly increasing x.
    keep = np.concatenate(([True], np.diff(a[:, 0]) > 0))
    a = a[keep]

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or '.', exist_ok=True)
    with open(dst, 'w') as f:
        f.write('# t x y z qx qy qz qw\n')
        f.write('# auto-extracted by gtdetect.py from %s topic %s (%s)%s\n'
                % (os.path.basename(os.path.normpath(bag)), topic, type_str,
                   ' child_frame=' + child_frame if is_tf else ''))
        np.savetxt(f, a, fmt='%.9f')

    span = float(a[-1, 0] - a[0, 0])
    res = {'found': True, 'path': dst, 'topic': topic, 'type': type_str,
           'child_frame': child_frame, 'poses': int(len(a)),
           'hz': round(len(a) / span, 2) if span > 0 else 0.0,
           'span_s': round(span, 2), 'bag_duration_s': round(dur, 2),
           'coverage': round(span / dur, 3) if dur > 0 else 0.0,
           'stamp_source': 'header' if n_bag == 0 else ('bag_receive_time' if n_hdr == 0 else 'mixed')}
    if tf_ranked is not None and len(tf_ranked) > 1:
        res['child_frames_considered'] = tf_ranked[:4]
    return res


def _quality_warnings(res):
    """Attach coverage / stamp-source warnings to a successful extract result, in place."""
    warns = []
    if res['coverage'] < MIN_COVERAGE:
        warns.append('ground truth spans only %.0f%% of the recording' % (100 * res['coverage']))
    if res['stamp_source'] != 'header':
        warns.append('stamps came from %s, not message headers — alignment may be biased'
                     % res['stamp_source'])
    if warns:
        res['warning'] = '; '.join(warns)
    return res


def autodetect(bag, dst, prefer_topic=None, hint=None):
    """-> stats dict on success, or {'found': False, 'reason': ...}. Never raises."""
    try:
        if prefer_topic:
            return _quality_warnings(extract(bag, prefer_topic, dst, hint=hint))
        cands, dur = candidates(bag, hint)
        if not cands:
            return {'found': False, 'reason': 'no pose-like topics in bag'}
        best = cands[0]
        if best['score'] <= 0:
            return {'found': False, 'reason': 'no plausible ground-truth topic',
                    'rejected': cands[:4]}
        res = extract(bag, best['topic'], dst, best['type'], hint=hint)
        res['considered'] = cands[:4]
        return _quality_warnings(res)
    except Exception as e:
        return {'found': False, 'reason': '%s: %s' % (type(e).__name__, e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag')
    ap.add_argument('--out', default=None, help='write TUM here (omit to only list candidates)')
    ap.add_argument('--topic', default=None, help='force this topic instead of auto-picking')
    ap.add_argument('--child-frame', default=None, help='for tf2_msgs/TFMessage sources')
    ap.add_argument('--hint', default=None,
                    help='vehicle name; boosts topics/frames containing it (run_tool.py passes --drone)')
    a = ap.parse_args()

    cands, dur = candidates(a.bag, a.hint)
    print('bag duration: %.1f s' % dur)
    if not cands:
        print('no pose-like topics found (looked for: %s)' % ', '.join(sorted(POSE_TYPES)))
    for c in cands:
        print('  %-40s %-42s n=%-7d %6.1f Hz  score=%.1f'
              % (c['topic'], c['type'], c['count'], c['hz'], c['score']))
    if not a.out:
        return 0
    if a.topic:
        res = extract(a.bag, a.topic, a.out, child_frame=a.child_frame, hint=a.hint)
    else:
        res = autodetect(a.bag, a.out, hint=a.hint)
    for k, v in res.items():
        if k != 'considered':
            print('%-16s %s' % (k + ':', v))
    return 0 if res.get('found') else 1


if __name__ == '__main__':
    sys.exit(main())
