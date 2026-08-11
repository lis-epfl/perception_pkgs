#!/usr/bin/env python3
"""Recording access: the one place that knows how a flight recording is laid out.

The study recorded one sqlite3 bag carrying /imu0 and /cam0../camN as
sensor_msgs/Imu and sensor_msgs/Image. A swarm-nxt vehicle records something else
entirely, and none of it is a preference we can ask it to change:

    <log>/bag/            rosbag2, mcap  -> /fmu/out/sensor_combined  (px4_msgs)
    <log>/cams/recording.mcap  bare mcap -> /oak_ffc_4p_driver_node/CAM_{A,B,C,D}/compressed
                                            (sensor_msgs/CompressedImage, jpeg 1280x800)

Both layouts are supported and auto-detected. Everything above this module works in
camera INDICES and decoded mono8 frames, never in topic names or storage ids.

TIMESTAMPS ARE THE POINT. Self-calibration estimates timeshift_cam_imu, so the
stamps used here must be the ones live VIO will use: the driver's header.stamp for
images, and SensorCombined.timestamp (already converted to host epoch by
uxrce_dds_client) for IMU. The bag's own log time is never used. The C++ side
derives them identically in ov_core/src/utils/swarmnxt_msgs.h -- keep the two in
step or every calibration silently describes a pipeline that does not exist.
"""
import os
import struct

import numpy as np
import rosbag2_py

# --- swarm-nxt defaults -------------------------------------------------------
# CAM_A..CAM_D map to cam0..cam3 in that order. The mapping is physical, not
# cosmetic: estimator_fleet.yaml gives cam3 a wider lens
# (mask_fisheye_theta_max3 1.87 vs 1.83), and cert_in_distribution compares each
# camera against the fleet mean for ITS mount position. A permuted mapping is
# caught by the principal-point certificate rather than shipping silently.
OAK_CAMS = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']
OAK_TOPIC = '/oak_ffc_4p_driver_node/%s/compressed'
PX4_IMU_TOPIC = '/fmu/out/sensor_combined'
PX4_IMU_TYPE = 'px4_msgs/msg/SensorCombined'

# px4_msgs/msg/SensorCombined CDR layout. Validated byte-exact against 2000 real
# fleet messages (52-byte payload) -- see swarmnxt_msgs.h for the field table.
SENSOR_COMBINED_SIZE = 52

# PX4 body frame is FRD (Forward-Right-Down); ROS/REP-103 and the seed extrinsics are
# FLU (Forward-Left-Up). The two differ by 180 deg about x.
FRD_TO_FLU = np.array([1.0, -1.0, -1.0])


def _storage_id(uri):
    """mcap vs sqlite3, read from the recording rather than assumed."""
    if uri.endswith('.mcap'):
        return 'mcap'
    meta = os.path.join(uri, 'metadata.yaml')
    if os.path.isfile(meta):
        for line in open(meta):
            if 'storage_identifier:' in line:
                return line.split('storage_identifier:')[1].strip().strip('"\'')
    return 'sqlite3'


def resolve_uri(uri):
    """A rosbag2 directory is only openable if it carries metadata.yaml. Some
    recordings ship a bare bag_N.mcap in a directory with no metadata (the file was
    copied out of its original bag, or the recorder was killed before writing it).
    rosbag2 cannot open such a directory, but it can open the .mcap itself, so point
    at the file.

    A metadata-less directory holding SEVERAL splits is refused rather than silently
    read in part: without metadata there is nothing that says how the splits order,
    and quietly calibrating on a fraction of a recording is worse than stopping.
    """
    if not os.path.isdir(uri):
        return uri
    if os.path.isfile(os.path.join(uri, 'metadata.yaml')):
        return uri
    mcaps = sorted(f for f in os.listdir(uri) if f.endswith('.mcap'))
    if len(mcaps) == 1:
        return os.path.join(uri, mcaps[0])
    if len(mcaps) > 1:
        raise ValueError('%s has %d .mcap splits but no metadata.yaml, so their order '
                         'is unknown. Restore metadata.yaml, or pass the split you '
                         'want directly.' % (uri, len(mcaps)))
    return uri


def fast_stamp(data):
    """Header stamp straight from CDR bytes. Valid for Image and CompressedImage
    alike -- in both, std_msgs/Header is the first field."""
    sec, nsec = struct.unpack_from('<iI', data, 4)
    return sec + nsec * 1e-9


def imu_from_cdr(data):
    """px4_msgs/msg/SensorCombined CDR -> (t_seconds, gyro(3), accel(3)).

    Hand-parsed rather than deserialized so the tool does not need px4_msgs on the
    PYTHONPATH, and so it cannot drift from the C++ reader. Raises on a payload
    that is not the expected size, instead of returning plausible garbage.
    """
    if len(data) < SENSOR_COMBINED_SIZE:
        raise ValueError('SensorCombined payload is %d bytes, expected >= %d; has the '
                         'px4_msgs definition changed?' % (len(data), SENSOR_COMBINED_SIZE))
    stamp_us, = struct.unpack_from('<Q', data, 4)
    gyro = np.array(struct.unpack_from('<3f', data, 12), float)
    accel = np.array(struct.unpack_from('<3f', data, 32), float)
    # FRD -> FLU. PX4's body frame is Forward-Right-Down; ROS/REP-103, the seed
    # extrinsics and the estimator are Forward-Left-Up, a 180 deg rotation about x.
    # Must match ov_core/src/utils/swarmnxt_msgs.h exactly -- the offline tool and
    # the estimator have to agree on what the IMU axes mean.
    return stamp_us * 1e-6, gyro * FRD_TO_FLU, accel * FRD_TO_FLU


def decode_image(data, compressed):
    """CDR payload -> mono8 ndarray."""
    if compressed:
        import cv2
        from sensor_msgs.msg import CompressedImage
        from rclpy.serialization import deserialize_message
        m = deserialize_message(data, CompressedImage)
        # IMREAD_GRAYSCALE lets libjpeg emit luma directly, markedly cheaper than
        # decoding to BGR and converting after.
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('could not decode compressed frame (format=%s)' % m.format)
        return img
    from sensor_msgs.msg import Image
    from rclpy.serialization import deserialize_message
    m = deserialize_message(data, Image)
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)


class Recording:
    """A flight recording, however many files it happens to span.

    Each consumer reads exactly one stream (gates read IMU-only or cameras-only,
    gtdetect reads the mocap topic), so this routes topic -> file and converts
    types. It deliberately does NOT merge streams: only the estimator needs a
    single interleaved ordering, and it does that in C++.
    """

    def __init__(self, uris, imu_topic, imu_type, cam_topics, cam_type, window=None):
        self.uris = [resolve_uri(u) for u in uris]
        self.imu_topic = imu_topic
        self.imu_type = imu_type
        self.cam_topics = dict(cam_topics)          # cam index -> topic
        self.cam_type = cam_type
        self.compressed = cam_type.endswith('CompressedImage')
        # Optional trim, in seconds. Semantics MUST match BagSource in
        # ov_msckf/src/utils/bag_source.h: measured from the instant every stream is
        # live, i.e. max(first IMU stamp, first camera stamp) -- on a fleet recording
        # the camera file starts seconds after the IMU file. If the two disagreed, the
        # gates and the circle fit would be measuring a different span than the
        # estimator, and the seed would not describe the data being calibrated on.
        self.window = float(window) if window else None
        self._anchor_cache = None
        self._topic_uri = {}                        # topic -> uri holding it
        self._meta = {}
        for uri in self.uris:
            info = rosbag2_py.Info().read_metadata(uri, _storage_id(uri))
            for t in info.topics_with_message_count:
                name = t.topic_metadata.name
                self._topic_uri[name] = uri
                self._meta[name] = {'type': t.topic_metadata.type, 'count': t.message_count}
            dur = info.duration.nanoseconds * 1e-9 if hasattr(info.duration, 'nanoseconds') \
                else float(info.duration) * 1e-9
            self._meta.setdefault('__dur__', {})
            self._meta['__dur__'][uri] = dur

    # -- discovery -------------------------------------------------------------
    @classmethod
    def open(cls, path, window=None):
        """Auto-detect the layout. `path` is a swarm-nxt log directory, or a plain
        rosbag2 recording in the study layout."""
        bag_dir = os.path.join(path, 'bag')
        cams_file = os.path.join(path, 'cams', 'recording.mcap')
        if os.path.isdir(bag_dir) and os.path.isfile(cams_file):
            return cls(uris=[bag_dir, cams_file],
                       imu_topic=PX4_IMU_TOPIC, imu_type=PX4_IMU_TYPE,
                       cam_topics={i: OAK_TOPIC % n for i, n in enumerate(OAK_CAMS)},
                       cam_type='sensor_msgs/msg/CompressedImage', window=window)
        return cls(uris=[path], imu_topic='/imu0', imu_type='sensor_msgs/msg/Imu',
                   cam_topics={i: '/cam%d/image_raw' % i for i in range(4)},
                   cam_type='sensor_msgs/msg/Image', window=window)

    # -- introspection ---------------------------------------------------------
    def topics(self):
        return {k: v for k, v in self._meta.items() if not k.startswith('__')}

    def duration(self):
        return max(self._meta.get('__dur__', {1: 0.0}).values())

    def has(self, topic):
        return topic in self._topic_uri

    # -- reading ---------------------------------------------------------------
    def _reader(self, topics):
        """One reader per distinct file, restricted to the topics we actually want.

        The storage-level filter matters: a fleet bag carries ~60 topics and 170k
        messages, of which the IMU gate wants one. Without it every message is
        fetched and copied before being discarded.
        """
        wanted = [t for t in topics if t in self._topic_uri]
        by_uri = {}
        for t in wanted:
            by_uri.setdefault(self._topic_uri[t], []).append(t)
        for uri, ts in by_uri.items():
            r = rosbag2_py.SequentialReader()
            r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=_storage_id(uri)),
                   rosbag2_py.ConverterOptions('', ''))
            r.set_filter(rosbag2_py.StorageFilter(topics=ts))
            yield r

    def _first_stamp(self, topic):
        """Sensor stamp of the first record on a topic, without applying the window."""
        for r in self._reader([topic]):
            while r.has_next():
                _, data, _ = r.read_next()
                return imu_from_cdr(data)[0] if topic == self.imu_topic and \
                    self.imu_type == PX4_IMU_TYPE else fast_stamp(data)
        return None

    def _anchor(self):
        """t0 = the instant every stream is live. Cached; two cheap single-record reads."""
        if self._anchor_cache is None:
            stamps = [self._first_stamp(t) for t in
                      [self.imu_topic] + [self.cam_topics[c] for c in sorted(self.cam_topics)][:1]]
            stamps = [x for x in stamps if x is not None]
            self._anchor_cache = max(stamps) if stamps else 0.0
        return self._anchor_cache

    def _in_window(self, t):
        if self.window is None:
            return True
        t0 = self._anchor()
        return t0 <= t <= t0 + self.window

    def imu(self):
        """-> (t, gyro(3), accel(3)) in recording order."""
        from rclpy.serialization import deserialize_message
        native = (self.imu_type == PX4_IMU_TYPE)
        for r in self._reader([self.imu_topic]):
            while r.has_next():
                _, data, _ = r.read_next()
                if native:
                    rec = imu_from_cdr(data)
                    if self._in_window(rec[0]):
                        yield rec
                    continue
                else:
                    from sensor_msgs.msg import Imu
                    m = deserialize_message(data, Imu)
                    _t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                    if not self._in_window(_t):
                        continue
                    yield (_t,
                           np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]),
                           np.array([m.linear_acceleration.x, m.linear_acceleration.y,
                                     m.linear_acceleration.z]))

    def images(self, cams=(0, 1, 2, 3)):
        """-> (cam_index, stamp, raw_cdr_payload).

        The payload is returned undecoded so callers that only need stamps (the
        timing gate) never pay for a decode, and callers that subsample (the image
        gate, circlefit) decode only the frames they keep. Use decode() on it.
        """
        topics = {self.cam_topics[c]: c for c in cams if c in self.cam_topics}
        for r in self._reader(list(topics)):
            while r.has_next():
                topic, data, _ = r.read_next()
                _t = fast_stamp(data)
                if not self._in_window(_t):
                    continue
                yield topics[topic], _t, data

    def decode(self, data):
        return decode_image(data, self.compressed)

    def read_topic(self, topic):
        """-> (stamp_or_None, raw payload) for an arbitrary topic (gtdetect)."""
        for r in self._reader([topic]):
            while r.has_next():
                _, data, t = r.read_next()
                yield t, data
