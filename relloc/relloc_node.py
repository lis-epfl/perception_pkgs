#!/usr/bin/env python3
"""relloc live node: cross-drone relative localization (relloc2 "General Route") on one drone, beside its VIO.

What it does, per drone (one process, three threads):
  sender     every 0.2 s: the four images of one instant + the VIO pose at that instant -> XFeat (3000 keypoints per image) -> rays
             (KB4 + lens terms) -> triangulation against the drone's own past keyframes -> v3 packet (about 43 kB in 36 datagrams) -> UDP
  receiver   the other drones' datagrams -> node15 (decode, pairing with own keyframes, solve on the GPU, lock rule, maintenance, pair
             protocol) -> per peer the transform of the peer's VIO world into the own VIO world, 4-DoF and 6-DoF
  udp        datagram intake
Inputs: the camera topics of the VIO (CompressedImage or Image), the VIO pose (run_online_msckf's /poseimu), the PX4 IMU (gyro between
        VIO poses), the flight folder.
Outputs (prefix RELLOC_TOPIC, default relloc):
  <prefix>/peer<id>/transform        geometry_msgs/TransformStamped, global <- peer<id>/global: the primary output (RELLOC_6DOF selects it)
  <prefix>/peer<id>/transform_4dof   the 4-DoF output (rotation about gravity only)
  <prefix>/peer<id>/transform_6dof   the 6-DoF output (full rotation)
  <prefix>/peer<id>/position         geometry_msgs/PointStamped: the peer's last header position mapped by the primary output
  <prefix>/status                    std_msgs/String, JSON, 1 Hz: per peer status (searching / provisional / confirmed / adopted), counts
  Transforms are published only while the own lock holds (status provisional or confirmed); 'confirmed' = the peer's own independent
  estimate agrees (20 cm / 2 deg). A consumer should act on confirmed only (DEPLOY_CONFIG of relloc2).
  RELLOC_LOG_DIR/relloc_<id>_<time>.jsonl: every own keyframe, received header, solve, lock, withdraw and output change (for grading).
Settings: environment variables, see config/relloc.env and README.md. `python3 relloc_node.py --check` prints them and exits."""
import os, sys, json, time, threading, collections, signal, math, random
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_k, '1')   # before numpy: the receiver's numerics
import numpy as np                                                                                       # otherwise take every core
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from live import paths, calib as CAL, vio as VIO, transport as TR
E = os.environ.get


def _id_default():
    import re, socket
    m = re.findall(r'\d+', socket.gethostname()); return m[-1] if m else ''


def config():
    c = {'id': E('RELLOC_ID', '') or _id_default(), 'flight_dir': E('RELLOC_FLIGHT_DIR', ''), 'pose': E('RELLOC_POSE_TOPIC', '/poseimu'),
         'pose_type': E('RELLOC_POSE_MSG_TYPE', 'geometry_msgs/msg/PoseWithCovarianceStamped'), 'imu': E('RELLOC_IMU_TOPIC'), 'imu_type': E('RELLOC_IMU_MSG_TYPE'),
         'cams': E('RELLOC_CAM_TOPICS', ''), 'cam_type': E('RELLOC_CAM_MSG_TYPE', ''), 'port': int(E('RELLOC_PORT', '47800')),
         'dest': E('RELLOC_DEST', '255.255.255.255:47800'), 'six': E('RELLOC_6DOF', '0') == '1', 'hold': float(E('RELLOC_TAKEOFF_HOLD_S', '10')),
         'dz': float(E('RELLOC_TAKEOFF_DZ', '0.3')), 'log_dir': os.path.expanduser(E('RELLOC_LOG_DIR', '~/relloc_logs')), 'td': E('RELLOC_TD_S', ''),
         'peers': E('RELLOC_PEERS', ''), 'max_queue_s': float(E('RELLOC_MAX_QUEUE_S', '1.0')), 'topic': E('RELLOC_TOPIC', 'relloc').strip('/'),
         'sim_time': E('RELLOC_SIM_TIME', '0') == '1', 'prio': E('RELLOC_SENDER_PRIO', '1') == '1', 'name': E('RELLOC_NODE_NAME', ''),
         'engine': E('RELLOC_XFEAT_ENGINE', '')}
    if not c['id'] or not c['id'].isdigit() or not 0 <= int(c['id']) <= 255: raise SystemExit(f'RELLOC_ID must be a drone number 0-255 (hostname gives {_id_default()!r})')
    c['id'] = int(c['id'])
    if not c['flight_dir']: raise SystemExit('RELLOC_FLIGHT_DIR: the flight folder of this drone (deploy_vio.py output) is required')
    return c


def main():
    C = config(); cal = CAL.load_flight_folder(C['flight_dir'])
    td = float(C['td']) if C['td'] else cal.td
    if C['imu'] is None: C['imu'] = cal.imu_topic or '/fmu/out/sensor_combined'     # the estimator's own IMU topic when its yaml names one
    if C['imu_type'] is None: C['imu_type'] = cal.imu_msg_type or 'px4_msgs/msg/SensorCombined'
    cams = [x.strip() for x in C['cams'].split(',') if x.strip()] if C['cams'] else (cal.topics or [f'/oak_ffc_4p_driver_node/CAM_{x}/compressed' for x in 'ABCD'[:cal.ncam]])
    cam_type = C['cam_type'] or cal.msg_type or ('sensor_msgs/msg/CompressedImage' if cams[0].endswith('compressed') else 'sensor_msgs/msg/Image')
    if len(cams) != cal.ncam: raise SystemExit(f'{len(cams)} camera topics for {cal.ncam} cameras')
    dest = TR.parse_dest(C['dest'], C['port']); peers_ok = {int(x) for x in C['peers'].split(',') if x.strip()} or None
    summary = {'drone': C['id'], 'calibration': cal.source, 'cameras': cal.ncam, 'image': [cal.W, cal.H], 'lens_terms': cal.has_terms, 'td_s': td,
               'camera_topics': cams, 'camera_msg_type': cam_type, 'vio_pose': f"{C['pose']} ({C['pose_type']})", 'imu': f"{C['imu']} ({C['imu_type']})" if C['imu'] else 'none: attitude interpolated between VIO poses', 'udp_port': C['port'], 'udp_dest': [f'{h}:{p}' for h, p in dest],
               'peers': sorted(peers_ok) if peers_ok else 'any', 'primary_output': '6dof' if C['six'] else '4dof', 'takeoff_hold_s': C['hold'],
               'takeoff_dz_m': C['dz'], 'log_dir': C['log_dir'], 'max_queue_s': C['max_queue_s'], 'topic_prefix': C['topic'], 'sim_time': C['sim_time'],
               'xfeat': f"TensorRT {C['engine']}" if C['engine'] else 'torch (batched)'}
    if C['engine'] and not os.path.exists(os.path.expanduser(C['engine'])): raise SystemExit(f"RELLOC_XFEAT_ENGINE {C['engine']}: no such file (scripts/build_xfeat_engine.py)")
    print('[relloc] ' + json.dumps(summary), flush=True)
    if any(h == '255.255.255.255' or h.endswith('.255') for h, _ in dest):
        print('[relloc] WARNING: broadcast destination: on Wi-Fi broadcast frames go at the basic rate without retries; ~1.7 Mbit/s per drone may not fit. Prefer RELLOC_DEST=<peer>:<port>,...', flush=True)
    if not C['imu']: print('[relloc] WARNING: no IMU topic: the attitude between VIO poses is interpolated (tilt error ~3x in the replay test)', flush=True)
    if '--check' in sys.argv: return
    os.environ.setdefault('CAPTURE_MODE', 'thread_local')

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import CompressedImage, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from sensor_msgs.msg import Imu
    from geometry_msgs.msg import TransformStamped, PointStamped
    from std_msgs.msg import String
    import cv2, torch, heapq
    from scipy.spatial.transform import Rotation as Rot
    cwd = os.getcwd()
    import orin_dump as RX, packet_codec as PC, node15                                        # orin_dump changes to its own directory
    os.chdir(cwd)
    from live import sender as SND, receiver as RCV, six as SIX
    from modules.xfeat import XFeat

    rclpy.init(args=[a for a in sys.argv if a != '--check'])
    ros = rclpy.create_node(C['name'] or f'relloc_{C["id"]}', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, C['sim_time'])])
    def now():
        return ros.get_clock().now().nanoseconds * 1e-9
    os.makedirs(C['log_dir'], exist_ok=True)
    LOG = open(os.path.join(C['log_dir'], f'relloc_{C["id"]}_{time.strftime("%Y%m%d_%H%M%S")}.jsonl'), 'w', buffering=1); LL = threading.Lock()
    def log(kind, **kw):
        def cv(v):
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, (np.floating, np.integer)): return v.item()
            if isinstance(v, (list, tuple)): return [cv(x) for x in v]
            if isinstance(v, dict): return {k: cv(x) for k, x in v.items()}
            return v
        with LL: LOG.write(json.dumps({'ev': kind, 't': now(), 'wall': time.time(), **{k: cv(v) for k, v in kw.items()}}) + '\n')
    log('start', **summary)

    # ---- GPU parts: the receiver's matcher graphs are captured before any other thread runs (DEPLOY_CONFIG, process layout)
    dev = 'cuda'; K_ = 2000 // cal.ncam; L_ = next((b for b in RX.G.BUCKETS if b >= K_), ((K_ + 255) // 256) * 256)
    for Bn in range(1, RX.CAMPAIRS + 1): RX.G._get(Bn, K_, K_, False); RX.G._get(Bn, L_, L_, True)
    torch.cuda.synchronize()
    xf = XFeat(top_k=3000); xf.dev = dev; XT = None
    if C['engine']:
        if cal.ncam != 4 or (cal.W, cal.H) != (1280, 800): raise SystemExit('RELLOC_XFEAT_ENGINE: the engine is built for 4 cameras of 1280 x 800')
        cwd = os.getcwd(); import xfeat_trt; os.chdir(cwd)                                    # xfeat_trt changes to its own directory
        XT = xfeat_trt.XFeatTRT(xf, os.path.expanduser(C['engine']))
    from concurrent.futures import ThreadPoolExecutor
    POOL = ThreadPoolExecutor(cal.ncam, thread_name_prefix='relloc-jpeg')
    def decode(raw):   # one serialized image message -> 8-bit grey image (H, W)
        from rclpy.serialization import deserialize_message
        if cam_type.endswith('CompressedImage'):
            d = bytes(deserialize_message(raw, CompressedImage).data); soi = d.find(b'\xff\xd8\xff'); d = d if soi < 0 else d[soi:]
            return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_GRAYSCALE)
        m = deserialize_message(raw, Image)
        if m.encoding not in ('mono8', '8UC1'): raise ValueError(f'image encoding {m.encoding}: mono8 expected')
        return np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width].copy()
    S = SND.LiveSender(cal.ncam, cal.W, cal.H, C['id']); S.CAPTURE_MODE = os.environ['CAPTURE_MODE']
    P_ = np.load(paths.PCA); codec = PC.Codec(P_['mu'], P_['basis'], cal.W, cal.H)
    own_pc = cal.peer_calib(node15)
    OWN = {}; OL = threading.Lock()
    def own(i):
        with OL: e = OWN.get(i)
        if e is None: return (float(nd.own_t[i]), (0.0,) * 6, {}, None)
        if e['A'] is None:
            e['A'] = node15.peer_side(codec.decode_v3(e['dg'], reconstruct=False), own_pc, codec)
        return (e['t'], e['pose'], e['A'], e['health'])
    nd = RCV.LiveNode(codec, RX.solve, own, [], win=0.25, rule=None, clock_free=True, own_id=C['id'], rig=CAL.RIG, own_fx=cal.fx, pair_protocol=True, six_primary=C['six'])
    SQ = node15.SolveQueue(nd)
    udp = TR.Udp(C['port'], dest, now)

    # ---- ROS inputs
    vb = VIO.VioBuffer(); st = collections.Counter(); ready_rx = threading.Event(); stop = threading.Event()
    def on_pose(m):   # PoseWithCovarianceStamped and Odometry both carry pose.pose
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9; q = m.pose.pose.orientation; p = m.pose.pose.position
        vb.add(t, (q.x, q.y, q.z, q.w), (p.x, p.y, p.z)); st['vio_poses'] += 1
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    PT = {'geometry_msgs/msg/PoseWithCovarianceStamped': PoseWithCovarianceStamped, 'nav_msgs/msg/Odometry': Odometry}
    if C['pose_type'] not in PT: raise SystemExit(f"RELLOC_POSE_MSG_TYPE {C['pose_type']}: one of {list(PT)}")
    ros.create_subscription(PT[C['pose_type']], C['pose'], on_pose, qos)
    if C['imu']:   # gyro for the attitude between VIO poses, in the estimator's IMU frame and clock (ov_core swarmnxt_msgs.h)
        if C['imu_type'] == 'px4_msgs/msg/SensorCombined':
            from px4_msgs.msg import SensorCombined
            def on_imu(m): vb.add_gyro(m.timestamp * 1e-6, (m.gyro_rad[0], -m.gyro_rad[1], -m.gyro_rad[2])); st['imu'] += 1   # FRD -> FLU
            ros.create_subscription(SensorCombined, C['imu'], on_imu, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200))
        elif C['imu_type'] == 'sensor_msgs/msg/Imu':
            def on_imu(m): vb.add_gyro(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)); st['imu'] += 1
            ros.create_subscription(Imu, C['imu'], on_imu, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200))
        else: raise SystemExit(f"RELLOC_IMU_MSG_TYPE {C['imu_type']}: px4_msgs/msg/SensorCombined or sensor_msgs/msg/Imu")
    FS = {}; FL = threading.Lock(); FQ = collections.deque(maxlen=1); FC = threading.Condition(FL); last_kf = [-1e18]
    # images arrive serialized (raw=True): the executor thread only reads the stamp from the CDR header (Header.stamp leads both message
    # types); the four images of a kept set are deserialized and decoded on the sender's pool. At 30 Hz x 4 cameras this keeps the
    # executor thread from losing images while the sender and receiver threads hold the interpreter.
    MT = CompressedImage if cam_type.endswith('CompressedImage') else Image
    import struct
    from rclpy.serialization import deserialize_message
    def stamp_of(raw):
        if len(raw) >= 12 and raw[0] == 0 and raw[1] == 1: sec, nsec = struct.unpack_from('<iI', raw, 4)   # CDR little endian
        else: h = deserialize_message(raw, MT).header.stamp; sec, nsec = h.sec, h.nanosec
        return sec * 1_000_000_000 + nsec
    def on_img(c, raw):
        ts = stamp_of(raw); st[f'img{c}'] += 1
        with FL:
            e = FS.setdefault(ts, [None] * cal.ncam); e[c] = raw
            if all(x is not None for x in e):
                del FS[ts]
                if (ts - last_kf[0]) * 1e-9 >= 0.2 - 1e-3:
                    last_kf[0] = ts
                    if FQ: st['kf_dropped_busy'] += 1
                    FQ.append((ts, e)); FC.notify()
            for k in [k for k in FS if k < ts - 1_000_000_000]: del FS[k]; st['img_incomplete'] += 1
    for c, tp in enumerate(cams): ros.create_subscription(MT, tp, lambda m, c=c: on_img(c, m), QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=30), raw=True)

    # ---- ROS outputs
    pubs = {}
    def pub_for(pid):
        if pid not in pubs:
            b = f'{C["topic"]}/peer{pid}'
            pubs[pid] = {k: ros.create_publisher(TransformStamped, f'{b}/{k}', 10) for k in ('transform', 'transform_4dof', 'transform_6dof')}
            pubs[pid]['position'] = ros.create_publisher(PointStamped, f'{b}/position', 10)
        return pubs[pid]
    pub_status = ros.create_publisher(String, f'{C["topic"]}/status', 10)
    def tf_msg(pid, R, t, stamp):
        m = TransformStamped(); m.header.stamp.sec = int(math.floor(stamp)); m.header.stamp.nanosec = int((stamp - math.floor(stamp)) * 1e9)
        m.header.frame_id = 'global'; m.child_frame_id = f'peer{pid}/global'; q = Rot.from_matrix(R).as_quat()
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in t)
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q); return m
    def Rz(yaw_deg): return Rot.from_euler('z', yaw_deg, degrees=True).as_matrix()
    LAST = {}
    def publish(pid, t):
        s = nd.state(pid); P = pub_for(pid)
        key = (s['status'], None if s['e4'] is None else tuple(s['e4']), None if s['e6'] is None else tuple(s['e6']), None if s['adopted'] is None else tuple(s['adopted']))
        if LAST.get(pid) == key: return s
        LAST[pid] = key
        if s['e4'] is not None: P['transform_4dof'].publish(tf_msg(pid, Rz(s['e4'][0]), s['e4'][1:4], t))
        if s['e6'] is not None: P['transform_6dof'].publish(tf_msg(pid, s['e6'][:9].reshape(3, 3), s['e6'][9:12], t))
        prim = (s['e6'][:9].reshape(3, 3), s['e6'][9:12]) if (C['six'] and s['e6'] is not None) else ((Rz(s['e4'][0]), s['e4'][1:4]) if s['e4'] is not None else None)
        if prim is not None: P['transform'].publish(tf_msg(pid, *prim, t))
        log('output', peer=pid, status=s['status'], confirmed=s['confirmed'], source=s['source'], e4=s['e4'], e6=s['e6'], adopted=s['adopted'], degraded=s['degraded'],
            tilt6_deg=None if s['e6'] is None else SIX.tilt_deg(s['e6']), n_lock=s['n_lock'], n_withdraw=s['n_withdraw'], n_shadow_withdraw=s['n_shadow_withdraw'])
        return s
    HDR = {}
    def publish_position(pid, h):
        s = nd.state(pid)
        if not s['locked']: return
        p = np.array([h['x'], h['y'], h['z']])
        x = (s['e6'][:9].reshape(3, 3) @ p + s['e6'][9:12]) if (C['six'] and s['e6'] is not None) else (Rz(s['e4'][0]) @ p + s['e4'][1:4])
        tp = nd._peer_time(nd.peers[pid], h)   # the peer's keyframe time on this drone's clock (node15's clock-offset estimate)
        m = PointStamped(); m.header.stamp.sec = int(math.floor(tp)); m.header.stamp.nanosec = int((tp - math.floor(tp)) * 1e9); m.header.frame_id = 'global'
        m.point.x, m.point.y, m.point.z = (float(v) for v in x); pub_for(pid)['position'].publish(m)

    # ---- sender thread
    hs = torch.cuda.Stream(priority=-1) if C['prio'] else torch.cuda.current_stream()
    OQ = collections.deque(); air = {'epoch': None, 't_off': None}; kidx = [0]; tim = collections.deque(maxlen=200); LAT = collections.deque(maxlen=50)
    def sender_loop():
        while not stop.is_set():
            with FL:
                while not FQ and not stop.is_set(): FC.wait(0.2)
                if stop.is_set(): return
                ts, imgs = FQ.popleft()
            t_img = ts * 1e-9; tq = t_img + td; t0 = time.perf_counter()
            while (vb.latest() is None or vb.latest() < tq) and time.perf_counter() - t0 < 0.6 and not stop.is_set(): time.sleep(0.005)
            P = vb.pose(tq)
            if P is None: st['kf_no_pose'] += 1; continue
            Rwb, pwb, ep, age, dz = P
            if air['t_off'] is None:   # once: the first take-off (VIO height above its first pose, or above the VIO origin, which is its
                if C['hold'] <= 0: air['t_off'] = t_img   # initialisation pose); a VIO restart later keeps it
                elif dz > C['dz'] or pwb[2] > C['dz']: air['t_off'] = t_img; log('takeoff', epoch=ep, dz=dz, z=float(pwb[2]))
                else: st['kf_ground'] += 1; continue
            sending = t_img >= air['t_off'] + C['hold']
            if t_img < air['t_off'] + C['hold'] - 2.5: st['kf_hold'] += 1; continue   # start 2.5 s before sending: triangulation partners
            try:
                ims = list(POOL.map(decode, imgs))
                for c, im in enumerate(ims):
                    if im is None or im.shape != (cal.H, cal.W): raise ValueError(f'cam{c}: image {None if im is None else im.shape} != {(cal.H, cal.W)}')
                t1 = time.perf_counter()
                with torch.cuda.stream(hs):
                    # all cameras in one batch (torch XFeat, or the TensorRT backbone of RELLOC_XFEAT_ENGINE); descriptors and scores
                    # rounded to fp16 as relloc2's md_prep.py stored them
                    x = torch.from_numpy(np.stack(ims)).float()[:, None].to(dev)
                    O = XT.detect(x, top_k=3000) if XT is not None else xf.detectAndCompute(x, top_k=3000)
                    feats = []
                    for c, o in enumerate(O):
                        kp = o['keypoints'].cpu().numpy().astype(np.float32); bear = cal.unproject(c, kp.astype(np.float64)).astype(np.float32)
                        feats.append((o['descriptors'].half().clone(), o['scores'].half().float().clone(), torch.from_numpy(bear).to(dev), kp))
                    Rm = Rwb.as_matrix(); eul = Rwb.as_euler('ZYX')
                    R_ = np.stack([Rm @ cal.R_bc[c] for c in range(cal.ncam)]).astype(np.float32); p_ = np.stack([pwb + Rm @ cal.p_bc[c] for c in range(cal.ncam)]).astype(np.float32)
                    pb = np.asarray(pwb, np.float32).astype(float); yaw = float(np.float32(eul[0])); rp = np.array([eul[2], eul[1]], np.float32).astype(float)
                    dg, info = S.keyframe(kidx[0], feats, R_, p_, pb, yaw, rp, tq, float(np.float32(age)), ep)   # header time: IMU clock
                    torch.cuda.current_stream().synchronize()
            except Exception as ex:
                st['kf_error'] += 1; log('error', where='sender', err=repr(ex)); continue
            t2 = time.perf_counter(); k = kidx[0]; kidx[0] += 1; ready_rx.set(); tim.append((t1 - t0, t2 - t1))
            pose6 = (float(rp[0]), float(rp[1]), yaw, *pb)
            if sending:
                udp.send(dg); st['kf_sent'] += 1; LAT.append(now() - tq)
                OQ.append({'t': tq, 'dg': dg, 'pose': pose6, 'health': info['hl'] / 255.0, 'epoch': ep, 'k': k})
            else: st['kf_warm'] += 1
            log('own_kf', k=k, t_img=t_img, t_kf=tq, epoch=ep, age=age, pose=pose6, sent=sending, n_dg=len(dg), bytes=sum(len(g) for g in dg), wait_ms=1e3 * (t1 - t0), proc_ms=1e3 * (t2 - t1),
                n_tri=info['n_tri'], n_mem=info['n_mem'], n3=info['n3_packet'], partners=info['n_partners'], health=info['hl'])

    # ---- receiver thread
    SESSION = random.SystemRandom().getrandbits(32); CAL_DG = cal.encode(C['id'], SESSION); PEER_CAL = {}; PEER_SES = {}; solve_ms = collections.deque(maxlen=200)
    def handle(g, t, addr):
        if CAL.Calib.is_calib(g):
            r = CAL.Calib.decode(g)
            if r is None: st['rx_bad_calib'] += 1; return
            pid, pc, ses = r
            if pid == C['id'] or (peers_ok and pid not in peers_ok): return
            if pc.W != cal.W or pc.H != cal.H: st['rx_other_rig'] += 1; return
            if pid in PEER_CAL and PEER_CAL[pid].same(pc) and PEER_SES.get(pid) == ses: return
            if pid in PEER_CAL:   # a new calibration or a restarted peer process: everything known of the peer is dropped
                log('peer_restart' if PEER_SES.get(pid) != ses else 'peer_calib_changed', peer=pid)
                SQ.q = [x for x in SQ.q if x[5] != pid]; heapq.heapify(SQ.q); LAST.pop(pid, None)
                for kk in [kk for kk in HDR if kk[0] == pid]: del HDR[kk]
            PEER_CAL[pid] = pc; PEER_SES[pid] = ses; nd.add_peer(pid, pc.peer_calib(node15)); pub_for(pid); log('peer', peer=pid, addr=f'{addr[0]}:{addr[1]}', lens_terms=pc.has_terms, session=ses); return
        if len(g) < 12 or g[:2] != PC.MAGIC: st['rx_foreign'] += 1; return
        try: drone, kf, cam, ck, nck, has = PC.Codec.peek(g)
        except Exception: st['rx_bad'] += 1; return
        if drone == C['id']: return
        if cam == PC.Codec.EST_CAM: nd.on_estimate(g, t); st['rx_est'] += 1; return
        if drone not in nd.peers: st['rx_no_calib'] += 1; return
        if nd.peers[drone].on_datagram(g, t) is not None and has:
            h = PC.Codec.keyframe_header(g)
            if HDR.get((drone, kf)) is None:
                HDR[(drone, kf)] = True; log('peer_kf', peer=drone, kf=kf, t_hdr=h['t'], t_arr=t, epoch=h['epoch'], pose=[h['roll'], h['pitch'], h['yaw'], h['x'], h['y'], h['z']], health=h['health'])
                publish_position(drone, h)
                for kk in [kk for kk in HDR if kk[0] == drone and kk[1] < kf - 50]: del HDR[kk]
    def receiver_loop():
        T = {'est': 0.0, 'cal': -1e9, 'stat': 0.0}; nerr = 0
        while not stop.is_set():
            try: busy = receiver_step(T)
            except Exception as ex:   # one bad datagram, keyframe or solve must not stop the receiver
                nerr += 1; st['rx_error'] += 1; busy = False
                if nerr <= 20 or nerr % 100 == 0:
                    import traceback; log('error', where='receiver', err=repr(ex), tb=traceback.format_exc()[-1500:], n=nerr)
            if not busy: time.sleep(0.003)
    def receiver_step(T):
        if True:
            t = now(); busy = False
            if LAT: nd.lat_min = max(0.02, min(list(LAT)))   # node15's clock offset assumes this minimum packet latency: the own sender's
            if nd.set_own_epoch(t, vb.epoch): log('own_epoch', epoch=vb.epoch)   # the epoch of the own packet headers
            if t - T['cal'] >= 1.0: udp.send([CAL_DG]); T['cal'] = t
            n = 0
            while not udp.q.empty() and n < 2000:
                g, ta, addr = udp.q.get_nowait(); handle(g, ta, addr); n += 1
            while OQ:
                e = OQ.popleft()
                if e['epoch'] != nd.own_epoch: st['own_kf_old_epoch'] += 1; continue
                e['A'] = None
                i, jobs = nd.add_own_keyframe(e['t'], t)
                with OL:
                    OWN[i] = e
                    for k in [k for k in OWN if k < i - 60]: del OWN[k]
                SQ.push_jobs(t, jobs); busy = True
            for pid, p in list(nd.peers.items()):
                for kf in list(p.kf):
                    if nd.keyframe_ready(pid, kf, t):
                        dec, cand = nd.requests(pid, kf)
                        if dec is not None and cand: SQ.push(t, pid, dec, cand)
                        busy = True
            if ready_rx.is_set():
                it = SQ.pop()
                while it is not None and t - it[0] > C['max_queue_s']: st['solve_stale'] += 1; it = SQ.pop()
                if it is not None:
                    t_av, pid, dec, i = it; t0 = time.perf_counter(); stt, s = nd.solve_pair(pid, dec, i); dt = time.perf_counter() - t0; solve_ms.append(dt * 1e3); st['solves'] += 1
                    t2 = now(); was = nd.peers[pid].lock.locked; nd.feed(pid, s, t2); p = nd.peers[pid]
                    h = dec['hdr']; z = float(nd_z(s)) if s is not None else None
                    log('solve', peer=pid, i=int(i), j=int(h['kf']), t_own=float(nd.own_t[i]), t_peer=float(h['t']), t_av=t_av, ms=dt * 1e3, ok=int(stt.get('ok', 0)),
                        ninl=stt.get('ninl'), ninl_raw=stt.get('ninl_raw'), amb=stt.get('amb'), sy=stt.get('sy'), st=stt.get('st'), z=z, npairs=stt.get('npairs'),
                        v4=None if s is None else [s['v_yaw'], s['v_x'], s['v_y'], s['v_z']], w6=None if s is None else s['w6'], x6=None if s is None else s['x6'],
                        p_peer=None if s is None else s['p_peer'], err=stt.get('error'))
                    if not was and p.lock.locked: log('lock', peer=pid, members=[[int(m['i']), int(m['j'])] for m in p.lock.members], e4=p.maint.est, e6=p.m6.est)
                    if was and not p.lock.locked: log('withdraw', peer=pid)
                    publish(pid, t2); busy = True
            if t - T['est'] >= 0.2:
                m = nd.estimate_messages(t); udp.send(m); T['est'] = t
                for pid in list(nd.peers): publish(pid, t)
            if t - T['stat'] >= 1.0:
                T['stat'] = t; out = {}
                for pid in list(nd.peers):
                    s = nd.state(pid); out[str(pid)] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k not in ('e6',)}
                    if s['e6'] is not None: out[str(pid)]['tilt6_deg'] = SIX.tilt_deg(s['e6'])
                pub_status.publish(String(data=json.dumps({'drone': C['id'], 't': t, 'primary': '6dof' if C['six'] else '4dof', 'peers': out})))
            return busy
    def nd_z(s):
        import rule15
        return rule15.zmargin(s)

    def census():
        while not stop.wait(5.0):
            try: census_once()
            except Exception as ex: log('error', where='census', err=repr(ex))
    def census_once():
        if True:
            tm = np.array(list(tim)) if tim else np.zeros((0, 2))
            msg = {'vio_poses': st['vio_poses'], 'imu': st['imu'], 'gyro_attitudes': vb.n_gyro, 'slerp_attitudes': vb.n_slerp, 'gyro_bias': [round(float(x), 5) for x in vb.bias], 'images': [st[f'img{c}'] for c in range(cal.ncam)], 'kf_sent': st['kf_sent'], 'kf_warm': st['kf_warm'], 'kf_ground': st['kf_ground'],
                   'kf_hold': st['kf_hold'], 'kf_no_pose': st['kf_no_pose'], 'img_incomplete': st['img_incomplete'], 'kf_dropped_busy': st['kf_dropped_busy'], 'kf_error': st['kf_error'],
                   'sender_wait_ms_med': float(np.median(tm[:, 0]) * 1e3) if len(tm) else None, 'sender_ms_med': float(np.median(tm[:, 1]) * 1e3) if len(tm) else None,
                   'udp_rx': udp.n_rx, 'udp_tx': udp.n_tx, 'udp_drop': udp.n_drop, 'rx_no_calib': st['rx_no_calib'], 'solves': st['solves'], 'solve_stale': st['solve_stale'],
                   'solve_ms_med': float(np.median(list(solve_ms))) if solve_ms else None, 'queue': len(SQ.q), 'lat_min_s': nd.lat_min, 'own_epoch': nd.own_epoch,
                   'rx_error': st['rx_error'], 'peers': {str(p): nd.state(p)['status'] for p in list(nd.peers)}}
            print('[relloc] census ' + json.dumps(msg), flush=True); log('census', **msg)
    ths = [threading.Thread(target=f, name=n, daemon=True) for f, n in ((sender_loop, 'relloc-sender'), (receiver_loop, 'relloc-receiver'), (census, 'relloc-census'))]
    for th in ths: th.start()
    def _stop(*_):
        stop.set()
    signal.signal(signal.SIGTERM, _stop)
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    ex = SingleThreadedExecutor(); ex.add_node(ros)
    try:
        while rclpy.ok() and not stop.is_set(): ex.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    stop.set(); [th.join(timeout=2.0) for th in ths]; udp.close()
    try: log('stop', **dict(st))
    except Exception: pass
    LOG.close()
    try: ros.destroy_node(); rclpy.shutdown()
    except Exception: pass


if __name__ == '__main__':
    main()
