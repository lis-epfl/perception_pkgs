"""The drone's own calibration, read from its flight folder (the folder selfcalib/tool/deploy_vio.py writes), and the calibration datagram
that tells the other drones how to turn this drone's packet pixels into rays.

Flight folder:
  kalibr_imucam_chain.yaml   per camera intrinsics [fx fy cx cy], distortion_coeffs [k1..k4] (equidistant = KB4), T_imu_cam, resolution,
                             timeshift_cam_imu (t_imu = t_cam + shift)
  flight_stiffness.env       its last line OV_NONRADIAL='p1,p2,s;...' = the lens terms of that calibration (also lens_terms.env); the
                             environment variable OV_NONRADIAL, when set, wins (the estimator reads the same variable)
  estimator_flight.yaml      cam_topic0..N and cam_msg_type, when they were added (vio/README.md)
The rig/body frame of relloc is the estimator's IMU frame: the VIO pose in the packet header is the IMU pose in the VIO world, and the
camera extrinsics are T_imu_cam."""
import os, re, struct
import numpy as np
from . import lens

RIG = 'drone4'


def _load_yaml(path):
    import yaml
    txt = open(path).read()
    txt = re.sub(r'^%YAML:[^\n]*\n', '', txt)   # OpenCV-style header of the estimator's files
    return yaml.safe_load(txt)


def parse_terms(s, ncam):
    """OV_NONRADIAL 'p1,p2,s;p1,p2,s;...' -> (ncam, 3); '' -> zeros."""
    s = (s or '').strip().strip("'\"")
    if not s: return np.zeros((ncam, 3))
    rows = [[float(x) for x in r.split(',')] for r in s.split(';') if r.strip()]
    if len(rows) != ncam or any(len(r) != 3 for r in rows):
        raise ValueError(f'OV_NONRADIAL must hold p1,p2,s for each of {ncam} cameras, got {s!r}')
    return np.array(rows, float)


def _env_value(path, key):
    if not os.path.exists(path): return None
    val = None
    for ln in open(path):
        m = re.match(r'\s*(?:export\s+)?' + key + r'=(.*)$', ln.strip())
        if m: val = m.group(1).strip()
    return val


class Calib:
    """ncam cameras: intr[c] (11 numbers: fx fy cx cy k1..k4 p1 p2 s), R_bc[c] / p_bc[c] (camera -> IMU), width, height, td (s), topics."""

    def __init__(self, ncam, intr, R_bc, p_bc, width, height, td=0.0, topics=None, msg_type=None, source='', imu_topic=None, imu_msg_type=None):
        self.ncam = ncam; self.intr = [np.asarray(v, float) for v in intr]; self.R_bc = [np.asarray(r, float) for r in R_bc]
        self.p_bc = [np.asarray(p, float) for p in p_bc]; self.W = int(width); self.H = int(height); self.td = float(td)
        self.topics = topics; self.msg_type = msg_type; self.source = source; self.imu_topic = imu_topic; self.imu_msg_type = imu_msg_type
        for c in range(ncam):
            assert len(self.intr[c]) == 11 and self.R_bc[c].shape == (3, 3) and self.p_bc[c].shape == (3,)
            assert abs(np.linalg.det(self.R_bc[c]) - 1.0) < 1e-3, f'cam{c}: T_imu_cam rotation is not a rotation'

    @property
    def fx(self): return [float(v[0]) for v in self.intr]

    @property
    def has_terms(self): return any(np.any(v[8:11] != 0) for v in self.intr)

    def unproject(self, c, uv):
        return lens.unproject(np.asarray(uv, np.float64), self.intr[c])

    def peer_calib(self, node15):
        return node15.PeerCalib(lambda c, uv: self.unproject(c, uv), self.R_bc, self.p_bc, self.W, self.H, fx=self.fx)

    # ---- calibration datagram: 'RC', version 1, drone id, camera count, width, height, session (uint32, random per process start: a
    #      receiver that sees it change treats the peer as new), then per camera 23 float64 (fx fy cx cy k1..k4 p1 p2 s, R_imu_cam
    #      row-major, p_cam_in_imu); 13 + 184 * ncam bytes (749 B for 4 cameras)
    HDR = struct.Struct('<2sBBBHHI'); MAGIC = b'RC'; VERSION = 1

    def encode(self, drone, session=0):
        body = [self.HDR.pack(self.MAGIC, self.VERSION, int(drone) & 0xFF, self.ncam, self.W, self.H, int(session) & 0xFFFFFFFF)]
        for c in range(self.ncam):
            body.append(struct.pack('<23d', *self.intr[c], *self.R_bc[c].ravel(), *self.p_bc[c]))
        return b''.join(body)

    @classmethod
    def is_calib(cls, g): return len(g) >= cls.HDR.size and g[:2] == cls.MAGIC

    @classmethod
    def decode(cls, g):
        """(drone id, Calib, session) or None for a malformed datagram."""
        try:
            magic, ver, drone, ncam, W, H, session = cls.HDR.unpack_from(g, 0)
            if magic != cls.MAGIC or ver != cls.VERSION or ncam == 0 or len(g) != cls.HDR.size + 184 * ncam: return None
            v = np.array(struct.unpack_from(f'<{23 * ncam}d', g, cls.HDR.size)).reshape(ncam, 23)
            if not np.isfinite(v).all(): return None
            return drone, Calib(ncam, v[:, :11], v[:, 11:20].reshape(-1, 3, 3), v[:, 20:23], W, H, source=f'datagram of drone {drone}'), session
        except Exception:
            return None

    def same(self, o):
        return o is not None and self.ncam == o.ncam and self.W == o.W and self.H == o.H and all(
            np.array_equal(a, b) for a, b in zip(self.intr + self.R_bc + self.p_bc, o.intr + o.R_bc + o.p_bc))


def load_flight_folder(folder, terms=None):
    """Calib of the drone from its flight folder. terms: OV_NONRADIAL string (None = environment, then the folder's env files)."""
    chain = os.path.join(folder, 'kalibr_imucam_chain.yaml')
    if not os.path.exists(chain): raise FileNotFoundError(f'{chain}: not a flight folder (deploy_vio.py writes it)')
    Y = _load_yaml(chain); cams = sorted((k for k in Y if re.fullmatch(r'cam\d+', k)), key=lambda k: int(k[3:]))
    if not cams: raise ValueError(f'{chain}: no cam0.. entries')
    ncam = len(cams); intr = []; R = []; p = []; res = set(); tds = []; topics_chain = []
    for k in cams:
        e = Y[k]
        if 'T_imu_cam' not in e: raise ValueError(f'{chain}:{k}: needs T_imu_cam (T_cam_imu is its inverse; selfcalib/README section 3)')
        if e.get('distortion_model', 'equidistant') != 'equidistant': raise ValueError(f'{chain}:{k}: relloc handles the equidistant (KB4) model only')
        T = np.array(e['T_imu_cam'], float); R.append(T[:3, :3]); p.append(T[:3, 3])
        intr.append([float(x) for x in e['intrinsics']] + [float(x) for x in e['distortion_coeffs']])
        res.add(tuple(int(x) for x in e['resolution'])); tds.append(float(e.get('timeshift_cam_imu', 0.0))); topics_chain.append(e.get('rostopic'))
    if len(res) != 1: raise ValueError(f'{chain}: cameras of different resolution {res}')
    W, H = res.pop()
    src = 'environment OV_NONRADIAL'
    if terms is None: terms = os.environ.get('OV_NONRADIAL')
    if terms is None:
        for f in ('flight_stiffness.env', 'lens_terms.env'):
            terms = _env_value(os.path.join(folder, f), 'OV_NONRADIAL'); src = f'{f} OV_NONRADIAL'
            if terms is not None: break
    if terms is None: src = 'none (zeros)'
    Q = parse_terms(terms, ncam); intr = [np.concatenate([np.asarray(intr[c], float), Q[c]]) for c in range(ncam)]
    topics = None; msg_type = None; imu_topic = None; imu_type = None; est = os.path.join(folder, 'estimator_flight.yaml')
    if os.path.exists(est):
        try:
            E = _load_yaml(est)
            t = [E.get(f'cam_topic{c}') for c in range(ncam)]
            if all(t): topics = [str(x) for x in t]
            msg_type = E.get('cam_msg_type'); imu_topic = E.get('imu_topic'); imu_type = E.get('imu_msg_type')
        except Exception:
            pass
    return Calib(ncam, intr, R, p, W, H, td=tds[0], topics=topics, msg_type=msg_type, source=f'{chain}; lens terms: {src}', imu_topic=imu_topic, imu_msg_type=imu_type)
