"""Byte-exact packet codec (round 14). One keyframe packet:
  header (24 B): magic 'R2', version, drone id, keyframe index (uint32), timestamp (float64), roll, pitch (float32, VIO tilt of the rig frame)
  per camera (4 B): keypoint count, count of leading keypoints that carry depth (3-D first)
  per keypoint: pixel (u, v) packed into ceil((bits_u + bits_v) / 8) bytes (whole pixels, bits from the image size: 3 B at 1280 x 800),
                descriptor = PCA-D coefficients at 4 bits (D/2 B) + scale (version 1: fp16, 2 B; version 2: 8-bit logarithmic code over
                [STEP_LO, STEP_HI], 1 B, <= 0.5 % step error, 65x below the 4-bit rounding noise), and fp16 depth along the pixel ray (2 B) for 3-D keypoints.
The receiver rebuilds the 64-D descriptor from the shared PCA basis and the 3-D point as camera centre + depth * ray(u, v) in the sender's
gravity-aligned rig frame (intrinsics and camera extrinsics are exchanged once). decode(encode(x)) reproduces packet.py's descriptor emulation.

Version 3 (round 15, transport format; encode_v3 / decode_v3): one keyframe = a list of self-contained datagrams of at most CHUNK bytes
(default 1200, below a 1500-B MTU with IP/UDP/RTPS headers), so a lost datagram costs its keypoints, not the keyframe.
  every datagram: compact header (12 B) magic, version 3, drone, keyframe (uint32), camera, chunk index, chunk count, flags
                  (bit0: this datagram carries the keyframe header, bit1: entropy-coded codes, reserved)
  chunk 0 of every camera also carries the keyframe header (40 B): time (float64), the sender's VIO pose of its rig/body frame at the
                  keyframe in its own VIO world (roll, pitch, yaw rad, x, y, z m; float32), VIO epoch (uint8, +1 on every VIO
                  (re)initialisation), VIO age (uint16, 0.1 s since that initialisation, saturating), health (uint8, sender's depth-quality code),
                  PCA basis id (uint8), camera count (uint8), keypoints per camera (uint16). Four copies per keyframe on a 4-camera rig.
  chunk body: count n, count n3 of leading keypoints with depth (uint16 each), index of the first keypoint in the camera (uint16), then
                  for 3-D keypoints one 32-bit word (u, v, log depth over [DEP_LO, DEP_HI] in the remaining 32 - bits_u - bits_v bits:
                  11 bits at 1280 x 800, step 0.34 %, error <= 0.17 %, below the 1.9 % triangulation error) and for 2-D keypoints the
                  v1/v2 pixel field; then the 4-bit PCA codes (D/2 B) and the one-byte step of every keypoint.
The receiver needs the sender's VIO pose for the deployed consensus (solves compared through both VIO worlds), for the VIO-world output and
for maintenance; the epoch tells it to drop an estimate after a VIO reset; the age replaces 'time since take-off' (VIO settling).
"""
import struct, numpy as np

MAGIC = b'R2'; VERSION = 2; VERSION_T = 3   # 2: one-byte logarithmic quantisation step (09-28); 1: fp16 step
STEP_LO, STEP_HI = 0.015, 0.2   # step = max|PCA coefficient| / 7; 0.023-0.113 on drones, TUM-VI and EuRoC
HDR = struct.Struct('<2sBBIdff')          # 24 bytes
CAM = struct.Struct('<HH')
HC3 = struct.Struct('<2sBBIBBBB')         # 12-byte compact datagram header (v3)
HK3 = struct.Struct('<dffffffBHBBBH')     # 40-byte keyframe header (v3)
CB3 = struct.Struct('<HHH')               # chunk body: n, n3, first index
DEP_LO, DEP_HI = 0.2, 200.0               # log-depth range of the 32-bit word (m)
CHUNK = 1200                              # max datagram bytes


def _bits(n): return int(np.ceil(np.log2(max(int(n), 2))))


def step_code(step):
    x = (np.log(np.asarray(step, np.float64)) - np.log(STEP_LO)) / (np.log(STEP_HI) - np.log(STEP_LO)) * 255.0
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
def step_value(code):
    return np.exp(np.log(STEP_LO) + np.asarray(code, np.float64) / 255.0 * (np.log(STEP_HI) - np.log(STEP_LO))).astype(np.float32)
class Codec:
    def __init__(self, mu, basis, width, height, dims=32, version=VERSION):
        self.mu = np.asarray(mu, np.float32); self.B = np.asarray(basis[:dims], np.float32); self.D = dims; self.version = version; self.step_bytes = 1 if version >= 2 else 2
        self.bu = _bits(width); self.bv = _bits(height); self.pix_bytes = (self.bu + self.bv + 7) // 8

    # ---- descriptors: PCA-D, 4-bit symmetric codes, one fp16 scale per keypoint (as packet.py PACKET_PCA) ----
    def quant_desc(self, desc64):
        z = (np.asarray(desc64, np.float32) - self.mu) @ self.B.T
        s = 7.0 / np.maximum(np.abs(z).max(1, keepdims=True), 1e-6); q = np.clip(np.round(z * s), -7, 7).astype(np.int8)
        if self.version >= 2: return q, step_code(1.0 / s[:, 0])        # step = 1/s, sent as a one-byte logarithmic code
        return q, (1.0 / s[:, 0]).astype(np.float16)                 # step = 1/s, sent as fp16

    def step_float(self, step):
        return step_value(step) if self.version >= 2 else np.asarray(step, np.float32)
    def dequant_desc(self, q, step):
        return ((q.astype(np.float32) * self.step_float(step)[:, None]) @ self.B + self.mu).astype(np.float16)

    # ---- encode / decode ----
    def encode(self, drone, kf, t, roll, pitch, cams):
        """cams: list of dicts {'kpts': (n, 2) whole pixels, 'desc': (n, 64) or precomputed 'q' (n, D) int8 codes + 'step' (n,) fp16,
        'depth': (n,) with finite values first (3-D first)}."""
        out = [HDR.pack(MAGIC, self.version, drone, kf, t, roll, pitch)]
        for cm in cams:
            kp = np.asarray(cm['kpts']); n = len(kp); dep = np.asarray(cm['depth'], np.float64) if cm.get('depth') is not None else np.full(n, np.nan)
            n3 = int(np.isfinite(dep).sum())
            assert np.isfinite(dep[:n3]).all(), 'depth-bearing keypoints must come first'
            u = np.round(kp[:, 0]).astype(np.uint64); v = np.round(kp[:, 1]).astype(np.uint64)
            assert (u < (1 << self.bu)).all() and (v < (1 << self.bv)).all()
            w = (u << np.uint64(self.bv)) | v
            pix = np.stack([(w >> np.uint64(8 * b)) & np.uint64(0xFF) for b in range(self.pix_bytes)], 1).astype(np.uint8)
            q, step = (np.asarray(cm['q'], np.int8), (step_code(cm['step']) if self.version >= 2 and np.asarray(cm['step']).dtype != np.uint8 else np.asarray(cm['step'], np.uint8 if self.version >= 2 else np.float16))) if 'q' in cm else self.quant_desc(cm['desc'])
            nib = (q & 0x0F).astype(np.uint8); packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
            out += [CAM.pack(n, n3), pix.tobytes(), packed.tobytes(), step.tobytes(), dep[:n3].astype(np.float16).tobytes()]
        return b''.join(out)

    def decode_codes(self, buf, n_cams):
        """as decode, but returns the 4-bit codes and steps instead of 64-D descriptors (reconstruct on the GPU: dequant_desc_torch)."""
        return self.decode(buf, n_cams, reconstruct=False)

    def dequant_desc_torch(self, q, step, torch, dev='cuda'):
        if not hasattr(self, '_Bt'): self._Bt = torch.from_numpy(self.B).to(dev); self._mut = torch.from_numpy(self.mu).to(dev)
        qt = torch.from_numpy(q).to(dev).float(); st = torch.from_numpy(self.step_float(step)).to(dev)
        return ((qt * st[:, None]) @ self._Bt + self._mut).half()

    def decode(self, buf, n_cams, reconstruct=True):
        magic, ver, drone, kf, t, roll, pitch = HDR.unpack_from(buf, 0); o = HDR.size
        assert magic == MAGIC and ver in (1, 2), ver
        sb = 1 if ver >= 2 else 2
        cams = []
        for _ in range(n_cams):
            n, n3 = CAM.unpack_from(buf, o); o += CAM.size
            pix = np.frombuffer(buf, np.uint8, n * self.pix_bytes, o).reshape(n, self.pix_bytes).astype(np.uint64); o += n * self.pix_bytes
            w = np.zeros(n, np.uint64)
            for b in range(self.pix_bytes): w |= pix[:, b] << np.uint64(8 * b)
            kp = np.stack([(w >> np.uint64(self.bv)).astype(np.float32), (w & np.uint64((1 << self.bv) - 1)).astype(np.float32)], 1)
            packed = np.frombuffer(buf, np.uint8, n * self.D // 2, o).reshape(n, self.D // 2); o += n * self.D // 2
            lo = (packed & 0x0F).astype(np.int8); hi = (packed >> 4).astype(np.int8); lo[lo > 7] -= 16; hi[hi > 7] -= 16
            q = np.empty((n, self.D), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
            step = (np.frombuffer(buf, np.uint8, n, o).copy() if sb == 1 else np.frombuffer(buf, np.float16, n, o).copy()); o += sb * n
            dep = np.full(n, np.nan, np.float32); dep[:n3] = np.frombuffer(buf, np.float16, n3, o).astype(np.float32); o += 2 * n3
            stepf = step_value(step) if sb == 1 else step.astype(np.float32)
            cams.append({'kpts': kp, 'desc': ((q.astype(np.float32) * stepf[:, None]) @ self.B + self.mu).astype(np.float16), 'depth': dep} if reconstruct else {'kpts': kp, 'q': q, 'step': step, 'depth': dep, 'version': ver})
        assert o == len(buf), (o, len(buf))
        return {'drone': drone, 'kf': kf, 't': t, 'roll': roll, 'pitch': pitch, 'cams': cams}

    # ---- version 3: self-contained datagrams ----
    def dep_bits(self): return 32 - self.bu - self.bv
    def _word(self, u, v, dep):
        bd = self.dep_bits(); assert bd >= 8, 'image too large for the 32-bit pixel+depth word'
        x = (np.log(np.clip(dep, DEP_LO, DEP_HI)) - np.log(DEP_LO)) / (np.log(DEP_HI) - np.log(DEP_LO)) * ((1 << bd) - 1)
        dq = np.clip(np.round(x), 0, (1 << bd) - 1).astype(np.uint64)
        return ((u.astype(np.uint64) << np.uint64(self.bv + bd)) | (v.astype(np.uint64) << np.uint64(bd)) | dq).astype(np.uint32)
    def _unword(self, w):
        bd = self.dep_bits(); w = w.astype(np.uint64)
        dq = (w & np.uint64((1 << bd) - 1)).astype(np.float64); v = ((w >> np.uint64(bd)) & np.uint64((1 << self.bv) - 1)).astype(np.float32); u = (w >> np.uint64(self.bv + bd)).astype(np.float32)
        dep = np.exp(np.log(DEP_LO) + dq / ((1 << bd) - 1) * (np.log(DEP_HI) - np.log(DEP_LO))).astype(np.float32)
        return u, v, dep
    def per_kp_bytes(self, has3d): return (4 if has3d else self.pix_bytes) + self.D // 2 + 1
    def encode_v3(self, drone, kf, hdr, cams, chunk=CHUNK):
        """hdr: {'t', 'roll', 'pitch', 'yaw', 'x', 'y', 'z', 'epoch', 'age_s', 'health', 'basis'}; cams as encode() (3-D keypoints first).
        Returns the list of datagrams (bytes) of one keyframe."""
        ncam = len(cams); K = max((len(c['kpts']) for c in cams), default=0)
        khdr = HK3.pack(float(hdr['t']), *[float(hdr[k]) for k in ('roll', 'pitch', 'yaw', 'x', 'y', 'z')], int(hdr.get('epoch', 0)) & 0xFF,
                        min(int(round(10 * float(hdr.get('age_s', 0.0)))), 0xFFFF), int(hdr.get('health', 0)) & 0xFF, int(hdr.get('basis', 0)) & 0xFF, ncam, K)
        per_cam = []
        for ci, cm in enumerate(cams):
            kp = np.asarray(cm['kpts']); n = len(kp); dep = np.asarray(cm['depth'], np.float64) if cm.get('depth') is not None else np.full(n, np.nan)
            n3 = int(np.isfinite(dep).sum()); assert np.isfinite(dep[:n3]).all(), 'depth-bearing keypoints must come first'
            u = np.round(kp[:, 0]).astype(np.uint64); v = np.round(kp[:, 1]).astype(np.uint64); assert (u < (1 << self.bu)).all() and (v < (1 << self.bv)).all()
            q, step = (np.asarray(cm['q'], np.int8), (step_code(cm['step']) if np.asarray(cm['step']).dtype != np.uint8 else np.asarray(cm['step'], np.uint8))) if 'q' in cm else self.quant_desc(cm['desc'])
            if self.version < 2: step = step_code(step.astype(np.float32))
            nib = (q & 0x0F).astype(np.uint8); packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
            w3 = self._word(u[:n3], v[:n3], dep[:n3]); w2 = (u[n3:] << np.uint64(self.bv)) | v[n3:]
            pix2 = np.stack([(w2 >> np.uint64(8 * b)) & np.uint64(0xFF) for b in range(self.pix_bytes)], 1).astype(np.uint8) if n > n3 else np.zeros((0, self.pix_bytes), np.uint8)
            # split into chunks: keypoints in order, as many as fit
            bounds = []; a = 0
            while True:
                room = chunk - HC3.size - CB3.size - (HK3.size if not bounds else 0); b = a; used = 0
                while b < n and used + self.per_kp_bytes(b < n3) <= room: used += self.per_kp_bytes(b < n3); b += 1
                if b == a and a < n: raise ValueError('chunk too small for one keypoint')
                bounds.append((a, b)); a = b
                if a >= n: break
            cur = []
            for k_, (a, b) in enumerate(bounds):
                a3, b3 = min(a, n3), min(b, n3)
                body = [HC3.pack(MAGIC, VERSION_T, drone, kf, ci, k_, len(bounds), 1 if k_ == 0 else 0)]
                if k_ == 0: body.append(khdr)
                body += [CB3.pack(b - a, b3 - a3, a), w3[a3:b3].tobytes(), pix2[max(a, n3) - n3:max(b, n3) - n3].tobytes(), packed[a:b].tobytes(), step[a:b].tobytes()]
                cur.append(b''.join(body))
            per_cam.append(cur)
        # round 16 (REVIEW15 item 5): cameras interleaved round-robin, so a burst cut short by a slow link loses keypoints of every camera,
        # not whole trailing cameras; and staggered (review16/loss 'stag'): camera c's chunks rotated by c x ceil(max chunks / cameras), key
        # ((chunk + c * s) mod chunks_c, c), so the header chunks (chunk 0 of each camera, the only copies of the keyframe header) are spread
        # over the burst instead of sent back to back (Gilbert 10 % bursts: 6.4 -> 0.4 % dropped keyframes). Decode is order-free.
        s_ = -(-max((len(x) for x in per_cam), default=1) // max(ncam, 1))
        return [per_cam[c][k_] for (_, c, k_) in sorted(((k_ + c * s_) % len(x), c, k_) for c, x in enumerate(per_cam) for k_ in range(len(x)))]
    # round 15: vote datagram (rule15.DepthVote summaries): compact header with camera 0xFF, flags bit 1; body = time (f8), entry count (u1),
    # per entry peer id (u1), median depth log-ratio x 1e4 (i2), solve count (u2) -> 5 B per peer (decode_v3 skips it)
    VOTE_CAM = 0xFF; VT = struct.Struct('<dB'); VE = struct.Struct('<BhH')
    @staticmethod
    def encode_vote(drone, t, entries):
        body = [HC3.pack(MAGIC, VERSION_T, drone, 0, Codec.VOTE_CAM, 0, 1, 2), Codec.VT.pack(float(t), len(entries))]
        body += [Codec.VE.pack(int(p) & 0xFF, int(np.clip(round(1e4 * m), -32767, 32767)), min(int(n), 0xFFFF)) for p, m, n in entries]
        return b''.join(body)
    @staticmethod
    def decode_vote(dgram):
        magic, ver, drone, kf, cam, ck, nck, fl = HC3.unpack_from(dgram, 0)
        if cam != Codec.VOTE_CAM or not fl & 2: return None
        t, n = Codec.VT.unpack_from(dgram, HC3.size); o = HC3.size + Codec.VT.size; out = []
        for _ in range(n): p, m, c = Codec.VE.unpack_from(dgram, o); o += Codec.VE.size; out.append((p, m / 1e4, c))
        return {'drone': drone, 't': t, 'entries': out}
    # round 16: estimate datagram of the pair protocol (rule15.PairLink): compact header with camera 0xFE, flags bit 2; body = time (f8),
    # peer id (u1), sender's VIO epoch (u1), the epoch of the peer's VIO the estimate refers to (u1), lock id (u2, +1 per (re)lock),
    # acknowledged peer lock id (u2, 0xFFFF = none), status (u1: 0 none, 1 provisional, 2 confirmed, 3 withdrawn; bit 7 = ack requested),
    # the sender's current estimate of the peer's VIO world in its own (yaw deg, x, y, z; f8) -> 60 B
    EST_CAM = 0xFE; ET = struct.Struct('<dBBBHHBdddd'); EST_STATUS = ('none', 'provisional', 'confirmed', 'withdrawn')
    @staticmethod
    def encode_est(drone, m):
        """m: {'t', 'peer', 'epoch', 'peer_epoch', 'lock_id', 'ack_id' (None = no ack), 'status', 'est' (4,) or None, 'need_ack' (bool)}."""
        e = m.get('est'); e = (np.nan,) * 4 if e is None else tuple(float(v) for v in e)
        return HC3.pack(MAGIC, VERSION_T, drone, 0, Codec.EST_CAM, 0, 1, 4) + Codec.ET.pack(float(m['t']), int(m['peer']) & 0xFF, int(m['epoch']) & 0xFF,
                int(m['peer_epoch']) & 0xFF, int(m['lock_id']) & 0xFFFF, 0xFFFF if m.get('ack_id') is None else int(m['ack_id']) & 0xFFFF,
                Codec.EST_STATUS.index(m['status']) | (0x80 if m.get('need_ack') else 0), *e)
    @staticmethod
    def decode_est(dgram):
        """estimate datagram -> dict (as encode_est plus 'drone'), None for anything else (never raises)."""
        try:
            if len(dgram) != HC3.size + Codec.ET.size: return None
            magic, ver, drone, kf, cam, ck, nck, fl = HC3.unpack_from(dgram, 0)
            if magic != MAGIC or ver != VERSION_T or cam != Codec.EST_CAM or not fl & 4: return None
            t, peer, ep, pep, lid, ack, st, y, x, y2, z = Codec.ET.unpack_from(dgram, HC3.size)
            need = bool(st & 0x80); st &= 0x7F
            if st >= len(Codec.EST_STATUS) or not np.isfinite(t): return None
            est = np.array([y, x, y2, z]) if np.isfinite([y, x, y2, z]).all() else None
            return {'drone': drone, 't': t, 'peer': peer, 'epoch': ep, 'peer_epoch': pep, 'lock_id': lid, 'ack_id': None if ack == 0xFFFF else ack,
                    'status': Codec.EST_STATUS[st], 'est': est, 'need_ack': need}
        except struct.error:
            return None
    @staticmethod
    def peek(dgram):
        """compact header of a datagram: (drone, keyframe, camera, chunk, chunk count, has keyframe header)."""
        magic, ver, drone, kf, cam, ck, nck, fl = HC3.unpack_from(dgram, 0); assert magic == MAGIC and ver == VERSION_T
        return drone, kf, cam, ck, nck, bool(fl & 1)
    @staticmethod
    def keyframe_header(dgram):
        drone, kf, cam, ck, nck, has = Codec.peek(dgram)
        if not has: return None
        t, r, p, y, x_, y_, z_, ep, age, hl, bs, ncam, K = HK3.unpack_from(dgram, HC3.size)
        return {'drone': drone, 'kf': kf, 't': t, 'roll': r, 'pitch': p, 'yaw': y, 'x': x_, 'y': y_, 'z': z_, 'epoch': ep, 'age_s': age / 10.0, 'health': hl, 'basis': bs, 'ncam': ncam, 'K': K}
    def validate(self, g):
        """round 16 (REVIEW15 S1 robustness): True when g is a well-formed v3 keyframe datagram for this codec (magic, version, length
        consistent with its counts and this codec's pixel bytes / PCA dims, finite header). A receiver drops anything else before storing it."""
        try:
            if len(g) < HC3.size: return False
            magic, ver, drone, kf, cam, ck, nck, fl = HC3.unpack_from(g, 0)
            if magic != MAGIC or ver != VERSION_T or cam in (Codec.VOTE_CAM, Codec.EST_CAM) or nck == 0 or ck >= nck: return False
            o = HC3.size
            if fl & 1:
                if len(g) < o + HK3.size: return False
                h = HK3.unpack_from(g, o); o += HK3.size
                if not all(np.isfinite(h[:7])) or h[11] == 0 or cam >= h[11]: return False
            if len(g) < o + CB3.size: return False
            n, n3, first = CB3.unpack_from(g, o); o += CB3.size
            return n3 <= n and len(g) == o + 4 * n3 + (n - n3) * self.pix_bytes + n * (self.D // 2 + 1)
        except (struct.error, TypeError, ValueError):
            return False
    def decode_v3(self, dgrams, reconstruct=True):
        """datagrams of one keyframe (any subset, any order) -> {'hdr', 'cams': per camera {'kpts', 'desc' or 'q'+'step', 'depth', 'idx'},
        'missing': chunks not received}. Keypoints of lost chunks are absent (idx gives each keypoint's position in the sender's order)."""
        hdr = None; per = {}; seen = set(); nck_of = {}
        for g in dgrams:
            drone, kf, cam, ck, nck, has = self.peek(g)
            if cam in (Codec.VOTE_CAM, Codec.EST_CAM): continue   # vote (decode_vote) / estimate (decode_est) datagram
            o = HC3.size; seen.add((cam, ck)); nck_of[cam] = nck
            if has:
                h = self.keyframe_header(g); o += HK3.size
                if hdr is None: hdr = h
            n, n3, first = CB3.unpack_from(g, o); o += CB3.size
            w3 = np.frombuffer(g, np.uint32, n3, o); o += 4 * n3
            pix = np.frombuffer(g, np.uint8, (n - n3) * self.pix_bytes, o).reshape(n - n3, self.pix_bytes).astype(np.uint64); o += (n - n3) * self.pix_bytes
            packed = np.frombuffer(g, np.uint8, n * self.D // 2, o).reshape(n, self.D // 2); o += n * self.D // 2
            step = np.frombuffer(g, np.uint8, n, o).copy(); o += n
            assert o == len(g), (o, len(g))
            u3, v3, d3 = self._unword(w3); w = np.zeros(n - n3, np.uint64)
            for b in range(self.pix_bytes): w |= pix[:, b] << np.uint64(8 * b)
            u2 = (w >> np.uint64(self.bv)).astype(np.float32); v2 = (w & np.uint64((1 << self.bv) - 1)).astype(np.float32)
            lo = (packed & 0x0F).astype(np.int8); hi = (packed >> 4).astype(np.int8); lo[lo > 7] -= 16; hi[hi > 7] -= 16
            q = np.empty((n, self.D), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
            dep = np.full(n, np.nan, np.float32); dep[:n3] = d3
            per.setdefault(cam, []).append((first, np.concatenate([np.stack([u3, v3], 1), np.stack([u2, v2], 1)]).astype(np.float32), q, step, dep))
        ncam = hdr['ncam'] if hdr is not None else (max(per) + 1 if per else 0); cams = []
        for c in range(ncam):
            parts = sorted(per.get(c, []), key=lambda x: x[0])
            if not parts: cams.append({'kpts': np.zeros((0, 2), np.float32), 'q': np.zeros((0, self.D), np.int8), 'step': np.zeros(0, np.uint8), 'depth': np.zeros(0, np.float32), 'idx': np.zeros(0, int)}); continue
            idx = np.concatenate([np.arange(x[0], x[0] + len(x[1])) for x in parts]); kp = np.concatenate([x[1] for x in parts]); q = np.concatenate([x[2] for x in parts])
            step = np.concatenate([x[3] for x in parts]); dep = np.concatenate([x[4] for x in parts])
            o3 = np.argsort(~np.isfinite(dep), kind='stable')   # keep 3-D first after reassembly (chunks are in sender order, so this is a no-op unless chunks were lost)
            kp, q, step, dep, idx = kp[o3], q[o3], step[o3], dep[o3], idx[o3]
            d = {'kpts': kp, 'depth': dep, 'idx': idx}
            if reconstruct: d['desc'] = ((q.astype(np.float32) * step_value(step)[:, None]) @ self.B + self.mu).astype(np.float16)
            else: d.update(q=q, step=step)
            cams.append(d)
        missing = [(c, k) for c in nck_of for k in range(nck_of[c]) if (c, k) not in seen]
        return {'hdr': hdr, 'cams': cams, 'missing': missing}

    def size(self, n_kp, n_3d):
        return HDR.size + sum(CAM.size + n * (self.pix_bytes + self.D // 2 + self.step_bytes) + 2 * m for n, m in zip(n_kp, n_3d))


def ray_table(unproject, intr, width, height):
    """unit rays of every whole pixel of one camera (height, width, 3) float32: built once per peer camera, then a packet's rays are a lookup."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)); uv = np.stack([u.ravel(), v.ravel()], 1)
    return unproject(uv, intr).astype(np.float32).reshape(height, width, 3)
