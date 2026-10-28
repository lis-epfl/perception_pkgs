"""The sender of one drone: every keyframe (5 Hz), triangulate keypoints against the drone's own past keyframes and build the v3 packet.

This is relloc2's orin_pkg/sender_lean.py (round 16) in its deployed configuration (GRAPH=1, PACKET=1, CODEC_V=3, EARLY=0, INTRA=0), turned
from a benchmark over a sequence file into a class fed one live keyframe at a time, with the parallax rule PT8F6G8 of relloc2 review24/fix
(section 8) in place of sender_lean's 3-deg floor. At sender_lean's settings (parmin = parmin_early = 3, mem_gate = 0,
mem_keep_fresh = False) the arithmetic is sender_lean's, expression for expression, except sender_lean's positive-depth test on every
view in use (dep > 0): the 0.35-deg residual test on the same views implies it (cos = dep / |v| >= cos 0.35 deg), so it is left out.
test/test_sender_parity.py feeds a sender_lean sequence file through this class and compares the packets byte for byte.
  partners   the same camera at keyframes -10, -7, -5, -4, -3, -2, -1 (0.2-2 s back at 5 Hz) with a baseline >= 8 cm
  matching   mutual nearest neighbour of the top 1500 keypoints (by score) against the partners' 3000, cosine > 0.75, epipolar penalty
             folded into the fp16 product (SOFT 0.1), then the exact 1-deg band on the chosen match
  3-D        >= 2 partner views, views off by > 0.35 deg pruned (2 passes), depth 0.3-12 m, parallax >= the floor: parmin_early (8 deg)
             before the switch time (set_floor_switch; the node sets it to the take-off latch + 8 s), parmin (6 deg) from it on
  memory     a camera with < 100 fresh 3-D points takes depths from its last stored keyframe (stored every 5 keyframes), 1-deg gate;
             a keyframe is stored only when >= 100 of its fresh points have parallax >= mem_gate (8 deg), and only those points
             (the climb keyframe whose points have little parallax no longer becomes the hover's memory); a keypoint with a fresh
             3-D point keeps it (mem_keep_fresh; sender_lean let a memory depth replace it)
  packet     per camera the 500 (= 2000 / cameras) best keypoints, 3-D first, PCA-32 4-bit codes, log depth; a camera with < 50 valid
             keypoints is sent empty; header = time, VIO roll / pitch / yaw / x / y / z, VIO epoch, VIO age, health, basis 0
Differences from sender_lean, all outside its arithmetic: keyframes come one at a time (ingest + triangulate + packet in keyframe());
old poses are dropped (sender_lean kept the whole sequence); reset() drops partners, depth memory and health on a new VIO epoch, as
sender_lean's docstring asks of a flight sender (with reset_on_epoch=False the epoch only restarts the health, as in sender_lean);
the floor is a 0-d tensor on the device, which the captured CUDA graph reads at every replay (a Python float would be baked into the
graph at capture), and gcompute also returns the parallax cosine of every query (OUT['par']) for the memory gate."""
import numpy as np, torch
from . import paths
import packet_codec


class LiveSender:
    DS = [-10, -7, -5, -4, -3, -2, -1]; TOPQ = 1500; BAND = 1.0; MINSIM = 0.75; RES_MAX = 0.35; MIN_BASE = 0.08; SOFT = 0.1
    MEM_MIN = 100; MEM_EVERY = 5; DEAD_MIN = 50; CAPTURE_MODE = 'global'

    def __init__(self, ncam, width, height, drone_id, np_max=3000, dev='cuda', reset_on_epoch=True, parmin=6.0, parmin_early=8.0, mem_gate=8.0,
                 mem_keep_fresh=True):
        self.NC = NC = ncam; self.dev = dev; self.DRONE_ID = int(drone_id); self.reset_on_epoch = reset_on_epoch
        self.K = 2000 // NC
        P_ = np.load(paths.PCA); self.MU = torch.from_numpy(P_['mu'].astype(np.float32)).to(dev); self.BAS = torch.from_numpy(P_['basis'][:32].astype(np.float32)).to(dev)
        self.CODEC = packet_codec.Codec(P_['mu'], P_['basis'], width, height)
        self.PARMIN = float(parmin); self.PARMIN_EARLY = float(parmin_early); self.t_switch = None; self.floor = self.PARMIN_EARLY
        self.MEM_GATE = float(mem_gate); self.cosgate = float(np.cos(np.radians(self.MEM_GATE))); self.mem_keep_fresh = bool(mem_keep_fresh)
        self.sinb = float(np.sin(np.radians(self.BAND))); self.cosres = float(np.cos(np.radians(self.RES_MAX)))
        assert 0 <= self.RES_MAX < 90, 'gcompute has no dep > 0 test: the residual test stands in for it only below 90 deg'
        self.cospar = torch.tensor(np.cos(np.radians(self.floor)), dtype=torch.float32, device=dev)   # read by the graph: written in place only
        self.NP = NP = int(np_max); self.NQ = NQ = self.TOPQ if self.TOPQ > 0 else NP
        self.DEPTH = DEPTH = max(-min(self.DS), 1) + 1
        self.RB_D = torch.zeros(DEPTH, NC, NP, 64, device=dev, dtype=torch.half); self.RB_B = torch.zeros(DEPTH, NC, NP, 3, device=dev); self.RB_V = torch.zeros(DEPTH, NC, NP, device=dev, dtype=torch.bool)
        self.RB_I = torch.zeros(DEPTH, NC, NP, device=dev, dtype=torch.long); self.RB_S = torch.zeros(DEPTH, NC, NP, device=dev); self.RB_R = torch.zeros(DEPTH, NC, 3, 3, device=dev); self.RB_P = torch.zeros(DEPTH, NC, 3, device=dev)
        self.POSE = {}; self.KPTS = {}
        self.PNX = PNX = len(self.DS)
        self.SB = {'D1': torch.zeros(NC, PNX, NP, 64, device=dev, dtype=torch.half), 'B1': torch.zeros(NC, PNX, NP, 3, device=dev), 'V1': torch.zeros(NC, PNX, NP, device=dev, dtype=torch.bool),
                   'pv': torch.zeros(NC, PNX, device=dev, dtype=torch.bool), 'Rr': torch.zeros(NC, PNX, 3, 3, device=dev), 'tr': torch.zeros(NC, PNX, 3, device=dev),
                   'd0': torch.zeros(NC, NQ, 64, device=dev, dtype=torch.half), 'r0': torch.zeros(NC, NQ, 3, device=dev), 'v0': torch.zeros(NC, NQ, device=dev, dtype=torch.bool),
                   'Rk': torch.zeros(NC, 3, 3, device=dev), 'pk': torch.zeros(NC, 3, device=dev), 'pb': torch.zeros(3, device=dev), 'Rz': torch.zeros(3, 3, device=dev),
                   'sc': torch.zeros(NC, NP, device=dev), 'vq': torch.zeros(NC, NP, device=dev, dtype=torch.bool), 'dq': torch.zeros(NC, NP, 64, device=dev, dtype=torch.half)}
        self.OUT = {}; self.GR = [None]; self.MEM = {}; self.cmem = float(np.cos(np.radians(1.0)))
        self.HEALTH = []; self.epoch = None

    def reset(self):
        """a new VIO world: no partner, no depth memory, health from 0 (the CUDA graph and its buffers stay; the floor switch stays)."""
        self.POSE.clear(); self.KPTS.clear(); self.MEM.clear(); self.HEALTH.clear(); self.RB_V.zero_()

    def set_floor_switch(self, t_switch):
        """keyframes with time tk >= t_switch (s, the clock of keyframe()'s tk) get the floor PARMIN, earlier ones PARMIN_EARLY;
        without a switch time every keyframe gets PARMIN_EARLY."""
        self.t_switch = float(t_switch)

    def _set_floor(self, tk):
        fl = self.PARMIN if (self.t_switch is not None and tk >= self.t_switch) else self.PARMIN_EARLY
        if fl != self.floor:   # on the current stream, before the graph replay that reads it
            self.cospar.fill_(float(np.cos(np.radians(fl)))); self.floor = fl

    # ---- sender_lean.ingest
    def ingest(self, k, feats, R_, p_, pb_, y_):
        """feats[c] = (desc (n, 64) fp16, score (n,) float32, bear (n, 3) float32 unit rays; torch on the device; kpts (n, 2) numpy);
        R_ (NC, 3, 3) / p_ (NC, 3) float32 camera -> VIO world, pb_ (3,) body position (float64 of the float32 value), y_ body yaw."""
        RB_D, RB_V, RB_I, RB_S, RB_B = self.RB_D, self.RB_V, self.RB_I, self.RB_S, self.RB_B; dev = self.dev
        s = k % self.DEPTH
        for c in range(self.NC):
            d, sc, b, kp = feats[c]
            o = torch.argsort(-sc)[:self.NP]; m = len(o)
            RB_D[s, c].zero_(); RB_V[s, c].zero_(); RB_I[s, c].zero_(); RB_S[s, c].zero_(); RB_S[s, c, :m] = sc[o].float(); RB_D[s, c, :m] = torch.nn.functional.normalize(d[o].float(), dim=1).half(); RB_B[s, c, :m] = b[o]; RB_V[s, c, :m] = True; RB_I[s, c, :m] = o
            self.KPTS[(k, c)] = np.asarray(kp)
        self.POSE[k] = (torch.from_numpy(R_).to(dev), torch.from_numpy(p_).to(dev), pb_, y_); self.RB_R[s] = self.POSE[k][0]; self.RB_P[s] = self.POSE[k][1]
        for kk in [kk for kk in self.POSE if kk < k - self.DEPTH + 1]: del self.POSE[kk]
        for kk in [kk for kk in self.KPTS if kk[0] < k]: del self.KPTS[kk]

    # ---- sender_lean.tri / inv3
    def inv3(self, A):
        a, b, c, d, e, f, g, h, i = [A[..., r, s] for r in range(3) for s in range(3)]
        co = torch.stack([e * i - f * h, -(b * i - c * h), b * f - c * e, -(d * i - f * g), a * i - c * g, -(a * f - c * d), d * h - e * g, -(a * h - b * g), a * e - b * d], -1).view(*A.shape[:-2], 3, 3)
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g); return co / det[..., None, None]

    def tri(self, C, Dd, W):
        dev = self.dev
        P = W[..., None, None] * (torch.eye(3, device=dev) - Dd[..., :, None] * Dd[..., None, :]); A = P.sum(-3) + 1e-9 * torch.eye(3, device=dev); bb = torch.einsum('...mij,...mj->...i', P, C)
        return (self.inv3(A) @ bb[..., None])[..., 0]

    # ---- sender_lean.gcompute (PACKET=1 branch)
    def gcompute(self):
        SB = self.SB; OUT = self.OUT; dev = self.dev; NC, NQ, PNX = self.NC, self.NQ, self.PNX; sinb, cosres, cospar = self.sinb, self.cosres, self.cospar
        D1, B1, V1, pv, Rr, tr, d0, r0, v0 = (SB[x] for x in ('D1', 'B1', 'V1', 'pv', 'Rr', 'tr', 'd0', 'r0', 'v0'))
        r1 = torch.einsum('cpij,cpnj->cpni', Rr, B1); nrm = torch.nn.functional.normalize(torch.cross(tr[:, :, None, :].expand_as(r1), r1, dim=3), dim=3)
        lam = self.SOFT / sinb ** 2
        qa = torch.stack([r0[..., 0] ** 2, r0[..., 1] ** 2, r0[..., 2] ** 2, 2 * r0[..., 0] * r0[..., 1], 2 * r0[..., 0] * r0[..., 2], 2 * r0[..., 1] * r0[..., 2]], -1)
        pa = -lam * torch.stack([nrm[..., 0] ** 2, nrm[..., 1] ** 2, nrm[..., 2] ** 2, nrm[..., 0] * nrm[..., 1], nrm[..., 0] * nrm[..., 2], nrm[..., 1] * nrm[..., 2]], -1)
        pa = torch.where(V1[..., None], pa, torch.full_like(pa, -1e3))
        sim = torch.matmul(torch.cat([d0, qa.half()], -1)[:, None], torch.cat([D1, pa.clamp(min=-6e4).half()], -1).transpose(2, 3))   # (NC, Pn, NQ, NP)
        s12, m12 = sim.max(3); m21 = sim.argmax(2)
        good = (torch.gather(m21, 2, m12) == torch.arange(NQ, device=dev)[None, None]) & v0[:, None, :] & pv[:, :, None]
        nsel = torch.gather(nrm, 2, m12[..., None].expand(-1, -1, -1, 3)); dsel_ = torch.gather(D1, 2, m12[..., None].expand(-1, -1, -1, 64))
        good &= ((nsel * r0[:, None]).sum(-1).abs() < sinb) & ((dsel_.float() * d0[:, None].float()).sum(-1) > self.MINSIM)
        g = good.permute(0, 2, 1).reshape(NC * NQ, PNX).float(); selc = torch.arange(NC, device=dev).repeat_interleave(NQ); selq = torch.arange(NQ, device=dev).repeat(NC)
        r1s = r1[selc[:, None], torch.arange(PNX, device=dev)[None], m12.permute(0, 2, 1).reshape(NC * NQ, PNX)]
        Dd = torch.cat([r0[selc, selq][:, None], r1s], 1); Cc = torch.cat([torch.zeros(NC * NQ, 1, 3, device=dev), tr[selc]], 1); W = torch.cat([torch.ones(NC * NQ, 1, device=dev), g], 1)
        X = self.tri(Cc, Dd, W)
        for _ in range(2):
            v = X[:, None, :] - Cc; cos = (v * Dd).sum(2) / v.norm(dim=2).clamp_min(1e-9); bad = (cos < cosres) & (W > 0); bad[:, 0] = False
            W = torch.where(bad, torch.zeros_like(W), W); X = self.tri(Cc, Dd, W)
        v = X[:, None, :] - Cc; dep = (v * Dd).sum(2); cos = dep / v.norm(dim=2).clamp_min(1e-9)
        nobs = (W[:, 1:] > 0).sum(1)   # no separate dep > 0 test: cos = dep / |v| >= cosres > 0 on every view in use implies it
        par = torch.where(W[:, 1:] > 0, (Dd[:, 1:] * Dd[:, :1]).sum(2), torch.ones_like(W[:, 1:])).min(1).values
        ok = (nobs >= 2) & (dep[:, 0] > 0.3) & (dep[:, 0] < 12) & torch.where(W > 0, cos >= cosres, torch.ones_like(cos, dtype=torch.bool)).all(1) & (par <= cospar) & torch.isfinite(X).all(1)
        Xw = (SB['Rk'][selc] @ X[..., None])[..., 0] + SB['pk'][selc]; Xrg = (Xw - SB['pb']) @ SB['Rz']
        OUT.update(par=par, ok=ok, Xrg=Xrg, qz=ok[:0], n3=ok[:0], o=ok[:0], n3v=(ok & (nobs >= 3)).sum(), nok=ok.sum())

    # ---- sender_lean.keyframe_graph (EARLY=0, PACKET=1)
    def keyframe(self, k, feats, R_, p_, pb_, y_, rp, tk, age, epoch):
        """one keyframe -> (datagrams, info). rp = (roll, pitch) of the body, tk = keyframe time (s), age = s since the VIO initialised,
        epoch = VIO epoch (any int; +1 per re-initialisation)."""
        if self.epoch is not None and epoch != self.epoch:
            if self.reset_on_epoch: self.reset()
            else: self.HEALTH.clear()
        self.epoch = epoch; self._set_floor(tk)
        RB_D, RB_B, RB_V, RB_R, RB_P, RB_S, RB_I, SB, OUT = self.RB_D, self.RB_B, self.RB_V, self.RB_R, self.RB_P, self.RB_S, self.RB_I, self.SB, self.OUT
        NC, NQ, PNX, dev = self.NC, self.NQ, self.PNX, self.dev
        self.ingest(k, feats, R_, p_, pb_, y_); Rk, pk, pbk, yaw = self.POSE[k]; s = k % self.DEPTH
        past = [k + d for d in self.DS if (k + d) in self.POSE and np.linalg.norm(self.POSE[k + d][2] - pbk) >= self.MIN_BASE]
        parts = [[((kk % self.DEPTH), c, kk) for kk in past] for c in range(NC)]
        sl = torch.tensor([[p[m][0] if m < len(p) else s for m in range(PNX)] for p in parts], device=dev); cm = torch.tensor([[p[m][1] if m < len(p) else c for m in range(PNX)] for c, p in enumerate(parts)], device=dev)
        pvv = torch.tensor([[m < len(p) for m in range(PNX)] for p in parts], device=dev)
        Rp = RB_R[sl, cm]; pp = RB_P[sl, cm]
        SB['D1'].copy_(RB_D[sl, cm]); SB['B1'].copy_(RB_B[sl, cm]); SB['V1'].copy_(RB_V[sl, cm] & pvv[..., None]); SB['pv'].copy_(pvv)
        SB['Rr'].copy_(Rk.transpose(1, 2)[:, None] @ Rp); SB['tr'].copy_((Rk.transpose(1, 2)[:, None] @ (pp - pk[:, None])[..., None])[..., 0])
        SB['d0'].copy_(RB_D[s, :, :NQ]); SB['r0'].copy_(RB_B[s, :, :NQ]); SB['v0'].copy_(RB_V[s, :, :NQ]); SB['Rk'].copy_(Rk); SB['pk'].copy_(pk)
        SB['pb'].copy_(torch.from_numpy(pbk).float()); cy, sy = np.cos(yaw), np.sin(yaw); SB['Rz'].copy_(torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float32))
        SB['sc'].copy_(torch.stack([RB_S[s, c] for c in range(NC)])); SB['vq'].copy_(RB_V[s]); SB['dq'].copy_(RB_D[s])
        if self.GR[0] is None:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): self.gcompute()
            torch.cuda.current_stream().wait_stream(st); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode=self.CAPTURE_MODE): self.gcompute()
            self.GR[0] = g   # only a completed capture is kept (a failed one is retried at the next keyframe)
        self.GR[0].replay()
        ok = OUT['ok'].cpu().numpy(); idx = np.where(ok)[0]; cams = idx // NQ; q = idx % NQ
        out = (RB_I[s].cpu().numpy()[cams, q], cams, OUT['n3'].cpu().numpy(), OUT['Xrg'][torch.from_numpy(idx).to(dev)].cpu().numpy(), OUT['qz'].cpu(), q)
        if self.MEM_MIN > 0: out = self.depth_memory(k, s, out, idx)
        n_tri = int(len(idx)); n_mem = int(len(out[5]) - len(idx))
        ki, cams_, n3c, Xrg_, b = self.build_packet(k, s, out, rp, tk, age, epoch)
        return b, {'n_tri': n_tri, 'n_mem': n_mem, 'n3_packet': [int(x) for x in n3c], 'n_partners': len(past), 'health': self.HEALTH[-1] if self.HEALTH else 0.0, 'hl': self._hl,
                   'floor_deg': self.floor}

    # ---- sender_lean.depth_memory
    def depth_memory(self, k, s, out, idx):
        RB_D, RB_B, RB_V, RB_I, MEM, dev = self.RB_D, self.RB_B, self.RB_V, self.RB_I, self.MEM, self.dev
        ki, cams, n3, Xrg, qz, rq = out; Rk, pk, pbk, yaw = self.POSE[k]; cy, sy_ = np.cos(yaw), np.sin(yaw); Rz = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
        Xw_all = Xrg @ Rz.T + pbk; add_i = []; add_c = []; add_X = []; add_q = []
        pgate = self.OUT['par'].cpu().numpy()[idx] <= self.cosgate if self.MEM_GATE > 0 else np.ones(len(idx), bool)   # fresh points >= mem_gate
        for c in range(self.NC):
            m = cams == c; mg = m & pgate; fresh = None
            if m.sum() >= self.MEM_MIN and (k % self.MEM_EVERY == 0 or c not in MEM) and mg.sum() >= self.MEM_MIN:
                ri = RB_I[s, c].cpu().numpy(); srt = np.argsort(ri); pos = srt[np.searchsorted(ri, ki[mg], sorter=srt)]
                fresh = (RB_D[s, c][torch.from_numpy(pos).to(dev)].clone(), torch.from_numpy(Xw_all[mg]).float().to(dev))
            if m.sum() >= self.MEM_MIN:
                if fresh is not None: MEM[c] = fresh
                continue
            prev = MEM.get(c)
            if fresh is not None: MEM[c] = fresh
            if prev is None: continue
            dm, Xm = prev
            R_ = Rk[c]; p_ = pk[c]; v = (Xm - p_) @ R_; dist = v.norm(dim=1); pred = v / dist[:, None].clamp_min(1e-6)
            sim = RB_D[s, c].float() @ dm.float().T; gate = (RB_B[s, c] @ pred.T) > self.cmem
            sim = torch.where(gate & RB_V[s, c][:, None], sim, torch.full_like(sim, -2.0)); m12 = sim.argmax(1); s12 = sim.gather(1, m12[:, None])[:, 0]; m21 = sim.argmax(0)
            good = (m21[m12] == torch.arange(len(sim), device=dev)) & (s12 > self.MINSIM) & (dist[m12] > 0.3)
            if self.mem_keep_fresh and m.any(): good[torch.from_numpy(rq[m]).to(dev)] = False   # a keypoint with a fresh 3-D point keeps it
            qi = torch.where(good)[0]
            if len(qi): add_i.append(RB_I[s, c][qi].cpu().numpy()); add_c.append(np.full(len(qi), c)); add_X.append(((Xm[m12[qi]].cpu().numpy() - pbk) @ Rz)); add_q.append(qi.cpu().numpy())
        if add_i: ki = np.concatenate([ki] + add_i); cams = np.concatenate([cams] + add_c); Xrg = np.concatenate([Xrg] + add_X); rq = np.concatenate([rq] + add_q)
        return ki, cams, n3, Xrg, qz, rq

    # ---- sender_lean.build_packet (CODEC_V=3)
    def build_packet(self, k, s, out, rp, tk, age, ep):
        RB_D, RB_V, RB_S, RB_I, dev, NC, NP, K = self.RB_D, self.RB_V, self.RB_S, self.RB_I, self.dev, self.NC, self.NP, self.K
        ki, cams, n3, Xrg, qz, rq = out; Rk, pk, pbk, yaw = self.POSE[k]; cy, sy_ = np.cos(yaw), np.sin(yaw); Rz = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
        has = torch.zeros(NC, NP, dtype=torch.bool, device=dev); row = np.full((NC, NP), -1, np.int64)
        if len(rq): has[torch.from_numpy(cams).to(dev), torch.from_numpy(rq).to(dev)] = True; row[cams, rq] = np.arange(len(rq))
        scs = torch.stack([RB_S[s, c] for c in range(NC)]).float(); key = scs * RB_V[s].float() + 10.0 * has.float(); o = torch.topk(key, K, dim=1).indices
        dsel = RB_D[s][torch.arange(NC, device=dev)[:, None], o].float(); z = (dsel - self.MU) @ self.BAS.T; sc_ = 7.0 / z.abs().amax(2, keepdim=True).clamp_min(1e-6)
        qz = torch.clamp(torch.round(z * sc_), -7, 7).to(torch.int8).cpu().numpy(); step = (1.0 / sc_[..., 0]).half().cpu().numpy(); o = o.cpu().numpy(); ri = RB_I[s].cpu().numpy()
        Xw = Xrg @ Rz.T + pbk; pkn = pk.cpu().numpy(); camsd = []; n3c = np.zeros(NC, int)
        for c in range(NC):
            rr = row[c, o[c]]; m = rr >= 0; dep = np.full(K, np.nan); dep[m] = np.linalg.norm(Xw[rr[m]] - pkn[c], axis=1); n3c[c] = int(m.sum())
            kp = self.KPTS[(k, c)][ri[c, o[c]]] if (k, c) in self.KPTS and len(self.KPTS[(k, c)]) else np.zeros((K, 2))
            va = RB_V[s, c][torch.from_numpy(o[c]).to(dev)].cpu().numpy()
            if va.sum() < self.DEAD_MIN: va[:] = False
            camsd.append({'kpts': kp[va], 'q': qz[c][va], 'step': step[c][va], 'depth': dep[va]}); n3c[c] = int(np.isfinite(dep[va]).sum())
        OUT = self.OUT; fr = float(OUT['n3v']) / max(float(OUT['nok']), 1.0) if 'n3v' in OUT else 0.0; self.HEALTH.append(fr); hl = int(round(255 * float(np.median(self.HEALTH[-10:])))); self._hl = hl
        hdr = {'t': tk, 'roll': rp[0], 'pitch': rp[1], 'yaw': yaw, 'x': pbk[0], 'y': pbk[1], 'z': pbk[2], 'epoch': ep & 0xFF, 'age_s': age, 'health': hl, 'basis': 0}
        b = self.CODEC.encode_v3(self.DRONE_ID, k, hdr, camsd)
        return ki, cams, n3c, Xrg, b
