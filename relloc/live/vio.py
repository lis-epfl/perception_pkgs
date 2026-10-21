"""The drone's own VIO poses (the estimator's poseimu: IMU pose in its world, stamp = IMU time) at an image's instant.

The flight estimator (run_online_msckf) publishes a pose after each filter update, ~7.5 per second. Between two poses the drone turns by
up to several degrees, so the attitude at the image's instant comes from the IMU, as relloc2's md.VioTraj computed it for every evaluated
result: the last VIO pose at or before the instant, propagated with the gyro samples between (sample and hold, R <- R Exp((w - b) dt)).
The gyro bias b is not published by the estimator; here it is the running mean (time constant `bias_tau` poses) of the rate that
separates two consecutive VIO attitudes from the gyro integral between them. Without gyro samples the attitude is interpolated between the
two poses (counted in n_slerp). The position is interpolated linearly between the two poses around the instant (the lookup waits for the
next pose).
Epochs: the VIO world exists from the first pose until the estimator restarts. A restart shows as a pose stream that goes back in time or
pauses for more than `gap_s`: the epoch then increments (packet header byte; node15 drops estimates of the old world) and the age (s since
the epoch's first pose; the header's 'VIO age') restarts."""
import threading, bisect, collections
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


class VioBuffer:
    def __init__(self, gap_s=2.0, keep_s=10.0, max_dt=0.5, bias_tau=30.0):
        self.lock = threading.Lock(); self.t = []; self.q = []; self.p = []; self.epoch = 0; self.t0 = None; self.z0 = None
        self.gap = gap_s; self.keep = keep_s; self.max_dt = max_dt; self.n = 0; self.n_slerp = 0; self.n_gyro = 0
        self.gt = collections.deque(); self.gw = collections.deque(); self.bias = np.zeros(3); self.bias_tau = bias_tau; self.n_bias = 0

    def add_gyro(self, t, w):
        """one gyro sample (IMU time s, rad/s in the estimator's IMU frame, FLU)."""
        with self.lock:
            if self.gt and t <= self.gt[-1]: return
            self.gt.append(float(t)); self.gw.append(np.asarray(w, float))
            while self.gt and self.gt[0] < t - self.keep: self.gt.popleft(); self.gw.popleft()

    def _integrate(self, R, t0, t1):
        """R propagated from t0 to t1 with the stored gyro samples (md.VioTraj.pose_imu); None if the samples do not cover [t0, t1]."""
        gt = self.gt
        if not gt or gt[0] > t0 or gt[-1] < t1 - 0.02: return None
        i0 = bisect.bisect_left(gt, t0); i1 = bisect.bisect_left(gt, t1)
        tt = [t0] + [gt[i] for i in range(i0, i1)] + [t1]; idx = [min(max(i, 0), len(gt) - 1) for i in range(i0 - 1, i1)]
        for n in range(len(tt) - 1):
            dt = tt[n + 1] - tt[n]
            if dt > 0: R = R * Rot.from_rotvec((self.gw[idx[min(n, len(idx) - 1)]] - self.bias) * dt)
        return R

    def add(self, t, quat_xyzw, p):
        with self.lock:
            if self.t and (t < self.t[-1] - 1e-6 or t - self.t[-1] > self.gap):
                self.epoch += 1; self.t, self.q, self.p = [], [], []; self.t0 = None; self.z0 = None
            if self.t and t <= self.t[-1]: return
            if self.t0 is None: self.t0 = t; self.z0 = float(p[2])
            if self.t:   # gyro bias from the two latest VIO attitudes
                Ra = Rot.from_quat(self.q[-1]); Rb = Rot.from_quat(quat_xyzw); dt = t - self.t[-1]; Rg = self._integrate(Ra, self.t[-1], t)
                if Rg is not None and 0.02 < dt < 0.5:
                    r = (Rg.inv() * Rb).as_rotvec() / dt; a = 1.0 / min(self.n_bias + 1, self.bias_tau)
                    if np.linalg.norm(r) < 0.05: self.bias = self.bias - a * r; self.n_bias += 1   # r: rate that (w - b) still misses
            self.t.append(float(t)); self.q.append(np.asarray(quat_xyzw, float)); self.p.append(np.asarray(p, float)); self.n += 1
            while self.t and self.t[0] < t - self.keep: self.t.pop(0); self.q.pop(0); self.p.pop(0)

    def latest(self):
        with self.lock: return self.t[-1] if self.t else None

    def pose(self, t):
        """(R (scipy Rotation, IMU -> world), p (3,), epoch, age s, z above the epoch's first pose) at IMU time t, or None."""
        with self.lock:
            if len(self.t) < 2 or t < self.t[0] or t > self.t[-1]: return None
            j = bisect.bisect_right(self.t, t) - 1; j = min(j, len(self.t) - 2); a, b = j, j + 1; ta, tb = self.t[a], self.t[b]
            if tb - ta > self.max_dt: return None
            w = (t - ta) / (tb - ta); p = (1 - w) * self.p[a] + w * self.p[b]
            R = self._integrate(Rot.from_quat(self.q[a]), ta, t)
            if R is None: R = Slerp([ta, tb], Rot.from_quat([self.q[a], self.q[b]]))([t])[0]; self.n_slerp += 1
            else: self.n_gyro += 1
            return R, p, self.epoch, t - self.t0, float(p[2] - self.z0)
