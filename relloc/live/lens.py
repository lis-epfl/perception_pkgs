"""Pixel -> unit ray of the fleet's cameras: Kannala-Brandt (KB4) with the two lens terms of the estimator (vio/ov_core/src/cam/CamEqui.h).

kb4_theta_peak / kb4_unproject_8 are relloc2's rl.kb4_theta_peak / rl.kb4_unproject (round 15), copied line for line: on a calibration whose
polynomial folds before 180 deg, a pixel beyond the fold gets the backward ray (0, 0, -1), which every test of the receiver rejects.
unproject (11 numbers) is relloc2's review21/ray/md21.kb4_unproject: the measured normalised point (x, y) = ((u - cx)/fx, (v - cy)/fy) loses
the lens terms N(x, y) explicitly, then the 8-number KB4 un-projection runs with unit focal length and zero centre:
    N_x = 2 p1 x y + p2 (r^2 + 2 x^2) + s y,   N_y = p1 (r^2 + 2 y^2) + 2 p2 x y,   r^2 = x^2 + y^2.
With p1 = p2 = s = 0 it equals the 8-number path bit for bit."""
import numpy as np

_KB4_PEAK = {}


def kb4_theta_peak(intr):
    key = tuple(float(v) for v in intr[4:8])
    if key not in _KB4_PEAK:
        k1, k2, k3, k4 = key
        def d(t): t2 = t * t; return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        g = np.linspace(1e-4, np.pi, 20001); neg = np.where(d(g) <= 0)[0]; tp = np.pi
        if len(neg):
            a, b = g[neg[0] - 1], g[neg[0]]
            for _ in range(60): m = 0.5 * (a + b); a, b = (m, b) if d(m) > 0 else (a, m)
            tp = a
        t2 = tp * tp; _KB4_PEAK[key] = (tp, tp * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))))
    return _KB4_PEAK[key]


def kb4_unproject_8(uv, intr, invalid='back'):
    fx,fy,cx,cy,k1,k2,k3,k4=intr
    x=(uv[:,0]-cx)/fx; y=(uv[:,1]-cy)/fy; thd=np.sqrt(x*x+y*y); th=thd.copy()
    tp,dp=kb4_theta_peak(intr); fold=tp<np.pi
    if fold: th=np.minimum(th,tp)
    for _ in range(12 if not fold else 30):
        th2=th*th; f=th*(1+th2*(k1+th2*(k2+th2*(k3+th2*k4))))-thd
        df=1+th2*(3*k1+th2*(5*k2+th2*(7*k3+th2*9*k4))); th=th-f/np.maximum(df,1e-9)
        if fold: th=np.clip(th,0,tp)
    th=np.clip(th,0,np.pi); s=np.where(thd>1e-9,np.sin(th)/np.maximum(thd,1e-9),1.0)
    b=np.stack([x*s,y*s,np.cos(th)],1); b=b/np.linalg.norm(b,axis=1,keepdims=True)
    if fold: b[thd>dp]=np.nan if invalid=='nan' else np.array([0.0,0.0,-1.0])
    return b


def nonrad_N(x, y, q):
    p1, p2, s = q; r2 = x * x + y * y
    return 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s * y, p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y


def unproject(uv, intr, invalid='back'):
    """unit rays (N, 3) in the camera frame of pixels uv (N, 2); intr = 8 numbers (fx fy cx cy k1..k4) or 11 (+ p1 p2 s)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2); n = len(intr)
    if n == 8: return kb4_unproject_8(uv, intr, invalid)
    if n != 11: raise ValueError(f'intr must hold 8 or 11 numbers, got {n}')
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]; q = (intr[8], intr[9], intr[10])
    x = (uv[:, 0] - cx) / fx; y = (uv[:, 1] - cy) / fy; nx, ny = nonrad_N(x, y, q)
    return kb4_unproject_8(np.stack([x - nx, y - ny], 1), [1.0, 1.0, 0.0, 0.0, intr[4], intr[5], intr[6], intr[7]], invalid)
