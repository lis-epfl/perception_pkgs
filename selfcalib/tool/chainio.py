#!/usr/bin/env python3
"""Shared kalibr-chain I/O + calibration distance for the calibration tool.
Reuses the parsing/writing patterns used throughout the basin study."""
import re
import numpy as np

# legacy residual weights (the study's criterion metric, budget_basin_<d>.json W)
DEFAULT_W = {'cx': 105.0, 'cy': 130.0, 'fx': 35.0, 'k1': 0.40, 'rotx': 20.0, 'transx': 180.0}


def parse_chain(path):
    """-> (cams, toff): cams[int c] = {'f':[fx,fy,cx,cy], 'k':[k1..k4], 'R':3x3, 'p':3}; toff = cam0 timeshift."""
    lines = open(path).read().splitlines()
    cams, cur, i, toff = {}, None, 0, None
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'\s*cam(\d+):', ln)
        if m:
            cur = int(m.group(1)); cams[cur] = {}
        if cur is not None:
            if re.match(r'\s*intrinsics:', ln):
                cams[cur]['f'] = [float(x) for x in re.search(r'\[(.*)\]', ln).group(1).split(',')]
            if re.match(r'\s*distortion_coeffs:', ln):
                cams[cur]['k'] = [float(x) for x in re.search(r'\[(.*)\]', ln).group(1).split(',')]
            if re.match(r'\s*timeshift_cam_imu:', ln) and toff is None:
                toff = float(re.search(r':\s*([-0-9.eE]+)', ln).group(1))
            if re.match(r'\s*T_imu_cam:', ln):
                T = np.array([[float(x) for x in re.search(r'\[(.*)\]', lines[i + 1 + r]).group(1).split(',')]
                              for r in range(4)])
                cams[cur]['R'] = T[:3, :3]; cams[cur]['p'] = T[:3, 3]
                i += 5
                continue
        i += 1
    # Only T_imu_cam is understood. A chain written with the T_cam_imu spelling parses with
    # intrinsics but no extrinsics, which used to surface much later as a bare KeyError during
    # seed assembly. The two are inverses, so silently reinterpreting would be worse than
    # refusing: it would publish a calibration that is wrong in a way nothing downstream flags.
    missing = sorted(c for c in cams if 'R' not in cams[c])
    if missing:
        alt = any(re.match(r'\s*T_cam_imu:', l) for l in lines)
        raise ValueError(
            '%s: no T_imu_cam block for cam%s.%s'
            % (path, ','.join(str(c) for c in missing),
               ' The file uses the T_cam_imu spelling instead; T_cam_imu is the INVERSE of'
               ' T_imu_cam, so invert the 4x4 matrices and rename the key rather than just'
               ' renaming it. See README section 3.' if alt else ''))
    return cams, toff


def write_chain(template_path, cams, toff, dst):
    """Write cams/toff into a copy of template chain yaml (all four timeshift slots get the rig toff)."""
    lines = open(template_path).read().splitlines()
    out, cur, i = [], None, 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'\s*cam(\d+):', ln)
        if m:
            cur = int(m.group(1))
        if re.match(r'\s*timeshift_cam_imu:', ln):
            out.append(re.sub(r':\s*[-0-9.eE]+', ': %.9f' % toff, ln)); i += 1; continue
        if cur in cams:
            v = cams[cur]
            if re.match(r'\s*intrinsics:', ln):
                out.append(re.sub(r'\[.*\]', '[%.6f, %.6f, %.6f, %.6f]' % tuple(v['f']), ln)); i += 1; continue
            if re.match(r'\s*distortion_coeffs:', ln):
                out.append(re.sub(r'\[.*\]', '[%.6f, %.6f, %.6f, %.6f]' % tuple(v['k']), ln)); i += 1; continue
            if re.match(r'\s*T_imu_cam:', ln):
                out.append(ln)
                R, p = v['R'], v['p']
                for r in range(3):
                    out.append('    - [%.10f, %.10f, %.10f, %.10f]' % (R[r, 0], R[r, 1], R[r, 2], p[r]))
                out.append('    - [0.0, 0.0, 0.0, 1.0]')
                i += 5
                continue
        out.append(ln); i += 1
    open(dst, 'w').write('\n'.join(out) + '\n')


def cams_from_caljson(cj):
    """Harvested estimate_tum.txt.calib.json -> cams dict in parse_chain format."""
    cams = {}
    for c, v in cj['cams'].items():
        cams[int(c)] = {'f': list(v['intr'][:4]), 'k': list(v['intr'][4:8]),
                        'R': np.array(v['R_CtoI']).reshape(3, 3), 'p': np.array(v['p_CinI'])}
    return cams


def calib_residual(cams_a, cams_b, W=DEFAULT_W):
    """The study's criterion metric between two camera calibrations (group means / legacy weights)."""
    DC, DY, DF, DK, DR, DT = [], [], [], [], [], []
    for c in cams_a:
        a, b = cams_a[c], cams_b[c]
        DC.append(abs(a['f'][2] - b['f'][2])); DY.append(abs(a['f'][3] - b['f'][3]))
        DF.append((abs(a['f'][0] / b['f'][0] - 1) + abs(a['f'][1] / b['f'][1] - 1)) / 2 * 100)
        DK.append(np.mean([abs(a['k'][j] - b['k'][j]) for j in range(4)]))
        DR.append(np.degrees(np.arccos(np.clip((np.trace(np.asarray(a['R']) @ np.asarray(b['R']).T) - 1) / 2, -1, 1))))
        DT.append(np.linalg.norm(np.asarray(a['p']) - np.asarray(b['p'])) * 100)
    return float(np.sqrt((np.mean(DC) / W['cx']) ** 2 + (np.mean(DY) / W['cy']) ** 2 +
                         (np.mean(DF) / W['fx']) ** 2 + (np.mean(DK) / W['k1']) ** 2 +
                         (np.mean(DR) / W['rotx']) ** 2 + (np.mean(DT) / W['transx']) ** 2))


def kb4_radius(f, k, theta_max):
    """Fisheye image-circle radius in px from KB4 params at the lens FOV edge angle theta_max."""
    fx, fy = f[0], f[1]
    k1, k2, k3, k4 = k
    th = theta_max
    thd = th + k1 * th ** 3 + k2 * th ** 5 + k3 * th ** 7 + k4 * th ** 9
    return 0.5 * (fx + fy) * thd


# The paper's convergence envelope (Sec. III-A): the width of the estimator's own settled oscillation
# around the reference calibration, one limit per parameter type -- principal point px, focal length %,
# signed mean of the four distortion coefficients, rotation-vector component deg, translation component
# cm, time offset ms.
ENVELOPE = {'c': 2.5, 'f%': 0.5, 'k': 0.001, 'r': 0.35, 't': 2.5, 'td': 6.0}


def step_types(cams_a, toff_a, cams_b, toff_b, env=ENVELOPE):
    """Movement from calibration a to b per parameter type, in envelopes, max over cameras (the
    paper's stopping rule, Sec. IV: converged when every type moved <= 2 envelopes since the
    previous pass). Same aggregation as the study (ov_reverify/multipass_eval.step_types)."""
    out = {t: 0.0 for t in env}
    for c in cams_a:
        x, y = cams_a[c], cams_b[c]
        out['c'] = max(out['c'], abs(y['f'][2] - x['f'][2]) / env['c'], abs(y['f'][3] - x['f'][3]) / env['c'])
        out['f%'] = max(out['f%'], abs(y['f'][0] / x['f'][0] - 1) * 100 / env['f%'])
        out['k'] = max(out['k'], abs(float(np.mean([y['k'][j] - x['k'][j] for j in range(4)]))) / env['k'])
        dR = np.asarray(y['R']) @ np.asarray(x['R']).T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if ang > 1e-12:
            rv = np.degrees(ang) * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
            out['r'] = max(out['r'], float(np.max(np.abs(rv))) / env['r'])
        out['t'] = max(out['t'], float(np.max(np.abs((np.asarray(y['p'], float).reshape(-1) - np.asarray(x['p'], float).reshape(-1)) * 100))) / env['t'])
    out['td'] = abs(toff_b - toff_a) * 1000 / env['td']
    return out
