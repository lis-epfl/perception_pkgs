#!/usr/bin/env python3
"""inflight_report.py RUN_DIR [--start 16.5] -- what a run of run_inflight_replay.sh measured.

Times are seconds after takeoff on the replay's clock: an instant c on the runner's clock (CLOCK_MONOTONIC) is
c - TAKEOFF_CLOCK, a record stamped s is s - (anchor stamp + takeoff). From RUN_DIR/replay.env, tool.log (each line stamped
with the clock as it arrived), rt_trace.txt (OV_RT_TRACE: one block per estimator pass) and the tool's output:

  pass 1        first set of images done, when it caught up with the live data (the first set done less than 0.5 s after
                it went live), its last set and how far behind the live data that was; its speed while working off the
                backlog
  pass 2        from the tool's "pass 2/" line to its VERDICT line: the calibration is ready at the VERDICT line
  as run        the times of this run
  common start  pass 1 replayed as a queue from a start at --start s after takeoff (default 16.5 s = 15 s of heuristics +
                1.5 s for the estimator to start, the rule behind the paper's "ready after takeoff"): set i is available at
                its stamp and costs what it cost in the run; the pass starts earlier or later by the difference. This
                removes the start of the tool's batch heuristics, which the replay cannot place exactly.

Calibration: with RUN_DIR/offline1 (and offline2 ...), the tool run AFTER the flight on the same recording, the published
calibrations (and the first passes) are compared: per camera position (mm) and rotation (deg) relative to the IMU, focal
length (%), principal point (px), distortion, the two lens terms; the time offset (ms); the camera pairs (change of the
distance between neighbouring cameras, mm, and of their relative rotation, deg); and the tool's own step measure in
envelopes (its stopping rule calls two calibrations the same at <= 2).

Board: with RUN_DIR/sampler.txt (the launcher samples the computer every second), the load, GPU load, CPU clocks and
temperature during the tool's run and during each offline run, and every process seen above 20 % CPU in the 5-s samples.
The run's own processes are run_serial_msck (the estimator), python3 (the tool and its image pass) and ros2 (the players
and the recorder); anything else was another job on the computer, and the times are then not those of a quiet board.

Writes RUN_DIR/inflight_report.json and inflight_report.txt.
"""
import argparse
import glob
import json
import os
import re
import shlex
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tool'))
sys.dont_write_bytecode = True
from chainio import parse_chain, cams_from_caljson, step_types   # noqa: E402

CAUGHT_UP_S = 0.5


def read_env(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            out[k] = shlex.split(v)[0] if v.strip() else ''
    return out


def stamped_lines(path):
    rows = []
    for line in open(path, errors='replace'):
        m = re.match(r'^(\d+\.\d+) (.*)$', line.rstrip('\n'))
        if m:
            rows.append((float(m.group(1)), m.group(2)))
    return rows


def trace_blocks(path):
    """OV_RT_TRACE file -> list of (header, array [stamp, clock, proc_ms]) in file order (one per estimator run)."""
    blocks, head, rows = [], None, []
    for line in open(path):
        if line.startswith('#'):
            if head is not None or rows:
                blocks.append((head, np.array(rows).reshape(-1, 3)))
            head, rows = line.strip(), []
        elif line.strip():
            try:
                v = [float(x) for x in line.split()[:3]]
            except ValueError:
                continue        # a line cut short by an estimator that was killed
            if len(v) == 3:
                rows.append(v)
    if head is not None or rows:
        blocks.append((head, np.array(rows).reshape(-1, 3)))
    return blocks


def queue_replay(stamp, done, first_done):
    """End of the pass and when it caught up, if its first set had been done at first_done (each set available at its
    stamp, costing what it cost in the run)."""
    cost = np.zeros(len(done))
    cost[1:] = np.maximum(done[1:] - np.maximum(done[:-1], stamp[1:]), 0.0)
    t, caught = first_done, None
    for i in range(1, len(done)):
        t = max(t, stamp[i]) + cost[i]
        if caught is None and t - stamp[i] < CAUGHT_UP_S:
            caught = t
    return t, caught


def times(run, start):
    env = read_env(os.path.join(run, 'replay.env'))
    tk_clock = float(env['TAKEOFF_CLOCK'])
    tk_stamp = float(env['REC_ANCHOR_STAMP']) + float(env['REC_TAKEOFF_S'])
    rel = lambda c: c - tk_clock
    L = stamped_lines(os.path.join(run, 'tool.log'))
    first = lambda pat: next((rel(t) for t, s in L if re.search(pat, s)), None)
    passes = sorted({int(m.group(1)) for _, s in L for m in [re.search(r'\[tool\] pass (\d+)/', s)] if m})
    verdict = next((s.split('VERDICT:', 1)[1].strip() for _, s in L if 'VERDICT:' in s), None)
    r = {'recording': env.get('RECORDING'), 'flight_s': float(env['FLIGHT_S']), 'start_after_s': float(env['START_AFTER']),
         'takeoff_after_live_s': float(env['REC_TAKEOFF_S']), 'tool_first_line': rel(L[0][0]) if L else None,
         'pass_lines': {k: first(r'\[tool\] pass %d/' % k) for k in passes}, 'verdict_line': first('VERDICT:'),
         'verdict': verdict, 'passes': len(passes), 'tool_rc': env.get('TOOL_RC')}
    blocks = trace_blocks(os.path.join(run, 'rt_trace.txt')) if os.path.exists(os.path.join(run, 'rt_trace.txt')) else []
    r['trace_blocks'] = len(blocks)
    if not blocks or len(blocks[0][1]) < 2:
        r['problem'] = 'no trace of pass 1'
        return r
    A = blocks[0][1]
    stamp, done = A[:, 0] - tk_stamp, rel(A[:, 1])
    lag = done - stamp
    k = np.where(lag < CAUGHT_UP_S)[0]
    p1 = {'sets': int(len(A)), 'first_set_stamp': float(stamp[0]), 'first_set_done': float(done[0]),
          'backlog_at_start_s': float(lag[0]), 'caught_up': float(done[k[0]]) if len(k) else None,
          'last_set_stamp': float(stamp[-1]), 'last_set_done': float(done[-1]), 'behind_at_end_s': float(lag[-1]),
          'max_proc_ms': float(A[:, 2].max())}
    if len(k) and k[0] > 0:     # speed while working off the backlog: recording seconds per second
        p1['backlog_speed'] = float((stamp[k[0]] - stamp[0]) / max(done[k[0]] - done[0], 1e-9))
        fl = (stamp[:k[0] + 1] >= 0.0)
        if fl.sum() > 30:
            i0 = int(np.argmax(fl))
            p1['backlog_speed_in_flight'] = float((stamp[k[0]] - stamp[i0]) / max(done[k[0]] - done[i0], 1e-9))
    r['pass1'] = p1
    if len(blocks) > 1 and len(blocks[1][1]) > 1:
        B = blocks[1][1]
        r['pass2_trace'] = {'sets': int(len(B)), 'first_set_done': float(rel(B[0, 1])), 'last_set_done': float(rel(B[-1, 1])),
                            'speed': float((B[-1, 0] - B[0, 0]) / max(B[-1, 1] - B[0, 1], 1e-9))}
    p2, vt = r['pass_lines'].get(2), r['verdict_line']
    if p2 is None or vt is None:
        r['problem'] = 'no second pass or no verdict'
        return r
    r['exit_after_pass1_s'] = p2 - p1['last_set_done']
    r['pass2_to_verdict_s'] = vt - p2
    r['end_of_pass2'] = r['pass_lines'].get(3, vt)
    r['ready_as_run'] = vt
    e, c = queue_replay(stamp, done, start)
    chk, _ = queue_replay(stamp, done, float(done[0]))
    r['common_start'] = {'start_s': start, 'pass1_caught_up': c, 'pass1_last_set_done': float(e),
                         'pass1_behind_at_end_s': float(e - stamp[-1]), 'ready': float(e + r['exit_after_pass1_s'] + r['pass2_to_verdict_s']),
                         'queue_check_s': float(chk - done[-1])}
    return r


OWN_PROCESSES = {'run_serial_msck', 'python3', 'ros2'}


def board(run, t0, t1):
    """What sampler.txt saw between clock t0 and t1 (uptime clock, the same as the estimator's on a computer that has not
    been suspended), or None."""
    path = os.path.join(run, 'sampler.txt')
    if not os.path.exists(path) or t0 is None or t1 is None:
        return None
    rows, names = [], {}
    for line in open(path, errors='replace'):
        p = line.split()
        try:
            if p and p[0] == '#top':
                if len(p) > 1 and t0 <= float(p[1]) <= t1:
                    for x in p[2:]:
                        n, c = x.rsplit(':', 1)
                        if float(c) >= 20.0:
                            names[n] = max(names.get(n, 0.0), float(c))
            elif len(p) == 6:
                v = [float(x) for x in p]
                if t0 <= v[0] <= t1:
                    rows.append(v)
        except ValueError:
            continue
    if not rows:
        return None
    A = np.array(rows)
    return {'seconds': int(len(A)), 'load1_mean': float(A[:, 1].mean()), 'load1_max': float(A[:, 1].max()),
            'gpu_load_pct_mean': float(A[:, 2].mean() / 10.0), 'gpu_load_pct_max': float(A[:, 2].max() / 10.0),
            'cpu_khz_min': float(A[:, 3].min()), 'cpu_khz_max': float(A[:, 4].max()), 'hottest_zone_c_max': float(A[:, 5].max() / 1000.0),
            'processes_over_20pct': {n: round(c, 1) for n, c in sorted(names.items())},
            'other_processes': sorted(n for n in names if n not in OWN_PROCESSES)}


def rotation_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(Ra) @ np.asarray(Rb).T) - 1.0) / 2.0, -1.0, 1.0))))


def read_terms(path):
    if not os.path.exists(path):
        return None
    return [[float(x) for x in e.split(',')] for e in open(path).read().strip().split(';')]


def compare(ca, ta, cb, tb, terms_a=None, terms_b=None):
    cams = sorted(ca)
    per = {}
    for c in cams:
        a, b = ca[c], cb[c]
        per[c] = {'position_mm': float(np.linalg.norm(np.asarray(a['p'], float) - np.asarray(b['p'], float)) * 1000.0),
                  'rotation_deg': rotation_deg(a['R'], b['R']),
                  'focal_pct': float(max(abs(a['f'][0] / b['f'][0] - 1.0), abs(a['f'][1] / b['f'][1] - 1.0)) * 100.0),
                  'principal_point_px': float(np.hypot(a['f'][2] - b['f'][2], a['f'][3] - b['f'][3])),
                  'distortion_max_abs': float(max(abs(x - y) for x, y in zip(a['k'], b['k'])))}
        if terms_a and terms_b:
            per[c]['lens_terms_max_abs'] = float(max(abs(terms_a[c][j] - terms_b[c][j]) for j in range(2)))
    pairs = {}
    for i, j in [(cams[n], cams[(n + 1) % len(cams)]) for n in range(len(cams))] if len(cams) > 2 else []:
        da = np.linalg.norm(np.asarray(ca[j]['p'], float) - np.asarray(ca[i]['p'], float))
        db = np.linalg.norm(np.asarray(cb[j]['p'], float) - np.asarray(cb[i]['p'], float))
        Ra = np.asarray(ca[i]['R']).T @ np.asarray(ca[j]['R'])
        Rb = np.asarray(cb[i]['R']).T @ np.asarray(cb[j]['R'])
        pairs['%d-%d' % (i, j)] = {'distance_mm': float(abs(da - db) * 1000.0), 'relative_rotation_deg': rotation_deg(Ra, Rb)}
    out = {'per_camera': per, 'pairs': pairs, 'time_offset_ms': float(abs(ta - tb) * 1000.0),
           'step_envelopes': {k: round(float(v), 3) for k, v in step_types(cb, tb, ca, ta).items()}}
    worst = lambda key, d: max(v[key] for v in d.values()) if d else None
    out['summary'] = {'position_mm_max': worst('position_mm', per), 'position_mm_mean': float(np.mean([v['position_mm'] for v in per.values()])),
                      'rotation_deg_max': worst('rotation_deg', per), 'principal_point_px_max': worst('principal_point_px', per),
                      'focal_pct_max': worst('focal_pct', per), 'time_offset_ms': out['time_offset_ms'],
                      'pair_distance_mm_max': worst('distance_mm', pairs), 'pair_rotation_deg_max': worst('relative_rotation_deg', pairs),
                      'step_envelopes_max': max(out['step_envelopes'].values())}
    if terms_a and terms_b:
        out['summary']['lens_terms_max_abs'] = worst('lens_terms_max_abs', per)
    return out


def published(tool_dir):
    rep = json.load(open(os.path.join(tool_dir, 'report.json')))
    head = {'verdict': rep.get('verdict'), 'passes': rep.get('passes_run'), 'published_toff': rep.get('published_toff'), 'cams': None}
    pub = rep.get('published')
    if not pub or not os.path.exists(os.path.join(tool_dir, os.path.basename(pub))):
        return head
    pub = os.path.join(tool_dir, os.path.basename(pub))
    cams, toff = parse_chain(pub)
    terms = read_terms(re.sub(r'_published_chain\.yaml$', '_published_terms.txt', pub))
    h1 = os.path.join(tool_dir, 'harvest_pass1.calib.json')
    p1 = None
    if os.path.exists(h1):
        cj = json.load(open(h1))
        p1 = (cams_from_caljson(cj), float(cj['toff']))
    head.update({'cams': cams, 'toff': toff, 'terms': terms, 'pass1': p1})
    return head


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--start', type=float, default=16.5)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    rep = {'run': run, 'times': times(run, a.start)}
    t = rep['times']
    try:
        tk_clock = float(read_env(os.path.join(run, 'replay.env'))['TAKEOFF_CLOCK'])
    except (OSError, KeyError, ValueError):
        tk_clock = None
    rep['board'] = {}
    if tk_clock is not None and t.get('tool_first_line') is not None and t.get('verdict_line') is not None:
        rep['board']['inflight'] = board(run, tk_clock + t['tool_first_line'], tk_clock + t['verdict_line'])
    for logf in sorted(glob.glob(os.path.join(run, 'offline[0-9]*.log'))):
        L = stamped_lines(logf)
        if L:
            rep['board'][os.path.basename(logf)[:-4]] = board(run, L[0][0], L[-1][0])
    runs = {'inflight': os.path.join(run, 'tool')}
    for d in sorted(glob.glob(os.path.join(run, 'offline[0-9]*'))):
        if os.path.isdir(d):
            runs[os.path.basename(d)] = d
    cal = {}
    for name, d in runs.items():
        try:
            cal[name] = published(d)
        except (OSError, ValueError, KeyError) as e:
            cal[name] = None
            rep.setdefault('problems', []).append('%s: %s: %s' % (name, type(e).__name__, e))
    rep['verdicts'] = {n: (c or {}).get('verdict') for n, c in cal.items()}
    rep['passes'] = {n: (c or {}).get('passes') for n, c in cal.items()}
    rep['compare'] = {}
    names = [n for n in runs if cal.get(n) and cal[n]['cams']]
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            A, B = cal[na], cal[nb]
            key = '%s vs %s' % (na, nb)
            rep['compare'][key] = {'published': compare(A['cams'], A['toff'], B['cams'], B['toff'], A['terms'], B['terms'])}
            if A['pass1'] and B['pass1']:
                rep['compare'][key]['pass1'] = compare(A['pass1'][0], A['pass1'][1], B['pass1'][0], B['pass1'][1])
    json.dump(rep, open(os.path.join(run, 'inflight_report.json'), 'w'), indent=1)

    t = rep['times']
    f = lambda x: 'never' if x is None else '%.1f' % x
    out = ['IN-FLIGHT SELF-CALIBRATION, real-time replay: %s' % run,
           'recording %s, %g s of flight, tool started %g s after takeoff (takeoff %.2f s after every stream is live)'
           % (t.get('recording'), t['flight_s'], t['start_after_s'], t['takeoff_after_live_s']),
           'times in seconds after takeoff on the replay clock']
    if 'pass1' in t:
        p = t['pass1']
        out.append('pass 1   line %s, first set done %.1f (%.1f s of data waiting), caught up %s, last set done %.1f (%.1f s behind the '
                   'live data); %d sets; backlog worked off at %s x real time (%s x in flight)'
                   % (f(t['pass_lines'].get(1)), p['first_set_done'], p['backlog_at_start_s'], f(p['caught_up']), p['last_set_done'],
                      p['behind_at_end_s'], p['sets'], '%.2f' % p['backlog_speed'] if 'backlog_speed' in p else '-',
                      '%.2f' % p['backlog_speed_in_flight'] if 'backlog_speed_in_flight' in p else '-'))
    if 'ready_as_run' in t:
        out.append('pass 2   line %.1f, %.1f s to the verdict (pass 2 trace: %s sets at %s x real time); passes %d; end of pass 2 %.1f'
                   % (t['pass_lines'][2], t['pass2_to_verdict_s'], t.get('pass2_trace', {}).get('sets', '-'),
                      '%.2f' % t['pass2_trace']['speed'] if 'pass2_trace' in t else '-', t['passes'], t['end_of_pass2']))
        cs = t['common_start']
        out.append('READY    as run %.1f s after takeoff | common start %.1f s: pass 1 caught up %s, ends %.1f, READY %.1f s after takeoff'
                   % (t['ready_as_run'], cs['start_s'], f(cs['pass1_caught_up']), cs['pass1_last_set_done'], cs['ready']))
    if t.get('problem'):
        out.append('PROBLEM  ' + t['problem'])
    out.append('verdicts %s ; passes %s' % (rep['verdicts'], rep['passes']))
    for key, cmp_ in rep['compare'].items():
        for what, c in cmp_.items():
            s = c['summary']
            out.append('%-22s %-9s cameras vs IMU: position max %.2f mm (mean %.2f), rotation max %.3f deg, principal point max %.2f px, '
                       'focal max %.3f %%, time offset %.2f ms%s | camera pairs: distance max %.2f mm, rotation max %.3f deg | '
                       'step %.2f envelopes' % (key, what, s['position_mm_max'], s['position_mm_mean'], s['rotation_deg_max'],
                                                s['principal_point_px_max'], s['focal_pct_max'], s['time_offset_ms'],
                                                (', lens terms max %.1e' % s['lens_terms_max_abs']) if 'lens_terms_max_abs' in s else '',
                                                s['pair_distance_mm_max'] or 0.0, s['pair_rotation_deg_max'] or 0.0, s['step_envelopes_max']))
    for name, b in rep['board'].items():
        if b:
            out.append('board    %-9s %d s: 1-min load mean %.2f max %.2f, GPU load mean %.0f %% max %.0f %%, CPU %.0f-%.0f MHz, hottest '
                       'zone %.1f C; other processes above 20 %% CPU: %s'
                       % (name, b['seconds'], b['load1_mean'], b['load1_max'], b['gpu_load_pct_mean'], b['gpu_load_pct_max'],
                          b['cpu_khz_min'] / 1000.0, b['cpu_khz_max'] / 1000.0, b['hottest_zone_c_max'],
                          ', '.join('%s (%.0f %%)' % (n, b['processes_over_20pct'][n]) for n in b['other_processes']) or 'none'))
    for p in rep.get('problems', []):
        out.append('PROBLEM  ' + p)
    open(os.path.join(run, 'inflight_report.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
