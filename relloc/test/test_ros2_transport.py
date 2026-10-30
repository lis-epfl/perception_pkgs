#!/usr/bin/env python3
"""live.transport.Ros2 in one process, no GPU: two transports with drone ids 3 and 6 exchange datagrams, in both domain modes and with
both receive loops (the wait set, used by the node, and the executor fallback):
  a   RELLOC_ROS_DOMAIN mode: each transport in its own rclpy Context with domain_id --domain-relloc (the default context is in another)
  b   the node's own domain: both transports in the default context (domain --domain-node)
Per mode and loop:
  discover   drone 6 has no peer list and finds /relloc/dg/d3 in the graph; drone 3 lists peer 6; neither subscribes to its own topic
  exact      every datagram arrives byte for byte and in order, both directions: 20 keyframes of 36 datagrams (random sizes 1-1,200 B,
             random content) with a 749-B calibration and a 60-B estimate datagram each; drone 3 sends from two threads at once (the
             sender and receiver threads of the node), which tests the send lock
  counters   n_tx of each sender = datagrams sent = n_rx of the receiver; n_drop = n_err = 0; address ('ros2', topic); stamps from now()
  foreign    a UInt8MultiArray with a non-empty layout (another publisher on /relloc/dg/d9) is deserialized, not sliced (mode a only)
  close      receive threads end; mode a: the own contexts are shut down; mode b: the default context stays up
Prints one line per check (plus the publish time per datagram), exit 1 on a failure. ROS_LOCALHOST_ONLY=1 is set (nothing leaves the host).
usage: python3 test_ros2_transport.py [--domain-node 60] [--domain-relloc 61]   (source the ROS 2 setup first)"""
import os, sys, time, random, threading, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('--domain-node', type=int, default=60); ap.add_argument('--domain-relloc', type=int, default=61)
A = ap.parse_args(); os.environ['ROS_LOCALHOST_ONLY'] = '1'
import rclpy
from live import transport as TR
FAIL = []
def check(name, ok, info=''):
    print(f'{"ok  " if ok else "FAIL"} {name} {info}', flush=True); (None if ok else FAIL.append(name))
T0 = 1000.0; now = lambda: T0 + time.monotonic()
rclpy.init(domain_id=A.domain_node)
rng = random.Random(7)

def traffic(drone, nkf=20):
    out = []
    for k in range(nkf):
        out.append([bytes([drone, k, i]) + rng.randbytes(rng.randint(0, 1197)) for i in range(36)])
        out[-1] += [bytes([drone, k, 200]) + rng.randbytes(746), bytes([drone, k, 201]) + rng.randbytes(57)]
    return out

def wait(cond, timeout=15.0):
    t = time.monotonic()
    while not cond() and time.monotonic() - t < timeout: time.sleep(0.02)
    return cond()

def drain(tr, n, timeout=10.0):
    got = []; t = time.monotonic()
    while len(got) < n and time.monotonic() - t < timeout:
        try: got.append(tr.q.get(timeout=0.1))
        except Exception: pass
    time.sleep(0.3)
    while not tr.q.empty(): got.append(tr.q.get_nowait())
    return got

for mode, rx in (('a', 'waitset'), ('b', 'waitset'), ('a', 'executor'), ('b', 'executor')):
    dom = A.domain_relloc if mode == 'a' else None
    T3 = TR.Ros2(3, now, domain=dom, peers={6}, rx=rx); T6 = TR.Ros2(6, now, domain=dom, peers=None, rx=rx)
    check(f'{mode}/{rx}: receive loop', T3.rx_loop == T6.rx_loop == {'waitset': 'wait set', 'executor': 'executor'}[rx], f'({T3.rx_loop})')
    want = A.domain_relloc if mode == 'a' else A.domain_node
    check(f'{mode}/{rx}: domain', T3.ctx.get_domain_id() == want and T6.ctx.get_domain_id() == want and (T3.ctx is not T6.ctx) == (mode == 'a'),
          f'({T3.ctx.get_domain_id()}, {T6.ctx.get_domain_id()}; {"own contexts" if mode == "a" else "default context"})')
    ok = wait(lambda: '/relloc/dg/d3' in T6.subs and T3.pub.get_subscription_count() >= 1 and T6.pub.get_subscription_count() >= 1)
    check(f'{mode}/{rx}: discover', ok and set(T6.subs) == {'/relloc/dg/d3'} and set(T3.subs) == {'/relloc/dg/d6'},
          f'(6 subscribes {sorted(T6.subs)}, 3 subscribes {sorted(T3.subs)}; matched readers {T3.pub.get_subscription_count()} / {T6.pub.get_subscription_count()})')
    time.sleep(0.5)   # both sides of the match settled
    c0 = {d: (t.n_tx, t.n_rx, t.n_drop, t.n_err) for d, t in ((3, T3), (6, T6))}
    K3 = traffic(3); K6 = traffic(6); half = len(K3) // 2; dt = []
    def snd(tr, kfs):
        for kf in kfs:
            t = time.perf_counter(); tr.send(kf); dt.append((time.perf_counter() - t) / len(kf)); time.sleep(0.005)
    t_send0 = now()
    th = [threading.Thread(target=snd, args=(T3, K3[:half])), threading.Thread(target=snd, args=(T3, K3[half:])), threading.Thread(target=snd, args=(T6, K6))]
    [x.start() for x in th]; [x.join() for x in th]; t_send1 = now()
    n3 = sum(len(k) for k in K3); n6 = sum(len(k) for k in K6)
    g6 = drain(T6, n3); g3 = drain(T3, n6)
    s3 = [g for k in K3 for g in k]; s6 = [g for k in K6 for g in k]
    def order_ok(got, sent_parts):   # two sending threads: each thread's datagrams in order, interleaving free
        it = [iter(p) for p in sent_parts]; cur = [next(i, None) for i in it]
        for g in got:
            j = next((j for j, c in enumerate(cur) if c == g), None)
            if j is None: return False
            cur[j] = next(it[j], None)
        return all(c is None for c in cur)
    p3 = [[g for k in K3[:half] for g in k], [g for k in K3[half:] for g in k]]
    check(f'{mode}/{rx}: exact 3 -> 6', len(g6) == n3 and sorted(x[0] for x in g6) == sorted(s3) and order_ok([x[0] for x in g6], p3),
          f'({len(g6)} of {n3} datagrams, byte for byte; each sending thread in order)')
    check(f'{mode}/{rx}: exact 6 -> 3', len(g3) == n6 and [x[0] for x in g3] == s6, f'({len(g3)} of {n6} datagrams, byte for byte, in order)')
    d3 = [a - b for a, b in zip((T3.n_tx, T3.n_rx, T3.n_drop, T3.n_err), c0[3])]; d6 = [a - b for a, b in zip((T6.n_tx, T6.n_rx, T6.n_drop, T6.n_err), c0[6])]
    check(f'{mode}/{rx}: counters', d3[0] == n3 == d6[1] and d6[0] == n6 == d3[1] and d3[2:] == [0, 0] and d6[2:] == [0, 0] and T3.n_bad == T6.n_bad == 0,
          f'(3: tx {d3[0]} rx {d3[1]} drop {d3[2]} err {d3[3]}; 6: tx {d6[0]} rx {d6[1]} drop {d6[2]} err {d6[3]})')
    adr = {x[2] for x in g6} | {x[2] for x in g3}
    check(f'{mode}/{rx}: address and stamps', adr == {('ros2', '/relloc/dg/d3'), ('ros2', '/relloc/dg/d6')} and all(t_send0 <= x[1] <= t_send1 + 15 for x in g6 + g3)
          and not any(x[0][0] == 6 for x in g6) and not any(x[0][0] == 3 for x in g3), f'({sorted(adr)}; no own datagram received)')
    st = T6.stats()
    print(f'     {mode}/{rx}: publish {1e6 * sorted(dt)[len(dt) // 2]:.0f} us per datagram (median per burst); drone 6 stats {st}', flush=True)
    if mode == 'a':   # another publisher with a non-empty layout on a peer topic: the receiver deserializes it
        from std_msgs.msg import UInt8MultiArray, MultiArrayDimension
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        X = rclpy.create_node('foreign_d9', context=T6.ctx); qx = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=16, durability=DurabilityPolicy.VOLATILE)
        px = X.create_publisher(UInt8MultiArray, '/relloc/dg/d9', qx); okx = wait(lambda: '/relloc/dg/d9' in T6.subs and px.get_subscription_count() >= 1)
        time.sleep(0.5); m = UInt8MultiArray(); m.layout.dim = [MultiArrayDimension(label='x', size=5, stride=5)]; m.data = bytes([9, 1, 2, 3, 4]); px.publish(m)
        gx = drain(T6, 1, 5.0)
        check(f'a/{rx}: foreign layout deserialized', okx and len(gx) == 1 and gx[0][0] == bytes([9, 1, 2, 3, 4]) and gx[0][2] == ('ros2', '/relloc/dg/d9'), f'({[(x[0], x[2]) for x in gx]})')
        X.destroy_node()
    ctxs = (T3.ctx, T6.ctx)
    t_c = time.monotonic(); T3.close(); T6.close(); t_c = time.monotonic() - t_c
    ok_close = not T3.th.is_alive() and not T6.th.is_alive() and (all(not c.ok() for c in ctxs) if mode == 'a' else rclpy.ok())
    check(f'{mode}/{rx}: close', ok_close, f'(threads ended; {"own contexts shut down" if mode == "a" else "default context still up"}; both closed in {t_c:.2f} s)')
rclpy.shutdown()
print('ALL PASSED' if not FAIL else f'FAILED: {FAIL}'); sys.exit(1 if FAIL else 0)
