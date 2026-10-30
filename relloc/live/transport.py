"""The link between the drones. Every datagram of relloc (keyframe packets, estimate messages, calibration) travels on its own, at most
1,200 B (packet_codec CHUNK), best effort: a lost datagram costs its keypoints, never the keyframe. Two transports with one interface
(send(list of bytes); q = a queue of (bytes, t_arrival, address) filled by a receive thread; counters n_rx, n_tx, n_drop, n_err; close()):

Udp (RELLOC_TRANSPORT=udp): one UDP datagram per datagram, to every destination in the list. Destinations: 'host:port,host:port,...'
(unicast to each peer, the PHY rate of the link) or the broadcast address ('255.255.255.255:port'; on Wi-Fi a broadcast frame goes at the
network's basic rate, which can be too slow for ~1.7 Mbit/s per drone). The receiving socket binds `port` on every interface; datagrams of
the drone itself (its own broadcast) are dropped by the caller (drone id in every datagram).

Ros2 (RELLOC_TRANSPORT=ros2): one ROS 2 message per datagram (relloc2 review24/ros2: as reliable as raw UDP; see the class)."""
import socket, threading, queue, array, collections, re, time
from contextlib import ExitStack


def parse_dest(s, default_port):
    out = []
    for item in (x.strip() for x in (s or '').split(',')):
        if not item: continue
        host, _, port = item.rpartition(':') if ':' in item else (item, '', '')
        out.append((host or item, int(port) if port else int(default_port)))
    return out


class Udp:
    def __init__(self, port, dest, now, rcvbuf=4 << 20, maxq=20000):
        self.dest = dest; self.now = now; self.q = queue.Queue(maxsize=maxq); self.n_rx = 0; self.n_tx = 0; self.n_drop = 0; self.n_err = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try: self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        except OSError: pass
        self.sock.bind(('0.0.0.0', int(port))); self.sock.settimeout(0.2); self._run = True
        self.th = threading.Thread(target=self._rx, name='relloc-udp-rx', daemon=True); self.th.start()

    def _rx(self):
        while self._run:
            try: g, addr = self.sock.recvfrom(65535)
            except socket.timeout: continue
            except OSError: break
            t = self.now(); self.n_rx += 1
            try: self.q.put_nowait((g, t, addr))
            except queue.Full: self.n_drop += 1

    def send(self, dgrams):
        for g in dgrams:
            for d in self.dest:
                try: self.sock.sendto(g, d); self.n_tx += 1
                except OSError: self.n_err += 1

    def close(self):
        self._run = False
        try: self.sock.close()
        except OSError: pass


class Ros2:
    """ROS 2 between the drones, with Udp's interface; the address of a received datagram is ('ros2', topic).

    - One message per datagram: std_msgs/msg/UInt8MultiArray, data = array.array('B', datagram), the setter's fast path (relloc2
      review24/ros2: 14 us per publish, 86-103 us with bytes or a list). Never one message per keyframe: a 43-kB message is lost with any
      of its 30-34 fragments (1.6-3.3 % of the keyframes arrive at 10 % loss).
    - QoS: best effort, keep last `depth` (default 128; at least 72 = two bursts of 36), volatile, on the publisher and every
      subscription. Not RELIABLE: with default timers its repairs miss the 0.19-s decode window.
    - Topics: the drone publishes /relloc/dg/d<own id> and subscribes to /relloc/dg/d<k> of every peer k: the `peers` given, or, without
      them, every such topic of the domain's graph (looked up every 0.25 s for the first 10 s, then every `discover_s` s). Never to its own topic.
    - domain=n: the topics live in a separate rclpy Context with domain_id n (one more DDS participant, in domain n), which only the
      drones' relloc processes join, so the drones' own topics never mix and no bridge is needed. domain=None: the node's own context
      (rclpy.init() first) and domain; the fleet's domain_bridge must then forward /relloc/dg/d<k> between the drones (README).
    - Receive: a thread waits on a bare rcl wait set over the subscriptions (rclpy's internal _rclpy.WaitSet, built as rclpy's executor
      builds it) and after each wake-up drains every subscription with take_message, as review24/ros2's drain receiver Rd (0 late
      datagrams in 2 runs against 53-69 with one callback per message). Messages are taken serialized and the payload is sliced out of the
      CDR. Each datagram is stamped with now() at its take (the node's clock: sim time in replays). Without the executor's per-wake-up
      work the receive CPU per keyframe fell from 6.5 to 2.0 ms with datagrams spaced as on a 24-Mbit/s link, and from 1.25 to 0.84 ms
      with back-to-back bursts (raw UDP 0.7 / 0.3 ms; workstation, review24/ros2tx bench). rx='auto' falls back to a SingleThreadedExecutor
      (spin_once, then the same drain) when this rclpy has no such wait set; rx='executor' forces it. The subscriptions' QoS events
      (incompatible QoS) are not serviced by the wait set.
    - Send: under a lock (the sender and receiver threads both send). A publish that raises (e.g. after the context shut down) counts
      in n_err.
    - stats(): matched readers of the own topic and, per subscribed topic, datagrams received and two latencies of the last 2,000:
      DDS (reception - source timestamp: the one-way delay when the drones' clocks agree) and take (thread take - reception)."""
    PREFIX = '/relloc/dg/d'
    MSG = 'std_msgs/msg/UInt8MultiArray'

    def __init__(self, own_id, now, domain=None, peers=None, depth=128, discover_s=2.0, maxq=20000, rx='auto'):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import UInt8MultiArray
        self.own = int(own_id); self.now = now; self.q = queue.Queue(maxsize=maxq); self.n_rx = 0; self.n_tx = 0; self.n_drop = 0; self.n_err = 0
        self.n_bad = 0; self.n_rx_err = 0
        self.domain = None if domain is None else int(domain); self.depth = int(depth); self.discover_s = float(discover_s)
        self.peers = sorted({int(p) for p in peers} - {self.own}) if peers else None
        if self.domain is None:
            self.ctx = rclpy.get_default_context(); self._own_ctx = False
            if not self.ctx.ok(): raise RuntimeError("Ros2 transport in the node's own domain: rclpy.init() first")
        else:
            self.ctx = Context(); self.ctx.init(args=[], domain_id=self.domain, initialize_logging=False); self._own_ctx = True
        self.node = rclpy.create_node(f'relloc_dg_d{self.own}', context=self.ctx, use_global_arguments=False, enable_rosout=False,
                                      start_parameter_services=False)
        self.qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=self.depth,
                              durability=DurabilityPolicy.VOLATILE)
        self._MT = UInt8MultiArray; self._deser = deserialize_message; self._msg = UInt8MultiArray()
        self.topic = f'{self.PREFIX}{self.own}'; self.pub = self.node.create_publisher(UInt8MultiArray, self.topic, self.qos)
        self._re = re.compile('^' + re.escape(self.PREFIX) + r'(\d{1,3})$')
        self.subs = {}; self.stat = {}; self._lock = threading.Lock(); self._run = True; self.ex = None; self._gc = None; self.rx_loop = None
        for p in self.peers or []: self._subscribe(p)
        if rx in ('auto', 'waitset'):
            try: self._wait = self._waiter(); self.rx_loop = 'wait set'
            except Exception:
                if rx == 'waitset': raise
        if self.rx_loop is None:
            self.ex = SingleThreadedExecutor(context=self.ctx); self.ex.add_node(self.node); self.rx_loop = 'executor'
            self._wait = lambda: self.ex.spin_once(timeout_sec=0.1)
        self.th = threading.Thread(target=self._rx, name='relloc-ros2-rx', daemon=True); self.th.start()

    def _subscribe(self, pid):
        tp = f'{self.PREFIX}{pid}'
        if pid == self.own or tp in self.subs: return
        self.stat[tp] = [0, collections.deque(maxlen=2000), collections.deque(maxlen=2000)]
        self.subs[tp] = self.node.create_subscription(self._MT, tp, lambda raw, tp=tp: self._got(raw, None, tp), self.qos, raw=True)

    def _waiter(self):
        """wait(): one wake-up of the receive thread: a data message on a subscription, close() (the guard condition) or 0.1 s"""
        from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
        from rclpy.guard_condition import GuardCondition
        self._gc = GuardCondition(None, None, context=self.ctx); W = [None, -1]
        def wait(timeout_ns=100_000_000):
            subs = list(self.subs.values())
            with ExitStack() as cs:   # the handles stay valid while waited on, as in rclpy's executor
                for s in subs: cs.enter_context(s.handle)
                cs.enter_context(self._gc.handle); cs.enter_context(self.ctx.handle)
                if W[1] != len(subs): W[0] = _rclpy.WaitSet(len(subs), 1, 0, 0, 0, 0, self.ctx.handle); W[1] = len(subs)
                W[0].clear_entities()
                for s in subs: W[0].add_subscription(s.handle)
                W[0].add_guard_condition(self._gc.handle); W[0].wait(timeout_ns)
        wait(0)   # fails here, at construction, when this rclpy has no such wait set
        return wait

    def _discover(self):
        for name, types in self.node.get_topic_names_and_types():
            m = self._re.match(name)
            if m and self.MSG in types and int(m.group(1)) != self.own: self._subscribe(int(m.group(1)))

    def _payload(self, raw):
        """the data of a serialized UInt8MultiArray: sliced out of little-endian CDR with an empty layout (what send() publishes), else
        deserialized"""
        if len(raw) >= 16 and raw[0] == 0 and raw[1] == 1 and raw[4:8] == b'\0\0\0\0':
            n = int.from_bytes(raw[12:16], 'little')
            if 16 + n <= len(raw): return raw[16:16 + n]
        try: return self._deser(raw, self._MT).data.tobytes()
        except Exception: return None

    def _got(self, raw, info, tp):
        t = self.now(); g = self._payload(raw)
        if g is None: self.n_bad += 1; return
        self.n_rx += 1; s = self.stat[tp]; s[0] += 1
        if info is not None and info.get('received_timestamp', 0) > 0 and info.get('source_timestamp', 0) > 0:
            r = info['received_timestamp']; s[1].append((r - info['source_timestamp']) * 1e-6); s[2].append((time.time_ns() - r) * 1e-6)
        try: self.q.put_nowait((g, t, ('ros2', tp)))
        except queue.Full: self.n_drop += 1

    def _drain(self):
        for tp, s in list(self.subs.items()):
            for _ in range(4 * self.depth + 64):   # bounded: messages arriving during the drain wait for the next pass
                with s.handle: r = s.handle.take_message(s.msg_type, s.raw)
                if r is None: break
                self._got(r[0], r[1], tp)

    def _rx(self):
        from rclpy.executors import ExternalShutdownException
        t_disc = -1e18; t0 = time.monotonic()
        while self._run and self.ctx.ok():   # a SIGINT shuts down every context of the process, this one too
            try:
                # every 0.25 s for the first 10 s (DDS needs a moment to see the peers; a 2-s step lost a late joiner's first 11
                # keyframes), then every discover_s s; one lookup costs ~6 us
                due = 0.25 if time.monotonic() - t0 < 10.0 else self.discover_s
                if self.peers is None and time.monotonic() - t_disc >= due:
                    t_disc = time.monotonic(); self._discover()
                self._wait(); self._drain()
            except ExternalShutdownException:
                break
            except Exception:
                if not self._run or not self.ctx.ok(): break
                self.n_rx_err += 1; time.sleep(0.05)

    def send(self, dgrams):
        with self._lock:
            m = self._msg
            for g in dgrams:
                try: m.data = array.array('B', g); self.pub.publish(m); self.n_tx += 1
                except Exception: self.n_err += 1

    def describe(self):
        return {'domain': self.ctx.get_domain_id(), 'context': 'own (RELLOC_ROS_DOMAIN)' if self._own_ctx else "the node's", 'receive': self.rx_loop,
                'topic': self.topic, 'peers': self.peers if self.peers else f'discovered every {self.discover_s:g} s',
                'qos': f'best effort, keep last {self.depth}, volatile', 'message': f'{self.MSG}, one per datagram'}

    def stats(self):
        def pc(d):
            v = sorted(d)
            return [round(v[min(len(v) - 1, int(q * len(v)))], 2) for q in (0.5, 0.9, 1.0)] if v else None
        try: readers = self.pub.get_subscription_count()
        except Exception: readers = None
        return {'readers': readers, 'bad': self.n_bad, 'rx_err': self.n_rx_err,
                'peers': {tp: {'n': s[0], 'dds_ms_p50_p90_max': pc(s[1]), 'take_ms_p50_p90_max': pc(s[2])} for tp, s in list(self.stat.items())}}

    def close(self):
        self._run = False
        try:
            if self._gc is not None: self._gc.trigger()   # wakes the wait at once
        except Exception: pass
        if self.th is not threading.current_thread(): self.th.join(timeout=2.0)
        for f in (lambda: self.ex is not None and self.ex.shutdown(timeout_sec=1.0), lambda: self._gc is not None and self._gc.destroy(), self.node.destroy_node):
            try: f()
            except Exception: pass
        if self._own_ctx:
            try:
                if self.ctx.ok(): self.ctx.shutdown()
            except Exception: pass
