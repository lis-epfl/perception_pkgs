"""UDP between the drones: every datagram of relloc (keyframe packets, estimate messages, calibration) is one UDP datagram of at most
1,200 B (packet_codec CHUNK), sent best effort to every destination in the list; a lost datagram costs its keypoints, never the keyframe.

Destinations: 'host:port,host:port,...' (unicast to each peer, the PHY rate of the link) or the broadcast address ('255.255.255.255:port';
on Wi-Fi a broadcast frame goes at the network's basic rate, which can be too slow for ~1.7 Mbit/s per drone). The receiving socket binds
`port` on every interface; datagrams of the drone itself (its own broadcast) are dropped by the caller (drone id in every datagram)."""
import socket, threading, queue


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
