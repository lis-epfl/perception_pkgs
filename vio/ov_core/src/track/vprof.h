#pragma once
// Fine-grained scoped profiler + value statistics, self-contained inside the shared library.
//
// It deliberately does NOT write into run_serial_msckf.cpp's g_vio_stage_secs: that symbol is
// not shared across the executable/library boundary here (which is why VioManager's own
// "marginalize_old_clone" probe never appears in the printed table). This keeps its own
// registry and dumps it from a static destructor at process exit.
//
// Everything lives inside Reg so there is no static-destruction-order hazard: a separate
// stats() static would be destroyed before Reg's destructor could read it.
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <cstdlib>

namespace ov_core {
namespace vprof {

struct Stat {
  long n = 0;
  double sum = 0;
  double lo = 1e300;
  double hi = -1e300;
};

struct Reg {
  std::mutex m;
  std::map<std::string, double> secs;
  std::map<std::string, long> calls;
  std::map<std::string, Stat> stats;
  ~Reg() {
    if (!secs.empty()) {
      fprintf(stderr, "\n[vprof]: %-30s %10s %12s %12s\n", "probe", "sec", "calls", "ms/call");
      for (const auto &kv : secs) {
        long n = calls[kv.first];
        fprintf(stderr, "[vprof]: %-30s %10.3f %12ld %12.4f\n", kv.first.c_str(), kv.second, n,
                n ? 1000.0 * kv.second / n : 0.0);
      }
    }
    if (!stats.empty()) {
      fprintf(stderr, "\n[vstat]: %-30s %10s %8s %8s %10s\n", "quantity", "mean", "min", "max", "n");
      for (const auto &kv : stats)
        fprintf(stderr, "[vstat]: %-30s %10.1f %8.0f %8.0f %10ld\n", kv.first.c_str(),
                kv.second.n ? kv.second.sum / kv.second.n : 0.0, kv.second.lo, kv.second.hi, kv.second.n);
    }
    fflush(stderr);
  }
};

inline Reg &reg() {
  static Reg r;
  return r;
}
inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void note(const char *k, double v) {
  Reg &r = reg();
  std::lock_guard<std::mutex> lk(r.m);
  Stat &s = r.stats[k];
  s.n++;
  s.sum += v;
  if (v < s.lo) s.lo = v;
  if (v > s.hi) s.hi = v;
}

inline bool vprof_off() {
  static const bool off = [] { const char *e = std::getenv("OV_NOPROF"); return e && *e == '1'; }();
  return off;
}
struct Scope {
  const char *k;
  double t0;
  explicit Scope(const char *key) : k(key), t0(vprof_off() ? 0.0 : now()) {}
  ~Scope() {
    if (vprof_off()) return;
    double dt = now() - t0;
    Reg &r = reg();
    std::lock_guard<std::mutex> lk(r.m);
    r.secs[k] += dt;
    r.calls[k] += 1;
  }
};

} // namespace vprof
} // namespace ov_core

#define VP_CAT2(a, b) a##b
#define VP_CAT(a, b) VP_CAT2(a, b)
#define VPROF(name) ov_core::vprof::Scope VP_CAT(_vp_, __COUNTER__)(name)
#define VSTAT(name, value) ov_core::vprof::note(name, (double)(value))
