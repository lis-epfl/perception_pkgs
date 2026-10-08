
// ======================================================================================================================
// OV_XCAM=1 -- cross-camera association in the grouped monocular path.
//
// After the per-camera feeds of ONE grouped message have returned, a feature of camera a gets, when it can be found, a
// second observation in a neighbouring camera b under the SAME feature id. The filter already handles a feature seen by
// several cameras (that is how upstream stereo works), so nothing changes in the update.
//
//  - pairs: every two cameras of the message whose optical axes (current extrinsics) are < OV_XCAM_PAIR_DEG apart
//  - candidates: points of a not yet shared with b and not tried against b (retry every OV_XCAM_RETRY frame-sets,
//    0 = only once), at most OV_XCAM_MAX per pair and direction per frame-set, newest first
//  - matcher: the curve on which the point must lie in b for inverse depth 0..1/DMIN is
//    predicted from the CURRENT calibration through the camera objects (calib_for), searched with a patch warped by the
//    local affine map (normalised correlation, uniqueness), then refined in 2-D; the refined position may leave the curve
//    by up to MAXOFF px. Reverse check: the same search from the found position back into a must land within MUT px.
//    RAW images of the message (identical in the CPU and the GPU tracking mode).
//  - commit: sequential, fixed order. Same scene point already tracked in b (within SAME px): the fresh one of the two
//    takes the other's id; otherwise the point is appended to b's list. Shared points are kept FIRST in the lists so that
//    the one-point-per-cell pruning of the detection step keeps them.
// With OV_XCAM unset none of this runs.
// ======================================================================================================================
namespace {
constexpr int XC_NCAM = 8;
inline double xc_envd(const char *name, double def) {
  const char *e = std::getenv(name);
  return (e && *e) ? atof(e) : def;
}
std::string xc_tail_stats();
void xc_host_want(bool on);
struct XcInfo {
  int born = 0;
  int sec = 0; // 0: the point carries the id it was detected with; 1: it took the id of another camera's point; 2: it was put here by the matcher
  int next_try[XC_NCAM]; // frame-set from which this point may be tried against camera d (-1: never again)
  int assoc[XC_NCAM];    // frame-set at which this point became shared with camera d (-1: not shared)
  int mchk[XC_NCAM];     // frame-set of the last re-registration check of the pair with camera d (-1: none yet)
  bool recv[XC_NCAM];    // in the association with camera d this point is the receiver (it was given the id; set in link(), cleared with assoc[d])
  XcInfo() {
    for (int d = 0; d < XC_NCAM; d++) {
      next_try[d] = 0;
      assoc[d] = -1;
      mchk[d] = -1;
      recv[d] = false;
    }
  }
  bool shared() const {
    for (int d = 0; d < XC_NCAM; d++)
      if (assoc[d] >= 0)
        return true;
    return false;
  }
};
struct XcMatch {
  int k;
  float px, py, qx, qy, qnx, qny;
};
struct XcTask {
  size_t a = 0, b = 0;
  int pair = 0, cand = 0, mutual_fail = 0, quota = 0;
  long rej[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  long fov_excl = 0;
  std::vector<int> tried;
  std::vector<XcMatch> matches;
};
struct XcGrid {
  std::vector<float> ox, oy;
  float ext = 0;
  int n = 0;
};
struct XcScratch {
  XcGrid G0, G0c, G1, G1c;
  std::vector<float> Tp, Tpc, P, Pxp, Pxm, Pyp, Pym;
  std::vector<float> score;
  std::vector<Eigen::Vector2d> ctr;
  std::vector<char> valid;
};
struct XcState {
  bool on = false, mutual = true, par = true, shared_first = true;
  int stats = 0, cap = 80, retry = 0, PS = 13, csub = 2, pts_every = 30, minage = 0;
  bool takeover = false;
  int maint = 0;                          // OV_XCAM_MAINT = K: every K frame-sets each shared pair is re-registered photometrically (0 = off)
  double MAINT_TOL = 0.3, MAINT_MAX = 3.0; // px: below TOL nothing changes; up to MAX the receiving point is moved; beyond (or no fit) the pair is dissolved
  long m_checked = 0, m_kept = 0, m_moved = 0, m_dissolved = 0;
  double m_shift_sum = 0;
  bool snap = false;                      // OV_XCAM_SNAP: an association made by relabelling moves camera b's point onto the matched position
  long s_moved = 0, s_third = 0;          // such moves; those of a point that was registered with a third camera already
  double s_shift_sum = 0;
  long m_no_origin = 0, m_three = 0;      // due re-registrations that were not made: both points or neither marked as the receiver of the pair (XcInfo::recv; cannot happen, counted to show it) / the point is shared by three cameras
  double ZMIN = 0.80, ZFIN = 0.85, UNIQ = 0.08, MAXOFF = 3.0, RHO_MAX = 1.0 / 0.30, MUT_PX = 1.0, SAME_PX = 1.5, FOV_DEG = 83.0, PAIR_DEG = 120.0;
  double cosfov = 0, tanfov = 0;
  std::vector<std::pair<size_t, size_t>> pairs_env;
  long fs = 0;
  std::vector<size_t> ids[XC_NCAM];
  std::vector<XcInfo> info[XC_NCAM];
  // totals
  long t_cand = 0, t_matched = 0, t_mfail = 0, t_relabel = 0, t_inject = 0, t_skip = 0, t_dup = 0, t_fov_excl = 0, t_considered = 0;
  long t_shared_sum[XC_NCAM][XC_NCAM];
  long rej[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<long> life_hist; // ended associations: number of frame-sets with both observations
  std::vector<long> pt_life_hist[3]; // ended list entries by XcInfo::sec: frame-sets in the list (0: all ordinary points = baseline)
  long end_cause[3] = {0, 0, 0};     // association ended because: both sides lost together / the receiving side lost first / the origin side lost first
  std::vector<float> ms;
  bool warned_single = false, warned_ext = false;
  // ---- switches that bound the cost of the step
  std::string preset;                 // OV_XCAM_PRESET (rt: the flight values, see xc_preset(); explicit switches override)
  bool fastproj = true;               // OV_XCAM_FASTPROJ: inline projection (0 = through the camera class)
  int every = 1;                      // OV_XCAM_EVERY = k: search and maintenance only when k frame-sets have passed since the last run
  bool readback = false;              // OV_XCAM_READBACK: bring GPU-decoded frames back to the host on the frame-sets where the step is due
  int rb_lead = 1;                    // OV_XCAM_READBACK_LEAD: frame-sets whose device upload is queued before the association of the current one ends
                                      // (start value; corrected from the first answered request on, see rb_lead_seen)
  int rb_lead_seen = -1;
  bool emu_ph = false;                // OV_XCAM_EMU_PLACEHOLDER: TEST ONLY, treat the message images as carrying no pixels
  bool rb_sync = false;               // OV_XCAM_READBACK_SYNC: TEST ONLY, read back with the synchronous device copy (waits for the tracker streams)
  int emu_blank = 0;                  // OV_XCAM_EMU_BLANK = n: TEST ONLY, from frame-set n on the images the matcher reads are zero pages
  int emu_blank_end = 0;              // OV_XCAM_EMU_BLANK_END = m: TEST ONLY, ... until frame-set m (exclusive; 0 = for ever)
  int budget = 0;                     // OV_XCAM_BUDGET = n: at most n searches per frame-set over ALL pairs and directions (0 = only the per-pair limit OV_XCAM_MAX)
  long t_budget_cut = 0;              // frame-sets on which the budget left eligible points for later
  long proj_test = 0;                 // OV_XCAM_PROJ_TEST = n: compare the inline projection with the camera class on n random directions per camera
  bool proj_verify = false;           // OV_XCAM_PROJ_VERIFY: recompute EVERY projection of the matcher through the class and count differences
  long last_run = 0, rb_req_fs = 0, t_runs = 0, t_notdue = 0, t_unusable = 0, t_blank = 0, t_rb_lost = 0, t_rb_used = 0;
  bool rb_out = false, warned_ph = false, warned_const = false, warned_blank = false, proj_tested = false;
  double search_ms_sum = 0;
  cv::Mat zero_page;
  static double preset_d(const std::string &p, const char *name, double def) {
    // OV_XCAM_PRESET=rt: values for flight. Anything not listed keeps its default. Three whole flights (nxt3_s1, nxt6_s2, nxt10_s1)
    // calibrated with these values gave the same result as with the defaults, at about half the mean and a third of the largest cost
    // of the step. Where bringing pixels back from the device is expensive: OV_XCAM_EVERY=2 OV_XCAM_BUDGET=40 (tested on the same flights).
    if (p == "rt") {
      static const struct { const char *n; double v; } T[] = {{"OV_XCAM_MAX", 20}, {"OV_XCAM_MINAGE", 3}, {"OV_XCAM_EVERY", 1}, {"OV_XCAM_BUDGET", 30}, {"OV_XCAM_MAINT", 0}, {"OV_XCAM_READBACK", 1}};
      for (const auto &e : T)
        if (!std::strcmp(e.n, name))
          return e.v;
    }
    return def;
  }
  double envp(const char *name, double def) const { return xc_envd(name, preset_d(preset, name, def)); }
  XcState() {
    const char *e = std::getenv("OV_XCAM");
    on = e && *e == '1';
    for (int i = 0; i < XC_NCAM; i++)
      for (int j = 0; j < XC_NCAM; j++)
        t_shared_sum[i][j] = 0;
    if (!on)
      return;
    if (const char *pz = std::getenv("OV_XCAM_PRESET"))
      preset = pz;
    if (!preset.empty() && preset != "rt")
      std::fprintf(stderr, "[xcam-warn] OV_XCAM_PRESET='%s' is not known (known: rt): defaults are used\n", preset.c_str());
    stats = (int)xc_envd("OV_XCAM_STATS", 0);
    cap = (int)envp("OV_XCAM_MAX", 80);
    every = std::max(1, (int)envp("OV_XCAM_EVERY", 1));
    fastproj = envp("OV_XCAM_FASTPROJ", 1) != 0;
    readback = envp("OV_XCAM_READBACK", 0) != 0;
    rb_sync = xc_envd("OV_XCAM_READBACK_SYNC", 0) != 0;
    ov_core::ov_nvjpg_host_copy(readback && !rb_sync); // GPU-decoded pictures are then also copied to host memory on the decode thread
    rb_lead = std::max(0, (int)envp("OV_XCAM_READBACK_LEAD", 1));
    emu_ph = xc_envd("OV_XCAM_EMU_PLACEHOLDER", 0) != 0;
    emu_blank = (int)xc_envd("OV_XCAM_EMU_BLANK", 0);
    emu_blank_end = (int)xc_envd("OV_XCAM_EMU_BLANK_END", 0);
    budget = std::max(0, (int)envp("OV_XCAM_BUDGET", 0));
    proj_test = (long)xc_envd("OV_XCAM_PROJ_TEST", 0);
    proj_verify = xc_envd("OV_XCAM_PROJ_VERIFY", 0) != 0;
    rb_out = (1 + rb_lead >= every); // the pixels of the first frame-set on which the step is due are asked for from the start
    xc_host_want(rb_out);
    retry = (int)xc_envd("OV_XCAM_RETRY", 0);
    PS = (int)xc_envd("OV_XCAM_PS", 13);
    csub = std::max(1, (int)xc_envd("OV_XCAM_CSUB", 2));
    ZMIN = xc_envd("OV_XCAM_ZNCC", 0.80);
    ZFIN = xc_envd("OV_XCAM_ZNCC_FIN", 0.85);
    UNIQ = xc_envd("OV_XCAM_UNIQ", 0.08);
    MAXOFF = xc_envd("OV_XCAM_MAXOFF", 3.0);
    RHO_MAX = 1.0 / xc_envd("OV_XCAM_DMIN", 0.30);
    mutual = xc_envd("OV_XCAM_MUTUAL", 1) != 0;
    MUT_PX = xc_envd("OV_XCAM_MUTUAL_PX", 1.0);
    SAME_PX = xc_envd("OV_XCAM_SAME_PX", 1.5);
    FOV_DEG = xc_envd("OV_XCAM_FOV_DEG", 83.0);
    PAIR_DEG = xc_envd("OV_XCAM_PAIR_DEG", 120.0);
    par = xc_envd("OV_XCAM_PAR", 1) != 0;
    shared_first = xc_envd("OV_XCAM_SHARED_FIRST", 1) != 0;
    minage = std::max(0, (int)envp("OV_XCAM_MINAGE", 0));
    maint = std::max(0, (int)envp("OV_XCAM_MAINT", 0));
    MAINT_TOL = xc_envd("OV_XCAM_MAINT_TOL", 0.3);
    MAINT_MAX = xc_envd("OV_XCAM_MAINT_MAX", 3.0);
    snap = xc_envd("OV_XCAM_SNAP", 0) != 0;
    takeover = xc_envd("OV_XCAM_TAKEOVER", minage > 0 ? 1 : 0) != 0;
    cosfov = std::cos(FOV_DEG * M_PI / 180.0);
    tanfov = std::tan(FOV_DEG * M_PI / 180.0);
    if (const char *pe = std::getenv("OV_XCAM_PAIRS")) { // e.g. "0-1,1-2,2-3,3-0"
      const char *s = pe;
      while (*s) {
        char *end = nullptr;
        long a = std::strtol(s, &end, 10);
        if (end == s || *end != '-')
          break;
        s = end + 1;
        long b = std::strtol(s, &end, 10);
        if (end == s)
          break;
        if (a >= 0 && b >= 0 && a < XC_NCAM && b < XC_NCAM && a != b)
          pairs_env.emplace_back((size_t)a, (size_t)b);
        s = end;
        while (*s == ',' || *s == ' ')
          s++;
      }
    }
    std::fprintf(stderr, "[xcam-cfg] on=1 cap=%d retry=%d ps=%d csub=%d zncc=%.2f zfin=%.2f uniq=%.2f maxoff=%.1f dmin=%.2f mutual=%d mut_px=%.2f same_px=%.2f fov_deg=%.1f pair_deg=%.1f par=%d pairs_env=%zu minage=%d takeover=%d shared_first=%d maint=%d maint_tol=%.2f maint_max=%.2f every=%d fastproj=%d readback=%d readback_lead=%d preset=%s budget=%d\n",
                 cap, retry, PS, csub, ZMIN, ZFIN, UNIQ, MAXOFF, 1.0 / RHO_MAX, (int)mutual, MUT_PX, SAME_PX, FOV_DEG, PAIR_DEG, (int)par, pairs_env.size(), minage, (int)takeover,
                 (int)shared_first, maint, MAINT_TOL, MAINT_MAX, every, (int)fastproj, (int)readback, rb_lead, preset.empty() ? "none" : preset.c_str(), budget);
    if (const char *se = std::getenv("OV_XCAM_SNAP")) {
      if (*se)
        std::fprintf(stderr, "[xcam-cfg] snap=%d\n", (int)snap);
    }
  }
  static void hist_add(std::vector<long> &h, int n) {
    if (n < 0)
      n = 0;
    if ((size_t)n >= h.size())
      h.resize(n + 1, 0);
    h[n]++;
  }
  void life_add(int n) { hist_add(life_hist, n); }
  static void hist_stat(const std::vector<long> &h, long &n, long &med, double &gt10) {
    n = 0;
    long g = 0;
    for (size_t i = 0; i < h.size(); i++) {
      n += h[i];
      if (i > 10)
        g += h[i];
    }
    med = 0;
    long acc = 0;
    for (size_t i = 0; i < h.size(); i++) {
      acc += h[i];
      if (2 * acc >= n) {
        med = (long)i;
        break;
      }
    }
    gt10 = n > 0 ? (double)g / (double)n : 0.0;
  }
  ~XcState() {
    if (!on)
      return;
    // associations still alive at exit count with their current length (a lower bound)
    long alive = 0;
    for (int c = 0; c < XC_NCAM; c++)
      for (const XcInfo &I : info[c])
        for (int d = c + 1; d < XC_NCAM; d++)
          if (I.assoc[d] >= 0) {
            life_add((int)(fs - I.assoc[d] + 1));
            alive++;
          }
    long n = 0, med = 0, pn[3], pmed[3];
    double gt10 = 0, pgt[3];
    hist_stat(life_hist, n, med, gt10);
    for (int i = 0; i < 3; i++)
      hist_stat(pt_life_hist[i], pn[i], pmed[i], pgt[i]);
    double mean = 0, p95 = 0, mx = 0, p99 = 0;
    if (!ms.empty()) {
      std::vector<float> s(ms);
      std::sort(s.begin(), s.end());
      for (float v : s)
        mean += v;
      mean /= s.size();
      p95 = s[(size_t)(0.95 * (s.size() - 1))];
      p99 = s[(size_t)(0.99 * (s.size() - 1))];
      mx = s.back();
    }
    std::string sh;
    for (int i = 0; i < XC_NCAM; i++)
      for (int j = 0; j < XC_NCAM; j++)
        if (t_shared_sum[i][j] > 0) {
          char buf[96];
          std::snprintf(buf, sizeof(buf), " shared_mean_%d_%d=%.2f", i, j, fs > 0 ? (double)t_shared_sum[i][j] / (double)fs : 0.0);
          sh += buf;
        }
    std::fprintf(stderr,
                 "[xcam-total] framesets=%ld considered=%ld cand=%ld matched=%ld mutual_fail=%ld relabel=%ld inject=%ld skip=%ld dup=%ld fov83_excluded=%ld rej_fov_a=%ld rej_cone=%ld "
                 "rej_aff=%ld rej_tex=%ld rej_weak=%ld rej_amb=%ld rej_ref=%ld rej_fin=%ld rej_off=%ld rej_edge=%ld rej_fov_b=%ld life_n=%ld life_alive_at_exit=%ld life_median=%ld "
                 "life_share_gt10=%.4f end_both=%ld end_receiver_first=%ld end_origin_first=%ld ptlife_plain_n=%ld ptlife_plain_median=%ld ptlife_plain_gt10=%.4f ptlife_relabeled_n=%ld "
                 "ptlife_relabeled_median=%ld ptlife_relabeled_gt10=%.4f ptlife_injected_n=%ld ptlife_injected_median=%ld ptlife_injected_gt10=%.4f maint_checked=%ld maint_kept=%ld maint_moved=%ld "
                 "maint_dissolved=%ld maint_moved_mean_px=%.3f ms_mean=%.4f ms_p95=%.4f ms_max=%.4f par=%d%s runs=%ld skip_notdue=%ld skip_unusable=%ld blank_tasks=%ld ms_p99=%.4f search_ms_sum=%.2f "
                 "us_per_cand=%.3f rb_used=%ld rb_lost=%ld budget_cut=%ld rb_lead_end=%d %s\n",
                 fs, t_considered, t_cand, t_matched, t_mfail, t_relabel, t_inject, t_skip, t_dup, t_fov_excl, rej[1], rej[2], rej[3], rej[4], rej[5], rej[6], rej[7], rej[8], rej[9],
                 rej[10], rej[11], n, alive, med, gt10, end_cause[0], end_cause[1], end_cause[2], pn[0], pmed[0], pgt[0], pn[1], pmed[1], pgt[1], pn[2], pmed[2], pgt[2], m_checked, m_kept, m_moved, m_dissolved,
                 m_moved > 0 ? m_shift_sum / (double)m_moved : 0.0, mean, p95, mx, (int)par, sh.c_str(), t_runs, t_notdue, t_unusable, t_blank, p99, search_ms_sum,
                 t_cand > 0 ? 1000.0 * search_ms_sum / (double)t_cand : 0.0, t_rb_used, t_rb_lost, t_budget_cut, rb_lead, xc_tail_stats().c_str());
    if (snap) // the two counters of the maintenance are in its own line below
      std::fprintf(stderr, "[xcam-snap] moved=%ld moved_mean_px=%.3f moved_third_camera=%ld\n", s_moved, s_moved > 0 ? s_shift_sum / (double)s_moved : 0.0, s_third);
    if (maint > 0)
      std::fprintf(stderr, "[xcam-maint] maint_skip_no_origin=%ld maint_skip_three_cameras=%ld\n", m_no_origin, m_three);
  }
};
XcState &xc() {
  static XcState s;
  return s;
}

// ---- host pixels for GPU-decoded frames (OV_XCAM_READBACK) and the projection check counters
struct XcHostCache {
  std::mutex m;
  struct E {
    double ts;
    cv::Mat img;
  };
  std::vector<E> e[XC_NCAM];
  std::atomic<bool> want{true}; // set by the association when its next run needs pixels; taken by the first camera of the next new timestamp
  double dec_ts = -1;
  bool dec = false;
  long n_ok = 0, n_fail = 0;
  double ms = 0;
};
XcHostCache &xc_host() {
  static XcHostCache h;
  return h;
}
void xc_host_want(bool on) { xc_host().want.store(on); }
bool xc_host_get(size_t c, double ts, cv::Mat &out) {
  XcHostCache &H = xc_host();
  std::lock_guard<std::mutex> lk(H.m);
  for (const XcHostCache::E &x : H.e[c])
    if (x.ts == ts) {
      out = x.img;
      return true;
    }
  return false;
}
std::atomic<long> xc_pv_n{0}, xc_pv_bad{0};
std::atomic<uint32_t> xc_pv_max{0}; // largest |difference| in px, as float bits (non-negative floats order like their bit patterns)
std::string xc_proj_test_line;
std::string xc_tail_stats() {
  XcHostCache &H = xc_host();
  char b[256];
  uint32_t u = xc_pv_max.load();
  float mxd;
  std::memcpy(&mxd, &u, 4);
  std::snprintf(b, sizeof(b), "rb_ok=%ld rb_fail=%ld rb_ms_mean=%.4f proj_verified=%ld proj_differ=%ld proj_maxdiff_px=%.3g", H.n_ok, H.n_fail, (H.n_ok + H.n_fail) > 0 ? H.ms / (double)(H.n_ok + H.n_fail) : 0.0,
                xc_pv_n.load(), xc_pv_bad.load(), (double)mxd);
  return std::string(b);
}

// A camera as the matcher reads it: the eight Kannala-Brandt values and the lens terms are copied ONCE per task and a
// direction is projected inline with the arithmetic of CamEqui::distort_f / distort_f_nonradial in the same order (float in, float out),
// without the heap copy of the parameter matrix and with products in place of the four pow calls. Any other camera class, or
// OV_XCAM_FASTPROJ=0: through the class as before.
struct XcCam {
  ov_core::CamBase *cam = nullptr;
  bool fast = false, nr = false, verify = false;
  double fx = 1, fy = 1, cx = 0, cy = 0, k1 = 0, k2 = 0, k3 = 0, k4 = 0, p1 = 0, p2 = 0, sk = 0;
  XcCam() {}
  XcCam(ov_core::CamBase *c, bool want_fast, bool want_verify) : cam(c) {
    const Eigen::MatrixXd v = c->get_value();
    const bool ok = v.rows() * v.cols() >= 8;
    if (ok) {
      fx = v(0), fy = v(1), cx = v(2), cy = v(3), k1 = v(4), k2 = v(5), k3 = v(6), k4 = v(7);
    }
    const Eigen::Vector3d q = c->get_nonradial();
    p1 = q(0), p2 = q(1), sk = q(2);
    nr = c->get_nonradial_estimated() || p1 != 0.0 || p2 != 0.0 || sk != 0.0; // = CamBase::has_nonradial
    fast = want_fast && ok && dynamic_cast<ov_core::CamEqui *>(c) != nullptr;
    verify = fast && want_verify;
  }
  inline Eigen::Vector2f distort(float ux, float uy) const {
    const double r = std::sqrt(ux * ux + uy * uy); // float product and float square root, as in the class
    const double theta = std::atan(r);
    const double t2 = theta * theta, t3 = t2 * theta, t5 = t3 * t2, t7 = t5 * t2, t9 = t7 * t2;
    const double theta_d = theta + k1 * t3 + k2 * t5 + k3 * t7 + k4 * t9;
    const double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
    const double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;
    double x1 = ux * cdist, y1 = uy * cdist;
    if (nr) { // CamEqui::add_nonradial: measured = ideal + N(measured), fixed-point iteration
      const double xi = x1, yi = y1;
      for (int it = 0; it < 20; it++) {
        const double r2 = x1 * x1 + y1 * y1;
        const double xn = xi + (2.0 * p1 * x1 * y1 + p2 * (r2 + 2.0 * x1 * x1) + sk * y1);
        const double yn = yi + (p1 * (r2 + 2.0 * y1 * y1) + 2.0 * p2 * x1 * y1);
        const double step = std::abs(xn - x1) + std::abs(yn - y1);
        x1 = xn;
        y1 = yn;
        if (it >= 3 && step < 1e-15)
          break;
      }
    }
    return Eigen::Vector2f((float)(fx * x1 + cx), (float)(fy * y1 + cy));
  }
};
inline void xc_pv_note(const Eigen::Vector2f &a, const Eigen::Vector2f &b) {
  xc_pv_n.fetch_add(1, std::memory_order_relaxed);
  if (std::memcmp(a.data(), b.data(), 2 * sizeof(float)) == 0)
    return;
  xc_pv_bad.fetch_add(1, std::memory_order_relaxed);
  const float dv = std::max(std::fabs(a.x() - b.x()), std::fabs(a.y() - b.y()));
  uint32_t u, cur = xc_pv_max.load();
  std::memcpy(&u, &dv, 4);
  while (std::isfinite(dv) && u > cur && !xc_pv_max.compare_exchange_weak(cur, u)) {
  }
}
// direction in the camera frame (any length) -> pixel; only within acos(cosmin) of the axis
inline bool xc_proj(const XcCam &C, const Eigen::Vector3d &d, double cosmin, Eigen::Vector2d &px) {
  const double n = d.norm();
  if (n < 1e-12 || d.z() < cosmin * n)
    return false;
  const float ux = (float)(d.x() / d.z()), uy = (float)(d.y() / d.z());
  const Eigen::Vector2f uv = C.fast ? C.distort(ux, uy) : C.cam->distort_f(Eigen::Vector2f(ux, uy));
  if (C.verify)
    xc_pv_note(uv, C.cam->distort_f(Eigen::Vector2f(ux, uy)));
  px = Eigen::Vector2d(uv.x(), uv.y());
  return std::isfinite(px.x()) && std::isfinite(px.y());
}
// local affine map tangent-plane offset (rad) -> pixel at direction d (axes e1, e2), central differences of width h
inline bool xc_affine(const XcCam &cam, const Eigen::Vector3d &d, const Eigen::Vector3d &e1, const Eigen::Vector3d &e2, double h, Eigen::Vector2d &a1,
                      Eigen::Vector2d &a2) {
  Eigen::Vector2d p, m;
  if (!xc_proj(cam, d + h * e1, 0.0872, p) || !xc_proj(cam, d - h * e1, 0.0872, m))
    return false;
  a1 = (p - m) / (2 * h);
  if (!xc_proj(cam, d + h * e2, 0.0872, p) || !xc_proj(cam, d - h * e2, 0.0872, m))
    return false;
  a2 = (p - m) / (2 * h);
  return true;
}
// OV_XCAM_PROJ_TEST = n: n random directions within 85 deg of the axis per camera, inline projection against the camera class
std::string xc_proj_selftest(ov_core::CamBase *cam, size_t cam_id, long n) {
  const XcCam C(cam, true, false);
  uint64_t st = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(cam_id + 1) * 0xD1B54A32D192ED03ULL;
  auto rnd = [&]() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return (double)(st >> 11) / 9007199254740992.0;
  };
  long differ = 0, done = 0;
  double maxd = 0, ns_fast = 0, ns_class = 0;
  std::vector<Eigen::Vector2f> in;
  in.reserve((size_t)n);
  for (long i = 0; i < n; i++) {
    const double th = std::acos(1.0 - rnd() * (1.0 - std::cos(85.0 * M_PI / 180.0))), ph = 2.0 * M_PI * rnd(); // uniform on the cap
    const Eigen::Vector3d d(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
    in.emplace_back((float)(d.x() / d.z()), (float)(d.y() / d.z()));
  }
  if (!C.fast)
    return "cam" + std::to_string(cam_id) + ":not_a_KB_camera";
  float acc = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (const Eigen::Vector2f &u : in)
    acc += C.distort(u.x(), u.y()).x();
  auto t1 = std::chrono::steady_clock::now();
  for (const Eigen::Vector2f &u : in)
    acc += cam->distort_f(u).x();
  auto t2 = std::chrono::steady_clock::now();
  ns_fast = std::chrono::duration<double, std::nano>(t1 - t0).count() / (double)std::max(1L, n);
  ns_class = std::chrono::duration<double, std::nano>(t2 - t1).count() / (double)std::max(1L, n);
  for (const Eigen::Vector2f &u : in) {
    const Eigen::Vector2f a = C.distort(u.x(), u.y()), b = cam->distort_f(u);
    done++;
    if (std::memcmp(a.data(), b.data(), 2 * sizeof(float)) != 0) {
      differ++;
      maxd = std::max(maxd, (double)std::max(std::fabs(a.x() - b.x()), std::fabs(a.y() - b.y())));
    }
  }
  char buf[256];
  std::snprintf(buf, sizeof(buf), " cam%zu:n=%ld,differ=%ld,maxdiff_px=%.3g,ns_inline=%.1f,ns_class=%.1f,lens_terms=%d%s", cam_id, done, differ, maxd, ns_fast, ns_class, (int)C.nr, acc == 12345.f ? "!" : "");
  return std::string(buf);
}
// three image rows, every 4th pixel: all equal -> the image carries no picture (zero page, saturated frame)
inline bool xc_constant(const cv::Mat &im) {
  if (im.rows < 8 || im.cols < 8)
    return true;
  uchar lo = 255, hi = 0;
  for (int q = 1; q <= 3; q++) {
    const uchar *r = im.ptr<uchar>(im.rows * q / 4);
    for (int x = 0; x < im.cols; x += 4) {
      lo = std::min(lo, r[x]);
      hi = std::max(hi, r[x]);
    }
  }
  return hi == lo;
}
// sample offsets of a PS x PS patch: sample (s, t) at centre + s a1 + t a2, spacing step
inline void xc_grid(const Eigen::Vector2d &a1, const Eigen::Vector2d &a2, int PS, double step, XcGrid &g) {
  g.n = PS * PS;
  g.ox.resize(g.n);
  g.oy.resize(g.n);
  const double hh = step * (PS - 1) / 2.0;
  float ext = 0;
  int idx = 0;
  for (int a = 0; a < PS; a++) {
    const double s = -hh + a * step;
    for (int b = 0; b < PS; b++, idx++) {
      const double t = -hh + b * step;
      g.ox[idx] = (float)(s * a1.x() + t * a2.x());
      g.oy[idx] = (float)(s * a1.y() + t * a2.y());
      ext = std::max(ext, std::max(std::fabs(g.ox[idx]), std::fabs(g.oy[idx])));
    }
  }
  g.ext = ext;
}
// bilinear patch at (cx, cy); zero mean, unit norm. false: outside the image or no texture (sd below 2 grey levels)
inline bool xc_sample(const cv::Mat &img, double cxd, double cyd, const XcGrid &g, std::vector<float> &outv) {
  const float cx = (float)cxd, cy = (float)cyd;
  if (!(cx - g.ext >= 1.0f && cy - g.ext >= 1.0f && cx + g.ext < (float)img.cols - 2.0f && cy + g.ext < (float)img.rows - 2.0f))
    return false;
  outv.resize(g.n);
  float *out = outv.data();
  const uchar *base = img.data;
  const size_t st = img.step;
  const float *ox = g.ox.data(), *oy = g.oy.data();
  float sum = 0;
  for (int i = 0; i < g.n; i++) {
    const float x = cx + ox[i], y = cy + oy[i];
    const int xi = (int)x, yi = (int)y;
    const float ax = x - (float)xi, ay = y - (float)yi;
    const uchar *p = base + (size_t)yi * st + (size_t)xi;
    const float v = (1.0f - ay) * ((1.0f - ax) * (float)p[0] + ax * (float)p[1]) + ay * ((1.0f - ax) * (float)p[st] + ax * (float)p[st + 1]);
    out[i] = v;
    sum += v;
  }
  const float mean = sum / (float)g.n;
  float nn = 0;
  for (int i = 0; i < g.n; i++) {
    out[i] -= mean;
    nn += out[i] * out[i];
  }
  if (nn < 4.0f * (float)g.n)
    return false;
  const float inv = 1.0f / std::sqrt(nn);
  for (int i = 0; i < g.n; i++)
    out[i] *= inv;
  return true;
}
inline double xc_dot(const std::vector<float> &a, const std::vector<float> &b) {
  float s = 0;
  const size_t n = a.size();
  for (size_t i = 0; i < n; i++)
    s += a[i] * b[i];
  return (double)s;
}

// One point of camera a (pixel p, normalised coordinates n0) searched in camera b; X_b = R X_a + t, step = 1 / f_a.
// Returns 0 and (q, qn) on success, else: 1 beyond the field limit in a, 2 curve not inside b's field, 3 affine, 4 no
// texture, 5 weak, 6 ambiguous, 7 refinement, 8 final score, 9 too far from the curve, 10 edge/mask, 11 field limit in b.
int xc_search(const XcState &S, const cv::Mat &img_a, const cv::Mat &img_b, const cv::Mat &mask_b, const XcCam &ca, const XcCam &cb,
              const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double step, const Eigen::Vector2d &p, const Eigen::Vector2f &n0, Eigen::Vector2d &q,
              Eigen::Vector2f &qn, XcScratch &W) {
  static const int WEXCL = 6;
  if (!std::isfinite(n0.x()) || !std::isfinite(n0.y()) || (double)n0.norm() > S.tanfov)
    return 1;
  Eigen::Vector3d b0(n0.x(), n0.y(), 1.0);
  b0.normalize();
  Eigen::Vector3d e1 = b0.cross(Eigen::Vector3d(0, 0, 1));
  if (e1.norm() < 1e-6)
    e1 = b0.cross(Eigen::Vector3d(0, 1, 0));
  e1.normalize();
  const Eigen::Vector3d e2 = b0.cross(e1);
  const Eigen::Vector3d Rb = R * b0, e1r = R * e1, e2r = R * e2;
  Eigen::Vector2d pinf, pnear;
  const bool okinf = xc_proj(cb, Rb, S.cosfov, pinf), oknear = xc_proj(cb, Rb + S.RHO_MAX * t, S.cosfov, pnear);
  if (!okinf && !oknear)
    return 2;
  const double h = 4.0 * step;
  Eigen::Vector2d a01, a02;
  if (!xc_affine(ca, b0, e1, e2, h, a01, a02))
    return 3;
  const int PS = S.PS, PSC = (S.csub > 1) ? ((PS - 1) / S.csub + 1) : PS;
  const double stepc = (S.csub > 1) ? step * (double)(PS - 1) / (double)(PSC - 1) : step;
  xc_grid(a01, a02, PS, step, W.G0);
  if (!xc_sample(img_a, p.x(), p.y(), W.G0, W.Tp))
    return 4;
  if (S.csub > 1) {
    xc_grid(a01, a02, PSC, stepc, W.G0c);
    if (!xc_sample(img_a, p.x(), p.y(), W.G0c, W.Tpc))
      return 4;
  }
  int N = 200;
  if (okinf && oknear)
    N = std::max(8, std::min(320, (int)std::ceil((pinf - pnear).norm()) + 2));
  // curve centres: exact projection every 4th sample, linear in between (the curve is smooth: error << 0.05 px)
  W.ctr.assign(N, Eigen::Vector2d(0, 0));
  W.valid.assign(N, 0);
  W.score.assign(N, -2.0f);
  auto dir = [&](int i) -> Eigen::Vector3d { return Rb + (S.RHO_MAX * (double)i / (double)(N - 1)) * t; };
  auto exact = [&](int i) {
    Eigen::Vector2d c;
    if (xc_proj(cb, dir(i), S.cosfov, c)) {
      W.ctr[i] = c;
      W.valid[i] = 1;
    }
  };
  for (int i = 0; i < N; i += 4)
    exact(i);
  if ((N - 1) % 4 != 0)
    exact(N - 1);
  for (int i0 = 0; i0 < N - 1; i0 += 4) {
    const int i1 = std::min(i0 + 4, N - 1);
    if (W.valid[i0] && W.valid[i1]) {
      for (int i = i0 + 1; i < i1; i++) {
        const double w = (double)(i - i0) / (double)(i1 - i0);
        W.ctr[i] = (1.0 - w) * W.ctr[i0] + w * W.ctr[i1];
        W.valid[i] = 1;
      }
    } else {
      for (int i = i0 + 1; i < i1; i++)
        exact(i);
    }
  }
  for (int i = 0; i < N; i++) {
    if (!W.valid[i])
      continue;
    const Eigen::Vector2d &c = W.ctr[i];
    if (c.x() < 8 || c.y() < 8 || c.x() >= img_b.cols - 8 || c.y() >= img_b.rows - 8)
      W.valid[i] = 0;
    else if (!mask_b.empty() && mask_b.at<uint8_t>((int)c.y(), (int)c.x()) > 127)
      W.valid[i] = 0;
  }
  // scores along the curve; the affine map is refreshed every 8th sample (plane through the point, perpendicular to the ray of a)
  Eigen::Vector2d a11, a12;
  bool have_aff = false;
  const std::vector<float> &Tsearch = (S.csub > 1) ? W.Tpc : W.Tp;
  XcGrid &Gsearch = (S.csub > 1) ? W.G1c : W.G1;
  int ibest = -1;
  double best = -2.0;
  for (int i = 0; i < N; i++) {
    if (!W.valid[i]) {
      have_aff = false;
      continue;
    }
    if (!have_aff || (i % 8) == 0) {
      have_aff = xc_affine(cb, dir(i), e1r, e2r, h, a11, a12);
      if (!have_aff)
        continue;
      xc_grid(a11, a12, PSC, stepc, Gsearch);
    }
    if (!xc_sample(img_b, W.ctr[i].x(), W.ctr[i].y(), Gsearch, W.P))
      continue;
    W.score[i] = (float)xc_dot(Tsearch, W.P);
    if (W.score[i] > best) {
      best = W.score[i];
      ibest = i;
    }
  }
  if (S.csub > 1) {
    // coarse scores nominate up to 4 separated maxima; each is re-scored with the full patch at i-1, i, i+1
    if (ibest < 0 || best < S.ZMIN - 0.25)
      return 5;
    int nom[4], nnom = 0;
    for (int r = 0; r < 4; r++) {
      int im = -1;
      float sm = (float)(S.ZMIN - 0.25);
      for (int i = 0; i < N; i++) {
        if (W.score[i] <= sm)
          continue;
        bool near = false;
        for (int u = 0; u < nnom; u++)
          if (std::abs(i - nom[u]) <= WEXCL)
            near = true;
        if (!near) {
          sm = W.score[i];
          im = i;
        }
      }
      if (im < 0)
        break;
      nom[nnom++] = im;
    }
    double fbest = -2.0, fsecond = -2.0;
    int fib = -1;
    for (int u = 0; u < nnom; u++) {
      if (!xc_affine(cb, dir(nom[u]), e1r, e2r, h, a11, a12))
        continue;
      xc_grid(a11, a12, PS, step, W.G1);
      double fu = -2.0;
      int iu = -1;
      for (int i = std::max(0, nom[u] - 1); i <= std::min(N - 1, nom[u] + 1); i++) {
        if (!W.valid[i] || !xc_sample(img_b, W.ctr[i].x(), W.ctr[i].y(), W.G1, W.P))
          continue;
        const double z = xc_dot(W.Tp, W.P);
        if (z > fu) {
          fu = z;
          iu = i;
        }
      }
      if (fu > fbest) {
        fsecond = fbest;
        fbest = fu;
        fib = iu;
      } else if (fu > fsecond) {
        fsecond = fu;
      }
    }
    if (fib < 0 || fbest < S.ZMIN)
      return 5;
    if (fsecond > fbest - S.UNIQ)
      return 6;
    ibest = fib;
    best = fbest;
  } else {
    if (ibest < 0 || best < S.ZMIN)
      return 5;
    double second = -2.0;
    for (int i = 0; i < N; i++)
      if (std::abs(i - ibest) > WEXCL && W.score[i] > second)
        second = W.score[i];
    if (second > best - S.UNIQ)
      return 6;
  }
  // ---- 2-D refinement around the best curve sample (Gauss-Newton on the normalised patches, free in both directions)
  if (!xc_affine(cb, dir(ibest), e1r, e2r, h, a11, a12))
    return 7;
  xc_grid(a11, a12, PS, step, W.G1);
  const Eigen::Vector2d cbest = W.ctr[ibest];
  Eigen::Vector2d d(0, 0);
  for (int it = 0; it < 10; it++) {
    const Eigen::Vector2d qq = cbest + d;
    if (!xc_sample(img_b, qq.x(), qq.y(), W.G1, W.P) || !xc_sample(img_b, qq.x() + 0.5, qq.y(), W.G1, W.Pxp) || !xc_sample(img_b, qq.x() - 0.5, qq.y(), W.G1, W.Pxm) ||
        !xc_sample(img_b, qq.x(), qq.y() + 0.5, W.G1, W.Pyp) || !xc_sample(img_b, qq.x(), qq.y() - 0.5, W.G1, W.Pym))
      return 7;
    double hxx = 0, hxy = 0, hyy = 0, gx = 0, gy = 0;
    const size_t np = W.P.size();
    for (size_t j = 0; j < np; j++) {
      const double jx = (double)W.Pxp[j] - (double)W.Pxm[j], jy = (double)W.Pyp[j] - (double)W.Pym[j], r = (double)W.Tp[j] - (double)W.P[j];
      hxx += jx * jx;
      hxy += jx * jy;
      hyy += jy * jy;
      gx += jx * r;
      gy += jy * r;
    }
    const double det = hxx * hyy - hxy * hxy;
    if (det < 1e-9 * (hxx + hyy) * (hxx + hyy) || det <= 0) // texture in one direction only
      return 7;
    const Eigen::Vector2d dd((hyy * gx - hxy * gy) / det, (hxx * gy - hxy * gx) / det);
    d += dd;
    if (d.norm() > 4.0)
      return 7;
    if (dd.norm() < 0.01)
      break;
  }
  const Eigen::Vector2d qf = cbest + d;
  if (!xc_sample(img_b, qf.x(), qf.y(), W.G1, W.P))
    return 7;
  const double zf = xc_dot(W.Tp, W.P);
  if (zf < S.ZFIN || zf < best - 0.02)
    return 8;
  // distance from the predicted curve (signed, along the curve normal)
  int ia = ibest, ib = ibest;
  if (ibest > 0 && W.valid[ibest - 1])
    ia = ibest - 1;
  if (ibest < N - 1 && W.valid[ibest + 1])
    ib = ibest + 1;
  Eigen::Vector2d tau = W.ctr[ib] - W.ctr[ia];
  if (ia == ib || tau.norm() < 1e-6)
    tau = (okinf && oknear) ? Eigen::Vector2d(pnear - pinf) : Eigen::Vector2d(1, 0);
  if (tau.norm() < 1e-9)
    tau = Eigen::Vector2d(1, 0);
  tau.normalize();
  const double off = -tau.y() * d.x() + tau.x() * d.y();
  if (std::abs(off) > S.MAXOFF)
    return 9;
  if (qf.x() < 10 || qf.y() < 10 || qf.x() >= img_b.cols - 10 || qf.y() >= img_b.rows - 10)
    return 10;
  if (!mask_b.empty() && mask_b.at<uint8_t>((int)qf.y(), (int)qf.x()) > 127)
    return 10;
  qn = cb.cam->undistort_f(Eigen::Vector2f((float)qf.x(), (float)qf.y()));
  if (!std::isfinite(qn.x()) || !std::isfinite(qn.y()) || (double)qn.norm() > S.tanfov)
    return 11;
  q = qf;
  return 0;
}

// Re-registration of an existing pair: the patch of camera a at p (normalised n0) is looked for in camera b by the same free 2-D
// refinement as in xc_search, started at the tracked position q0 (normalised qn0). 0 and (q, qn) on success.
int xc_refine(const XcState &S, const cv::Mat &img_a, const cv::Mat &img_b, const cv::Mat &mask_b, const XcCam &ca, const XcCam &cb, const Eigen::Matrix3d &R,
              double step, const Eigen::Vector2d &p, const Eigen::Vector2f &n0, const Eigen::Vector2d &q0, const Eigen::Vector2f &qn0, Eigen::Vector2d &q, Eigen::Vector2f &qn,
              XcScratch &W) {
  if (!std::isfinite(n0.x()) || !std::isfinite(n0.y()) || (double)n0.norm() > S.tanfov || !std::isfinite(qn0.x()) || !std::isfinite(qn0.y()) || (double)qn0.norm() > S.tanfov)
    return 1;
  Eigen::Vector3d b0(n0.x(), n0.y(), 1.0);
  b0.normalize();
  Eigen::Vector3d e1 = b0.cross(Eigen::Vector3d(0, 0, 1));
  if (e1.norm() < 1e-6)
    e1 = b0.cross(Eigen::Vector3d(0, 1, 0));
  e1.normalize();
  const Eigen::Vector3d e2 = b0.cross(e1);
  const Eigen::Vector3d e1r = R * e1, e2r = R * e2;
  Eigen::Vector3d Dq(qn0.x(), qn0.y(), 1.0);
  Dq.normalize();
  const double h = 4.0 * step;
  Eigen::Vector2d a01, a02, a11, a12;
  if (!xc_affine(ca, b0, e1, e2, h, a01, a02) || !xc_affine(cb, Dq, e1r, e2r, h, a11, a12))
    return 3;
  xc_grid(a01, a02, S.PS, step, W.G0);
  if (!xc_sample(img_a, p.x(), p.y(), W.G0, W.Tp))
    return 4;
  xc_grid(a11, a12, S.PS, step, W.G1);
  Eigen::Vector2d d(0, 0);
  for (int it = 0; it < 10; it++) {
    const Eigen::Vector2d qq = q0 + d;
    if (!xc_sample(img_b, qq.x(), qq.y(), W.G1, W.P) || !xc_sample(img_b, qq.x() + 0.5, qq.y(), W.G1, W.Pxp) || !xc_sample(img_b, qq.x() - 0.5, qq.y(), W.G1, W.Pxm) ||
        !xc_sample(img_b, qq.x(), qq.y() + 0.5, W.G1, W.Pyp) || !xc_sample(img_b, qq.x(), qq.y() - 0.5, W.G1, W.Pym))
      return 7;
    double hxx = 0, hxy = 0, hyy = 0, gx = 0, gy = 0;
    const size_t np = W.P.size();
    for (size_t j = 0; j < np; j++) {
      const double jx = (double)W.Pxp[j] - (double)W.Pxm[j], jy = (double)W.Pyp[j] - (double)W.Pym[j], r = (double)W.Tp[j] - (double)W.P[j];
      hxx += jx * jx;
      hxy += jx * jy;
      hyy += jy * jy;
      gx += jx * r;
      gy += jy * r;
    }
    const double det = hxx * hyy - hxy * hxy;
    if (det < 1e-9 * (hxx + hyy) * (hxx + hyy) || det <= 0)
      return 7;
    const Eigen::Vector2d dd((hyy * gx - hxy * gy) / det, (hxx * gy - hxy * gx) / det);
    d += dd;
    if (d.norm() > 4.0)
      return 7;
    if (dd.norm() < 0.01)
      break;
  }
  const Eigen::Vector2d qf = q0 + d;
  if (!xc_sample(img_b, qf.x(), qf.y(), W.G1, W.P))
    return 7;
  if (xc_dot(W.Tp, W.P) < S.ZFIN)
    return 8;
  if (qf.x() < 10 || qf.y() < 10 || qf.x() >= img_b.cols - 10 || qf.y() >= img_b.rows - 10)
    return 10;
  if (!mask_b.empty() && mask_b.at<uint8_t>((int)qf.y(), (int)qf.x()) > 127)
    return 10;
  qn = cb.cam->undistort_f(Eigen::Vector2f((float)qf.x(), (float)qf.y()));
  if (!std::isfinite(qn.x()) || !std::isfinite(qn.y()) || (double)qn.norm() > S.tanfov)
    return 11;
  q = qf;
  return 0;
}

// Statistic only (never used for a measurement): direction of a raw pixel from the plain Kannala-Brandt values of the
// camera object, valid over the whole lens (the class's own undistortion saturates near 85 deg).
inline bool xc_wide_dir(ov_core::CamBase *cam, double u, double v, Eigen::Vector3d &dirv, double &theta) {
  const Eigen::MatrixXd cv_ = cam->get_value();
  const double x = (u - cv_(2)) / cv_(0), y = (v - cv_(3)) / cv_(1), rd = std::sqrt(x * x + y * y);
  double th = rd;
  for (int i = 0; i < 12; i++) {
    const double t2 = th * th;
    const double f = th * (1 + t2 * (cv_(4) + t2 * (cv_(5) + t2 * (cv_(6) + t2 * cv_(7))))) - rd;
    const double df = 1 + t2 * (3 * cv_(4) + t2 * (5 * cv_(5) + t2 * (7 * cv_(6) + t2 * 9 * cv_(7))));
    if (std::abs(df) < 1e-9)
      return false;
    th -= f / df;
  }
  if (!std::isfinite(th) || th < 0 || th > 2.2)
    return false;
  theta = th;
  const double s = std::sin(th), inv = rd > 1e-12 ? 1.0 / rd : 0.0;
  dirv = Eigen::Vector3d(s * x * inv, s * y * inv, std::cos(th));
  return true;
}
} // namespace

// The settings the estimator asks for (TrackBase::xcam_settings()): the values the association itself uses, the preset included. With
// OV_XCAM=1 the state exists since the library was loaded (xc_boot_at_load at the end of this file); without it nothing is created.
TrackBase::XcamSettings TrackKLT::xcam_settings() const {
  XcamSettings s;
  if (!xcam_enabled() || use_stereo)
    return s;
  const XcState &S = xc();
  s.on = S.on;
  s.maint = S.maint;
  s.snap = S.snap;
  return s;
}

void TrackKLT::xcam_associate(const CameraData &message, const std::vector<size_t> &pend_base, size_t pend_base_shared) {
  XcState &S = xc();
  const auto t0 = std::chrono::steady_clock::now();
  const double ts = message.timestamp;

  // cameras of this message that can take part
  std::vector<size_t> cams;
  size_t msg_of[XC_NCAM];
  for (size_t m = 0; m < message.sensor_ids.size(); m++) {
    const size_t c = message.sensor_ids[m];
    if (c >= (size_t)XC_NCAM || message.images[m].empty() || message.images[m].type() != CV_8UC1)
      continue;
    if (!xcam_R_ItoC.count(c) || !xcam_p_IinC.count(c)) {
      if (!S.warned_ext) {
        S.warned_ext = true;
        std::fprintf(stderr, "[xcam-warn] OV_XCAM=1 but the tracker has no extrinsics for camera %zu: no association\n", c);
      }
      continue;
    }
    cams.push_back(c);
    msg_of[c] = m;
  }
  if (cams.size() < 2)
    return;
  std::sort(cams.begin(), cams.end());
  const int fs = (int)(++S.fs);
  static const bool defer_env = [] {
    const char *e = std::getenv("OV_ASYNC_UPDATE");
    return e && *e == '1';
  }();
  const bool deferred = defer_env || defer_runtime.load(std::memory_order_relaxed);
  // the pending buffers are only touched by this thread between the feeds and flush_pending(); hold the lock anyway
  std::unique_lock<std::mutex> lkp(pending_mtx, std::defer_lock);
  if (deferred)
    lkp.lock();
  std::lock_guard<std::mutex> lckv(mtx_last_vars);

  // ---- 1. carry the per-point bookkeeping over to the new lists (the lists keep their order: survivors, then new points)
  struct XcDead {
    size_t c, id;
    XcInfo I;
  };
  std::vector<XcDead> dead;
  for (size_t c : cams) {
    const std::vector<size_t> &ids = ids_last[c];
    const std::vector<size_t> &old = S.ids[c];
    const std::vector<XcInfo> &oi = S.info[c];
    std::vector<XcInfo> ni(ids.size());
    std::vector<char> hit(old.size(), 0);
    std::unordered_map<size_t, size_t> rest;
    size_t o = 0;
    bool tail = false;
    for (size_t k = 0; k < ids.size(); k++) {
      if (!tail) {
        size_t o2 = o;
        while (o2 < old.size() && old[o2] != ids[k])
          o2++;
        if (o2 < old.size()) {
          ni[k] = oi[o2];
          hit[o2] = 1;
          o = o2 + 1;
          continue;
        }
        tail = true;
        for (size_t r = 0; r < old.size(); r++)
          if (!hit[r])
            rest.emplace(old[r], r);
      }
      auto it = rest.find(ids[k]);
      if (it != rest.end() && !hit[it->second]) {
        ni[k] = oi[it->second];
        hit[it->second] = 1;
      } else {
        ni[k].born = fs; // new in this camera
      }
    }
    for (size_t r = 0; r < old.size(); r++)
      if (!hit[r]) {
        XcState::hist_add(S.pt_life_hist[oi[r].sec < 3 ? oi[r].sec : 0], fs - oi[r].born);
        if (oi[r].shared())
          dead.push_back({c, old[r], oi[r]});
      }
    S.info[c].swap(ni);
  }
  // an association ends when the partner camera has lost the point
  {
    std::unordered_set<uint64_t> claim[XC_NCAM];
    for (size_t c : cams)
      for (size_t k = 0; k < S.info[c].size(); k++)
        for (int d = 0; d < XC_NCAM; d++)
          if (S.info[c][k].assoc[d] >= 0)
            claim[c].insert((uint64_t)ids_last[c][k] * XC_NCAM + (uint64_t)d);
    for (const XcDead &D : dead)
      for (size_t d : cams) {
        if (d == D.c || D.I.assoc[d] < 0)
          continue;
        const bool partner_alive = claim[d].count((uint64_t)D.id * XC_NCAM + (uint64_t)D.c) > 0;
        if (!partner_alive && !(D.c < d))
          continue; // both sides lost in this frame-set: counted once
        S.life_add(fs - D.I.assoc[d]);
        // by XcInfo::sec, NOT by recv[d] as the maintenance decides it: left on purpose, recv[d] would alter end_receiver_first / end_origin_first in the [xcam-total] line of the flight build
        S.end_cause[!partner_alive ? 0 : (D.I.sec ? 1 : 2)]++;
      }
    for (size_t c : cams)
      for (size_t k = 0; k < S.info[c].size(); k++)
        for (size_t d : cams) {
          XcInfo &I = S.info[c][k];
          if (d == c || I.assoc[d] < 0)
            continue;
          if (!claim[d].count((uint64_t)ids_last[c][k] * XC_NCAM + (uint64_t)c)) {
            I.assoc[d] = -1;
            I.recv[d] = false;
          }
        }
  }

  // ---- 2. camera pairs
  std::vector<std::pair<size_t, size_t>> pairs;
  auto present = [&](size_t c) { return std::find(cams.begin(), cams.end(), c) != cams.end(); };
  if (!S.pairs_env.empty()) {
    for (auto &pr : S.pairs_env)
      if (present(pr.first) && present(pr.second))
        pairs.push_back(pr);
  } else {
    const double cmin = std::cos(S.PAIR_DEG * M_PI / 180.0);
    for (size_t x = 0; x < cams.size(); x++)
      for (size_t y = x + 1; y < cams.size(); y++) {
        const size_t a = cams[x], b = cams[y];
        // optical axis of camera c in the IMU frame = third row of R_ItoC
        const double cang = xcam_R_ItoC.at(a).row(2).dot(xcam_R_ItoC.at(b).row(2));
        if (cang > cmin) {
          if (cams.size() > 2 && x == 0 && y == cams.size() - 1)
            pairs.emplace_back(b, a); // ring closure, written as on the rig: 3,0
          else
            pairs.emplace_back(a, b);
        }
      }
    std::sort(pairs.begin(), pairs.end());
  }
  if (pairs.empty()) {
    for (size_t c : cams)
      S.ids[c] = ids_last[c];
    return;
  }

  // ---- 2a. Is the step due on this frame-set (OV_XCAM_EVERY), and are there host pixels to work on?
  // Not due or no pixels: no search and no re-registration; NOTHING is marked as tried, the shared points keep being tracked by
  // their cameras and the bookkeeping above and below runs as on every frame-set, so candidates are only delayed.
  const cv::Mat *himg[XC_NCAM];
  cv::Mat hold[XC_NCAM];
  for (int c = 0; c < XC_NCAM; c++)
    himg[c] = nullptr;
  bool any_ph = false, all_rb = true;
  for (size_t c : cams) {
    const cv::Mat &im = message.images.at(msg_of[c]);
    himg[c] = &im;
    if (S.emu_ph || (ov_core::ov_nvjpg_enabled() && ov_core::ov_nvjpg_is_placeholder(im.data))) { // header over the shared page: no pixels
      any_ph = true;
      if (xc_host_get(c, ts, hold[c]))
        himg[c] = &hold[c];
      else
        all_rb = false;
    }
  }
  int unusable = 0; // 1: GPU-decoded frame-set without host pixels, 2: an image without a picture
  size_t bad_cam = 0;
  bool due = (fs - S.last_run >= S.every);
  if (any_ph) {
    if (all_rb)
      due = true; // the cadence was applied when the pixels were asked for (see the request at the end)
    else if (S.readback)
      due = false; // waiting for the pixels that were asked for
    else
      unusable = 1;
  }
  if (due && unusable == 0) {
    const int eb = std::abs(S.emu_blank);
    if (eb > 0 && fs >= eb && (S.emu_blank_end <= 0 || fs < S.emu_blank_end)) { // TEST ONLY
      if (S.zero_page.empty())
        S.zero_page = cv::Mat::zeros(himg[cams[0]]->rows, himg[cams[0]]->cols, CV_8UC1);
      for (size_t c : cams)
        himg[c] = &S.zero_page;
    }
    if (S.emu_blank >= 0)
      for (size_t c : cams)
        if (xc_constant(*himg[c])) {
          unusable = 2;
          bad_cam = c;
        }
  }
  const bool run = due && unusable == 0;
  if (!run) {
    if (due) {
      S.t_unusable++;
      if (unusable == 1 && !S.warned_ph) {
        S.warned_ph = true;
        std::fprintf(stderr, "[xcam-warn] ts=%.6f: the frames are decoded on the GPU (OV_NVJPG=1) and the message carries no host pixels: the association is skipped "
                             "on such frame-sets and nothing is marked as tried (OV_XCAM_READBACK=1 brings the pixels back when the step is due)\n", ts);
      }
      if (unusable == 2 && !S.warned_const) {
        S.warned_const = true;
        std::fprintf(stderr, "[xcam-warn] ts=%.6f: the image of camera %zu is constant (no picture): the association is skipped on such frame-sets and nothing is "
                             "marked as tried\n", ts, bad_cam);
      }
    } else {
      S.t_notdue++;
    }
  }
  std::vector<XcCam> CC(XC_NCAM);
  if (run) {
    for (size_t c : cams)
      CC[c] = XcCam(calib_for(c), S.fastproj, S.proj_verify);
    if (S.proj_test > 0 && !S.proj_tested) {
      S.proj_tested = true;
      std::string line;
      for (size_t c : cams)
        line += xc_proj_selftest(calib_for(c), c, S.proj_test);
      std::fprintf(stderr, "[xcam-proj]%s\n", line.c_str());
    }
  }

  // where each camera's observations of this timestamp start in the pending buffers (deferred writes only)
  long pstart[XC_NCAM];
  std::vector<PendingObs> *pvec[XC_NCAM];
  for (int c = 0; c < XC_NCAM; c++) {
    pstart[c] = -1;
    pvec[c] = nullptr;
  }
  if (deferred) {
    for (size_t c : cams) {
      if (det_ids_on() && c < PEND_NCAM) {
        pvec[c] = &pending_by_cam[c];
        const size_t base = c < pend_base.size() ? pend_base[c] : 0;
        if (pvec[c]->size() >= base && pvec[c]->size() - base == ids_last[c].size() && !ids_last[c].empty())
          pstart[c] = (long)base;
      } else {
        pvec[c] = &pending_obs;
        for (size_t i = pend_base_shared; i < pending_obs.size(); i++)
          if (pending_obs[i].cam == c) {
            pstart[c] = (long)i;
            break;
          }
      }
    }
  }
  // pending observation of list entry (c, k) at this timestamp, or -1
  auto pend_idx = [&](size_t c, size_t k) -> long {
    if (pstart[c] < 0)
      return -1;
    const size_t i = (size_t)pstart[c] + k;
    if (i >= pvec[c]->size())
      return -1;
    const PendingObs &e = (*pvec[c])[i];
    if (e.cam != c || e.id != ids_last[c][k] || e.ts != ts)
      return -1;
    return (long)i;
  };

  // ---- 2b. maintenance (OV_XCAM_MAINT = K > 0): the two cameras track a shared point independently and the tracks drift apart.
  // Every K frame-sets a pair is re-registered: patch of the origin camera at its tracked position, free 2-D refinement in the
  // receiving camera from its tracked position. Shift below TOL: nothing. Up to MAX: the receiving point and its observation of
  // this timestamp are moved there. No fit or a larger shift: the pair is dissolved (the receiving point continues under a new id).
  if (S.maint > 0 && run) {
    XcScratch W;
    for (size_t pi = 0; pi < pairs.size(); pi++) {
      const size_t ci = pairs[pi].first, cj = pairs[pi].second;
      std::unordered_map<size_t, size_t> where;
      for (size_t k = 0; k < S.info[cj].size(); k++)
        if (S.info[cj][k].assoc[ci] >= 0)
          where.emplace(ids_last[cj][k], k);
      for (size_t k = 0; k < S.info[ci].size(); k++) {
        const int as = S.info[ci][k].assoc[cj];
        if (as < 0)
          continue;
        // due when OV_XCAM_MAINT frame-sets have passed since the association or since the last check. With the step run on every
        // frame-set this is the rule (fs - as) % MAINT == 0; with OV_XCAM_EVERY > 1 or skipped frame-sets the check comes later, never earlier.
        if (fs - std::max(as, S.info[ci][k].mchk[cj]) < S.maint)
          continue;
        S.info[ci][k].mchk[cj] = fs;
        auto itw = where.find(ids_last[ci][k]);
        if (itw == where.end())
          continue;
        // origin and receiver come from the association itself (XcInfo::recv, set in link()): the receiver is the point that was
        // given the id of the other one. (XcInfo::sec does not tell: a point that received its id and outlived its origin keeps sec != 0.)
        size_t a = ci, ka = k, b = cj, kb = itw->second;
        if (S.info[a][ka].recv[b])
          std::swap(a, b), std::swap(ka, kb);
        if (S.info[a][ka].recv[b] || !S.info[b][kb].recv[a]) {
          S.m_no_origin++; // both or neither marked as receiver: cannot happen, counted to show it
          continue;
        }
        int nsh = 0;
        for (int d = 0; d < XC_NCAM; d++)
          if (S.info[b][kb].assoc[d] >= 0)
            nsh++;
        if (nsh != 1) {
          S.m_three++;
          continue; // a point shared by three cameras is left alone
        }
        const XcCam &ca = CC[a], &cb = CC[b];
        const Eigen::Matrix3d R = xcam_R_ItoC.at(b) * xcam_R_ItoC.at(a).transpose();
        const Eigen::Vector2d p(pts_last[a][ka].pt.x, pts_last[a][ka].pt.y), q0(pts_last[b][kb].pt.x, pts_last[b][kb].pt.y);
        const long pia = pend_idx(a, ka), pib = pend_idx(b, kb);
        const Eigen::Vector2f n0 = pia >= 0 ? Eigen::Vector2f((*pvec[a])[pia].un, (*pvec[a])[pia].vn) : ca.cam->undistort_f(Eigen::Vector2f(pts_last[a][ka].pt.x, pts_last[a][ka].pt.y));
        const Eigen::Vector2f qn0 = pib >= 0 ? Eigen::Vector2f((*pvec[b])[pib].un, (*pvec[b])[pib].vn) : cb.cam->undistort_f(Eigen::Vector2f(pts_last[b][kb].pt.x, pts_last[b][kb].pt.y));
        Eigen::Vector2d q;
        Eigen::Vector2f qn;
        const int r = xc_refine(S, *himg[a], *himg[b], message.masks.at(msg_of[b]), ca, cb, R, 1.0 / ca.fx, p, n0, q0, qn0, q, qn, W);
        S.m_checked++;
        const double shift = (r == 0) ? (q - q0).norm() : 1e9;
        if (r == 0 && shift <= S.MAINT_TOL) {
          S.m_kept++;
          continue;
        }
        const size_t idb = ids_last[b][kb];
        if (r == 0 && shift <= S.MAINT_MAX) {
          // move the receiving point and its observation of this timestamp
          pts_last[b][kb].pt.x = (float)q.x();
          pts_last[b][kb].pt.y = (float)q.y();
          if (deferred) {
            if (pib >= 0) {
              PendingObs &e = (*pvec[b])[pib];
              e.u = (float)q.x();
              e.v = (float)q.y();
              e.un = qn.x();
              e.vn = qn.y();
            }
          } else {
            std::shared_ptr<Feature> f = database->get_feature(idb, false);
            if (f != nullptr) {
              auto itt = f->timestamps.find(b);
              if (itt != f->timestamps.end() && !itt->second.empty() && itt->second.back() == ts) {
                f->uvs.at(b).back() = Eigen::Vector2f((float)q.x(), (float)q.y());
                f->uvs_norm.at(b).back() = qn;
              }
            }
          }
          S.m_moved++;
          S.m_shift_sum += shift;
          continue;
        }
        // dissolve: the receiving point continues as an ordinary point under a new id (its earlier observations stay with the shared feature)
        const size_t nid = next_id(b);
        if (deferred) {
          if (pib >= 0)
            (*pvec[b])[pib].id = nid;
        } else {
          std::shared_ptr<Feature> f = database->get_feature(idb, false);
          if (f != nullptr) {
            auto itt = f->timestamps.find(b);
            if (itt != f->timestamps.end() && !itt->second.empty() && itt->second.back() == ts) {
              const Eigen::Vector2f uv = f->uvs.at(b).back(), un = f->uvs_norm.at(b).back();
              itt->second.pop_back();
              f->uvs.at(b).pop_back();
              f->uvs_norm.at(b).pop_back();
              database->update_feature(nid, ts, b, uv(0), uv(1), un(0), un(1));
            }
          }
        }
        ids_last[b][kb] = nid;
        XcState::hist_add(S.life_hist, fs - as);
        S.end_cause[1]++;
        S.info[b][kb].assoc[a] = -1;
        S.info[b][kb].recv[a] = false;
        S.info[b][kb].sec = 0;
        S.info[b][kb].next_try[a] = -1;
        S.info[a][ka].assoc[b] = -1;
        S.info[a][ka].recv[b] = false;
        S.info[a][ka].next_try[b] = -1;
        where.erase(idb);
        S.m_dissolved++;
      }
    }
  }

  // ---- 3. search (reads only; one task per pair and direction)
  std::vector<XcTask> tasks;
  for (size_t pi = 0; run && pi < pairs.size(); pi++) {
    XcTask T;
    T.pair = (int)pi;
    T.a = pairs[pi].first;
    T.b = pairs[pi].second;
    tasks.push_back(T);
    T.a = pairs[pi].second;
    T.b = pairs[pi].first;
    tasks.push_back(T);
  }
  for (XcTask &T : tasks)
    T.quota = S.cap;
  if (S.budget > 0 && !tasks.empty()) {
    // OV_XCAM_BUDGET: one limit for the whole frame-set, shared out before the tasks run (deterministic): each task's share
    // is at most its number of points that may be tried now and at most OV_XCAM_MAX; the shares grow one by one, starting with
    // another task on every frame-set. What does not fit stays untried and is searched on a later frame-set.
    std::vector<int> elig(tasks.size(), 0);
    for (size_t ti = 0; ti < tasks.size(); ti++) {
      const size_t b = tasks[ti].b;
      for (const XcInfo &I : S.info[tasks[ti].a])
        if (!(I.assoc[b] >= 0 || I.next_try[b] < 0 || I.next_try[b] > fs || fs - I.born < S.minage))
          elig[ti]++;
      tasks[ti].quota = 0;
    }
    int left = S.budget;
    bool any = true, cut = false;
    while (left > 0 && any) {
      any = false;
      for (size_t j = 0; j < tasks.size() && left > 0; j++) {
        const size_t ti = (j + (size_t)fs) % tasks.size();
        if (tasks[ti].quota < std::min(elig[ti], S.cap)) {
          tasks[ti].quota++;
          left--;
          any = true;
        }
      }
    }
    for (size_t ti = 0; ti < tasks.size(); ti++)
      if (tasks[ti].quota < std::min(elig[ti], S.cap))
        cut = true;
    if (cut)
      S.t_budget_cut++;
  }
  auto run_task = [&](XcTask &T) {
    const size_t a = T.a, b = T.b;
    const XcCam &ca = CC[a], &cb = CC[b];
    // relative pose from the 3x3 rotation blocks: X_b = R X_a + t
    const Eigen::Matrix3d R = xcam_R_ItoC.at(b) * xcam_R_ItoC.at(a).transpose();
    const Eigen::Vector3d t = xcam_p_IinC.at(b) - R * xcam_p_IinC.at(a);
    const Eigen::Matrix3d Rinv = R.transpose();
    const Eigen::Vector3d tinv = -(Rinv * t);
    const cv::Mat &img_a = *himg[a], &img_b = *himg[b];
    const cv::Mat &mask_a = message.masks.at(msg_of[a]), &mask_b = message.masks.at(msg_of[b]);
    const double step_a = 1.0 / ca.fx, step_b = 1.0 / cb.fx;
    const std::vector<cv::KeyPoint> &pts = pts_last.at(a);
    const std::vector<XcInfo> &inf = S.info[a];
    XcScratch W;
    for (int k = (int)pts.size() - 1; k >= 0 && T.cand < T.quota; k--) {
      const XcInfo &I = inf[k];
      if (I.assoc[b] >= 0 || I.next_try[b] < 0 || I.next_try[b] > fs || fs - I.born < S.minage)
        continue;
      T.tried.push_back(k);
      const Eigen::Vector2d p(pts[k].pt.x, pts[k].pt.y);
      Eigen::Vector2f n0;
      const long pix = pend_idx(a, (size_t)k);
      if (pix >= 0)
        n0 = Eigen::Vector2f((*pvec[a])[pix].un, (*pvec[a])[pix].vn); // already undistorted by the feed, same camera object
      else
        n0 = ca.cam->undistort_f(Eigen::Vector2f(pts[k].pt.x, pts[k].pt.y));
      Eigen::Vector2d q;
      Eigen::Vector2f qn;
      const int r = xc_search(S, img_a, img_b, mask_b, ca, cb, R, t, step_a, p, n0, q, qn, W);
      if (r == 1 || r == 2) { // outside the common field (limit FOV_DEG in both cameras): costs no search, not counted against the cap
        T.rej[r]++;
        if (S.stats > 0) { // would the point be in both lenses' fields (1.83 rad) without the limit?
          Eigen::Vector3d dw;
          double th = 0;
          if (xc_wide_dir(ca.cam, p.x(), p.y(), dw, th) && th < 1.83) {
            const Eigen::Vector3d d0 = R * dw, d1 = R * dw + S.RHO_MAX * t;
            const double c183 = std::cos(1.83);
            if (d0.z() > c183 * d0.norm() || d1.z() > c183 * d1.norm())
              T.fov_excl++;
          }
        }
        continue;
      }
      T.cand++;
      if (r != 0) {
        T.rej[r]++;
        continue;
      }
      if (S.mutual) {
        Eigen::Vector2d pb;
        Eigen::Vector2f pbn;
        const int rr = xc_search(S, img_b, img_a, mask_a, cb, ca, Rinv, tinv, step_b, q, qn, pb, pbn, W);
        if (rr != 0 || (pb - p).norm() > S.MUT_PX) {
          T.mutual_fail++;
          continue;
        }
      }
      T.matches.push_back({k, (float)p.x(), (float)p.y(), (float)q.x(), (float)q.y(), qn.x(), qn.y()});
    }
  };
  const auto tsearch0 = std::chrono::steady_clock::now();
  if (S.par && tasks.size() > 1) {
    parallel_for_(cv::Range(0, (int)tasks.size()), LambdaBody([&](const cv::Range &range) {
                    for (int i = range.start; i < range.end; i++)
                      run_task(tasks[i]);
                  }));
  } else {
    for (XcTask &T : tasks)
      run_task(T);
  }
  const double search_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tsearch0).count();
  S.search_ms_sum += search_ms;
  // Last line of defence: when the own patch of EVERY candidate of a task had no texture, that camera's image is
  // not the picture its tracker works on. Nothing that involves the camera is marked as tried, so the points are searched again.
  bool blank_cam[XC_NCAM];
  for (int c = 0; c < XC_NCAM; c++)
    blank_cam[c] = false;
  for (const XcTask &T : tasks)
    if (T.cand >= 4 && T.rej[4] == (long)T.cand)
      blank_cam[T.a] = true;

  // ---- 4. commit: sequential, fixed order (pair, direction, position in the list)
  struct PairStat {
    int cand = 0, matched = 0, mfail = 0, relabel = 0, inject = 0, skip = 0;
  };
  std::vector<PairStat> pst(pairs.size());
  auto move_obs = [&](size_t c, size_t k, size_t from, size_t to) { // the observation of (c, k) at this timestamp changes its id
    if (deferred) {
      long i = pend_idx(c, k);
      if (i < 0 && pvec[c] != nullptr) { // not where expected: look for it
        for (long j = (long)pvec[c]->size() - 1; j >= 0; j--) {
          const PendingObs &e = (*pvec[c])[j];
          if (e.ts != ts)
            break;
          if (e.cam == c && e.id == from) {
            i = j;
            break;
          }
        }
      }
      if (i >= 0)
        (*pvec[c])[i].id = to;
    } else {
      // writes are immediate here: nothing else touches the database between the feeds and the update (sequential path)
      std::shared_ptr<Feature> f = database->get_feature(from, false);
      if (f != nullptr) {
        auto itt = f->timestamps.find(c);
        if (itt != f->timestamps.end() && !itt->second.empty() && itt->second.back() == ts) {
          const Eigen::Vector2f uv = f->uvs.at(c).back(), un = f->uvs_norm.at(c).back();
          itt->second.pop_back();
          f->uvs.at(c).pop_back();
          f->uvs_norm.at(c).pop_back();
          bool any = false;
          for (const auto &kv : f->timestamps)
            if (!kv.second.empty())
              any = true;
          if (!any)
            database->get_feature(from, true); // it had this one observation only
          database->update_feature(to, ts, c, uv(0), uv(1), un(0), un(1));
        }
      }
    }
  };
  auto write_obs = [&](size_t id, size_t c, float u, float v, float un, float vn) {
    if (deferred) {
      std::vector<PendingObs> &dst = (det_ids_on() && c < PEND_NCAM) ? pending_by_cam[c] : pending_obs;
      dst.push_back({id, ts, c, u, v, un, vn});
    } else {
      database->update_feature(id, ts, c, u, v, un, vn);
    }
  };
  // OV_XCAM_SNAP: list entry (c, k) and its observation of this timestamp move onto the matched position, the way the maintenance
  // step moves a point. Called before the relabelling, while the observation still carries the id of the list entry.
  auto snap_obs = [&](size_t c, size_t k, const XcMatch &m) {
    S.s_moved++;
    S.s_shift_sum += std::hypot((double)pts_last[c][k].pt.x - (double)m.qx, (double)pts_last[c][k].pt.y - (double)m.qy);
    pts_last[c][k].pt.x = m.qx;
    pts_last[c][k].pt.y = m.qy;
    if (deferred) {
      const long i = pend_idx(c, k);
      if (i >= 0) {
        PendingObs &e = (*pvec[c])[i];
        e.u = m.qx;
        e.v = m.qy;
        e.un = m.qnx;
        e.vn = m.qny;
      }
    } else {
      std::shared_ptr<Feature> f = database->get_feature(ids_last[c][k], false);
      if (f != nullptr) {
        auto itt = f->timestamps.find(c);
        if (itt != f->timestamps.end() && !itt->second.empty() && itt->second.back() == ts) {
          f->uvs.at(c).back() = Eigen::Vector2f(m.qx, m.qy);
          f->uvs_norm.at(c).back() = Eigen::Vector2f(m.qnx, m.qny);
        }
      }
    }
  };
  // entry (x, kx) holds the id already; entry (y, ky) has just received it
  auto link = [&](size_t x, size_t kx, size_t y, size_t ky) {
    const size_t id = ids_last[x][kx];
    for (int d = 0; d < XC_NCAM; d++) {
      if (S.info[x][kx].assoc[d] < 0 || (size_t)d == y || !present((size_t)d))
        continue;
      const std::vector<size_t> &idd = ids_last[d]; // the id also lives in camera d
      for (size_t j = 0; j < idd.size() && j < S.info[d].size(); j++)
        if (idd[j] == id) {
          S.info[d][j].assoc[y] = fs;
          S.info[d][j].recv[y] = false;
          break;
        }
      S.info[y][ky].assoc[d] = fs;
      S.info[y][ky].recv[d] = true; // camera d held the id before y
    }
    S.info[x][kx].assoc[y] = fs;
    S.info[x][kx].recv[y] = false;
    S.info[y][ky].assoc[x] = fs;
    S.info[y][ky].recv[x] = true;
    if (S.info[y][ky].sec == 0)
      S.info[y][ky].sec = 1;
  };
  const float same2 = (float)(S.SAME_PX * S.SAME_PX);
  int ncand_all = 0;
  for (XcTask &T : tasks) {
    PairStat &P = pst[T.pair];
    const size_t a = T.a, b = T.b;
    P.cand += T.cand;
    P.mfail += T.mutual_fail;
    P.matched += (int)T.matches.size();
    ncand_all += T.cand;
    S.t_considered += (long)T.tried.size();
    S.t_fov_excl += T.fov_excl;
    for (int r = 0; r < 12; r++)
      S.rej[r] += T.rej[r];
    if (blank_cam[a] || blank_cam[b]) {
      S.t_blank++;
      if (!S.warned_blank) {
        S.warned_blank = true;
        std::fprintf(stderr, "[xcam-warn] ts=%.6f: no candidate of camera %zu has texture at its own position: the image the association reads is not the "
                             "picture the tracker works on; nothing is marked as tried for the pairs of this camera\n", ts, blank_cam[a] ? a : b);
      }
      T.matches.clear();
    } else {
      for (int k : T.tried)
        S.info[a][k].next_try[b] = (S.retry > 0) ? fs + S.retry : -1;
    }
    std::sort(T.matches.begin(), T.matches.end(), [](const XcMatch &x, const XcMatch &y) { return x.k < y.k; });
    for (const XcMatch &m : T.matches) {
      const size_t ka = (size_t)m.k;
      if (S.info[a][ka].assoc[b] >= 0) { // the opposite direction has made this association already
        S.t_dup++;
        continue;
      }
      const size_t A = ids_last[a][ka];
      std::vector<cv::KeyPoint> &pb = pts_last[b];
      std::vector<size_t> &idb = ids_last[b];
      long kb = -1;
      float bd2 = same2;
      for (size_t j = 0; j < pb.size(); j++) {
        const float dx = pb[j].pt.x - m.qx, dy = pb[j].pt.y - m.qy, d2 = dx * dx + dy * dy;
        if (d2 < bd2 || (kb < 0 && d2 <= bd2)) {
          bd2 = d2;
          kb = (long)j;
        }
      }
      if (kb >= 0) {
        // 5a: camera b tracks this scene point already
        const size_t B = idb[kb];
        if (B == A) {
          S.t_dup++;
          continue;
        }
        const XcInfo &IA = S.info[a][ka], &IB = S.info[b][kb];
        if (IB.assoc[a] >= 0) { // B has a partner in camera a already
          P.skip++;
          continue;
        }
        const bool a_fresh = (IA.born == fs) && !IA.shared(), b_fresh = (IB.born == fs) && !IB.shared();
        // OV_XCAM_TAKEOVER (on with OV_XCAM_MINAGE > 0): neither point is fresh -> the YOUNGER unshared one continues under the id of
        // the other from this timestamp on; its earlier observations stay under its old id (a track that ends here).
        bool a_takes = a_fresh, b_takes = !a_fresh && b_fresh;
        if (!a_takes && !b_takes && S.takeover) {
          const bool a_can = !IA.shared(), b_can = !IB.shared();
          if (a_can && (IA.born >= IB.born || !b_can))
            a_takes = true;
          else if (b_can)
            b_takes = true;
        }
        if (S.snap && (a_takes || b_takes)) {
          // camera b's point may be a little off the position the matcher found: put it there, so that the pair starts registered
          if (IB.shared())
            S.s_third++;
          snap_obs(b, (size_t)kb, m);
        }
        if (a_takes) {
          move_obs(a, ka, A, B);
          ids_last[a][ka] = B;
          link(b, (size_t)kb, a, ka);
          P.relabel++;
        } else if (b_takes) {
          move_obs(b, (size_t)kb, B, A);
          idb[kb] = A;
          link(a, ka, b, (size_t)kb);
          P.relabel++;
        } else {
          P.skip++;
        }
      } else {
        // 5b: new point in camera b under the id of camera a's feature
        cv::KeyPoint kp;
        kp.pt.x = m.qx;
        kp.pt.y = m.qy;
        pb.push_back(kp);
        idb.push_back(A);
        XcInfo NI;
        NI.born = fs;
        NI.sec = 2;
        S.info[b].push_back(NI);
        link(a, ka, b, pb.size() - 1);
        write_obs(A, b, m.qx, m.qy, m.qnx, m.qny);
        P.inject++;
      }
    }
  }

  // ---- 5. shared points first (stable), so that the detection step's one-point-per-cell pruning keeps them
  for (size_t c : cams) {
    std::vector<XcInfo> &I = S.info[c];
    const size_t n = I.size();
    bool need = false, seen_non = false;
    for (size_t k = 0; k < n && !need; k++) {
      if (!I[k].shared())
        seen_non = true;
      else if (seen_non)
        need = true;
    }
    if (need && S.shared_first) {
      std::vector<cv::KeyPoint> np;
      std::vector<size_t> nid;
      std::vector<XcInfo> nI;
      np.reserve(n);
      nid.reserve(n);
      nI.reserve(n);
      for (int pass = 0; pass < 2; pass++)
        for (size_t k = 0; k < n; k++)
          if (I[k].shared() == (pass == 0)) {
            np.push_back(pts_last[c][k]);
            nid.push_back(ids_last[c][k]);
            nI.push_back(I[k]);
          }
      pts_last[c].swap(np);
      ids_last[c].swap(nid);
      I.swap(nI);
    }
    S.ids[c] = ids_last[c];
  }
  if (run) {
    S.last_run = fs;
    S.t_runs++;
    if (any_ph) {
      S.t_rb_used++;
      if (S.rb_out && S.rb_req_fs > 0) { // pixels asked for at the end of frame-set rb_req_fs arrived now: that many frame-sets were already uploaded
        const int seen = (int)(fs - S.rb_req_fs) - 1;
        if (seen >= 0 && seen <= 4 && seen != S.rb_lead) {
          S.rb_lead = seen;
          S.rb_lead_seen = seen;
        }
      }
      S.rb_out = false;
    }
  }
  if (S.readback) {
    // ask for the host pixels of the frame-set on which the step is due next (one frame-set per request; with OV_XCAM_EVERY=1 every
    // frame-set is read back, see the hook). A request that brought nothing within LEAD + 3 frame-sets is repeated.
    const bool again = S.rb_out && (fs - S.rb_req_fs > S.rb_lead + 3);
    if (again && any_ph) {
      S.t_rb_lost++;
      if (S.t_rb_lost == 3)
        std::fprintf(stderr, "[xcam-warn] ts=%.6f: OV_XCAM_READBACK=1 but the host pixels that were asked for did not arrive three times: the association does not "
                             "run on GPU-decoded frame-sets (see rb_ok / rb_fail in [xcam-total])\n", ts);
    }
    if (again || (!S.rb_out && fs + 1 + S.rb_lead - S.last_run >= S.every)) {
      xc_host_want(true);
      S.rb_out = true;
      S.rb_req_fs = fs;
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  S.ms.push_back((float)ms);

  // ---- 6. statistics
  for (size_t pi = 0; pi < pairs.size(); pi++) {
    const size_t i = pairs[pi].first, j = pairs[pi].second;
    int shared = 0;
    for (const XcInfo &I : S.info[i])
      if (I.assoc[j] >= 0)
        shared++;
    const PairStat &P = pst[pi];
    S.t_cand += P.cand;
    S.t_matched += P.matched;
    S.t_mfail += P.mfail;
    S.t_relabel += P.relabel;
    S.t_inject += P.inject;
    S.t_skip += P.skip;
    S.t_shared_sum[i][j] += shared;
    if (S.stats >= 1)
      std::fprintf(stderr, "[xcam] ts=%.6f pair=%zu,%zu cand=%d matched=%d mutual_fail=%d relabel=%d inject=%d skip=%d shared=%d\n", ts, i, j, P.cand, P.matched, P.mfail,
                   P.relabel, P.inject, P.skip, shared);
    if (S.stats >= 2 && (fs % S.pts_every) == 0) {
      std::unordered_map<size_t, size_t> where;
      for (size_t k = 0; k < S.info[j].size(); k++)
        if (S.info[j][k].assoc[i] >= 0)
          where.emplace(ids_last[j][k], k);
      for (size_t k = 0; k < S.info[i].size(); k++) {
        if (S.info[i][k].assoc[j] < 0)
          continue;
        auto it = where.find(ids_last[i][k]);
        if (it == where.end())
          continue;
        std::fprintf(stderr, "[xcam-pts] %.6f %zu %zu %zu %.3f %.3f %.3f %.3f %d\n", ts, i, j, ids_last[i][k], pts_last[i][k].pt.x, pts_last[i][k].pt.y,
                     pts_last[j][it->second].pt.x, pts_last[j][it->second].pt.y, fs - S.info[i][k].assoc[j]);
      }
    }
  }
  if (S.stats >= 1)
    std::fprintf(stderr, "[xcam-time] ts=%.6f ms=%.3f ncand=%d run=%d search_ms=%.3f\n", ts, ms, ncand_all, (int)run, search_ms);
}

// OV_XCAM_READBACK=1: with OV_NVJPG=1 the images of a message are headers over one shared page without pixels;
// the decoded picture lives in a ring slot on the device until gpu_prepare takes it. Called for every camera right BEFORE that
// (pre-seed and feed): when the association has asked for the pixels of its next run, the Y plane of this (camera, timestamp)
// is copied to the host and kept for xcam_associate. Does nothing unless the image is such a header.
static void xcam_host_hook(size_t cam_id, double ts, const cv::Mat &im) {
  XcState &S = xc();
  if (!S.on || !S.readback || cam_id >= (size_t)XC_NCAM || im.empty() || im.type() != CV_8UC1)
    return;
  if (!(S.emu_ph || (ov_core::ov_nvjpg_enabled() && ov_core::ov_nvjpg_is_placeholder(im.data))))
    return;
  XcHostCache &H = xc_host();
  {
    std::lock_guard<std::mutex> lk(H.m);
    if (ts > H.dec_ts) { // first camera of a NEWER frame-set decides for all of them
      H.dec_ts = ts;
      H.dec = (S.every <= 1) ? true : H.want.exchange(false);
    }
    if (ts != H.dec_ts || !H.dec) // an older frame-set (its surface is taken already) never uses up a request
      return;
    for (const XcHostCache::E &x : H.e[cam_id])
      if (x.ts == ts)
        return; // already here (the pre-seed and the feed both pass this point)
  }
  const auto t0 = std::chrono::steady_clock::now();
  cv::Mat buf(im.rows, im.cols, CV_8UC1); // a new buffer every time: the association may still read the previous one
  bool ok = false;
  if (S.emu_ph) {
    im.copyTo(buf);
    ok = true;
  } else {
    if (S.rb_sync) {
      int w = 0, h = 0;
      ok = ov_core::ov_nvjpg_readback_ts((int)cam_id, ts, buf.data, &w, &h) && w == im.cols && h == im.rows; // synchronous, on the default stream
    } else {
      ok = ov_core::ov_nvjpg_host_get((int)cam_id, ts, buf.data, im.cols, im.rows); // host copy made on the decode thread: no device call here
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::lock_guard<std::mutex> lk(H.m);
  H.ms += ms;
  if (!ok) {
    H.n_fail++;
    return;
  }
  H.n_ok++;
  if (H.e[cam_id].size() >= 3)
    H.e[cam_id].erase(H.e[cam_id].begin());
  H.e[cam_id].push_back({ts, buf});
}

// The decode side must know BEFORE the first picture is staged that host copies are wanted: pictures are decoded several frame-sets
// ahead of the first feed. So the switches are read when the library is loaded (only with OV_XCAM=1; nothing happens otherwise).
namespace {
const struct XcBoot {
  XcBoot() {
    if (xcam_enabled())
      (void)xc();
  }
} xc_boot_at_load;
} // namespace
