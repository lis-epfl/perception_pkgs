#include "OvLogger.h"
#include <cstdlib>
#include <cstring>

namespace ov_msckf {

OvLogger::OvLogger() : fp_(nullptr) {
  const char* path = std::getenv("OPENVINS_DUMP_PATH");
  if (!path || std::strlen(path) == 0) return;
  fp_ = std::fopen(path, "w");
  if (fp_)
    std::fprintf(stderr, "[OvLogger] writing events to %s\n", path);
}
OvLogger::~OvLogger() {
  if (fp_) { std::fflush(fp_); std::fclose(fp_); }
}
void OvLogger::flush() {
  std::lock_guard<std::mutex> lk(mu_);
  if (fp_) std::fflush(fp_);
}

static void write_id_array(std::FILE* fp, const std::vector<long>& v) {
  std::fputc('[', fp);
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) std::fputc(',', fp);
    std::fprintf(fp, "%ld", v[i]);
  }
  std::fputc(']', fp);
}

void OvLogger::log_frame(double ts, int frame_idx, const double twc[7],
                         const std::vector<long>& observed_feat_ids) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_,
               "{\"e\":\"frame\",\"t\":%.9f,\"fid\":%d,\"twc\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],\"mps\":",
               ts, frame_idx, twc[0], twc[1], twc[2], twc[3], twc[4], twc[5], twc[6]);
  write_id_array(fp_, observed_feat_ids);
  std::fprintf(fp_, "}\n");
}

void OvLogger::log_clone_create(double ts, long clone_id, const double twc[7]) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_,
               "{\"e\":\"kf_create\",\"t\":%.9f,\"kfid\":%ld,\"twc\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]}\n",
               ts, clone_id, twc[0], twc[1], twc[2], twc[3], twc[4], twc[5], twc[6]);
}

void OvLogger::log_clone_cull(long clone_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_, "{\"e\":\"kf_cull\",\"kfid\":%ld}\n", clone_id);
}

void OvLogger::log_feat_create(long feat_id, const double pos[3], long clone_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_,
               "{\"e\":\"mp_create\",\"mpid\":%ld,\"pos\":[%.6f,%.6f,%.6f],\"kfid\":%ld}\n",
               feat_id, pos[0], pos[1], pos[2], clone_id);
}

void OvLogger::log_feat_destroy(long feat_id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_, "{\"e\":\"mp_destroy\",\"mpid\":%ld}\n", feat_id);
}

void OvLogger::log_feat_snapshot(double ts, const std::vector<long>& ids,
                                 const std::vector<double>& xyz) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_, "{\"e\":\"mp_snap\",\"t\":%.9f,\"ids\":", ts);
  write_id_array(fp_, ids);
  std::fprintf(fp_, ",\"xyz\":[");
  for (size_t i = 0; i < xyz.size(); ++i) {
    if (i) std::fputc(',', fp_);
    std::fprintf(fp_, "%.5f", xyz[i]);
  }
  std::fprintf(fp_, "]}\n");
}

void OvLogger::log_clone_snapshot(double ts, const std::vector<long>& ids,
                                  const std::vector<double>& twc) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_, "{\"e\":\"kf_snap\",\"t\":%.9f,\"ids\":", ts);
  write_id_array(fp_, ids);
  std::fprintf(fp_, ",\"twc\":[");
  for (size_t i = 0; i < twc.size(); ++i) {
    if (i) std::fputc(',', fp_);
    std::fprintf(fp_, "%.6f", twc[i]);
  }
  std::fprintf(fp_, "]}\n");
}

void OvLogger::log_window(double ts, const std::vector<long>& clone_ids) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!fp_) return;
  std::fprintf(fp_, "{\"e\":\"lba\",\"t\":%.9f,\"local_kfs\":", ts);
  write_id_array(fp_, clone_ids);
  std::fprintf(fp_, ",\"fixed_kfs\":[]}\n");
}

}  // namespace ov_msckf
