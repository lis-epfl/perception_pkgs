/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

// Portable stand-in for nvjpg_decode.cpp, selected by cmake/ROS2.cmake when the Jetson
// multimedia API (nvbufsurface.h) is not present at build time -- i.e. on every non-Jetson
// host. It satisfies the ov_nvjpg_* API so TrackKLT / gpu_track link unchanged, and reports
// the decoder as unavailable so OV_NVJPG=1 falls back to the CPU decode path. No behaviour
// changes on the Orin, where the real implementation is compiled instead.
#include "nvjpg_decode.h"

namespace ov_core {

void ov_nvjpg_set_lookahead(int) {}
bool ov_nvjpg_enabled() { return false; }
void ov_nvjpg_refuse_if(bool, const char *) {}
bool ov_nvjpg_selftest(const unsigned char *, size_t) { return false; }
const unsigned char *ov_nvjpg_stage(int, double, const unsigned char *, size_t, int *, int *, size_t *) { return nullptr; }
bool ov_nvjpg_is_placeholder(const unsigned char *) { return false; }
bool ov_nvjpg_take(int, double, const void **, size_t *, int *, int *, int *) { return false; }
void ov_nvjpg_mark_consumed(int, void *) {}
void ov_nvjpg_watermark(double) {}
bool ov_nvjpg_readback(int, unsigned char *) { return false; }
bool ov_nvjpg_readback_ts(int, double, unsigned char *, int *, int *) { return false; }
bool ov_nvjpg_readback_cpu_ts(int, double, unsigned char *, int *, int *) { return false; }
std::string ov_nvjpg_stats() { return "nvjpg: unavailable (built without the Jetson multimedia API)"; }

} // namespace ov_core
