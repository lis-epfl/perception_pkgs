#!/bin/bash
# Build the VIO package (pruned OpenVINS + vio_deploy) into a colcon workspace.
# Usage: [OV_WS=/custom/path] bash vio_deploy/scripts/build_openvins.sh
#
# The four ament packages live side by side in the vio/ directory:
#   ov_core  ov_init  ov_msckf  vio_deploy
# They are symlinked into $OV_WS/src so colcon discovers all of them. (They must stay
# siblings: colcon does not descend into a directory that has its own package.xml.)
#
# Default workspace is $HOME/ov_ws_vio, NOT $HOME/ov_ws — using the latter would
# silently repoint an existing workspace's src/ symlinks at this checkout.
# run_serial.sh uses the same default.
set -eo pipefail
VIO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # .../vio
OV_WS="${OV_WS:-$HOME/ov_ws_vio}"
source /opt/ros/humble/setup.bash

# OpenVINS declares cmake_minimum_required(VERSION 3.3); CMake >= 4.0 rejects that outright.
if cmake --version 2>/dev/null | head -1 | grep -qE 'version [4-9]\.'; then
  if /usr/bin/cmake --version 2>/dev/null | head -1 | grep -qE 'version 3\.'; then
    echo "note: $(command -v cmake) is too new for OpenVINS; using /usr/bin/cmake"
    PATH="/usr/bin:$PATH"
  else
    echo "warning: cmake >= 4.0 on PATH and no 3.x fallback; relying on CMAKE_POLICY_VERSION_MINIMUM"
  fi
fi

PKGS="ov_core ov_init ov_msckf vio_deploy"
mkdir -p "$OV_WS/src"
for pkg in $PKGS; do
  [ -f "$VIO/$pkg/package.xml" ] || { echo "build_openvins.sh: missing package $VIO/$pkg" >&2; exit 66; }
  ln -sfn "$VIO/$pkg" "$OV_WS/src/$pkg"
done
cd "$OV_WS"

# Cap build parallelism by MEMORY, not by core count. An OpenVINS translation unit
# pulls in Eigen and Ceres templates and peaks around 1.1 GB in cc1plus, so an
# unbounded `make -j$(nproc)` sizes the build off cores it has and RAM it does not.
# Measured failure: a 20-core / 15 GiB host asked for ~22 GB and the kernel
# OOM-killed cc1plus mid-build. Jetsons are the same shape of trap (Orin NX: 8
# cores, 16 GB shared with the GPU). Override with OV_BUILD_JOBS if you know better.
if [ -z "${OV_BUILD_JOBS:-}" ]; then
  _cores=$(nproc 2>/dev/null || echo 4)
  # MemAvailable is what we can actually use without swapping; 1.5 GB per job.
  _avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 4096)
  _mem_jobs=$(( _avail_mb / 1500 ))
  [ "$_mem_jobs" -lt 1 ] && _mem_jobs=1
  OV_BUILD_JOBS=$(( _cores < _mem_jobs ? _cores : _mem_jobs ))
fi
# NOTE: colcon's --parallel-workers and MAKEFLAGS MULTIPLY -- N packages in flight,
# each running make -jN, is N*N concurrent compilers. Pinning workers to 1 makes the
# ceiling exactly OV_BUILD_JOBS: packages build one at a time, and the parallelism goes
# where the memory actually goes (many translation units inside one package).
echo "build_openvins.sh: building with $OV_BUILD_JOBS parallel compile jobs"
export MAKEFLAGS="-j${OV_BUILD_JOBS}"

# shellcheck disable=SC2086
colcon build --packages-select $PKGS \
    --parallel-workers 1 \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo "OK. Estimator invocation goes through: $VIO/vio_deploy/scripts/run_serial.sh"
echo "  bash run_serial.sh BAG CONFIG OUT_DIR [NCAM] [STEREO] [SEED] [DOMAIN]"
echo "  (export OV_WS=$OV_WS if not the default \$HOME/ov_ws_vio)"
