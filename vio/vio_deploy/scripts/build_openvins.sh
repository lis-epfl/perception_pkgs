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
# shellcheck disable=SC2086
colcon build --packages-select $PKGS \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo "OK. Estimator invocation goes through: $VIO/vio_deploy/scripts/run_serial.sh"
echo "  bash run_serial.sh BAG CONFIG OUT_DIR [NCAM] [STEREO] [SEED] [DOMAIN]"
echo "  (export OV_WS=$OV_WS if not the default \$HOME/ov_ws_vio)"
