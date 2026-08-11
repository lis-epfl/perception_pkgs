#!/bin/bash
# Offline OpenVINS run — the ONLY supported way to invoke the estimator.
# Args: BAG CFG OUT_DIR [NCAM] [STEREO] [SEED] [DOMAIN]
#
# Note on reproducibility: the RNG seed is pinned and the serial node is used, but the
# estimator is not bit-reproducible on long recordings (see README "Reproducibility
# caveat"). Compare published chains against thresholds, not byte-for-byte.
# NOTE: deliberately no `set -u`. ROS 2's setup.bash dereferences unset variables
# (AMENT_TRACE_SETUP_FILES and friends), so `set -u` makes sourcing it fail. Defaults are
# handled explicitly with ${VAR:-...} below instead.
set -eo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: run_serial.sh BAG CFG OUT_DIR [NCAM] [STEREO] [SEED] [DOMAIN]" >&2
  exit 64
fi
BAG="$1"; CFG="$2"; OUT="$3"; NCAM="${4:-4}"; STEREO="${5:-true}"; SEED="${6:-42}"; DOM="${7:-0}"
OV_WS="${OV_WS:-$HOME/ov_ws_vio}"

# Fail loudly on a missing prerequisite. Sourcing a nonexistent setup.bash used to be
# swallowed by >/dev/null, leaving `ros2 run` to fail with an unrelated message — or, worse,
# to pick up a DIFFERENT ov_msckf from the ambient environment.
# BAG may be a comma-separated list of URIs: a swarm-nxt recording keeps IMU and
# cameras in separate files. Each part must exist, either as a directory (rosbag2)
# or as a file (a bare .mcap, e.g. cams/recording.mcap).
IFS=',' read -r -a _BAG_PARTS <<< "$BAG"
for _p in "${_BAG_PARTS[@]}"; do
  [ -d "$_p" ] || [ -f "$_p" ] || { echo "run_serial.sh: bag not found: $_p" >&2; exit 66; }
done
[ -f "$CFG" ] || { echo "run_serial.sh: config not found: $CFG" >&2; exit 66; }
[ -f /opt/ros/humble/setup.bash ] || { echo "run_serial.sh: ROS 2 Humble not found at /opt/ros/humble" >&2; exit 69; }
if [ ! -f "$OV_WS/install/setup.bash" ]; then
  echo "run_serial.sh: no OpenVINS workspace at $OV_WS" >&2
  echo "  build it first:  bash scripts/build_openvins.sh" >&2
  echo "  or point OV_WS at an existing workspace." >&2
  exit 69
fi
case "$DOM" in
  ''|*[!0-9]*) echo "run_serial.sh: DOMAIN must be an integer 0-232, got '$DOM'" >&2; exit 64 ;;
esac
[ "$DOM" -le 232 ] || { echo "run_serial.sh: DOMAIN must be 0-232 (higher silently fails in DDS), got $DOM" >&2; exit 64; }

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$OV_WS/install/setup.bash"
export ROS_DOMAIN_ID="$DOM"
export ROS_LOCALHOST_ONLY=1
mkdir -p "$OUT"

set +e
OV_RNG_SEED="$SEED" ros2 run ov_msckf run_serial_msckf "$CFG" "$BAG" "$OUT/estimate_tum.txt" \
    --ros-args -p use_stereo:="$STEREO" -p max_cameras:="$NCAM" > "$OUT/run.log" 2>&1
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "run_serial.sh: estimator exited $rc — see $OUT/run.log" >&2
  tail -n 5 "$OUT/run.log" >&2 || true
  exit "$rc"
fi
if [ ! -s "$OUT/estimate_tum.txt" ]; then
  echo "run_serial.sh: estimator exited 0 but wrote no trajectory — see $OUT/run.log" >&2
  exit 70
fi
echo "DONE $OUT"
