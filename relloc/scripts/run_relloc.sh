#!/bin/bash
# Start relloc on this drone, beside the VIO (run_online_msckf). Does nothing unless RELLOC=1.
#   set -a; source relloc/config/relloc.env; set +a      # then set RELLOC=1, RELLOC_FLIGHT_DIR, RELLOC_DEST (README.md)
#   bash relloc/scripts/run_relloc.sh [--check]           # --check: print the resolved settings and exit
# Needs the ROS 2 environment of the flight workspace sourced (rclpy, px4_msgs), python3 with torch (CUDA), kornia, scipy, opencv.
# Exit codes: 0 off (RELLOC != 1) or clean stop; 1 bad settings (the message names the variable); 64 no rclpy.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${RELLOC:-0}" != "1" ]; then echo "[relloc] off (RELLOC=${RELLOC:-unset}); set RELLOC=1 to run"; exit 0; fi
python3 -c "import rclpy" 2>/dev/null || { echo "[relloc] rclpy not found: source the ROS 2 / flight workspace setup first"; exit 64; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1   # the receiver's numerics otherwise take every core
exec python3 "$HERE/relloc_node.py" "$@"
