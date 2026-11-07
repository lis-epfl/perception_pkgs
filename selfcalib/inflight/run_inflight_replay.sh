#!/bin/bash
# run_inflight_replay.sh -- the self-calibration during the flight, shown on a recorded flight that is replayed in real time on
# the flight computer. The replay plays the recording's IMU and camera streams at their recorded rate (and records them again,
# as the vehicle does); the tool is started during the replay; pass 1 of its estimator may not use a record before the
# instant that record went live in the replay (OV_REALTIME=1 with the pacing clock anchored to the replay: OV_RT_T0_BAG /
# OV_RT_T0_WALL), so it works off the backlog at full speed and then follows the data; pass 2 starts when pass 1 has read
# the last record of the window (the 90th second of flight); the tool publishes. README.txt in this folder: the scheme, the
# measured times, the CPU requirement and what this does NOT do (it reads the finished recording, not a growing one).
#
# USAGE
#   run_inflight_replay.sh --recording DIR --drone NAME --ws DIR --out DIR [options]
#
#   --recording DIR      flight recording in the vehicle layout: DIR/bag (PX4 rosbag2) + DIR/cams/recording.mcap. It must be
#                        complete (every .mcap closed): a recording still being written is refused (README, "Live use").
#   --drone NAME         vehicle name in selfcalib/fleet_reference (seed, --fleet-exclude, default template and IMU chain)
#   --ws DIR             colcon workspace built from THIS checkout (vio/vio_deploy/scripts/build_openvins.sh with OV_WS=DIR);
#                        its run_serial_msckf must know OV_RT_T0_BAG
#   --out DIR            run folder, new or empty
#   --flight-time S      seconds of flight the calibration uses (tool --after-takeoff), default 90
#   --image-time S       seconds of flight of the image pass (tool --image-after-takeoff), default 15
#   --start-after S      start the tool S seconds after takeoff on the replay clock, default 3 (README, "Start of the tool")
#   --min-still S        tool --min-still (the tool's default 6 s; 4.5 for the 4 June 2026 recordings of nxt10 and nxt3b)
#   --template FILE      tool --template, default selfcalib/fleet_reference/NAME/theta_star.yaml
#   --imu-chain FILE     tool --imu-chain, default selfcalib/fleet_reference/NAME/kalibr_imu_chain.yaml
#   --vio-root DIR       the vio/ package, default the sibling of selfcalib/
#   --arrival-stamps     the images carry their ARRIVAL time (recordings made before the camera driver stamped the capture
#                        time, 15 Oct 2026): seed and judge the camera-IMU time offset with the fleet's arrival-stamp centres
#                        (a copy of configs/ and fleet_reference/ in OUT/sct_root, td_center.json "center" :=
#                        "center_arrival_stamps", given to the tool as SCT_ROOT). Without it such a recording starts 40 ms
#                        from its offset and ends HARDWARE-CHANGED (out of distribution: toff).
#   --tool-cpus LIST     run the tool and its estimator on these CPUs only: taskset -c LIST and OV_AFFINITY=LIST:LIST
#                        (e.g. 4-7). REQUIRED with --flight-folder.
#   --flight-folder DIR  also run the flight estimator (run_online_msckf) on the replayed topics, flying the calibration of
#                        the folder deploy_vio.py wrote (its flight_stiffness.env; campaign.env on aarch64)
#   --offline N          after the replay, run the same tool N times AFTER the flight on the same recording (no pacing, no CPU
#                        restriction), into OUT/offline1..N, to compare the calibrations (default 0)
#   --domain N           ROS domain of the replay (players, recorder, flight estimator), default 49
#   --tool-domain N      ROS domain of the tool's estimator passes (tool --domain), default 122
#   --fastdds-profile F  FastDDS profile of the replay, default fastdds_replay.xml beside this script
#   --ros-setup FILE     ROS 2 setup, default /opt/ros/humble/setup.bash
#   --setup FILE         another setup.bash to source after it (repeatable), e.g. the workspace with px4_msgs, which
#                        `ros2 bag play` needs to publish the PX4 IMU messages
#   --no-recorder        do not record the replayed topics (the measured runs recorded them, as the vehicle does)
#   --keep-copy          keep the recorder's copy (default: removed at the end; 1-5 GB)
#   --wait-idle S        wait up to S seconds for a quiet computer before starting: 1-min load < 0.5 and, on a Jetson, GPU
#                        load < 5 %, both for 60 s; default 0 = do not wait. Another estimator or tool run always makes it
#                        refuse. Either way the board is sampled every second during the run (sampler.txt; the report
#                        says what else ran).
#   --tool-args "ARGS"   more options for run_tool.py (split on spaces)
#   --dry-run            print what would run and stop
# END USAGE
#
# Writes OUT/: launcher.log, versions.txt, replay.env (the facts the report reads), first_frame.txt, play_imu.log,
# play_cam.log, record.log, sampler.txt (load, GPU load, CPU clocks and temperature every second; the busiest processes
# every 5 s), tool.cmd.sh (the exact tool command), tool.log (its output, each line stamped with the clock),
# rt_trace.txt (one block per estimator pass), tool/ (the tool's --out: published chain, lens terms, report.json),
# offline<k>/ + offline<k>.log, flight/ (with --flight-folder), inflight_report.txt/.json (inflight_report.py).
# One run at a time. Nothing outside OUT is written. Exit code: the tool's (selfcalib/README.md, "Exit codes"); 64 usage,
# 65 refused, 66-72 set-up failures.
set -o pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELFCALIB=$(dirname "$HERE")
CLOCK=$HERE/replay_clock.py

usage() { sed -n '/^# USAGE/,/^# END USAGE/p' "$0" | sed -e '/USAGE/d' -e 's/^# \{0,1\}//'; }
die() { local rc=$1; shift; echo "run_inflight_replay.sh: $*" >&2; [ -n "$LOG" ] && echo "[$(stamp)] ERROR: $*" >> "$LOG"; exit "$rc"; }
stamp() { echo "$(date +%H:%M:%S) $(cut -d' ' -f1 /proc/uptime)"; }
GPU_LOAD=; for f in /sys/devices/platform/17000000.gpu/load /sys/devices/platform/gpu.0/load; do [ -r $f ] && { GPU_LOAD=$f; break; }; done
gpu_load() { if [ -n "$GPU_LOAD" ]; then cat "$GPU_LOAD"; else echo 0; fi; }   # per mille (Jetson); 0 elsewhere
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

LOG=; REC=; DRONE=; WS=; OUT=; FLIGHT_S=90; IMAGE_S=15; START_AFTER=3; MIN_STILL=; TEMPLATE=; IMU=; VIO=
ARRIVAL=0; TOOL_CPUS=; FLIGHT=; OFFLINE=0; DOMAIN=49; TOOL_DOMAIN=122; FASTDDS=$HERE/fastdds_replay.xml
ROS_SETUP=/opt/ros/humble/setup.bash; SETUPS=(); RECORD=1; KEEP=0; WAIT_IDLE=0; TOOL_ARGS=; DRY=0
num() { python3 -c "import sys; v = float(sys.argv[1]); sys.exit(0 if v == v and v >= 0 else 1)" "$1" 2>/dev/null; }
while [ $# -gt 0 ]; do
  case $1 in
    --recording) REC=$2; shift 2 ;;        --drone) DRONE=$2; shift 2 ;;
    --ws) WS=$2; shift 2 ;;                --out) OUT=$2; shift 2 ;;
    --flight-time) FLIGHT_S=$2; shift 2 ;; --image-time) IMAGE_S=$2; shift 2 ;;
    --start-after) START_AFTER=$2; shift 2 ;; --min-still) MIN_STILL=$2; shift 2 ;;
    --template) TEMPLATE=$2; shift 2 ;;    --imu-chain) IMU=$2; shift 2 ;;
    --vio-root) VIO=$2; shift 2 ;;         --arrival-stamps) ARRIVAL=1; shift ;;
    --tool-cpus) TOOL_CPUS=$2; shift 2 ;;  --flight-folder) FLIGHT=$2; shift 2 ;;
    --offline) OFFLINE=$2; shift 2 ;;      --domain) DOMAIN=$2; shift 2 ;;
    --tool-domain) TOOL_DOMAIN=$2; shift 2 ;; --fastdds-profile) FASTDDS=$2; shift 2 ;;
    --ros-setup) ROS_SETUP=$2; shift 2 ;;  --setup) SETUPS+=("$2"); shift 2 ;;
    --no-recorder) RECORD=0; shift ;;      --keep-copy) KEEP=1; shift ;;
    --wait-idle) WAIT_IDLE=$2; shift 2 ;;  --tool-args) TOOL_ARGS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;             -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
[ -n "$REC" ] && [ -n "$DRONE" ] && [ -n "$WS" ] && [ -n "$OUT" ] || { usage >&2; exit 64; }
for v in "$FLIGHT_S" "$IMAGE_S" "$START_AFTER" "$OFFLINE" "$DOMAIN" "$TOOL_DOMAIN" "$WAIT_IDLE" ${MIN_STILL:+"$MIN_STILL"}; do
  num "$v" || { echo "not a number >= 0: $v" >&2; exit 64; }
done
[[ "$OFFLINE" =~ ^[0-9]+$ ]] || { echo "--offline takes a whole number" >&2; exit 64; }
[ -z "$TOOL_CPUS" ] || [[ "$TOOL_CPUS" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || { echo "--tool-cpus takes a CPU list like 4-7" >&2; exit 64; }
REC=$(cd "$REC" 2>/dev/null && pwd) || { echo "no recording folder: $REC" >&2; exit 66; }
WS=$(cd "$WS" 2>/dev/null && pwd) || { echo "no workspace folder: $WS" >&2; exit 66; }
VIO=${VIO:-$(dirname "$SELFCALIB")/vio}
TEMPLATE=${TEMPLATE:-$SELFCALIB/fleet_reference/$DRONE/theta_star.yaml}
IMU=${IMU:-$SELFCALIB/fleet_reference/$DRONE/kalibr_imu_chain.yaml}
for f in "$TEMPLATE" "$IMU" "$VIO/vio_deploy/scripts/run_serial.sh" "$FASTDDS" "$ROS_SETUP" "${SETUPS[@]}"; do
  [ -f "$f" ] || { echo "missing file: $f" >&2; exit 66; }
done
EST=$WS/install/ov_msckf/lib/ov_msckf/run_serial_msckf
[ -x "$EST" ] || { echo "no estimator in $WS (build it: OV_WS=$WS bash $VIO/vio_deploy/scripts/build_openvins.sh)" >&2; exit 66; }
grep -q OV_RT_T0_BAG "$EST" || { echo "$EST does not know OV_RT_T0_BAG: it was not built from this checkout" >&2; exit 66; }
if [ -n "$FLIGHT" ]; then
  [ -n "$TOOL_CPUS" ] || { echo "--flight-folder needs --tool-cpus: two estimators on the same CPUs fall behind (README)" >&2; exit 64; }
  for f in estimator_flight.yaml kalibr_imucam_chain.yaml kalibr_imu_chain.yaml flight_stiffness.env; do
    [ -f "$FLIGHT/$f" ] || { echo "$FLIGHT is not a folder written by deploy_vio.py (no $f)" >&2; exit 66; }
  done
  [ -x "$WS/install/ov_msckf/lib/ov_msckf/run_online_msckf" ] || { echo "no run_online_msckf in $WS" >&2; exit 66; }
fi
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT exists and is not empty" >&2; exit 65; fi
if [ $DRY = 1 ]; then LOG=/dev/null; OUT=$(realpath -m "$OUT")
else mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd) && LOG=$OUT/launcher.log || exit 66; fi

# ---- environment of the replay ----
# shellcheck disable=SC1090
source "$ROS_SETUP" || die 66 "cannot source $ROS_SETUP"
for f in "${SETUPS[@]}"; do source "$f" || die 66 "cannot source $f"; done
source "$WS/install/setup.bash" || die 66 "cannot source $WS/install/setup.bash"
export ROS_DOMAIN_ID=$DOMAIN FASTRTPS_DEFAULT_PROFILES_FILE=$FASTDDS ROS_LOG_DIR=$OUT/roslog
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] [{time}] [{name}]: {message}'   # the player's "Sleep" line carries its time
unset ROS_LOCALHOST_ONLY      # the profile keeps discovery on the loopback; ROS_LOCALHOST_ONLY=1 would also drop shared memory
case ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp} in rmw_fastrtps_cpp) ;; *) say "WARNING: RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION: the FastDDS profile does not apply; images may be lost" ;; esac

# ---- the recording ----
INFO=$(python3 -B "$CLOCK" info --recording "$REC") || die 66 "cannot read $REC"
eval "$INFO"
[ -z "$REC_INCOMPLETE" ] || die 65 "the recording is not complete (no MCAP footer: still being written?): $REC_INCOMPLETE. Live use, on a file being written, is not implemented (README, \"Live use\")"
[ -n "$REC_TAKEOFF_S" ] || die 66 "no takeoff in $REC (PX4 land detector): the scheme counts its seconds from the takeoff"
ros2 interface show "$REC_IMU_TYPE" > /dev/null 2>&1 || die 66 "message type $REC_IMU_TYPE unknown here: pass --setup <workspace with it>/install/setup.bash"
read -r CAM0_TOPIC _ <<< "$REC_CAM_TOPICS"
OFF=$(python3 -c "print('%.6f' % ($REC_CAM_START - $REC_IMU_START))")
PLAY_S=$(python3 -c "print('%.1f' % (max(0.0, $OFF) + $REC_TAKEOFF_S + 1.8 * $FLIGHT_S + 30.0))")
SCT=
if [ $ARRIVAL = 1 ]; then
  SCT=$OUT/sct_root
  if [ $DRY = 0 ]; then
    mkdir -p "$SCT" && cp -r "$SELFCALIB/configs" "$SELFCALIB/fleet_reference" "$SCT/" || die 66 "cannot copy the fleet reference"
  fi
  [ $DRY = 1 ] || python3 - "$SCT/fleet_reference/td_center.json" <<'PY' || die 66 "cannot write the arrival-stamp centres"
import json, sys
p = sys.argv[1]; d = json.load(open(p)); n = 0
for k, v in d.items():
    if isinstance(v, dict) and 'center_arrival_stamps' in v:
        v['center'] = v['center_arrival_stamps']; n += 1
d['_note_inflight'] = 'run_inflight_replay.sh --arrival-stamps: "center" := "center_arrival_stamps" for %d vehicles' % n
json.dump(d, open(p, 'w'), indent=1)
PY
fi

# ---- the commands ----
UNSET=(-u OV_XCAM -u OV_XCAM_PRESET -u OV_PRIOR_NONRAD_SIG -u OV_NONRAD_SKEW_FIXED -u OV_NONRADIAL -u OV_DETERMINISTIC
       -u OV_REALTIME -u OV_RT_T0_BAG -u OV_RT_T0_WALL -u OV_RT_TRACE -u OV_AFFINITY -u ROS_DOMAIN_ID -u FASTRTPS_DEFAULT_PROFILES_FILE)
TOOL_ENV=(PYTHONDONTWRITEBYTECODE=1 "ROS_LOG_DIR=$OUT/roslog_tool" "OV_WS=$WS" "VIO_ROOT=$VIO")
[ -n "$SCT" ] && TOOL_ENV+=("SCT_ROOT=$SCT")
TOOL_ARGV=(python3 -B "$SELFCALIB/tool/run_tool.py" --drone "$DRONE" --bag "$REC" --template "$TEMPLATE" --imu-chain "$IMU"
           --fleet-exclude "$DRONE" --domain "$TOOL_DOMAIN" --after-takeoff "$FLIGHT_S" --image-after-takeoff "$IMAGE_S" --no-gt-autodetect)
[ -n "$MIN_STILL" ] && TOOL_ARGV+=(--min-still "$MIN_STILL")
# shellcheck disable=SC2206
[ -n "$TOOL_ARGS" ] && TOOL_ARGV+=($TOOL_ARGS)
write_cmd() {   # write_cmd FILE|- TOOL_OUT [VAR=VALUE ...] [taskset ...]: the tool's command line, every word quoted
  local file=$1 tout=$2; shift 2
  cmd_text() { echo '#!/bin/bash'; echo '# written by run_inflight_replay.sh: the exact tool command of this run'
    printf 'cd %q || exit 1\nexec env' "$OUT"; printf ' %q' "${UNSET[@]}" "${TOOL_ENV[@]}" "$@" "${TOOL_ARGV[@]}" --out "$tout"
    printf ' 2>&1\n'; }
  if [ "$file" = - ]; then cmd_text "$@"; else cmd_text "$@" > "$file"; fi
}
IMU_PLAY=(ros2 bag play "$REC_IMU_URI" --topics "$REC_IMU_TOPIC" --disable-keyboard-controls)
[[ $REC_IMU_URI == *.mcap ]] && IMU_PLAY+=(-s mcap)
# shellcheck disable=SC2206
CAM_PLAY=(ros2 bag play "$REC_CAM_URI" -s mcap --topics $REC_CAM_TOPICS --disable-keyboard-controls --read-ahead-queue-size 1000)
# shellcheck disable=SC2206
RECORDER=(ros2 bag record -s mcap -o "$OUT/rec" "$REC_IMU_TOPIC" $REC_CAM_TOPICS)

[ $DRY = 1 ] || { echo "run_inflight_replay.sh $(stamp)"; echo "checkout: $(git -C "$SELFCALIB" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')"
  echo "run_tool.py md5 $(md5sum < "$SELFCALIB/tool/run_tool.py" | cut -c1-32)"
  echo "estimator $EST md5 $(md5sum < "$EST" | cut -c1-32)"
  echo "host $(uname -nmr) cpus $(nproc) load $(cut -d' ' -f1-3 /proc/loadavg)"
  G=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu   # Jetson Orin GPU (read only)
  [ -r $G/max_freq ] && echo "gpu max $(cat $G/max_freq) cur $(cat $G/cur_freq)"
  echo "cpu max kHz $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq 2>/dev/null | sort -u | tr '\n' ' ')"
} > "$OUT/versions.txt"
say "recording $REC: takeoff $REC_TAKEOFF_S s after every stream is live, camera file starts $OFF s after the IMU file, $REC_SPAN_S s of data"
say "flight time $FLIGHT_S s, image pass $IMAGE_S s, tool started $START_AFTER s after takeoff${TOOL_CPUS:+, tool on CPUs $TOOL_CPUS}${FLIGHT:+, flight estimator $FLIGHT}${SCT:+, arrival-stamp centres}"
if [ $DRY = 1 ]; then
  echo "--- dry run: nothing started, nothing written ---"; echo "recorder:   ${RECORDER[*]}"; echo "IMU player: ${IMU_PLAY[*]} -d <s>"
  echo "cam player: ${CAM_PLAY[*]} -d <s>"; echo "first image of $CAM0_TOPIC -> tool started $START_AFTER s after takeoff:"
  write_cmd - "$OUT/tool" OV_REALTIME=1 OV_RT_T0_BAG='<stamp of the first image>' OV_RT_T0_WALL='<its arrival clock>' \
            "OV_RT_TRACE=$OUT/rt_trace.txt" ${TOOL_CPUS:+"OV_AFFINITY=$TOOL_CPUS:$TOOL_CPUS" taskset -c "$TOOL_CPUS"}
  exit 0
fi

# ---- one run at a time ----
busy() { pgrep -x run_serial_msck; pgrep -x run_online_msck; pgrep -f '^python3 .*run_tool\.py'; }
[ -z "$(busy)" ] || die 65 "another estimator or tool run is running ($(busy | tr '\n' ' ')): not started"
if [ "${WAIT_IDLE%.*}" -gt 0 ]; then
  t_end=$(( $(cut -d. -f1 /proc/uptime) + ${WAIT_IDLE%.*} )); ok=0
  while [ $ok -lt 12 ]; do
    [ "$(cut -d. -f1 /proc/uptime)" -lt $t_end ] || die 65 "the computer did not stay quiet (1-min load < 0.5, GPU load < 5 %) for 60 s within $WAIT_IDLE s"
    [ -z "$(busy)" ] || die 65 "another estimator or tool run started while waiting"
    if awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v g="$(gpu_load)" 'BEGIN{exit !(l < 0.5 && g < 50)}'; then ok=$((ok + 1)); else ok=0; fi
    sleep 5
  done
  say "quiet: 1-min load $(cut -d' ' -f1 /proc/loadavg), GPU load $(gpu_load) per mille"
fi

# ---- processes: each in its own session, so that it can be stopped with its children; stopped on any exit ----
LEADERS=()
start() { setsid "$@" & LEADERS+=($!); }                 # start CMD...: background, own process group
stop_group() { kill -0 -- "-$1" 2>/dev/null || return 0; kill -INT -- "-$1" 2>/dev/null
               for _ in $(seq 1 "${2:-20}"); do kill -0 -- "-$1" 2>/dev/null || return 0; sleep 0.5; done
               kill -KILL -- "-$1" 2>/dev/null; }
cleanup() { local p; for p in "${LEADERS[@]}"; do stop_group "$p" 4; done; }
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

# the board, every second: clock load1 gpu_load_permille cpu_cur_min_kHz cpu_cur_max_kHz hottest_zone_mC; every 5 s the three
# busiest processes ("#top clock name:%cpu ..."), so that a run shows whether anything else was running
sampler() {
  local n=0 t f mn mx x z hot
  while :; do
    mn=99999999; mx=0; hot=-273000
    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do read -r x < "$f" 2>/dev/null || continue
      [ "$x" -lt $mn ] && mn=$x; [ "$x" -gt $mx ] && mx=$x; done
    for z in /sys/class/thermal/thermal_zone*/temp; do read -r x < "$z" 2>/dev/null || continue
      [[ $x =~ ^-?[0-9]+$ ]] && [ "$x" -gt $hot ] && hot=$x; done
    read -r t _ < /proc/uptime
    echo "$t $(cut -d' ' -f1 /proc/loadavg) $(gpu_load) $mn $mx $hot"
    if [ $((n % 5)) = 0 ]; then
      echo "#top $t $(ps -eo comm,pcpu --sort=-pcpu --no-headers | head -3 | awk '{printf "%s:%s ", $1, $2}')"
    fi
    n=$((n + 1)); sleep 1
  done
}
start bash -c "GPU_LOAD='$GPU_LOAD'; $(declare -f gpu_load sampler); sampler" > "$OUT/sampler.txt" 2>/dev/null < /dev/null
SPID=$!

NPID=
if [ -n "$FLIGHT" ]; then
  F=$OUT/flight; mkdir -p "$F"
  cp "$FLIGHT"/estimator_flight.yaml "$FLIGHT"/kalibr_imucam_chain.yaml "$FLIGHT"/kalibr_imu_chain.yaml "$F"/ || die 66 "cannot copy $FLIGHT"
  if ! grep -q '^imu_topic:' "$F/estimator_flight.yaml"; then   # a deployed config names no topics (deploy_vio.py MANIFEST)
    { printf '\n# --- topics of the replayed recording (run_inflight_replay.sh) ---\nimu_topic: "%s"\nimu_msg_type: "%s"\ncam_msg_type: "%s"\n' \
        "$REC_IMU_TOPIC" "$REC_IMU_TYPE" "$REC_CAM_TYPE"; i=0; for t in $REC_CAM_TOPICS; do printf 'cam_topic%d: "%s"\n' $i "$t"; i=$((i + 1)); done
    } >> "$F/estimator_flight.yaml"
  fi
  CAMPAIGN=; [ "$(uname -m)" = aarch64 ] && [ -f "$FLIGHT/campaign.env" ] && CAMPAIGN=$FLIGHT/campaign.env
  start bash -c 'set -a; source "$1"; [ -n "$2" ] && source "$2"; set +a; shift 2; exec "$@"' flight \
        "$FLIGHT/flight_stiffness.env" "$CAMPAIGN" env "OV_ONLINE_LAT=$F/lat.txt" "OV_ONLINE_POSE_LAT=$F/pose_lat.txt" \
        "$WS/install/ov_msckf/lib/ov_msckf/run_online_msckf" "$F/estimator_flight.yaml" "$F/estimate_tum.txt" \
        --ros-args -p use_stereo:=false -p max_cameras:="$REC_NCAM" > "$F/run.log" 2>&1 < /dev/null
  NPID=$!
  for _ in $(seq 1 180); do grep -q '\[online\]: qos=' "$F/run.log" 2>/dev/null && break; sleep 1; done
  grep -q '\[online\]: qos=' "$F/run.log" || die 70 "the flight estimator did not come up (see $F/run.log)"
  sleep 4
  say "flight estimator up (pid $NPID)"
fi
RECPID=
if [ $RECORD = 1 ]; then start "${RECORDER[@]}" > "$OUT/record.log" 2>&1 < /dev/null; RECPID=$!; sleep 3; fi
start python3 -B "$CLOCK" first-frame --topic "$CAM0_TOPIC" --out "$OUT/first_frame.txt" > "$OUT/first_frame.log" 2>&1 < /dev/null
FCPID=$!
sleep 2

# ---- the players: the IMU must lead the cameras by 0-60 ms, as on the vehicle; restarted until it does ----
BASE_DELAY=12; DELTA=0
for try in 1 2 3 4 5 6 7 8; do
  DI=$(python3 -c "print('%.3f' % ($BASE_DELAY + max(0.0, -$OFF)))")
  DC=$(python3 -c "print('%.3f' % ($BASE_DELAY + max(0.0, $OFF) + $DELTA / 1000.0))")
  T_KILL=$(python3 -c "print(int($DI + $PLAY_S + 0.5))")
  start timeout -s INT -k 5 "$T_KILL" "${IMU_PLAY[@]}" -d "$DI" > "$OUT/play_imu.log" 2>&1 < /dev/null; PI=$!
  start timeout -s INT -k 5 "$T_KILL" "${CAM_PLAY[@]}" -d "$DC" > "$OUT/play_cam.log" 2>&1 < /dev/null; PC=$!
  for _ in $(seq 1 100); do grep -q 'Sleep' "$OUT/play_imu.log" 2>/dev/null && grep -q 'Sleep' "$OUT/play_cam.log" 2>/dev/null && break; sleep 0.1; done
  TIS=$(grep -a 'Sleep' "$OUT/play_imu.log" | grep -o '\[[0-9]*\.[0-9]*\]' | tr -d '[]' | head -1)
  TCS=$(grep -a 'Sleep' "$OUT/play_cam.log" | grep -o '\[[0-9]*\.[0-9]*\]' | tr -d '[]' | head -1)
  LEAD=$(python3 -c "print(round(1000 * ((float('${TCS:-0}') + $DC) - (float('${TIS:-1e12}') + $DI) - $OFF), 1))")
  if python3 -c "import sys; sys.exit(0 if 0.0 <= $LEAD <= 60.0 else 1)"; then break; fi
  say "players, try $try: IMU leads the cameras by $LEAD ms, not 0-60: restarting them"
  DELTA=$(python3 -c "print(round(max(-900.0, min(900.0, $DELTA + 20.0 - $LEAD)), 1))")
  stop_group $PI 6; stop_group $PC 6; wait $PI $PC 2>/dev/null; sleep 0.5
done
python3 -c "import sys; sys.exit(0 if 0.0 <= $LEAD <= 60.0 else 1)" || die 71 "the two players could not be started within 0-60 ms of each other"
say "players started (try $try, IMU leads by $LEAD ms; delays $DI / $DC s; they stop $T_KILL s after their start)"

# ---- the tool, START_AFTER seconds after the takeoff ----
SCHED=$(python3 -B "$CLOCK" wait --first "$OUT/first_frame.txt" --anchor-stamp "$REC_ANCHOR_STAMP" --takeoff "$REC_TAKEOFF_S" --after "$START_AFTER")
[ "$SCHED" != NOFRAME ] && [ -n "$SCHED" ] || die 72 "no image of $CAM0_TOPIC arrived from the replay"
read -r T0_BAG T0_CLOCK TAKEOFF_CLOCK GO_CLOCK NOW <<< "$SCHED"
RT=(OV_REALTIME=1 "OV_RT_T0_BAG=$T0_BAG" "OV_RT_T0_WALL=$T0_CLOCK" "OV_RT_TRACE=$OUT/rt_trace.txt")
[ -n "$TOOL_CPUS" ] && RT+=("OV_AFFINITY=$TOOL_CPUS:$TOOL_CPUS" taskset -c "$TOOL_CPUS")
write_cmd "$OUT/tool.cmd.sh" "$OUT/tool" "${RT[@]}"
start bash -c 'bash "$1" | python3 -B "$2" stamp > "$3"; echo "${PIPESTATUS[0]}" > "$4"' tool \
      "$OUT/tool.cmd.sh" "$CLOCK" "$OUT/tool.log" "$OUT/tool.rc" < /dev/null
TOOLPID=$!
say "tool started (first image: stamp $T0_BAG at clock $T0_CLOCK; takeoff at clock $TAKEOFF_CLOCK; start at $GO_CLOCK, now $NOW)"
{ printf '%s=%q\n' RECORDING "$(basename "$REC")" RECORDING_PATH "$REC" DRONE "$DRONE" FLIGHT_S "$FLIGHT_S" IMAGE_S "$IMAGE_S" \
    START_AFTER "$START_AFTER" TOOL_CPUS "$TOOL_CPUS" FLIGHT_FOLDER "$FLIGHT" ARRIVAL_STAMPS "$ARRIVAL" OFF "$OFF" \
    PLAYER_DELAY_IMU "$DI" PLAYER_DELAY_CAM "$DC" IMU_LEAD_MS "$LEAD" T0_BAG "$T0_BAG" T0_CLOCK "$T0_CLOCK" \
    TAKEOFF_CLOCK "$TAKEOFF_CLOCK" GO_CLOCK "$GO_CLOCK"; echo "$INFO"; } > "$OUT/replay.env"

wait $TOOLPID; TOOL_RC=$(cat "$OUT/tool.rc" 2>/dev/null || echo 99)
echo "TOOL_RC=$TOOL_RC" >> "$OUT/replay.env"
say "tool done, rc $TOOL_RC: $(grep -a 'VERDICT:' "$OUT/tool.log" | tail -1 | cut -d' ' -f2-)"
if [ "$TOOL_RC" != 0 ]; then say "stopping the replay early"; stop_group $PI 6; stop_group $PC 6; fi
wait $PI $PC 2>/dev/null
say "players done"
if [ -n "$RECPID" ]; then sleep 2; stop_group "$RECPID" 20; wait "$RECPID" 2>/dev/null
  say "recorder stopped: $(du -sh "$OUT/rec" 2>/dev/null | cut -f1) written"; fi
if [ -n "$NPID" ]; then stop_group "$NPID" 240; wait "$NPID" 2>/dev/null; say "flight estimator stopped"; fi
stop_group "$FCPID" 2; wait "$FCPID" 2>/dev/null
[ $KEEP = 1 ] || rm -rf "$OUT/rec"

# ---- the same tool after the flight, for the comparison ----
for k in $(seq 1 "$OFFLINE"); do
  [ -z "$(busy)" ] || { say "offline run $k not started: another estimator is running"; break; }
  write_cmd "$OUT/offline$k.cmd.sh" "$OUT/offline$k"
  start bash -c 'bash "$1" | python3 -B "$2" stamp > "$3"; echo "${PIPESTATUS[0]}" > "$4"' offline \
        "$OUT/offline$k.cmd.sh" "$CLOCK" "$OUT/offline$k.log" "$OUT/offline$k.rc" < /dev/null
  wait $!
  say "offline run $k done, rc $(cat "$OUT/offline$k.rc" 2>/dev/null), $(python3 -c "import sys; L = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]; print('%.1f s' % (float(L[-1]) - float(L[0])))" "$OUT/offline$k.log" 2>/dev/null): $(grep -a 'VERDICT:' "$OUT/offline$k.log" | tail -1 | cut -d' ' -f2-)"
done

stop_group "$SPID" 2; wait "$SPID" 2>/dev/null
python3 -B "$HERE/inflight_report.py" "$OUT" | tee -a "$OUT/launcher.log"
exit "$TOOL_RC"
