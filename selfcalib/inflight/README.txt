selfcalib/inflight -- the self-calibration during the flight
===========================================================

This folder (branch selfcalib-inflight) holds the scheme in which the self-calibration starts while the vehicle is still
flying, and the scripts that demonstrate it on the flight computer. The released tool (../tool/run_tool.py on main) runs
after landing, on the closed recording, and that does not change here: the scheme uses the same tool, and nothing changes
when its switches are not set.


1. THE SCHEME
-------------

After landing (the released tool): the recording is closed, then the tool reads it. With 90 s of flight it needs about
145-150 s on the flight computer (two passes; Orin NX, 26 Oct 2026), so the calibration exists at the earliest that long
after the landing.

During the flight (this folder), as the paper describes it: the tool computes the image-centre prior during the first
15 s of the flight and runs the first pass during the flight. A pass takes less time than the flight, so the first pass
is done by the end of the 90 s, and the second follows.

  1. Heuristics on the first 15 s of flight: static-start and timing checks, image-centre prior, image check
     (run_tool.py --image-after-takeoff 15).
  2. Pass 1 starts about 15 s after takeoff. It reads the recording from its start (the standstill before takeoff), works
     off what has already arrived faster than real time, catches up with the incoming data, and then follows them until
     the 90th second of flight (--after-takeoff 90). It never uses a record before that record has arrived.
  3. Pass 2 starts as soon as pass 1 has read its last record. It starts from pass 1's calibration and reads the same 90 s
     again, as the released tool's second pass does.
  4. The released stopping rule decides (every parameter type moved by at most two envelopes between the passes), and the
     tool publishes the calibration (<drone>_published_chain.yaml and its lens terms).

The calibration logic is the released tool's. What makes pass 1 an in-flight pass is the estimator's real-time mode
(OV_REALTIME=1) with its clock tied to the instant the data went live, which this branch adds to the replay runner:

  OV_RT_T0_BAG + OV_RT_T0_WALL   the record stamped OV_RT_T0_BAG (sensor clock, s) is released at OV_RT_T0_WALL, read on
                                 the runner's own clock: CLOCK_MONOTONIC in seconds, Python's time.monotonic(), not the
                                 wall clock despite the name. A pass started late works off its backlog at full speed and
                                 then follows the data as they arrive. Needs OV_REALTIME=1 (and no OV_DETERMINISTIC=1).
                                 A value that is not a number, only one of the two, a run that is not paced, or an anchor
                                 that would hold the first record back more than a minute (wrong clock): a warning in the
                                 estimator's log, and the anchor is ignored.
  OV_RT_TRACE=<file>             one line per set of images handed to the filter: its stamp, the clock when the filter was
                                 done with it, and the milliseconds that took. Appended after one '#' line per run, so that
                                 the passes of one calibration, which share the environment, keep one block each.

Both are in vio/ov_msckf/src/run_serial_msckf.cpp. Unset, the runner is unchanged (section 6).


2. FILES
--------

  run_inflight_replay.sh   the demonstration on the flight computer (section 3); --help lists every option
  replay_clock.py          its helpers: facts of the recording (takeoff, file start times, complete or not), arrival of
                           the first image in the replay, the start instant, a clock stamp on every line of the tool's log
  inflight_report.py       what a run measured: times after takeoff (as run, and from the common start of section 4), and
                           the calibration against the same tool run after the flight
  fastdds_replay.xml       FastDDS profile of the replay: shared memory, discovery on the loopback only (a 1280x800 JPEG
                           does not fit the Orin's UDP receive buffer)
  README.txt               this file


3. HOW TO RUN IT (on the flight computer)
-----------------------------------------

Build the estimator of this branch once, with ROS 2 Humble and a workspace that has px4_msgs sourced (the vehicle's flight
workspace has it; `ros2 bag play` needs it to publish the PX4 IMU messages, and the built workspace chains it):

    source /opt/ros/humble/setup.bash
    source <workspace with px4_msgs>/install/setup.bash
    OV_WS=<new workspace> bash vio/vio_deploy/scripts/build_openvins.sh    # 8 min on an Orin NX (OV_BUILD_JOBS=6)

Replay a recorded flight in real time and run the scheme on it:

    bash selfcalib/inflight/run_inflight_replay.sh \
        --recording <log folder: bag/ + cams/recording.mcap> --drone <name in fleet_reference> \
        --ws <the workspace> --out <new run folder> [--offline 1] [--wait-idle 900]

What it does, in this order: refuses to start if another estimator or tool run is on the computer (and with --wait-idle
waits for a 1-min load below 0.5 and a GPU load below 5 % for 60 s); samples the computer every second until the end
(sampler.txt); starts a recorder of the five topics (the vehicle records them too) and a listener for the first image;
starts two `ros2 bag play` for the IMU file and the camera file, with the IMU leading the cameras by 0-60 ms (restarted
until it does); waits until the replay is --start-after seconds past the takeoff; starts the tool with --after-takeoff 90
--image-after-takeoff 15 --no-gt-autodetect --fleet-exclude <drone> and OV_REALTIME=1, OV_RT_T0_BAG / OV_RT_T0_WALL =
stamp and arrival of the first image of camera A, OV_RT_TRACE; waits for the tool; stops the players, the recorder (its
copy is deleted unless --keep-copy) and the listener; with --offline N runs the same tool N times after the flight on the
same recording, without pacing; and writes inflight_report.txt/.json. The run folder keeps the exact tool command
(tool.cmd.sh), the tool's log with a clock stamp on every line (tool.log), the trace (rt_trace.txt) and the tool's output
(tool/). The exit code is the tool's.

The times are those of a quiet computer only if nothing else ran. The quiet check passes before the start, but nothing
stops another job from starting during the run (it happened in the first run of section 6): the report lists every
process above 20 % CPU that was not the run's own, and the load and GPU load during the tool's run.

Options that matter:
  --flight-time 90 / --image-time 15   the two lengths of the scheme (tool --after-takeoff / --image-after-takeoff)
  --start-after S        the tool is started S s after takeoff (default 3; section 4)
  --min-still S          the tool's --min-still; 4.5 for recordings with a short standstill (nxt10, 4 June 2026)
  --arrival-stamps       for recordings made before 15 Oct 2026, whose images carry their arrival time on the flight
                         computer: the fleet's time-offset centres of that time are used (a copy of configs/ and
                         fleet_reference/ in the run folder, given to the tool as SCT_ROOT). Without it the tool starts
                         such a recording 40 ms from its offset and ends HARDWARE-CHANGED (toff out of distribution).
  --tool-cpus 4-7 --flight-folder DIR   the flight estimator beside the calibration (section 5)
  --dry-run              prints the players, the recorder and the tool command, writes nothing

Afterwards, deploy the calibration as usual (../tool/deploy_vio.py --calib-out <run folder>/tool --out <flight folder>).

Example (the runs of section 6):

    bash selfcalib/inflight/run_inflight_replay.sh --recording data/nxt10_s1 --drone nxt10 --ws ws --out runs/nxt10_s1 \
        --min-still 4.5 --arrival-stamps --offline 1 --wait-idle 600


4. WHAT WAS MEASURED
--------------------

The paper's numbers come from replays on the flight computer (Orin NX, GPU at the fleet's 612 MHz, CPUs at 1.98 GHz)
with the tool 1ff4d12 (matching between cameras and lens terms on) and an estimator build of the release dc8d058 plus the
same two switches (test harness scripts onl15.sh / onl2p.sh). This folder is those harness scripts, cleaned and
parameterised; the switches are the same, ported to main (section 1).

Ready after takeoff, two passes as one sequence, 90 s of flight, nothing else estimating on the computer: median 158.3 s
after takeoff, range 155.9-168.2 s, 9 replays (3 rounds of nxt3_s1, nxt6_s2, nxt10_s1; 13 Oct 2026). From the common
start below, pass 1 ended 90.3 s after takeoff in 7 of the 9 replays and 1.9 and 2.1 s behind the live data in the other
two; pass 2 took 65.6-76.0 s (median 68.0) to the verdict.
Other flight times, same protocol: 20 s 49.8, 30 s 63.2, 60 s 105.9, 120 s 213.1, 150 s 257.6 s (medians).
  Study records (not in this repository): the replays of 13 Oct and the script that computed these times from their
  traces; how each cell of the paper's flight-time table was produced, and its independent recomputation (column "Ready
  after takeoff").

When pass 1 caught up with the live data (one pass, 12 Oct 2026, 11 runs on nxt3_s1, nxt6_s2, nxt10_s1, standstill before
takeoff 11 / 8 / 4 s): 55-62 s after takeoff alone (n = 4), 69-77 s beside the flight estimator on separate cores (n = 3),
both from the common start below. In flight the pass ran at 1.46-1.51 x real time alone and 1.35-1.38 x beside the flight
estimator; the standstill replays at 2.2-2.9 x. The calibration of the in-flight pass was as close to the whole-flight
calibration as a pass run after the flight (0.6-1.5 mm and 0.03-0.10 deg, against 0.5-1.3 mm and 0.05-0.09 deg; two runs
of the same pass differ by about 0.9 mm).
  Study records (not in this repository): the test of 12 Oct, its notes and its table of runs.

The common start. In the replay the heuristics run as a batch on the recorded file, so the tool is started a few seconds
after takeoff (--start-after, default 3) and pass 1 usually begins 13-17 s after takeoff, earlier or later than in the
scheme.
The reported time is therefore also given for a common start, "first set of images done 16.5 s after takeoff" (15 s of
heuristics + 1.5 s for the estimator to start): pass 1's own trace is replayed as a queue (each set available at its stamp,
costing what it cost in the run) from that start, and pass 2 is added as measured. All paper numbers above use it;
inflight_report.py prints both. At 90 s they are equal in 8 of the 9 replays; in the ninth pass 1 began 28 s after
takeoff and ended 8.7 s behind the live data (166.9 s as run, 158.3 s from the common start).

What the computer does otherwise: the released tool after landing needs about 145-150 s at 90 s of flight on the same
computer. In the scheme that work is done between takeoff and about 158 s after takeoff, most of it while the vehicle flies.


5. CPU AFFINITY: REQUIRED WHEN THE FLIGHT ESTIMATOR RUNS BESIDE IT
-------------------------------------------------------------------

On the vehicle the flight estimator (run_online_msckf) runs during the whole flight. campaign.env binds the main threads of
every estimator to CPUs 0-3 (OV_AFFINITY=0-3:4-7), so the calibration's estimator lands on the same four CPUs as the
flight estimator. Measured (12 Oct 2026, one run, nxt10_s1): the pass then ran at about 0.6 x real time and never caught
up, and the flight estimator's poses came out 12 s late in the median and up to 51 s late, while CPUs 4-7 were 7 % busy.

The calibration must run on the other four CPUs: `taskset -c 4-7` for the tool and OV_AFFINITY=4-7:4-7 for its estimator.
run_inflight_replay.sh --tool-cpus 4-7 does both (and refuses --flight-folder without it). Measured that way (3 runs): pass
1 caught up 69-77 s after takeoff (common start), the flight estimator's pose delay rose by 1-5 ms in the median and 7-10
ms at the 90th percentile, it processed the same sets of images (none dropped), and its tracking error changed by +0.5,
+0.2 and -0.6 cm. With the computer otherwise idle (no flight estimator) no affinity is needed.

Two passes beside the flight estimator were not run as one sequence. Applying the one-pass runs' slowdown (8-9 % beside
it) to the nine measured two-pass replays of section 4 puts the result about 164 s after takeoff (161-175 s; estimate).


6. THIS BRANCH ON THE FLIGHT COMPUTER
-------------------------------------

Tested on 7 Nov 2026 on the Orin NX flight computer (GPU at the fleet's 612 MHz, CPUs at 1.98 GHz): estimator built
from this branch (8 min, the same compiler warnings as the release build, none in the changed file), recording nxt10_s1
(4 June 2026, arrival stamps, 4.1 s on the ground before takeoff), the example command of section 3 (--flight-time 90
--image-time 15 --start-after 3 --min-still 4.5 --arrival-stamps), nothing else estimating.

Switches unset change nothing. The deterministic runner (OV_DETERMINISTIC=1, campaign.env, matching and lens terms, the
calibration config on 20 s of flight) wrote byte-identical trajectories and calibrations with the release build of main
(3a96bc0), with this branch's build, and with this branch's build and all the switches set (the trace written; the anchor
ignored with a warning, because a deterministic run is not paced).

The replay on a quiet computer (sampler: no other process above 20 % CPU; GPU load 20 % on average, 53 % at most, which
is the estimator's own tracker; CPUs at 1.98 GHz throughout):
  pass 1  began 12.1 s after takeoff (first set of images done at 13.5 s, with 17.6 s of data waiting), worked off the
          backlog at 1.50 x real time (1.44 x in flight), caught up with the live data 47.6 s after takeoff (54.3 s from the
          common start of section 4) and read its last set 90.0 s after takeoff, 0.015 s behind the live data
  pass 2  started 90.3 s after takeoff; verdict 65.0 s later: HEALTHY, two passes (step 0.16 envelopes)
  READY   155.3 s after takeoff, as run and from the common start. The replays behind the paper: 155.9-168.2 s, median
          158.3 s (nxt10_s1 alone: 155.9, 156.4 and 158.3 s).
The same tool after the flight on the same recording, on the same quiet computer: 133.0 s from its start (without the
motion-capture read of 7-9 s, --no-gt-autodetect).

The calibration against the tool after the flight. Five calibrations of the same data were made: two in flight (this
replay and the first one below) and three after the flight, all HEALTHY with two passes, time offsets -42.99 to -43.06 ms.
  in flight vs after the flight (6 pairs): camera positions relative to the IMU differ by at most 2.48-2.71 mm (mean of
    the four cameras 1.74-2.04 mm), rotations by at most 0.036-0.072 deg, principal points by at most 0.19-0.46 px, focal
    lengths by at most 0.04-0.08 %, the time offset by 0.02-0.08 ms; distances between neighbouring cameras by at most
    0.61-1.94 mm; the tool's own step measure 0.11-0.18 envelopes (it calls two calibrations the same at 2)
  after the flight vs after the flight (3 pairs): 2.35-2.58 mm (mean 1.45-1.86 mm), 0.057-0.069 deg, 0.23-0.35 px,
    0.04-0.07 %, 0.01-0.05 ms, 1.10-1.91 mm, 0.15-0.18 envelopes
So the calibration made during the flight differs from one made after the flight as much as two runs after the flight
differ from each other: the estimator's run-to-run scatter.

The first replay, same set-up 45 min earlier, ran while another job built TensorRT engines on the same computer (trtexec,
through both passes and both runs after the flight; it started 2 s before the tool, after the quiet check had passed,
which is why the launcher now samples the computer): pass 1 at 1.33 x real time in flight, caught up 69.0 s
after takeoff (70.5 s from the common start), pass 2 76.0 s, READY 166.3 s after takeoff; after the flight 154.3 and
142.0 s. Its calibrations are in the comparison above; its times are not those of a quiet computer.

  Records (not in this repository): the scripts that ran, their logs, and the run folders with inflight_report.txt/.json.


7. WHAT IT DOES NOT DO (limitations)
------------------------------------

  - It reads the finished recording. The replay plays a closed recording and the tool's passes read that same closed file
    under the clock rule above; reading a file that is still being written was not emulated.
  - No hand-over to live topics. A pass that has caught up keeps reading the file under the clock rule; it does not switch
    to the live topics as a flight node would.
  - The heuristics are not computed frame by frame. The static-start and timing checks, the image-centre prior and the
    image check run as a batch on the recorded file when the tool starts (a few seconds after takeoff), not while the
    images arrive; the timing check reads the whole 90 s window. Hence the common start of section 4. With the batch
    started only at the 15th second, pass 1 would begin 26-29 s after takeoff: alone it would still catch up (at about
    78-88 s), beside the flight estimator it would miss the 90th second by 3-9 s (estimate, 12 Oct 2026).
  - Live use. Pointing the launcher at the recording a vehicle is writing does NOT work without new code, and the launcher
    refuses it (a .mcap that does not end with the MCAP footer is still being written): the tool finds the topics and
    the length of the recording in its metadata (the bag's metadata.yaml, the MCAP summary), which a recorder writes
    when it closes the file, and the estimator's reader (vio/ov_msckf/src/utils/bag_source.h) stops at the end of the
    file instead of waiting for more data. A live version needs a reader that follows a growing recording or hands
    over to the live topics, the takeoff from the land detector as it happens, and the heuristics computed as the
    images arrive. None of this is implemented.
  - Standstill before takeoff. Pass 1 also replays the standstill (at 2.2-2.9 x real time); the tested recordings stood
    4-11 s. Estimate (12 Oct 2026): beyond about 17-20 s of standstill pass 1 no longer catches up by the 90th second
    beside the flight estimator; alone the limit is about 35 s.
  - It was replayed, not flown: the camera driver, the autopilot link and the rest of the onboard software were not
    running.
  - The flight estimator beside it (--flight-folder) is the 12 Oct harness's set-up ported to deploy_vio.py folders; it
    was not run with this branch (the replays of section 6 ran without it).
  - One run at a time, recordings in the vehicle layout (bag/ + cams/recording.mcap with PX4 and OAK topics) only.
