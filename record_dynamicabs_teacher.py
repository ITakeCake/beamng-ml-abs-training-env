"""Record DynamicABS_Trainer 400 Hz demonstrations for distillation.

DynamicABS_Trainer is a classical PID slip regulator running at 400 Hz inside
BeamNG. This script drives brake stops with it active, captures per-tick raw
data via a Lua recorder extension, and post-processes into the same (obs, act)
npz format that distill_teacher.py consumes.

    python record_dynamicabs_teacher.py --reps 6 --out teacher_dabs_400hz.npz
    python distill_teacher.py --data teacher_dabs_400hz.npz --config .gui-configs/PPO-63.json
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_config

HERE = os.path.dirname(os.path.abspath(__file__))
MPH_TO_MS = 0.44704
DT = 1.0 / 400
TRAINER_PC = "vehicles/etk800/DynamicABS_Trainer.pc"

# CSV column indices (must match trainerRecorder.lua header)
C_GS = 0
C_WS_FR, C_WS_FL, C_WS_RR, C_WS_RL = 1, 2, 3, 4
C_GY, C_GX, C_YAW = 5, 6, 7
C_SLAM_FIRED, C_SLAM_FIRE_SPEED = 8, 9
C_INST_SPEED = 10
C_DIST, C_AVGG = 11, 12
C_FUSED, C_BACT = 13, 14
C_BRK = 15
C_WGY, C_WGYMIN, C_WGYMAX, C_WYAW = 16, 17, 18, 19
C_RPM, C_GEAR, C_STEER = 20, 21, 22
C_PITCH, C_ROLL = 23, 24
C_THR, C_GZ = 25, 26
C_ABSSPD = 27
C_BRK_NM_FR, C_BRK_NM_FL, C_BRK_NM_RR, C_BRK_NM_RL = 28, 29, 30, 31
C_CMD_FR, C_CMD_FL, C_CMD_RR, C_CMD_RL = 32, 33, 34, 35

FRAME_DIM = 35
N_STACK = 64
FRAME_LOW = np.array([
    0.0, 0.0, 0.0, 0.0,
    -50.0, -50.0, -10.0,
    0.0, 0.0, 0.0, 0.0,
    -50.0, -50.0,
    0.0, -2.0, -2.0,
    -np.pi, -np.pi,
    0.0, 0.0,
    -50.0, -10.0, -10.0,
    -200.0, -200.0, -200.0, -200.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)
FRAME_HIGH = np.array([
    100.0, 100.0, 100.0, 100.0,
    50.0, 50.0, 10.0,
    1.0, 1.0, 1.0, 1.0,
    50.0, 50.0,
    15000.0, 12.0, 2.0,
    np.pi, np.pi,
    1.0, 1.0,
    50.0, 10.0, 10.0,
    200.0, 200.0, 200.0, 200.0,
    1.0, 1.0, 1.0, 1.0,
    5000.0, 5000.0, 5000.0, 5000.0,
], dtype=np.float32)


def build_frame(row, prev_ws, prev_pitch, prev_roll, prev_brakes):
    ws = np.array([row[C_WS_FR], row[C_WS_FL], row[C_WS_RR], row[C_WS_RL]],
                  dtype=np.float32)
    pitch = float(row[C_PITCH])
    roll = float(row[C_ROLL])

    if prev_ws is None:
        wa = np.zeros(4, dtype=np.float32)
        pitch_rate = roll_rate = 0.0
    else:
        wa = (ws - prev_ws) / DT
        pitch_rate = (pitch - prev_pitch) / DT
        roll_rate = (roll - prev_roll) / DT

    abs_ref = max(float(row[C_ABSSPD]), 0.5)
    slip = np.clip(1.0 - ws / abs_ref, 0.0, 1.0)

    raw = np.array([
        ws[0], ws[1], ws[2], ws[3],
        float(row[C_WGY]), float(row[C_GX]), float(row[C_WYAW]),
        prev_brakes[0], prev_brakes[1], prev_brakes[2], prev_brakes[3],
        float(row[C_WGYMIN]), float(row[C_WGYMAX]),
        float(row[C_RPM]), float(row[C_GEAR]), float(row[C_STEER]),
        pitch, roll,
        float(row[C_BRK]), float(row[C_THR]),
        float(row[C_GZ]), pitch_rate, roll_rate,
        wa[0], wa[1], wa[2], wa[3],
        slip[0], slip[1], slip[2], slip[3],
        float(row[C_BRK_NM_FR]), float(row[C_BRK_NM_FL]),
        float(row[C_BRK_NM_RR]), float(row[C_BRK_NM_RL]),
    ], dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(raw, FRAME_LOW, FRAME_HIGH).astype(np.float32), ws, pitch, roll


def process_recording(csv_path, wheel_mode="independent"):
    """Read a trainerRecorder CSV, build stacked obs + teacher release pairs."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    braking_mask = data[:, C_BRK] > 0.5
    if not np.any(braking_mask):
        return np.empty((0, FRAME_DIM * N_STACK), np.float32), np.empty((0, 4), np.float32)

    first_brake = int(np.argmax(braking_mask))
    data = data[first_brake:]

    stack = [np.zeros(FRAME_DIM, np.float32) for _ in range(N_STACK)]
    prev_ws = None
    prev_pitch = prev_roll = 0.0
    prev_brakes = (0.0, 0.0, 0.0, 0.0)
    obs_list = []
    act_list = []

    for i in range(len(data)):
        row = data[i]
        frame, prev_ws, prev_pitch, prev_roll = build_frame(
            row, prev_ws, prev_pitch, prev_roll, prev_brakes)
        stack.pop(0)
        stack.append(frame)
        obs = np.concatenate(stack).astype(np.float32)

        cmd_fr = float(row[C_CMD_FR])
        cmd_fl = float(row[C_CMD_FL])
        cmd_rr = float(row[C_CMD_RR])
        cmd_rl = float(row[C_CMD_RL])
        prev_brakes = (cmd_fr, cmd_fl, cmd_rr, cmd_rl)

        if row[C_BRK] < 0.5:
            continue

        rel_fr = 1.0 - cmd_fr
        rel_fl = 1.0 - cmd_fl
        rel_rr = 1.0 - cmd_rr
        rel_rl = 1.0 - cmd_rl

        if wheel_mode == "independent":
            act = [rel_fr, rel_fl, rel_rr, rel_rl]
        else:
            act = [0.5 * (rel_fr + rel_fl), 0.5 * (rel_rr + rel_rl)]

        obs_list.append(obs.copy())
        act_list.append(act)

    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32))


def recording_path(sc):
    up = sim_config.content_userpath(sc)
    return os.path.join(up, "trainer_recording.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", default="teacher_dabs_400hz.npz")
    ap.add_argument("--speeds", default="80",
                    help="comma-separated target speeds in mph")
    ap.add_argument("--wheel-mode", choices=("independent", "axle"),
                    default="independent")
    ap.add_argument("--pc", default=TRAINER_PC)
    args = ap.parse_args()

    sc = sim_config.load(os.path.join(HERE, "settings.json"))
    sc.game = "drive"
    sc.headless = False
    port = args.port or sc.port
    speeds = [float(s) for s in args.speeds.split(",")]
    rec_path = recording_path(sc)

    bng = BeamNGpy("localhost", port, home=sc.game_folder,
                   user=sim_config.resolved_userpath(sc))
    bng.open(None, launch=True)
    veh = Vehicle("teacher", model="etk800", part_config=args.pc)
    veh.sensors.attach("electrics", Electrics())
    scenario = Scenario("smallgrid", "teacher_rec")
    scenario.add_vehicle(veh, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(3.0)

    veh.queue_lua_command("extensions.load('abstelemetry')")
    veh.queue_lua_command("extensions.load('trainerRecorder')")
    time.sleep(1.0)

    all_obs = []
    all_act = []

    try:
        for speed_mph in speeds:
            target_ms = speed_mph * MPH_TO_MS
            arm_ms = target_ms + 3.0 * MPH_TO_MS
            for rep in range(args.reps):
                print("speed=%d mph  rep=%d/%d" % (speed_mph, rep + 1, args.reps),
                      end="  ", flush=True)

                veh.teleport((0.0, 0.0, 0.0), rot_quat=(0.0, 0.0, 0.0, 1.0),
                             reset=True)
                time.sleep(0.5)
                veh.queue_lua_command("extensions.abstelemetry.resetAccum()")
                veh.queue_lua_command(
                    "extensions.abstelemetry.setTargetSpeed(%f)" % target_ms)
                time.sleep(0.2)

                rec_lua = rec_path.replace("\\", "/")
                veh.queue_lua_command(
                    'extensions.trainerRecorder.startRecording("%s")' % rec_lua)

                veh.control(gear=2, throttle=1.0, steering=0, brake=0)
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    veh.sensors.poll()
                    e = veh.sensors["electrics"]
                    spd = float(e.get("airspeed", 0) or 0)
                    if spd >= arm_ms:
                        break
                else:
                    print("TIMEOUT reaching target speed")
                    veh.queue_lua_command(
                        "extensions.trainerRecorder.stopRecording()")
                    continue

                veh.control(throttle=0.0, brake=1.0, gear=0)

                stop_frames = 0
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    veh.sensors.poll()
                    e = veh.sensors["electrics"]
                    spd = float(e.get("airspeed", 0) or 0)
                    if spd < 0.5:
                        stop_frames += 1
                        if stop_frames >= 10:
                            break
                    else:
                        stop_frames = 0

                time.sleep(0.3)
                veh.queue_lua_command(
                    "extensions.trainerRecorder.stopRecording()")
                time.sleep(0.5)

                veh.sensors.poll()
                e = veh.sensors["electrics"]
                avg_g = float(e.get("tel_last_brake_avg_g", 0) or 0)
                dist = float(e.get("tel_last_brake_dist", 0) or 0)
                print("avg_g=%.4f  dist=%.2f m" % (avg_g, dist), flush=True)

                if os.path.isfile(rec_path):
                    obs, act = process_recording(rec_path, args.wheel_mode)
                    if len(obs) > 0:
                        all_obs.append(obs)
                        all_act.append(act)
                        print("  -> %d (obs, act) pairs" % len(obs))
                    else:
                        print("  -> no braking data in recording")
                    os.remove(rec_path)
                else:
                    print("  -> recording file not found")

    finally:
        bng.close()

    if all_obs:
        obs = np.concatenate(all_obs)
        act = np.concatenate(all_act)
        out_path = os.path.join(HERE, args.out)
        np.savez_compressed(out_path, obs=obs, act=act)
        print("\nrecorded %d (obs, act) pairs -> %s" % (len(obs), args.out))
    else:
        print("\nno data recorded")


if __name__ == "__main__":
    main()
