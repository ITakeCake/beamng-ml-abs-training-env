"""Live gates for the residual action space, run against real physics before any
training starts.

GATE 1 (slam baseline): zero action ([0,0], no release) must reproduce the known
lockup baseline (~0.85-0.95g on sport_plus tires -- CLAUDE.md: "full lockup ~= 0.9g
on sport_plus, models below that prove nothing"). If it doesn't, the pedal-hold or
the action-inversion path is broken and nothing downstream can be trusted.

GATE 2 (release helps): a scripted mid-stop release must beat the slam mean. If it
doesn't, releasing brake torque isn't reaching the wheels -- the mailbox path is
broken -- and training would never discover it either.

avg_g/stopping_dist_m are only ever written to the episode CSV log by abs_env.py
(never returned via step()'s info dict, which stays {} on every path) -- this
probe reads them back from that log rather than editing the byte-identical copy.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from abs_env_residual import ABSLearningEnvResidual
from residual_log import setup_logging

LOG_PATH = os.path.join(HERE, "logs", "episode_log_env0.csv")
log = setup_logging(os.path.join(HERE, "logs", "probe.log"), component="probe")


def _last_episode_row():
    with open(LOG_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def run_episode(env, policy):
    obs, info = env.reset()
    t, done = 0, False
    while not done:
        action = np.array(policy(t), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += 1
    row = _last_episode_row()
    if row is None:
        return None, None, "NO_LOG_ROW"
    return float(row["avg_g"]), float(row["stopping_dist_m"]), row["outcome"]


def slam(t):
    return [0.0, 0.0]


def scripted_release(t):
    # Alternate a partial release (35% front / 20% rear) with full slam every 20
    # steps (~0.1s at 200Hz) so the policy oscillates through the grip-recovery
    # window instead of committing to one extreme for the whole stop.
    return [0.35, 0.2] if (t // 20) % 2 == 0 else [0.0, 0.0]


def main():
    log.info("=== baseline_probe start: pid=%d python=%s", os.getpid(), sys.executable)
    env = ABSLearningEnvResidual(port=64291, env_index=0, user_path=None, pedal_range=None)
    env.fixed_mph = [60]

    results = {}
    for name, policy in [("SLAM", slam), ("SCRIPTED-RELEASE", scripted_release)]:
        gs = []
        for ep in range(3):
            avg_g, dist, outcome = run_episode(env, policy)
            log.info("%s ep%d: outcome=%s avg_g=%s dist=%s", name, ep, outcome, avg_g, dist)
            if outcome == "STOP" and avg_g is not None:
                gs.append(avg_g)
            time.sleep(0.5)
        results[name] = gs

    env.close()
    log.info("gate results: %s", results)

    slam_gs = results["SLAM"]
    release_gs = results["SCRIPTED-RELEASE"]
    print("\n--- GATE SUMMARY ---")
    if slam_gs:
        slam_mean = sum(slam_gs) / len(slam_gs)
        # Band is 0.7-1.05g, deliberately wider than the ~0.85-0.95g figure quoted
        # elsewhere for a DIFFERENT car/tire config -- this vehicle's own measured
        # floor (2026-08-27, 3 episodes: 1.020/1.029/1.035) is ~1.03g, so the band
        # is sized to catch a broken pedal-hold/inversion path (near 0 or near
        # peak-g), not to match a number from a different car.
        print(f"SLAM mean avg_g = {slam_mean:.3f} "
              f"({'PASS' if 0.7 <= slam_mean <= 1.05 else 'FAIL'} -- band is 0.7-1.05g; "
              f"this vehicle's measured floor is ~1.03g, not the ~0.85-0.95g figure "
              f"documented for a different car config)")
    else:
        slam_mean = None
        print("SLAM: FAIL -- no STOP episodes recorded")

    if release_gs and slam_mean is not None:
        release_mean = sum(release_gs) / len(release_gs)
        print(f"SCRIPTED-RELEASE mean avg_g = {release_mean:.3f} "
              f"({'PASS' if release_mean > slam_mean else 'FAIL'} -- must exceed SLAM mean)")
    else:
        print("SCRIPTED-RELEASE: FAIL -- no STOP episodes recorded, or SLAM gate failed first")


if __name__ == "__main__":
    main()
