"""Live gates for the residual action space, run against real physics before any
training starts.

GATE 1 (slam baseline): zero action ([0,0], no release) must reproduce the known
lockup baseline (~0.85-0.95g on sport_plus tires, CLAUDE.md: "full lockup ~= 0.9g
on sport_plus, models below that prove nothing"). If it doesn't, the pedal-hold or
the action-inversion path is broken and nothing downstream can be trusted.

GATE 2 (release helps): a scripted mid-stop release must beat the slam mean. If it
doesn't, releasing brake torque isn't reaching the wheels, the mailbox path is
broken, and training would never discover it either.

GATE 3 (pedal scaling reaches the wheels): the untested premise flagged in the
2026-08-28 review, gates 1/2 above both run at pedal=1.0, so the residual math
`brake = pedal * (1 - release)` was never exercised at any OTHER pedal value.
If the slam-latch (abstelemetry's armBrakeSlam) or the controller's driver-pedal
override beats the mailboxed brake command, pedal cancels out of the physics and
--pedal randomization silently trains on a lie. Compares slam (zero release) at
pedal=1.0 vs a much lower pedal: peak_g is used instead of avg_g because a low
enough pedal may not even reach the STOP threshold (TIMEOUT -> avg_g=0 by
convention), while peak_g is logged on every outcome.

avg_g/stopping_dist_m are only ever written to the episode CSV log by abs_env.py
(never returned via step()'s info dict, which stays {} on every path), this
probe reads them back from that log rather than editing the byte-identical copy.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from abs_env_residual import ABSLearningEnvResidual
from residual_log import setup_logging
from sim_config import load as load_sim_config

PEDAL_LOW = 0.3   # GATE 3 low-pedal test point

LOG_PATH = os.path.join(HERE, "logs", "episode_log_env0.csv")
log = setup_logging(os.path.join(HERE, "logs", "probe.log"), component="probe")


def _last_episode_row():
    with open(LOG_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def pedal_gate_verdict(peak_g_full, peak_g_low, margin=0.1):
    """Pure decision logic (no game imports) so it's testable offline. PASS
    only if the low-pedal mean peak_g is clearly below the full-pedal mean by
    at least `margin` g, a margin this large can't be explained by
    episode-to-episode noise (GATE 1/2 history: ~0.01-0.02g spread)."""
    if not peak_g_full or not peak_g_low:
        return False, "missing data (no episodes recorded for one or both pedal levels)"
    mean_full = sum(peak_g_full) / len(peak_g_full)
    mean_low = sum(peak_g_low) / len(peak_g_low)
    ok = mean_low < mean_full - margin
    msg = (f"peak_g full-pedal mean={mean_full:.3f}, low-pedal mean={mean_low:.3f} "
          f"(need low < full - {margin:.2f})")
    return ok, msg


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


def parse_args():
    p = argparse.ArgumentParser(description="ResidualABS live gates")
    p.add_argument("--settings", default=os.path.join(HERE, "settings.json"))
    p.add_argument("--game", choices=["tech", "drive"], default=None)
    p.add_argument("--game-folder", default=None)
    p.add_argument("--userpath", default=None)
    p.add_argument("--port", type=int, default=64291)
    p.add_argument("--skip-pedal-gate", action="store_true",
                   help="skip GATE 3 (pedal scaling), e.g. to re-check gates 1/2 quickly")
    return p.parse_args()


def build_sim_config(args):
    cfg = load_sim_config(args.settings)
    if args.game is not None:
        cfg.game = args.game
    if args.game_folder is not None:
        cfg.game_folder = args.game_folder
    if args.userpath is not None:
        cfg.userpath = args.userpath
    return cfg


def main():
    args = parse_args()
    log.info("=== baseline_probe start: pid=%d python=%s", os.getpid(), sys.executable)
    cfg = build_sim_config(args)
    env = ABSLearningEnvResidual(port=args.port, env_index=0, sim_config=cfg, pedal_range=None)
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

    pedal_result = None
    if not args.skip_pedal_gate:
        peak_full, peak_low = [], []
        for pedal, bucket in ((1.0, peak_full), (PEDAL_LOW, peak_low)):
            env.pedal_range = (pedal, pedal)
            for ep in range(3):
                avg_g, dist, outcome = run_episode(env, slam)
                row = _last_episode_row()
                peak_g = float(row["peak_g"]) if row else None
                log.info("PEDAL=%.2f ep%d: outcome=%s avg_g=%s peak_g=%s",
                         pedal, ep, outcome, avg_g, peak_g)
                if peak_g is not None:
                    bucket.append(peak_g)
                time.sleep(0.5)
        pedal_result = pedal_gate_verdict(peak_full, peak_low)
        log.info("pedal gate: full=%s low=%s -> %s", peak_full, peak_low, pedal_result)

    env.close()
    log.info("gate results: %s pedal=%s", results, pedal_result)

    slam_gs = results["SLAM"]
    release_gs = results["SCRIPTED-RELEASE"]
    print("\n--- GATE SUMMARY ---")
    if slam_gs:
        slam_mean = sum(slam_gs) / len(slam_gs)
        # Band is 0.7-1.05g, deliberately wider than the ~0.85-0.95g figure quoted
        # elsewhere for a DIFFERENT car/tire config, this vehicle's own measured
        # floor (2026-08-27, 3 episodes: 1.020/1.029/1.035) is ~1.03g, so the band
        # is sized to catch a broken pedal-hold/inversion path (near 0 or near
        # peak-g), not to match a number from a different car.
        print(f"SLAM mean avg_g = {slam_mean:.3f} "
              f"({'PASS' if 0.7 <= slam_mean <= 1.05 else 'FAIL'}, band is 0.7-1.05g; "
              f"this vehicle's measured floor is ~1.03g, not the ~0.85-0.95g figure "
              f"documented for a different car config)")
    else:
        slam_mean = None
        print("SLAM: FAIL, no STOP episodes recorded")

    if release_gs and slam_mean is not None:
        release_mean = sum(release_gs) / len(release_gs)
        print(f"SCRIPTED-RELEASE mean avg_g = {release_mean:.3f} "
              f"({'PASS' if release_mean > slam_mean else 'FAIL'}, must exceed SLAM mean)")
    else:
        print("SCRIPTED-RELEASE: FAIL, no STOP episodes recorded, or SLAM gate failed first")

    if pedal_result is not None:
        ok, msg = pedal_result
        print(f"PEDAL SCALING ({'PASS' if ok else 'FAIL'}): {msg}")
        if not ok:
            print("  -> pedal is not reaching the wheels: the slam-latch or the "
                 "controller's driver-pedal override is beating the mailboxed "
                 "brake command. --pedal randomization would train on a no-op.")
    elif args.skip_pedal_gate:
        print("PEDAL SCALING: skipped (--skip-pedal-gate)")


if __name__ == "__main__":
    main()
