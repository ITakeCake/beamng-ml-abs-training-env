"""One episode, every stage of the brake-event state machine logged.

PPO-12 ran at 259 steps/s and scored zero: every episode TIMEOUT, dist=0.0 m,
avg_g=0, while peak_g read 1.4-1.6. So the car brakes and the MEASUREMENT does
not finish -- which of those stages fails is the question, and guessing at it
produced one wrong explanation already.

Stages checked, in order:

  1. armBrakeSlam acknowledged      (tel_slam_armed)
  2. slam fired / event started     (tel_slam_fired, tel_accum_active)
  3. sim time advances 5 ms/step    (unique timestamps == steps taken)
  4. speed actually falls           (monotone-ish decrease in tel_inst_speed)
  5. stop threshold crossed         (speed <= 1.0 m/s)
  6. event finalises                (tel_last_brake_dist > 0, avg_g > 0)

Plus the staleness check: how many DISTINCT observations arrive over N steps.
N steps yielding far fewer distinct readings means the trainer is consuming
repeated telemetry, which would explain a stop that is never detected without
any of the above being individually broken.

    python stage_probe.py --uncap both --mph 80
"""
import argparse
import time

import residual_log
import sim_clock
from abs_env_residual import ABSLearningEnvResidual
from sim_config import load as load_cfg

log = residual_log.setup_logging("logs/stage_probe.log", "stage")

CHANNELS = ("tel_inst_speed", "airspeed", "tel_slam_armed", "tel_slam_fired",
            "tel_last_brake_dist", "tel_last_brake_avg_g_arc", "mlabs_tickseq",
            "mlabs_active")


def snapshot(env):
    e = env.vehicle.sensors["electrics"]
    return {k: e.get(k) for k in CHANNELS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mph", type=int, default=80)
    ap.add_argument("--uncap", choices=["off", "on"], default="on",
                    help="off = leave BeamNG's frame limiter alone (the old, "
                         "working-but-slow behaviour) as the control")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--port", type=int, default=64291)
    args = ap.parse_args()

    cfg = load_cfg("settings.json")
    if args.uncap == "off":
        # Neutralise the uncap for this run so the two can be compared without
        # any other difference. This is the control arm.
        sim_clock.uncap_frame_rate = lambda bng, verify=True: "SKIPPED"
        sim_clock.uncap_and_verify = lambda bng, log=None: None

    env = ABSLearningEnvResidual(
        port=args.port, env_index=0, sim_config=cfg,
        vehicle_pc="vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc",
        deterministic=True)
    env.fixed_mph = [args.mph]

    log.info("=== stage probe: mph=%d uncap=%s ===", args.mph, args.uncap)
    t0 = time.monotonic()
    env.reset()
    log.info("STAGE 1/2 after reset: %s", snapshot(env))

    seen_speed, seen_tick, rows = set(), set(), []
    fired_at = armed_at = stopped_at = None
    for i in range(args.steps):
        env.step([0.0, 0.0])                      # pure slam, the simplest case
        s = snapshot(env)
        rows.append(s)
        seen_speed.add(s["tel_inst_speed"])
        seen_tick.add(s["mlabs_tickseq"])
        if armed_at is None and s.get("tel_slam_armed"):
            armed_at = i
        if fired_at is None and s.get("tel_slam_fired"):
            fired_at = i
            log.info("STAGE 2 slam FIRED at step %d: %s", i, s)
        spd = float(s.get("tel_inst_speed") or 99)
        if stopped_at is None and spd <= 1.0:
            stopped_at = i
            log.info("STAGE 5 stop threshold crossed at step %d: %s", i, s)
            break
    wall = time.monotonic() - t0

    n = len(rows)
    log.info("--- results over %d steps (%.1fs wall, %.1f steps/s) ---",
             n, wall, n / max(wall, 1e-9))
    log.info("STAGE 1 armed at step: %s", armed_at)
    log.info("STAGE 2 fired at step: %s", fired_at)
    log.info("STAGE 3/staleness: %d distinct tel_inst_speed, %d distinct "
             "mlabs_tickseq, over %d steps", len(seen_speed), len(seen_tick), n)
    if len(seen_tick) < n * 0.9:
        log.error("STALE TELEMETRY: %d distinct controller ticks for %d python "
                  "steps -- the env is reading the same publication more than "
                  "once, so it cannot see the car stop.", len(seen_tick), n)
    speeds = [float(r.get("tel_inst_speed") or 0) for r in rows]
    log.info("STAGE 4 speed: first=%.2f last=%.2f min=%.2f",
             speeds[0], speeds[-1], min(speeds))
    log.info("STAGE 5 stopped at step: %s", stopped_at)
    last = rows[-1]
    log.info("STAGE 6 finalise: dist=%s avg_g=%s",
             last.get("tel_last_brake_dist"), last.get("tel_last_brake_avg_g_arc"))

    ok = (fired_at is not None and stopped_at is not None
          and float(last.get("tel_last_brake_dist") or 0) > 0)
    log.info("VERDICT: %s", "PASS" if ok else "FAIL -- see the first stage above that is None/0")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
