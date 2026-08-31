"""Where does a training step's wall clock actually go?

A deterministic step costs ~62 ms to buy 5 ms of simulation. That is attributed
to "three TCP round trips", but round trip is a guess, not a measurement: a
localhost handshake has no business costing 20 ms, and if the cost is really in
the engine's per-step work then no amount of protocol tuning will help.

This times each operation the env performs, separately, against its OWN BeamNG
instance on its own port -- a live training run is left alone.

    python latency_probe.py --port 64299 --reps 200

The interesting comparison is not the absolute numbers but their ratio:

  * If a bare socket ping is ~0.1 ms and step() is 20 ms, the time is inside
    BeamNG (per-step engine work, or its socket handler waking up), and the fix
    is fewer/larger requests -- action repeat, batched steps, in-Lua rollout.
  * If the bare ping is itself ~20 ms, the time is transport (Nagle/delayed
    ACK, loopback path) and the fix is socket configuration.
"""
import argparse
import socket
import statistics
import time

from beamngpy import BeamNGpy, Scenario, Vehicle

import sim_config


def _stats(name, samples_ms, note=""):
    s = sorted(samples_ms)
    n = len(s)
    return {
        "name": name, "n": n,
        "mean": statistics.fmean(s),
        "p50": s[n // 2],
        "p90": s[int(n * 0.9)],
        "min": s[0], "max": s[-1],
        "note": note,
    }


def _time(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def raw_socket_ping(host, port, reps):
    """Transport floor: connect to the same port and time a send/recv pair.

    BeamNG will reject the payload, but a rejection still travels the whole
    path, which is exactly what is being measured. Failure here is not fatal --
    it just means this line is unavailable.
    """
    try:
        skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        skt.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        skt.settimeout(2.0)
        skt.connect((host, port))
    except OSError as e:
        return None, f"unavailable: {type(e).__name__}"
    samples = []
    try:
        for _ in range(min(reps, 50)):
            t0 = time.perf_counter()
            skt.sendall(b"\x00\x00\x00\x04ping")
            try:
                skt.recv(4096)
            except socket.timeout:
                pass
            samples.append((time.perf_counter() - t0) * 1000.0)
    except OSError:
        pass
    finally:
        skt.close()
    return (samples or None), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=64299)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--settings", default="settings.json")
    args = ap.parse_args()

    cfg = sim_config.load(args.settings)
    home = cfg.game_folder
    user = sim_config.resolved_userpath(cfg)
    print(f"home={home}\nuser={user}\nport={args.port} reps={args.reps}\n")

    bng = BeamNGpy("localhost", args.port, home=home, user=user)
    # Attach to an instance already listening before launching a second one:
    # two instances cannot share a userpath (the launcher fails to rotate its
    # own log and dies), and a freshly stopped training run leaves a perfectly
    # good instance behind.
    try:
        bng.open(launch=False)
        print(f"attached to existing BeamNG on port {args.port}.")
    except Exception:
        bng.open(None, "-headless", "-gfx", "null", "-no-sound", launch=True)
        print(f"launched a new BeamNG on port {args.port}.")

    veh = Vehicle("probe", model="etk800")
    sc = Scenario("smallgrid", "latency_probe")
    sc.add_vehicle(veh, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
    sc.make(bng)
    bng.scenario.load(sc)
    bng.scenario.start()
    bng.control.pause()
    bng.settings.set_deterministic(200)
    print("scenario up, deterministic 200Hz.\n")

    rows = []

    ping, note = raw_socket_ping("localhost", args.port, args.reps)
    if ping:
        rows.append(_stats("raw socket send+recv", ping,
                           "transport floor, no engine work"))
    else:
        rows.append({"name": "raw socket send+recv", "n": 0, "mean": float("nan"),
                     "p50": float("nan"), "p90": float("nan"), "min": float("nan"),
                     "max": float("nan"), "note": note})

    rows.append(_stats("bng.control.step(1)",
                       _time(lambda: bng.control.step(1), args.reps),
                       "1 physics tick = 5 ms of sim"))
    rows.append(_stats("bng.control.step(20)",
                       _time(lambda: bng.control.step(20), max(20, args.reps // 4)),
                       "20 ticks = 100 ms of sim"))
    rows.append(_stats("vehicle.sensors.poll()",
                       _time(lambda: veh.sensors.poll(), args.reps),
                       "reads electrics"))
    rows.append(_stats("vehicle.control(brake=1)",
                       _time(lambda: veh.control(brake=1.0), args.reps),
                       "the call the env makes every step"))
    rows.append(_stats("vehicle.queue_lua_command",
                       _time(lambda: veh.queue_lua_command("local _ = 1"),
                             args.reps),
                       "fire and forget?"))
    rows.append(_stats("full env step (3 calls)",
                       _time(lambda: (veh.control(brake=1.0),
                                      bng.control.step(1),
                                      veh.sensors.poll()), args.reps),
                       "what training actually pays"))

    # techCore.lua:559-568 returns early from onPreRender while a step is
    # blocking, but on the frame the step COMPLETES it falls through to
    # "while tcom.checkMessages do end". So a request queued during the step is
    # drained on that same frame -- if Python did not stop to wait for the
    # step's own ACK first. wait=False is exactly that.
    def pipelined():
        veh.control(brake=1.0)
        bng.control.step(1, wait=False)
        veh.sensors.poll()
    rows.append(_stats("pipelined step (wait=False)",
                       _time(pipelined, args.reps),
                       "same 3 calls, no ACK wait on step"))

    print(f"{'operation':<28}{'n':>5}{'mean':>9}{'p50':>9}{'p90':>9}"
          f"{'min':>9}{'max':>9}   note")
    print("-" * 110)
    for r in rows:
        print(f"{r['name']:<28}{r['n']:>5}{r['mean']:>9.2f}{r['p50']:>9.2f}"
              f"{r['p90']:>9.2f}{r['min']:>9.2f}{r['max']:>9.2f}   {r['note']}")

    step1 = next((r for r in rows if r["name"] == "bng.control.step(1)"), None)
    step20 = next((r for r in rows if r["name"] == "bng.control.step(20)"), None)
    if step1 and step20 and step1["p50"] > 0:
        # If 20 ticks cost about the same as 1, the per-call overhead dominates
        # and batching is the whole answer. If they cost 20x, the engine's
        # per-tick work is the cost and batching buys nothing.
        ratio = step20["p50"] / step1["p50"]
        print(f"\nstep(20)/step(1) = {ratio:.2f}x")
        if ratio < 3:
            print("  -> per-CALL overhead dominates. Batching ticks per action "
                  "(action repeat) converts almost 1:1 into throughput.")
        elif ratio > 12:
            print("  -> per-TICK engine work dominates. Batching buys little; "
                  "the sim itself is the floor.")
        else:
            print("  -> mixed: both per-call overhead and per-tick work matter.")

    try:
        bng.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
