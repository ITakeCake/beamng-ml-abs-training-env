"""Set the fps limiter OFF at RUNTIME (settings.setValue), read it back to
prove it took, then time step(1). Editing settings.json offline never proved
the running process honoured it."""
import statistics, time, sys
from beamngpy import BeamNGpy, Scenario, Vehicle
import sim_config

cfg = sim_config.load("settings.json")
bng = BeamNGpy("localhost", 64299, home=cfg.game_folder,
               user=sim_config.resolved_userpath(cfg))
try:
    bng.open(launch=False); print("attached")
except Exception:
    bng.open(None, "-headless", "-gfx", "null", "-no-sound", launch=True)
    print("launched")

veh = Vehicle("probe", model="etk800")
sc = Scenario("smallgrid", "fps_test")
sc.add_vehicle(veh, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
sc.make(bng); bng.scenario.load(sc); bng.scenario.start()
bng.control.pause(); bng.settings.set_deterministic(200)

def read_back(tag):
    q = ("local s=settings.getValue('fpsLimitEnabled'); "
         "local b=settings.getValue('fpsLimitBackgroundEnabled'); "
         "local l=settings.getValue('fpsLimit'); "
         "log('E','FPSTEST', tostring(s)..'|'..tostring(b)..'|'..tostring(l))")
    bng.control.queue_lua_command(q)
    print(f"  [{tag}] queued read-back -> see BeamNG log")

def timeit(label, reps=120):
    for _ in range(5): bng.control.step(1)
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter(); bng.control.step(1)
        xs.append((time.perf_counter()-t0)*1000)
    xs.sort()
    print(f"  {label:<34} p50={xs[len(xs)//2]:7.2f} ms   min={xs[0]:6.2f}  "
          f"-> {1000/xs[len(xs)//2]:5.1f} Hz")
    return xs[len(xs)//2]

print("\n-- limiter as found --")
read_back("before")
before = timeit("step(1) limiter AS FOUND")

print("\n-- disabling BOTH at runtime --")
bng.control.queue_lua_command(
    "settings.setValue('fpsLimitEnabled', false); "
    "settings.setValue('fpsLimitBackgroundEnabled', false); "
    "settings.setValue('fpsLimit', 2000)")
time.sleep(1.0)
read_back("after")
after = timeit("step(1) limiter OFF (runtime)")

print(f"\nchange: {before:.2f} -> {after:.2f} ms  ({before/after:.2f}x)")
try: bng.close()
except Exception: pass
