"""
ABS Learning Environment, standalone module for single or parallel training.
Accepts port, env_index, and user_path for multi-instance support.
"""
import gymnasium as gym
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
import time
import os
import math
import random
import csv
import json

import sim_config

# --- CONFIGURATION (BeamNG.tech, headless / no-graphics) ---
BNG_HOME = sim_config.autodetect_game_folder("tech") or ""

MAP_NAME   = 'smallgrid'                                 # flat gridmap
START_POS  = (0, 0, 0.5)
START_ROT  = (0, 0, 0)
VEHICLE_PC = 'vehicles/etk800/Machine-Trainer-Boy.pc'   # custom no-ABS etk800 (etk_DSE="")
HEADLESS   = True                                        # -headless -gfx null (no window, no GPU)

# --- ENVIRONMENT CONFIG ---
DETERM_HZ  = 200    # ML braking-control rate (changed 250 -> 200)
FRAME_SKIP = 1
MIN_MPH = 25
MAX_MPH = 100

# --- LSTM DATA RECORDING ---
# run EXACTLY ONE recorder, the 99-col 2kHz one.
#   RECORD_LSTM=True      -> lstm2khz.lua, 99-col raw 2kHz format -> recorded_data_2khz/SAC/
RECORD_LSTM     = True
RECORD_SAC_DATA = False
LSTM_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'SpeedLSTM', 'recorded_data_2khz', 'SAC'))
LSTM_LUA_SRC  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'SpeedLSTM', 'lstm2khz.lua')

# --- REWARD CONFIG (v5.0 2026-05-08: 33/33/33 split at "okayish" reference) ---
# Three positive sources sized to ~equal contribution at reference episode (1.05g, perfect yaw, ~900 steps):
#   1) Per-step g-force      ~1000 contribution
GATEKEEPER_G        = 1.06       # threshold above which the step jump + quadratic fire
TERMINAL_STEP_BONUS = 500.0      # instant jump at the gatekeeper crossing
TERMINAL_QUAD_K     = 200000.0   # quadratic coefficient for explosive growth above gatekeeper
TERMINAL_RAMP_NEG_K = 2000.0     # below 0.5g ramp slope (per-g): -200 at 0.4g, -400 at 0.3g
TERMINAL_RAMP_POS_K = 1818.18    # above 0.5g ramp slope (1000/0.55): 0 at 0.5g, +1000 at 1.05g
PER_STEP_K          = 0.0011     # per-step g reward = K × terminal_shape(g_step) ⇒ ~1000 over 900 steps
PER_STEP_G_GATE     = 0.30       # below this gy_avg per step, NO per-step yaw bonus (anti-coast)

# Yaw rewards (NEW v5.0). Metric: cumulative integral ∫|yaw_rate|·dt < 0.1 rad ⇒ "perfect"
YAW_BONUS_K_STEP    = 1.0        # per-step Gaussian bonus, max at yaw_rate=0
YAW_BONUS_ALPHA     = 20.0       # Gaussian decay: yaw_rate=0.05 rad/s ⇒ ~37% of max
YAW_BONUS_K_TERMINAL = 300.0     # terminal one-shot if cumulative integral stays under threshold
YAW_BONUS_THRESHOLD  = 0.1       # cumulative |yaw_rate|·dt under this ⇒ full bonus, scales linearly to 0
YAW_PEN_K_TERMINAL   = 5000.0    # KEPT: terminal accumulator (catastrophic-yaw backstop, on excess²)
YAW_RATE_DEADZONE_RAD_S = 0.05   # only excess above this counts toward catastrophic accumulator
# REMOVED v5.0: YAW_PEN_K_STEP (replaced by positive yaw bonus, cleaner inverse signal)

CRASH_HEADING       = 1.571      # ~90° heading deviation = crash terminate
SLIP_THRESHOLD      = 0.3        # legacy, UNUSED (slip reward removed)
SLIP_REWARD_K       = 0.5        # legacy, UNUSED

# --- CRASH DETECTION ---
MAX_EPISODE_STEPS = 5000  # 25s at 200Hz (comment fixed 2026-07-11; was stale "20s at 250Hz")
STOP_FRAMES       = 15    # frames at <0.05m/s before terminal (75ms at 200Hz, fast stop detection)
CRASH_PENALTY     = -2000.0  # bumped from -500: scaled to v5.0 reward magnitudes


def _terminal_g_shape(g):
    """v5.0 terminal g-force reward shape (used for both per-step and terminal).
    Anchored at: -400 (0.3g), 0 (0.5g), +1000 (1.05g), +1500 (1.06g jump),
    then quadratic growth above. Open-ended past 2g (no cap).
    """
    g = max(0.0, min(g, 2.5))  # safety clamp
    if g < 0.5:
        ramp = -TERMINAL_RAMP_NEG_K * (0.5 - g)
    else:
        ramp = TERMINAL_RAMP_POS_K * (g - 0.5)
    extra = 0.0
    if g >= GATEKEEPER_G:
        d = g - GATEKEEPER_G
        extra = TERMINAL_STEP_BONUS + TERMINAL_QUAD_K * d * d
    return ramp + extra


def get_quat(x, y, z):
    roll = math.radians(x)
    pitch = math.radians(y)
    yaw = math.radians(z)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


class ABSLearningEnv(gym.Env):
    """
    4-wheel independent brake control ABS environment.
    Supports multi-instance via port, env_index, and user_path parameters.
    """

    def __init__(self, port=64256, env_index=0, user_path=None, bng_home=BNG_HOME,
                 fixed_mph=None):
        super().__init__()
        self.env_index = env_index
        self.port = port
        self.fixed_mph = fixed_mph  # None = random, set a number to lock speed
        self._prefix = f"[ENV {env_index}]"

        self._user_path = user_path
        kwargs = dict(host='localhost', port=port, home=bng_home)
        if user_path:
            kwargs['user'] = user_path

        self.bng = BeamNGpy(**kwargs)
        # Headless / no-graphics launch (.tech): CLI flags go as positional *args
        # (NOT extra_opts). Proven pattern from lstm_data_env.py:378.
        #   -headless : no window     -gfx null : no GPU rendering     -no-sound : no audio
        try:
            self.bng.open(launch=False)
            print(f"{self._prefix} Connected to existing BeamNG on port {port}.")
        except Exception:
            if HEADLESS:
                self.bng.open(None, '-headless', '-gfx', 'null', '-no-sound', launch=True)
                print(f"{self._prefix} Launched BeamNG.tech HEADLESS (-gfx null) on port {port}.")
            else:
                self.bng.open(None, '-no-sound', launch=True)
                print(f"{self._prefix} Launched BeamNG.tech (windowed) on port {port}.")

        # Install .tech assets (telemetry + LSTM recorder + custom car) into the live
        # userpath so extensions.load(...) and partConfig spawn can resolve them.
        self._install_tech_assets()

        # speedup: Python on P-core 0 (cores 0,1), BeamNG on remaining (cores 2-15)
        # 12600K layout: 0-11 = 6 P-cores w/ HT, 12-15 = 4 E-cores
        # TCP_NODELAY already set by beamngpy 1.35 internally, no extra work needed.
        self._apply_performance_tuning(
            python_cores=[0, 1], beamng_cores=list(range(2, 16)))

        self.vehicle = Vehicle(f'ego_{env_index}', model='etk800',
                               partConfig=VEHICLE_PC)

        unique_name = f'abs_{env_index}_{random.randint(10000, 99999)}'
        self.scenario = Scenario(MAP_NAME, unique_name)

        self.start_quat = get_quat(*START_ROT)
        self.scenario.add_vehicle(self.vehicle, pos=START_POS, rot_quat=self.start_quat)
        self.scenario.make(self.bng)

        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()

        # EXPERIMENTAL: set_velocity did not actually move the vehicle on BeamNG.drive
        # consumer (API call accepted but had no physical effect). Disabled for now.
        # try:

        # (TCP_NODELAY already set by beamngpy internally, no extra work needed)

        # .tech: set freeroam gamestate so vehicle controls actually take effect,
        # then hide the UI overlay.
        self.bng.control.queue_lua_command(
            'core_gamestate.setGameState("scenario", "freeroam", "freeroam")')
        self.bng.control.queue_lua_command('ui_visibility.set(false)')


        # Let sim run LIVE (not paused) for vehicle to fully initialize
        # This is critical on fast-loading maps like smallgrid
        time.sleep(3)

        self.vehicle.sensors.attach('electrics', Electrics())
        self.vehicle.switch()
        self.vehicle.focus()

        # Load telemetry while still running live
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        if RECORD_LSTM:
            # Second extension: 2kHz LSTM recorder (coexists with abstelemetry,
            # both run their own onPhysicsStep). Recording armed per-episode in reset().
            self.vehicle.queue_lua_command("extensions.load('lstm2khz')")
            os.makedirs(LSTM_DATA_DIR, exist_ok=True)
            self._lstm_ep = 0
        time.sleep(3)

        # NOW pause and set deterministic
        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        print(f"{self._prefix} set_deterministic({DETERM_HZ}) called.")
        self.vehicle.control(gear=2)
        self.bng.step(30)

        # Verify telemetry
        tel_status = 'NOT_FOUND'
        for attempt in range(10):
            self.vehicle.sensors.poll()
            e = self.vehicle.sensors['electrics']
            tel_status = e.get('tel_status', 'NOT_FOUND')
            if 'OK' in str(tel_status):
                break
            self.bng.step(30)
        print(f"{self._prefix} Telemetry: {tel_status}")
        if 'OK' not in str(tel_status):
            raise RuntimeError(
                f"{self._prefix} Telemetry failed: {tel_status}\n"
                ">>> KNOWN BUG: BeamNG shows a menu/popup on fresh launch that blocks\n"
                ">>> vehicle init. Dismiss the menu in the BeamNG window, then retry.\n"
                ">>> If restarting overnight, connect to existing instances (launch=False)\n"
                ">>> instead of launching new ones."
            )

        # Disable ABS via official API (MLTrainerVBoy has no DSE, but safety net)
        self.vehicle.queue_lua_command('wheels.setABSBehavior("off")')
        self.bng.step(5)

        # Action: 4-wheel brakes only (speed-guesser removed per Rule 2, only g-forces in reward)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        # Obs: 31 − 4 (brk_actual_* dropped: physics cheat, redundant w/ prev_brakes) = 27
        # Per-sensor sanity bounds applied via _clip_obs() before returning.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        # Sanity bounds for each obs dim, clipping prevents NaN/inf/garbage from poisoning the policy.
        # Order matches the obs vector layout below. Wide-but-finite ranges; anything outside = sensor fault.
        self._OBS_LOW = np.array([
            0.0, 0.0, 0.0, 0.0,                         # 0-3:  wheel speeds (m/s)
            -50.0, -50.0, -10.0,                        # 4-6:  gy_avg, lateral_g, yaw_rate
            0.0, 0.0, 0.0, 0.0,                         # 7-10: prev_brakes (env enforces [0.05, 1])
            -50.0, -50.0,                               # 11-12: gy_min, gy_max
            0.0, -2.0, -720.0,                          # 13-15: rpm, gear, steering (PPO_V2: measured wheel angle, deg)
            -math.pi, -math.pi,                         # 16-17: pitch, roll (rad)
            0.0, 0.0,                                   # 18-19: input_brake, input_throttle
            -50.0, -10.0, -10.0,                        # 20-22: gz, pitch_rate, roll_rate
            -200.0, -200.0, -200.0, -200.0,             # 23-26: wheel accels (m/s²)
        ], dtype=np.float32)
        self._OBS_HIGH = np.array([
            100.0, 100.0, 100.0, 100.0,                 # 0-3:  wheel speeds (~224mph max)
            50.0, 50.0, 10.0,                           # 4-6:  ~5g longitudinal/lateral, ~570°/s yaw
            1.0, 1.0, 1.0, 1.0,                         # 7-10: prev_brakes
            50.0, 50.0,                                 # 11-12: gy_min, gy_max
            15000.0, 12.0, 720.0,                       # 13-15: rpm, gear, steering (PPO_V2: measured wheel angle, deg)
            math.pi, math.pi,                           # 16-17: pitch, roll
            1.0, 1.0,                                   # 18-19: input_brake, input_throttle
            50.0, 10.0, 10.0,                           # 20-22: gz, pitch/roll rates
            200.0, 200.0, 200.0, 200.0,                 # 23-26: wheel accels (~20g spike protection)
        ], dtype=np.float32)

        self.steps_taken = 0
        self.target_mph = 0.0
        self.stop_timer = 0
        self.prev_brakes = np.zeros(4, dtype=np.float32)
        self.episode_count = 0
        self.ep_step_g_sum = 0.0
        self.ep_yaw_penalty = 0.0
        self.start_heading = 0.0     # captured at episode start from physics engine
        self.current_heading = 0.0   # updated each step from physics engine
        self.target_heading = 0.0    # TODO: driven by steering input, 0 for now
        self.target_yaw_rate = 0.0   # NEW v5.0: requested yaw rate (rad/s). Currently
                                     # always 0 (steering locked). When steering is wired,
                                     # compute via bicycle model: v·tan(δ)/wheelbase.
        self.prev_heading_error = 0.0  # for recovery reward
        self.start_speed_ms = 0.0
        self._last_gps_speed = 999.0
        self._reset_mode = 1
        self._mode3_throttle_countdown = 0

        # State for derivative-based features (pitch/roll/wheel rates).
        # First step after reset → all rates = 0 (no prior sample yet).
        self._dt = 1.0 / DETERM_HZ  # game time per Python step: 1/200 = 5 ms (exactly 200Hz)
        self._prev_ws = np.zeros(4, dtype=np.float32)
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._has_prev_state = False         # set True after first build_obs call

        # Per-episode stats tracking
        self.ep_peak_g = 0.0
        self.ep_peak_slip = 0.0
        self.ep_max_yaw = 0.0
        self.ep_stopping_dist = 0.0
        self.ep_yaw_sq_sum = 0.0    # ∫ (excess yaw_rate above deadzone)² dt, catastrophic backstop
        self.ep_yaw_abs_sum = 0.0   # NEW v5.0: ∫|yaw_rate|·dt, matches "perfect yaw < 0.1" metric
        self._ep_wall_start = time.monotonic()

        # CSV episode log
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"episode_log_env{env_index}.csv")
        file_exists = os.path.exists(log_path)
        self._csv_file = open(log_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow([
                'episode', 'target_mph', 'start_speed_ms', 'steps', 'stop_time_s',
                'stopping_dist_m', 'stopping_dist_ft', 'avg_g', 'peak_g',
                'peak_slip', 'max_yaw_rate', 'outcome',
                'reward_step_g', 'reward_yaw',
                'reward_terminal',
                'reward_total', 'wall_clock_s',
            ])
        print(f"{self._prefix} Episode log: {log_path}")

        # Speed prediction log
        spd_log_path = os.path.join(log_dir, f"speed_prediction_env{env_index}.csv")
        spd_exists = os.path.exists(spd_log_path)
        self._spd_file = open(spd_log_path, 'a', newline='')
        self._spd_writer = csv.writer(self._spd_file)
        if not spd_exists:
            self._spd_writer.writerow([
                'episode', 'step', 'predicted_ms', 'actual_ms',
                'error_ms', 'error_pct', 'yaw_rate', 'skipped',
            ])
        self._spd_ep_errors = []  # collect per-episode for summary

        # Detailed per-step audit log (toggled via audit_log param)
        self._audit_writer = None
        self._audit_file = None

        # ─── LSTM training data export (27-col SAC-Data, OPTIONAL) ────
        # DISABLED by default only ONE recorder may run,
        # and it's the 99-col lstm2khz layer (RECORD_LSTM above). Flip
        self._lstm_file = None
        self._lstm_writer = None
        self._lstm_frame = 0   # per-episode frame counter (resets on new episode)
        if RECORD_SAC_DATA:
            lstm_dir = os.path.normpath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'SpeedLSTM', 'SAC-Data'))
            os.makedirs(lstm_dir, exist_ok=True)
            run_ts = time.strftime('%Y-%m-%d_%H%M%S')
            self._lstm_run_id = run_ts
            self._lstm_vehicle_model = self.vehicle.model
            lstm_path = os.path.join(
                lstm_dir, f"sac_session_{run_ts}_{self._lstm_vehicle_model}_env{env_index}.csv")
            self._lstm_file = open(lstm_path, 'w', newline='')
            self._lstm_writer = csv.writer(self._lstm_file)
            self._lstm_writer.writerow([
                'timestamp', 'session_id', 'vehicle_id', 'vehicle_model', 'frame',
                'ws_fr', 'ws_fl', 'ws_rr', 'ws_rl',
                'gy_avg', 'gy_min', 'gy_max',
                'gx_avg', 'gx_min', 'gx_max',
                'gz_avg', 'gz_min', 'gz_max',
                'yaw_avg', 'steering', 'rpm', 'gear',
                'driver_brake', 'driver_throttle',
                'airspeed', 'pitch', 'roll',
            ])
            print(f"{self._prefix} LSTM data: {lstm_path}")
        else:
            print(f"{self._prefix} 27-col SAC-Data export OFF; 99-col 2kHz recorder "
                  f"{'ON -> ' + LSTM_DATA_DIR if RECORD_LSTM else 'ALSO OFF (no LSTM recording!)'}")

    def _install_tech_assets(self):
        """Copy telemetry + LSTM recorder + custom car into the live .tech userpath so
        extensions.load(...) and partConfig spawns resolve. Mirrors lstm_data_env.py."""
        import shutil
        from pathlib import Path
        try:
            paths = self.bng.system.get_environment_paths()
            user_root = Path(paths["user"])
        except Exception:
            user_root = (Path(self._user_path) if self._user_path
                         else Path.home() / "AppData" / "Local" / "BeamNG.tech" / "current")
        self._lstm_userdata_dir = user_root   # where vlua writes 2kHz CSVs (sandbox-relative)
        ext_dir = user_root / "lua" / "vehicle" / "extensions"
        veh_dir = user_root / "vehicles" / "etk800"
        ext_dir.mkdir(parents=True, exist_ok=True)
        veh_dir.mkdir(parents=True, exist_ok=True)
        here = os.path.dirname(os.path.abspath(__file__))
        # abstelemetry.lua, telemetry bridge (required)
        src_tel = os.path.join(here, "abstelemetry.lua")
        if os.path.exists(src_tel):
            shutil.copy2(src_tel, ext_dir / "abstelemetry.lua")
        # lstm2khz.lua, 2kHz recorder (optional)
        if RECORD_LSTM and os.path.exists(LSTM_LUA_SRC):
            shutil.copy2(LSTM_LUA_SRC, ext_dir / "lstm2khz.lua")
        # Machine-Trainer-Boy.pc, custom no-ABS car
        src_car = os.path.join(here, "Machine-Trainer-Boy.pc")
        if os.path.exists(src_car):
            shutil.copy2(src_car, veh_dir / "Machine-Trainer-Boy.pc")
        print(f"{self._prefix} Installed .tech assets -> {user_root}")

    def _finish_lstm(self):
        """Stop the 2kHz recorder and move the CSV out of the .tech userdata dir.
        Fully guarded, never raises into the training loop."""
        if not (RECORD_LSTM and getattr(self, '_lstm_recording', False)):
            return
        self._lstm_recording = False
        try:
            self.vehicle.queue_lua_command(
                "if extensions.lstm2khz then extensions.lstm2khz.stopRecording() end")
            self.bng.step(5)
            import shutil
            src = self._lstm_userdata_dir / self._lstm_fname
            if src.exists():
                shutil.move(str(src), os.path.join(LSTM_DATA_DIR, self._lstm_fname))
        except Exception as _e:
            print(f"{self._prefix} LSTM stop/move failed (continuing): {_e}")

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._lstm_frame = 0
        self.stop_timer = 0
        self.prev_brakes = np.zeros(4, dtype=np.float32)
        self.ep_step_g_sum = 0.0
        self.ep_yaw_penalty = 0.0
        self.start_heading = 0.0
        self.current_heading = 0.0
        self.target_heading = 0.0
        self.target_yaw_rate = 0.0   # reset to 0 each episode (no steering yet)
        self.prev_heading_error = 0.0
        self._last_gps_speed = 999.0
        self._spd_ep_errors = []
        self.ep_peak_g = 0.0
        self.ep_peak_slip = 0.0
        self.ep_max_yaw = 0.0
        self.ep_stopping_dist = 0.0
        self.ep_yaw_sq_sum = 0.0    # accumulated above-deadzone yaw_rate² × dt (catastrophic backstop)
        self.ep_yaw_abs_sum = 0.0   # NEW v5.0: cumulative ∫|yaw_rate|·dt (matches user's perfect-yaw metric)

        # Reset derivative-state so first step's rates = 0
        self._prev_ws = np.zeros(4, dtype=np.float32)
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._has_prev_state = False

        # fixed_mph supports None (full random), a single number (locked), or a list (random pick from list)
        if self.fixed_mph is None:
            self.target_mph = round(random.uniform(MIN_MPH, MAX_MPH))
        elif isinstance(self.fixed_mph, (list, tuple)):
            self.target_mph = random.choice(self.fixed_mph)
        else:
            self.target_mph = self.fixed_mph
        target_ms = self.target_mph * 0.44704

        self.vehicle.teleport(pos=START_POS, rot_quat=self.start_quat, reset=True)
        self.vehicle.focus()

        # Setup: extension + brakes zero + ABS off
        self.vehicle.control(throttle=0, steering=0, gear=0, parkingbrake=0, brake=0)
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        if RECORD_LSTM:
            # teleport(reset=True) unloads vehicle extensions, reload the 2kHz
            # recorder beside abstelemetry (it gets verified at the arm point).
            self.vehicle.queue_lua_command("extensions.load('lstm2khz')")
        self.vehicle.queue_lua_command("extensions.abstelemetry.setBrakes(0,0,0,0)")
        self.bng.step(15)
        self.vehicle.queue_lua_command('wheels.setABSBehavior("off")')
        self.bng.step(2)

        # --- Pick reset mode: 50% Mode1 (instant brake), 50% Mode2 (coast down). ---
        mode_roll = random.random()
        if mode_roll < 0.50:
            self._reset_mode = 1
        else:
            self._reset_mode = 2

        # Arm g-force state machine BEFORE throttle
        # measure_target is the brake event START speed (state machine arms when
        # instSpeed crosses below this). Set well below target_ms so the state
        measure_target = target_ms - 1.0  # ~2.2 mph below target
        self.vehicle.queue_lua_command("extensions.abstelemetry.resetAccum()")
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.setTargetSpeed({measure_target})")
        self.bng.step(1)

        # --- Throttle up (NON-DETERMINISTIC: sim runs free, no Python gating) ---
        # Determinism gates physics on bng.step() calls, so accel-up under determinism
        # crawls at ~25x slow-mo. Switch to free-running, poll sensors with time.sleep,
        self.bng.settings.set_nondeterministic()
        self.bng.control.resume()  # un-pause sim so it actually advances in free-run

        if self._reset_mode == 2:
            accel_target_ms = (self.target_mph + 1.5) * 0.44704
        else:
            accel_target_ms = target_ms

        self.vehicle.control(gear=2, throttle=1.0, steering=0, brake=0)
        accel_deadline = time.monotonic() + 20.0  # 20s wall-clock max
        while time.monotonic() < accel_deadline:
            time.sleep(0.02)  # 20ms poll interval = 50Hz check rate
            self.vehicle.sensors.poll()
            if self.vehicle.sensors['electrics'].get('airspeed', 0.0) >= accel_target_ms:
                break

        # --- Mode-specific handoff (still non-deterministic) ---
        if self._reset_mode == 1:
            # Mode 1: Instant brake, throttle off, neutral, ML takes over
            self.vehicle.control(throttle=0.0, steering=0, gear=0)

        elif self._reset_mode == 2:
            # Mode 2: Coast down, stay in gear, throttle off, wait for target speed
            self.vehicle.control(throttle=0.0, steering=0)  # still in gear
            coast_deadline = time.monotonic() + 10.0  # 10s wall-clock max
            while time.monotonic() < coast_deadline:
                time.sleep(0.02)
                self.vehicle.sensors.poll()
                if self.vehicle.sensors['electrics'].get('airspeed', 0.0) <= target_ms:
                    break
            # Now: neutral, ML takes over
            self.vehicle.control(throttle=0.0, steering=0, gear=0)

        elif self._reset_mode == 3:
            # Mode 3: Brake while throttle on, ML brakes now, throttle stays
            self._mode3_throttle_countdown = 250  # 1 second at 250Hz

        # --- Flip BACK to deterministic, pause, hand off to ML at TRUE 200Hz ---
        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        self.bng.step(1)  # let one tick settle under determinism

        # --- LSTM 2kHz recording: arm capture of THIS braking event ---
        # vlua sandbox writes a bare filename into the userdata dir; it gets moved out
        # at episode end. Wrapped so a recorder hiccup never kills training.
        self._lstm_recording = False
        if RECORD_LSTM:
            try:
                # Verify the recorder survived the reset-reload (lstm_data_env.py
                # pattern). One loud retry if it didn't.
                chk = self.vehicle.queue_lua_command(
                    "return tostring(extensions.lstm2khz ~= nil)", response=True)
                if str(chk).strip().lower() != 'true':
                    print(f"{self._prefix} lstm2khz missing (got {chk!r}), reloading...")
                    self.vehicle.queue_lua_command("extensions.load('lstm2khz')")
                    self.bng.step(5)
                self._lstm_ep += 1
                self._lstm_fname = f"sac_brake_{self.target_mph:.0f}mph_ep{self._lstm_ep:05d}.csv"
                cfg = ("{vehicle_model='etk800',grip_pattern='smallgrid_default'"
                       ",safety_variant='no_abs'}")
                # Direct call (NOT wrapped in "if extensions... then") so a missing
                # extension errors loudly in the BeamNG log instead of silently
                # no-opping, that silent guard is what hid the original failure.
                self.vehicle.queue_lua_command(
                    f"extensions.lstm2khz.startRecording('{self._lstm_fname}',{cfg})")
                self._lstm_recording = True
            except Exception as _e:
                print(f"{self._prefix} LSTM startRecording failed (continuing): {_e}")

        # Read start speed + heading
        self.vehicle.sensors.poll()
        e = self.vehicle.sensors['electrics']
        self.start_speed_ms = e.get('airspeed', 0.0)
        # Capture start heading from telemetry (ground truth from physics engine)
        start_data = json.loads(self.vehicle.queue_lua_command(
            "return extensions.abstelemetry.readAll()", response=True))
        self.start_heading = float(start_data.get('heading', 0.0))
        self.current_heading = self.start_heading
        self.target_heading = self.start_heading

        # resetAccum + setTargetSpeed already called BEFORE throttle-up.
        # Do NOT call resetAccum again, it would reset the brake state machine
        # which Mode 2 may have already armed during coast-down.
        self.bng.step(2)

        self.steps_taken = 0
        self.episode_count += 1
        self._ep_wall_start = time.monotonic()
        return self._get_obs(), {}

    def step(self, action):
        # v5.0 (revised): action ∈ [0,1] maps to brake ∈ [0.01, 1.0], tiny 1%
        # floor so the lua brake-event state machine never resets mid-measurement.
        # Lua threshold is 0.001 (v3.2), ours is 0.01, comfortable safety margin.
        brakes = 0.01 + 0.99 * action[:4].astype(np.float64)
        # Speed-guesser action removed (Rule 2). predicted_speed retained as constant 0
        # so downstream CSV/logging code that references it continues to work harmlessly.
        predicted_speed = 0.0

        # Axle clamp REMOVED v5.0: full 4-wheel independent brake control.
        # Each wheel can take any value in [0.01, 1.0] independently.
        brakes = np.clip(brakes, 0.01, 1.0)

        fr, fl, rr, rl = brakes

        # --- 3 round-trips, zero delay ---
        # RT 1: Set brakes BEFORE step (takes effect immediately)
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.setBrakes({fr},{fl},{rr},{rl})")

        # Mode 3: throttle stays on for 1 second, then off + neutral
        if self._mode3_throttle_countdown > 0:
            self._mode3_throttle_countdown -= 1
            self.vehicle.control(brake=float(max(brakes)), throttle=1.0)
            if self._mode3_throttle_countdown == 0:
                self.vehicle.control(throttle=0.0, gear=0)
        else:
            # Set input.brake so Lua brake event state machine can detect braking
            self.vehicle.control(brake=float(max(brakes)))

        # RT 2: Step physics (brakes are applied during this step)
        self.bng.step(FRAME_SKIP)
        self.steps_taken += 1

        # RT 3: Read sensors (result of this step with our brakes)
        data = json.loads(self.vehicle.queue_lua_command(
            "return extensions.abstelemetry.readAll()", response=True))

        obs = self._build_obs_from_data(data)
        # gps_speed not in obs, read from data dict for internal tracking
        gps_speed = self._last_gps_speed  # set by _build_obs_from_data
        # NOTE: indices below are post-reorg (slip removed, brk_actual removed, heading_error removed).
        # gy_avg moved from obs[8] (22-dim) → obs[4] (27-dim). yaw_rate moved from obs[10] → obs[6].
        braking_g_ms2 = float(obs[4])   # gy_avg (2000Hz poll-averaged decel, m/s^2)
        yaw_rate = float(obs[6])        # yaw_avg
        self.prev_brakes = brakes.astype(np.float32)

        # --- Speed prediction tracking ---
        spd_error = abs(predicted_speed - gps_speed)
        high_yaw = abs(yaw_rate) > 0.3
        if not high_yaw and gps_speed > 1.0:
            spd_error_pct = (spd_error / gps_speed) * 100.0 if gps_speed > 0.5 else 0.0
            self._spd_ep_errors.append(spd_error)
        else:
            spd_error_pct = 0.0
        # speedup #3: per-step speed_prediction CSV write skipped (diagnostic-only,
        # predicted_speed is hardcoded 0 since removal of the speed-guesser action).
        # If you ever need this back, uncomment:

        # ─── LSTM data row (27-col SAC-Data, only when RECORD_SAC_DATA) ───
        # Per-episode session_id so LSTM training respects brake-event boundaries.
        self._lstm_frame += 1
        if self._lstm_writer is not None:
            lstm_session_id = (f"{self._lstm_run_id}_{self._lstm_vehicle_model}_"
                               f"ep{self.episode_count}_{int(self.target_mph)}mph")
            self._lstm_writer.writerow([
                time.time(), lstm_session_id, f"sac_env{self.env_index}",
                self._lstm_vehicle_model, self._lstm_frame,
                round(float(data.get('ws3', 0.0)), 4),    # ws_fr
                round(float(data.get('ws4', 0.0)), 4),    # ws_fl
                round(float(data.get('ws1', 0.0)), 4),    # ws_rr
                round(float(data.get('ws2', 0.0)), 4),    # ws_rl
                round(float(data.get('gy_avg', 0.0)), 4),
                round(float(data.get('gy_min', 0.0)), 4),
                round(float(data.get('gy_max', 0.0)), 4),
                round(float(data.get('gx_avg', 0.0)), 4),
                round(float(data.get('gx_min', 0.0)), 4),
                round(float(data.get('gx_max', 0.0)), 4),
                round(float(data.get('gz_avg', 0.0)), 4),
                round(float(data.get('gz_min', 0.0)), 4),
                round(float(data.get('gz_max', 0.0)), 4),
                round(float(data.get('yaw_avg', 0.0)), 4),
                round(float(data.get('steering', 0.0)), 4),
                round(float(data.get('rpm', 0.0)), 1),
                int(data.get('gear', 0)),
                round(float(data.get('input_brake', 0.0)), 4),
                round(float(data.get('input_throttle', 0.0)), 4),
                round(gps_speed, 4),                      # airspeed (LABEL)
                round(float(data.get('pitch', 0.0)), 4),
                round(float(data.get('roll', 0.0)), 4),
            ])

        # Track per-episode stats
        braking_g_gs = abs(braking_g_ms2) / 9.81
        self.ep_peak_g = max(self.ep_peak_g, braking_g_gs)
        # Slip removed from obs (real cars don't see body-speed-derived slip).
        # We still compute slip here from gps_speed (game-engine ground truth) for grading only.
        slip_vals = self._last_slip_vals  # set inside _build_obs_from_data
        self.ep_peak_slip = max(self.ep_peak_slip, float(np.max(slip_vals)))
        self.ep_max_yaw = max(self.ep_max_yaw, abs(yaw_rate))

        reward = 0.0

        # SLIP REWARD REMOVED per Rule 2 (g-forces-only). slip_vals still tracked above
        # for ep_peak_slip diagnostic logging but no longer contributes to reward.

        # SPEED-PREDICTION REWARD REMOVED per Rule 2. spd_rew kept as 0 so audit-log
        # downstream code that references it continues to work.
        spd_rew = 0.0

        # ─── v5.0 PER-STEP G-FORCE REWARD ────────────────────────────
        # Per-step = PER_STEP_K × terminal_shape(g_instantaneous).
        # ~1000 contribution at the okayish reference (1.05g, 900 steps).
        step_g_rew = PER_STEP_K * _terminal_g_shape(braking_g_gs)
        reward += step_g_rew
        self.ep_step_g_sum += step_g_rew

        # heading_error still computed, grading-side use ONLY (crash terminal at >CRASH_HEADING).
        self.current_heading = float(data.get('heading', self.current_heading))
        heading_error = abs(self.current_heading - self.target_heading)
        self.prev_heading_error = heading_error  # legacy state, kept so existing refs don't break

        # ─── v5.0 YAW: target-driven (deviation from requested yaw rate) ──
        # All three signals (per-step bonus, cumulative, catastrophic backstop)
        # operate on `yaw_error = |actual − target|`. With target=0 (current),
        yaw_rate_mag = abs(yaw_rate)                       # diagnostics only (ep_max_yaw)
        yaw_error = abs(yaw_rate - self.target_yaw_rate)   # deviation from requested

        # Per-step bonus: Gaussian, max at yaw_error=0. Gated on braking
        # so a perfectly-tracking coast doesn't farm reward.
        if braking_g_gs > PER_STEP_G_GATE:
            yaw_bonus = YAW_BONUS_K_STEP * math.exp(-yaw_error * YAW_BONUS_ALPHA)
        else:
            yaw_bonus = 0.0
        reward += yaw_bonus
        self.ep_yaw_penalty += yaw_bonus  # field-name kept for CSV-compat; semantically the yaw signal

        # Cumulative tracking (for terminal bonus + catastrophic backstop)
        self.ep_yaw_abs_sum += yaw_error * self._dt           # ∫|yaw_error|·dt, matches "perfect <0.1" metric
        if yaw_error > YAW_RATE_DEADZONE_RAD_S:
            excess = yaw_error - YAW_RATE_DEADZONE_RAD_S
            self.ep_yaw_sq_sum += excess * excess * self._dt  # catastrophic backstop input (excess²·dt)

        # Aliased for audit-log code below (was `yaw_pen` in v4.0; now positive bonus).
        yaw_pen = yaw_bonus

        # --- Audit log (per-step, non-terminal rows) ---
        force_override = 0  # FORCE_BRAKE_THRESH removed, model handles all stopping
        self._audit_step_data = (obs, brakes, predicted_speed, gps_speed,
                                 braking_g_gs, force_override,
                                 step_g_rew, yaw_pen, spd_rew, reward)

        if self._audit_writer:
            self._audit_writer.writerow([
                self.env_index, self.episode_count, self.steps_taken, self.target_mph,
                # 27 obs
                *[round(float(v), 4) for v in obs],
                # 4 actions (speed-guesser removed)
                round(fr, 4), round(fl, 4), round(rr, 4), round(rl, 4),
                # what happened
                round(gps_speed, 2), round(gps_speed / 0.44704, 1),
                round(predicted_speed, 2),
                round(braking_g_gs, 4), force_override,
                # rewards
                round(step_g_rew, 4), round(yaw_pen, 4), round(spd_rew, 4),
                round(reward, 4),
                # terminal (empty for non-terminal steps)
                '', '', '', '',
            ])

        terminated = False

        if self.steps_taken >= MAX_EPISODE_STEPS:
            # v5.0: catastrophic-yaw backstop still applies on timeout
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum
            reward += acc_yaw_pen
            print(f"{self._prefix} [EP {self.episode_count:4d}] TIMEOUT: "
                  f"{self.target_mph:.0f}mph, {self.steps_taken} steps "
                  f"| YawInt: {self.ep_yaw_abs_sum:.3f} AccYaw: {acc_yaw_pen:+.1f}")
            self._log_episode('TIMEOUT', avg_g=0.0, terminal_rew=acc_yaw_pen)
            self._write_audit_terminal(0.0, 0.0, acc_yaw_pen, 'TIMEOUT')
            self._finish_lstm()
            terminated = True
            return obs, reward, terminated, False, {}

        if heading_error > CRASH_HEADING:
            # v5.0: -2000 crash penalty + catastrophic-yaw backstop
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum
            terminal_rew = CRASH_PENALTY + acc_yaw_pen
            reward += terminal_rew
            print(f"{self._prefix} [EP {self.episode_count:4d}] CRASH (heading): "
                  f"{self.target_mph:.0f}mph, heading={math.degrees(heading_error):.1f}° "
                  f"| YawInt: {self.ep_yaw_abs_sum:.3f} AccYaw: {acc_yaw_pen:+.1f}")
            self._log_episode('CRASH', avg_g=0.0, terminal_rew=terminal_rew)
            self._write_audit_terminal(0.0, 0.0, terminal_rew, 'CRASH')
            self._finish_lstm()
            terminated = True
            return obs, reward, terminated, False, {}

        if gps_speed < 0.05:
            self.stop_timer += 1
        else:
            self.stop_timer = 0

        if self.stop_timer >= STOP_FRAMES:
            terminated = True

            # Terminal: release brakes, step, read final stats (3 RT, runs once per episode)
            self.vehicle.queue_lua_command(
                "extensions.abstelemetry.setBrakes(0,0,0,0)")
            self.bng.step(5)
            term_data = json.loads(self.vehicle.queue_lua_command(
                "return extensions.abstelemetry.readAll()", response=True))

            # v2.1: last_brake_avg_g is already in g's (BeamNG kinematic method).
            # SAME stopping mechanics abstelemetry uses, do NOT self-compute distance.
            avg_g = float(term_data.get('last_brake_avg_g', 0.0))
            self.ep_stopping_dist = float(term_data.get('last_brake_dist', 0.0))

            # Stop the 2kHz LSTM recorder now that the brake event is fully measured.
            self._finish_lstm()

            # ─── v5.0 TERMINAL G-FORCE REWARD ────────────────────────
            # Linear ramp -400→+1000 across [0.3g, 1.05g], +500 jump at 1.06g,
            # quadratic growth above (open-ended past 2g, no cap).
            terminal_g_rew = _terminal_g_shape(avg_g)

            # ─── v5.0 TERMINAL YAW BONUS ─────────────────────────────
            # One-shot bonus if cumulative ∫|yaw_rate|·dt stayed under threshold.
            # Gated on avg_g > 0.30g so a coast-to-stop doesn't farm this either.
            if avg_g > PER_STEP_G_GATE:
                yaw_clean_factor = max(0.0, 1.0 - self.ep_yaw_abs_sum / YAW_BONUS_THRESHOLD)
                terminal_yaw_bonus = YAW_BONUS_K_TERMINAL * yaw_clean_factor
            else:
                terminal_yaw_bonus = 0.0

            # ─── KEPT: catastrophic-yaw accumulator backstop ─────────
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum

            terminal_rew = terminal_g_rew + terminal_yaw_bonus + acc_yaw_pen
            reward += terminal_rew

            stop_time = self.steps_taken / DETERM_HZ
            total_ep_reward = (self.ep_step_g_sum + self.ep_yaw_penalty
                               + terminal_rew)

            # Speed prediction accuracy for this episode
            spd_acc = ""
            if self._spd_ep_errors:
                avg_err = sum(self._spd_ep_errors) / len(self._spd_ep_errors)
                spd_acc = f" | SpdErr: {avg_err:.1f}m/s"
            self._spd_file.flush()

            print(f"{self._prefix} [EP {self.episode_count:4d}] M{self._reset_mode} "
                  f"{self.target_mph:3.0f}mph ({self.start_speed_ms:.1f}m/s) | "
                  f"{self.steps_taken:4d} steps ({stop_time:.2f}s) | "
                  f"Dist: {self.ep_stopping_dist:.1f}m ({self.ep_stopping_dist * 3.281:.0f}ft) | "
                  f"Avg G: {avg_g:.3f}g Peak G: {self.ep_peak_g:.3f}g | "
                  f"YawInt: {self.ep_yaw_abs_sum:.3f} | "
                  f"StepG: {self.ep_step_g_sum:+.0f} "
                  f"YawBon: {self.ep_yaw_penalty:+.0f} "
                  f"TermG: {terminal_g_rew:+.0f} TermYaw: {terminal_yaw_bonus:+.0f} "
                  f"AccYaw: {acc_yaw_pen:+.0f} | "
                  f"TOTAL: {total_ep_reward:+.0f}{spd_acc}")
            self._log_episode('STOP', avg_g=avg_g, terminal_rew=terminal_rew)
            self._write_audit_terminal(avg_g, self.ep_stopping_dist, terminal_rew, 'STOP')

        return obs, reward, terminated, False, {}

    def _clip_obs(self, raw):
        """Replace NaN/inf with safe values, then clip each dim to its declared bound.
        Defensive layer, protects PPO from sensor faults that could poison policy weights."""
        safe = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(safe, self._OBS_LOW, self._OBS_HIGH).astype(np.float32)

    def _get_obs(self):
        """Initial obs via sensors.poll (used in reset only).
        v2.1: tel_gy_inst is 60Hz stale, fine for initial obs, step() uses gy_avg."""
        self.vehicle.sensors.poll()
        e = self.vehicle.sensors['electrics']

        gps_speed = float(e.get('airspeed', 0.0))

        ws_rr = abs(float(e.get('tel_ws_1', 0.0)))
        ws_rl = abs(float(e.get('tel_ws_2', 0.0)))
        ws_fr = abs(float(e.get('tel_ws_3', 0.0)))
        ws_fl = abs(float(e.get('tel_ws_4', 0.0)))

        safe_speed = max(gps_speed, 0.5)
        slip_fr = np.clip(1.0 - ws_fr / safe_speed, 0.0, 1.0)
        slip_fl = np.clip(1.0 - ws_fl / safe_speed, 0.0, 1.0)
        slip_rr = np.clip(1.0 - ws_rr / safe_speed, 0.0, 1.0)
        slip_rl = np.clip(1.0 - ws_rl / safe_speed, 0.0, 1.0)

        braking_g = float(e.get('tel_gy_inst', 0.0))
        lateral_g = float(e.get('tel_gx_inst', 0.0))
        yaw_rate = float(e.get('tel_yaw_rate_inst', 0.0))

        brk_fr = float(e.get('tel_brk_actual_fr', 0.0))
        brk_fl = float(e.get('tel_brk_actual_fl', 0.0))
        brk_rr = float(e.get('tel_brk_actual_rr', 0.0))
        brk_rl = float(e.get('tel_brk_actual_rl', 0.0))

        self._last_gps_speed = gps_speed

        # Slip channels removed from obs (cheat, derived from ground-truth airspeed).
        # slip_fr/fl/rr/rl still computed above for reward grading via _last_slip_vals.
        self._last_slip_vals = np.array([slip_fr, slip_fl, slip_rr, slip_rl], dtype=np.float32)
        raw = np.array([
            ws_fr, ws_fl, ws_rr, ws_rl,                  # 0-3: wheel speeds
            braking_g, lateral_g, yaw_rate,              # 4-6: g-forces + yaw
            self.prev_brakes[0], self.prev_brakes[1],    # 7-10: prev brake cmds
            self.prev_brakes[2], self.prev_brakes[3],
            # brk_actual_* DROPPED 2026-05-02 (physics cheat, redundant w/ prev_brakes)
            0.0, 0.0,                                    # 11-12: gy_min, gy_max, no poll data yet at reset
            0.0,                                         # 13: rpm (none at reset)
            0.0,                                         # 14: gear (none at reset)
            0.0,                                         # 15: steering (none at reset)
            0.0,                                         # 16: pitch (none at reset)
            0.0,                                         # 17: roll (none at reset)
            0.0,                                         # 18: input_brake (none at reset)
            0.0,                                         # 19: input_throttle (none at reset)
            0.0,                                         # 20: gz (none at reset)
            0.0,                                         # 21: pitch_rate (no prev sample)
            0.0,                                         # 22: roll_rate (no prev sample)
            0.0, 0.0, 0.0, 0.0,                          # 23-26: wheel accels (no prev sample)
        ], dtype=np.float32)
        return self._clip_obs(raw)

    def _build_obs_from_data(self, data):
        """Build observation array from readAll return dict.
        Same layout as _get_obs() but without a sensors.poll() round-trip.
        v2.1: uses gy_avg/yaw_avg (2000Hz poll-averaged) + gy_min/gy_max."""
        # Prefer physics-rate inst_speed; fall back to airspeed if missing (older lua).
        gps_speed = float(data.get('inst_speed', data.get('airspeed', 0.0)))

        ws_rr = abs(float(data.get('ws1', 0.0)))
        ws_rl = abs(float(data.get('ws2', 0.0)))
        ws_fr = abs(float(data.get('ws3', 0.0)))
        ws_fl = abs(float(data.get('ws4', 0.0)))

        safe_speed = max(gps_speed, 0.5)
        slip_fr = np.clip(1.0 - ws_fr / safe_speed, 0.0, 1.0)
        slip_fl = np.clip(1.0 - ws_fl / safe_speed, 0.0, 1.0)
        slip_rr = np.clip(1.0 - ws_rr / safe_speed, 0.0, 1.0)
        slip_rl = np.clip(1.0 - ws_rl / safe_speed, 0.0, 1.0)

        braking_g = float(data.get('gy_avg', 0.0))    # 2000Hz poll-averaged decel (m/s^2)
        braking_g_min = float(data.get('gy_min', 0.0)) # min decel between polls (smoothed)
        braking_g_max = float(data.get('gy_max', 0.0)) # max decel between polls (smoothed)
        lateral_g = float(data.get('gx_inst', 0.0))
        yaw_rate = float(data.get('yaw_avg', 0.0))     # 2000Hz poll-averaged yaw

        # Actual brake torque being applied per wheel (Nm, from physics engine)
        brk_fr = float(data.get('brk_actual_fr', 0.0))
        brk_fl = float(data.get('brk_actual_fl', 0.0))
        brk_rr = float(data.get('brk_actual_rr', 0.0))
        brk_rl = float(data.get('brk_actual_rl', 0.0))

        # gps_speed intentionally excluded, model must infer speed from wheel/g data
        self._last_gps_speed = gps_speed  # still tracked internally for stop detection

        # New real-car sensors added 2026-05-02 (Rule 1 compliant, all standard CAN/IMU outputs)
        rpm = float(data.get('rpm', 0.0))
        gear = float(data.get('gear', 0.0))
        steering = float(data.get('steering', 0.0))      # normalized [-1, 1] steering-wheel angle
        pitch = float(data.get('pitch', 0.0))            # radians, positive = nose up
        roll = float(data.get('roll', 0.0))              # radians, positive = right side down
        input_brake = float(data.get('input_brake', 0.0))      # driver brake pedal [0, 1]
        input_throttle = float(data.get('input_throttle', 0.0))  # driver throttle pedal [0, 1]

        # 3rd IMU axis + derivative-rate features added 2026-05-02
        gz = float(data.get('gz_inst', 0.0))             # vertical accel (m/s², raw, includes gravity)

        # Derivatives (rates), computed as (current - prev) / dt. dt fixed at 1/250s.
        # First step after reset has no prev → emit 0.
        cur_ws = np.array([ws_fr, ws_fl, ws_rr, ws_rl], dtype=np.float32)
        if self._has_prev_state:
            wa_fr, wa_fl, wa_rr, wa_rl = (cur_ws - self._prev_ws) / self._dt
            pitch_rate = (pitch - self._prev_pitch) / self._dt
            roll_rate = (roll - self._prev_roll) / self._dt
        else:
            wa_fr = wa_fl = wa_rr = wa_rl = 0.0
            pitch_rate = 0.0
            roll_rate = 0.0
        self._prev_ws = cur_ws
        self._prev_pitch = pitch
        self._prev_roll = roll
        self._has_prev_state = True

        # Slip channels removed from obs (cheat, derived from ground-truth airspeed).
        # slip_fr/fl/rr/rl still computed above for reward grading via _last_slip_vals.
        self._last_slip_vals = np.array([slip_fr, slip_fl, slip_rr, slip_rl], dtype=np.float32)
        raw = np.array([
            ws_fr, ws_fl, ws_rr, ws_rl,           # 0-3: wheel speeds (m/s)
            braking_g, lateral_g, yaw_rate,       # 4-6: gy_avg, gx_inst, yaw_avg
            self.prev_brakes[0], self.prev_brakes[1],  # 7-10: prev brake cmds
            self.prev_brakes[2], self.prev_brakes[3],
            # brk_actual_* DROPPED 2026-05-02 (physics cheat, redundant w/ prev_brakes)
            braking_g_min, braking_g_max,         # 11-12: gy poll-window min/max
            rpm,                                  # 13: engine RPM (CAN)
            gear,                                 # 14: gear index (CAN)
            steering,                             # 15: steering input [-1, 1] (SAS)
            pitch,                                # 16: pitch angle (rad)
            roll,                                 # 17: roll angle (rad)
            input_brake,                          # 18: driver brake pedal [0, 1] (BPS)
            input_throttle,                       # 19: driver throttle pedal [0, 1] (TPS)
            gz,                                   # 20: vertical accel (m/s², IMU)
            pitch_rate,                           # 21: pitch rate (rad/s)
            roll_rate,                            # 22: roll rate (rad/s)
            wa_fr,                                # 23: wheel accel FR (m/s²)
            wa_fl,                                # 24: wheel accel FL (m/s²)
            wa_rr,                                # 25: wheel accel RR (m/s²)
            wa_rl,                                # 26: wheel accel RL (m/s²)
        ], dtype=np.float32)
        return self._clip_obs(raw)

    def enable_audit_log(self, path=None):
        """Enable per-step audit CSV. Call before training/testing."""
        if path is None:
            log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f'audit_env{self.env_index}.csv')
        self._audit_file = open(path, 'w', newline='')
        self._audit_writer = csv.writer(self._audit_file)
        self._audit_writer.writerow([
            # --- WHO ---
            'env', 'episode', 'step', 'target_mph',
            # --- WHAT THE MODEL SEES (27 obs, brk_actual_* dropped: physics cheat) ---
            'see_ws_fr', 'see_ws_fl', 'see_ws_rr', 'see_ws_rl',
            'see_gy_avg', 'see_lateral_g', 'see_yaw',
            'see_prev_brk_fr', 'see_prev_brk_fl', 'see_prev_brk_rr', 'see_prev_brk_rl',
            'see_gy_min', 'see_gy_max',
            'see_rpm', 'see_gear', 'see_steering', 'see_pitch', 'see_roll',
            'see_input_brake', 'see_input_throttle',
            'see_gz', 'see_pitch_rate', 'see_roll_rate',
            'see_wa_fr', 'see_wa_fl', 'see_wa_rr', 'see_wa_rl',
            # --- WHAT THE MODEL DOES (4 actions, speed-guesser removed) ---
            'do_brake_fr', 'do_brake_fl', 'do_brake_rr', 'do_brake_rl',
            # --- WHAT ACTUALLY HAPPENED ---
            'actual_speed_ms', 'actual_speed_mph', 'predicted_speed_ms',
            'actual_gy_gs', 'force_brake_override',
            # --- REWARDS THIS STEP ---
            'rew_step_g', 'rew_yaw', 'rew_speed', 'rew_total_step',
            # --- EPISODE END (only on last step) ---
            'terminal_avg_g', 'terminal_dist_m', 'terminal_reward', 'outcome',
        ])
        print(f"{self._prefix} Audit log: {path}")

    def _write_audit_terminal(self, avg_g, dist, terminal_rew, outcome):
        """Overwrite the last audit row's terminal columns."""
        if not self._audit_writer:
            return
        # Write a summary row marking the episode end
        self._audit_writer.writerow([
            self.env_index, self.episode_count, 'END', self.target_mph,
            # obs: empty
            *([''] * 21),
            # actions: empty
            '', '', '', '', '',
            # what happened: empty
            '', '', '', '', '',
            # rewards: empty
            '', '', '', '',
            # terminal
            round(avg_g, 4), round(dist, 2), round(terminal_rew, 1), outcome,
        ])
        self._audit_file.flush()

    def _log_episode(self, outcome, avg_g, terminal_rew):
        stop_time = self.steps_taken / DETERM_HZ
        wall_clock = time.monotonic() - self._ep_wall_start
        total_rew = (self.ep_step_g_sum + self.ep_yaw_penalty + terminal_rew)
        self._csv_writer.writerow([
            self.episode_count, self.target_mph, round(self.start_speed_ms, 2),
            self.steps_taken, round(stop_time, 2),
            round(self.ep_stopping_dist, 2), round(self.ep_stopping_dist * 3.281, 1),
            round(avg_g, 4), round(self.ep_peak_g, 4),
            round(self.ep_peak_slip, 4), round(self.ep_max_yaw, 4), outcome,
            round(self.ep_step_g_sum, 1), round(self.ep_yaw_penalty, 1),
            round(terminal_rew, 1),
            round(total_rew, 1), round(wall_clock, 2),
        ])
        self._csv_file.flush()

    def _shrink_window(self):
        """Resize BeamNG window to minimum via Windows API (ctypes)."""
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32

            def enum_callback(hwnd, results):
                if user32.IsWindowVisible(hwnd):
                    pid = ctypes.wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if 'BeamNG' in buf.value:
                            results.append(hwnd)
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.c_void_p)
            results = []
            user32.EnumWindows(WNDENUMPROC(enum_callback), None)

            x_offset = self.env_index * 420
            for hwnd in results:
                user32.MoveWindow(hwnd, x_offset, 0, 400, 300, True)

            if results:
                print(f"{self._prefix} Resized {len(results)} BeamNG window(s) to 400x300")
            else:
                print(f"{self._prefix} No BeamNG windows found to resize")
        except Exception as e:
            print(f"{self._prefix} Window resize failed: {e}")

    def _apply_performance_tuning(self, python_cores, beamng_cores):
        """Pin Python to specific cores + HIGH priority, do the same for spawned
        BeamNG processes. Built-in so users don't need to manually set priority/
        affinity per launch.

        - Python: small core list (1-2 threads of a single P-core)
        - BeamNG: all remaining cores
        - Both: HIGH priority class (Windows)
        """
        try:
            import psutil
        except ImportError:
            print(f"{self._prefix} psutil missing, skipping perf tuning (pip install psutil)")
            return

        # --- Python: priority + affinity ---
        try:
            py = psutil.Process()
            py.nice(psutil.HIGH_PRIORITY_CLASS)
            py.cpu_affinity(python_cores)
            print(f"{self._prefix} Python: HIGH priority, affinity={python_cores}")
        except Exception as e:
            print(f"{self._prefix} Python perf tuning failed: {e}")

        # --- BeamNG: find spawned processes, set priority + affinity ---
        bng_procs = []
        for p in psutil.process_iter(['name']):
            try:
                n = (p.info.get('name') or '').lower()
                if 'beamng' in n:
                    bng_procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        for p in bng_procs:
            try:
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                p.cpu_affinity(beamng_cores)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"{self._prefix} BeamNG perf tuning failed on pid={p.pid}: {e}")
        print(f"{self._prefix} BeamNG: HIGH priority, affinity={beamng_cores} "
              f"(applied to {len(bng_procs)} processes)")

    def close(self):
        try:
            self._csv_file.close()
        except Exception:
            pass
        try:
            self._spd_file.close()
        except Exception:
            pass
        try:
            self._lstm_file.close()
        except Exception:
            pass
        try:
            if self._audit_file:
                self._audit_file.close()
        except Exception:
            pass
        try:
            self.bng.close()
        except Exception:
            pass
