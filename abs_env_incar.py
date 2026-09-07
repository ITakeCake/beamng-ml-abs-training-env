"""
Deployment-Clock In-Car ABS Learning Environment.

Subclass of abs_env.ABSLearningEnv (BUILD_SPEC_incar_training.md §4). The point of
this env: make SAC train against the EXACT loop that deploys.

In the parent env (abs_env.py) Python drives the loop, it calls
`abstelemetry.readAll()` every step, which runs `buildSensorData()` and DRAINS a
clean 1-step poll window. The NN therefore sees pristine, training-distribution
obs. The deployed in-car controller (MTB-ML-ABS.lua) instead builds obs IN-TICK on
its own 200Hz tick (gy_min/gy_max smoothing seeds, wheel-accel derivs, engage
warmup, prev_brake latch), so the obs the policy actually consumes drift from
training. Same run-2 weights => 1.15g in the TCP loop, 0.89g deployed.

This env closes that gap: the in-car CONTROLLER is the SOLE obs-builder and SOLE
poll-window drainer. Python only:
  - mailboxes the 4 action floats via `controller.getControllerSafe(...).setExtCmd`
  - reads the controller's PUBLISHED obs (`mlabs_o0..o26`) and non-draining
    telemetry electrics (`tel_fused_speed`, `tel_last_brake_avg_g`,
    `tel_last_brake_dist`, `mlabs_heading`)

The env NEVER calls readAll/buildSensorData/exchangeData, doing so would steal the
controller's poll window (corrupting its obs) AND recreate the old TCP loop.

Reward v5.0 is COPIED BYTE-IDENTICAL from abs_env.step(): every constant and
`_terminal_g_shape` are IMPORTED from abs_env (cannot drift); only the INPUT SOURCE
of each reward term changes (published electrics instead of drained readAll).

DO NOT EDIT abs_env.py / train.py / abstelemetry.lua. This is a standalone subclass.
"""
import math
import time
import numpy as np

import abs_env
from abs_env import (
    ABSLearningEnv, _terminal_g_shape,
    DETERM_HZ, FRAME_SKIP,
    GATEKEEPER_G, TERMINAL_STEP_BONUS, TERMINAL_QUAD_K, TERMINAL_RAMP_NEG_K,
    TERMINAL_RAMP_POS_K, PER_STEP_K, PER_STEP_G_GATE,
    YAW_BONUS_K_STEP, YAW_BONUS_ALPHA, YAW_BONUS_K_TERMINAL, YAW_BONUS_THRESHOLD,
    YAW_PEN_K_TERMINAL, YAW_RATE_DEADZONE_RAD_S,
    CRASH_HEADING, MAX_EPISODE_STEPS, STOP_FRAMES, CRASH_PENALTY,
    MIN_MPH, MAX_MPH, START_POS,
)

# --- This is a training loop, NOT a data-collection run. Disable both recorders ---
# (spec §4.1). These are module-level flags read by the parent env at __init__/reset.
abs_env.RECORD_LSTM = False
abs_env.RECORD_SAC_DATA = False

# Controller that MTB-ML-ABS.lua registers as. `controller.getControllerSafe(NAME)`
# resolves to the loaded controller table M.
CONTROLLER_NAME = 'MTB-ML-ABS'

# The DEPLOY car: this .pc has the ABS slot + MTB-ML-ABS controller loaded. The
# training car (Machine-Trainer-Boy.pc, parent default) has NO ABS slot, so the
# controller would never load. We MUST force the MLABS car here.
# (PPO_V2, 2026-07-11): switched to Machine-Trainer-Boy-V2, etk800 SEDAN
# Rennspecht build on sport_plus tires (same platform class as the 1FEX 1.283g
# baseline). V2-MLABS = V2 byte-identical except etk_DSE_ABS -> etk_DSE_ABS_MTB_ML.
VEHICLE_PC_INCAR = 'vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc'

# Hard ceiling on engage+warmup ride-through (spec §4.3): 400 ticks @200Hz = 2.0s.
ENGAGE_RIDE_TICKS = 400

# Number of published raw-obs channels (mlabs_o0..o26).
OBS_DIM = 27


class ABSLearningEnvIncar(ABSLearningEnv):
    """In-car (deployment-clock) ABS env. Subclass overrides __init__/reset/step;
    everything else (_clip_obs, _log_episode, _write_audit_terminal, close, perf
    tuning, action/obs spaces, the entire __init__ machinery) is inherited."""

    def __init__(self, *args, **kwargs):
        # Force the MLABS car BEFORE super().__init__ spawns the vehicle, so
        # MTB-ML-ABS.lua loads as a controller on the ego vehicle.
        abs_env.VEHICLE_PC = VEHICLE_PC_INCAR
        super().__init__(*args, **kwargs)

        # Python-side monotonic seq written into each mailbox cmd. Reset to 0 each reset.
        self._seq = 0
        self._ctrl_call = f"controller.getControllerSafe('{CONTROLLER_NAME}')"

        # Put the controller into ext mode (mailbox actions, skip NN forward, force
        # debugObs so the 27 raw obs publish every tick). Re-asserted after every
        # teleport(reset=True) inside reset() because that re-inits the controller.
        self.vehicle.queue_lua_command(f"{self._ctrl_call}.setExtMode(true)")
        self.bng.step(2)

    # ------------------------------------------------------------------ helpers
    def _poll_electrics(self):
        """Poll the already-attached 'electrics' sensor (TCP read of electrics.values).
        Does NOT call any Lua function on the vehicle VM and does NOT drain the poll
        window, this is the ONLY per-step Python<->VM read besides the mailbox write."""
        self.vehicle.sensors.poll()
        return self.vehicle.sensors['electrics']

    def _published_obs_vector(self, e):
        """The 27 raw obs the controller built + published THIS tick (mlabs_o0..o26).
        No Python-side recompute of any channel. Parent's _clip_obs is idempotent here
        (the controller already clipped), kept only as a NaN/inf safety net."""
        raw = np.array(
            [float(e.get(f'mlabs_o{i}', 0.0)) for i in range(OBS_DIM)],
            dtype=np.float32)
        return self._clip_obs(raw)

    def _assert_ext_mode(self, e):
        """Re-arm ext mode if the controller dropped it (teleport reset re-inits it)."""
        if int(e.get('mlabs_extmode', 0)) != 1:
            self.vehicle.queue_lua_command(f"{self._ctrl_call}.setExtMode(true)")

    # ------------------------------------------------------------------- reset
    def reset(self, seed=None):
        """Bring the car to target speed (same machinery as parent), then hand off by
        SLAMMING the driver brake to trigger the controller's engage state machine,
        ride through engage+warmup, and return the controller-PUBLISHED obs0.

        This deliberately re-implements the parent reset body (rather than calling
        super().reset() then patching) because the parent returns self._get_obs(), a
        sensors.poll()-based obs that does NOT match the controller's published obs and
        would corrupt the frame stack's first frame. Only these differ from parent:
          (a) handoff = slam driver brake (the controller owns brakes in-car),
          (b) ext-mode re-assert after teleport,
          (c) engage+warmup ride-through before obs0,
          (d) _seq = 0,
          (e) return controller-published obs0.
        The speed-attainment + reset-mode machinery is identical to parent.
        """
        # Standard gym RNG seeding (parent calls super().reset(seed=seed)).
        super(ABSLearningEnv, self).reset(seed=seed)

        # --- per-episode state reset (mirrors parent.reset) ---
        self._lstm_frame = 0
        self.stop_timer = 0
        self.prev_brakes = np.zeros(4, dtype=np.float32)
        self.ep_step_g_sum = 0.0
        self.ep_yaw_penalty = 0.0
        self.start_heading = 0.0
        self.current_heading = 0.0
        self.target_heading = 0.0
        self.target_yaw_rate = 0.0
        self.prev_heading_error = 0.0
        self._last_gps_speed = 999.0
        self._spd_ep_errors = []
        self.ep_peak_g = 0.0
        self.ep_peak_slip = 0.0
        self.ep_max_yaw = 0.0
        self.ep_stopping_dist = 0.0
        self.ep_yaw_sq_sum = 0.0
        self.ep_yaw_abs_sum = 0.0

        # Reset derivative-state so first step's rates = 0 (parent semantics; obs are
        # built in-Lua now, but keep the Python mirror consistent).
        self._prev_ws = np.zeros(4, dtype=np.float32)
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._has_prev_state = False

        # --- target speed pick (identical to parent) ---
        if self.fixed_mph is None:
            self.target_mph = round(np.random.uniform(MIN_MPH, MAX_MPH))
        elif isinstance(self.fixed_mph, (list, tuple)):
            import random as _random
            self.target_mph = _random.choice(self.fixed_mph)
        else:
            self.target_mph = self.fixed_mph
        target_ms = self.target_mph * 0.44704

        # --- teleport + bring extensions/ABS to a known state (parent semantics) ---
        self.vehicle.teleport(pos=START_POS, rot_quat=self.start_quat, reset=True)
        self.vehicle.focus()
        # teleport(reset=True) UNLOADS vehicle lua extensions AND re-inits the
        # controller -> M.extMode drops back to false. Re-assert it below (and inside
        # the ride-through loop) so ext mode is live before any obs counts.
        self.vehicle.queue_lua_command(f"{self._ctrl_call}.setExtMode(true)")

        self.vehicle.control(throttle=0, steering=0, gear=0, parkingbrake=0, brake=0)
        self.vehicle.queue_lua_command("extensions.load('abstelemetry')")
        self.vehicle.queue_lua_command("extensions.abstelemetry.setBrakes(0,0,0,0)")
        self.bng.step(15)
        self.vehicle.queue_lua_command('wheels.setABSBehavior("off")')
        self.bng.step(2)

        # --- (PPO_V2, 2026-07-11) SINGLE simple regime (the spec): accelerate to
        # target+2mph, coast down IN GEAR, brakes begin AT the target crossing.
        # (Was 50/50 instant-brake / coast-down.)
        self._reset_mode = 2
        self._neutral_dropped = False  # gear stays in DRIVE until ~2.5 mph (see step())

        # Arm the brake-event distance state machine BEFORE throttle (parent semantics).
        # The controller's onEngage() will call resetAccum() again, but setTargetSpeed
        # here is what makes last_brake_dist / last_brake_avg_g (the terminal reward
        # inputs) get measured exactly as the parent does.
        measure_target = target_ms - 1.0
        self.vehicle.queue_lua_command("extensions.abstelemetry.resetAccum()")
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.setTargetSpeed({measure_target})")
        self.bng.step(1)

        # --- throttle up under NON-determinism (parent semantics) ---
        self.bng.settings.set_nondeterministic()
        self.bng.control.resume()

        # (PPO_V2) +2.0 mph overshoot so the coast-down target crossing arms cleanly
        accel_target_ms = (self.target_mph + 2.0) * 0.44704

        self.vehicle.control(gear=2, throttle=1.0, steering=0, brake=0)
        accel_deadline = time.monotonic() + 20.0
        while time.monotonic() < accel_deadline:
            time.sleep(0.02)
            self.vehicle.sensors.poll()
            if self.vehicle.sensors['electrics'].get('airspeed', 0.0) >= accel_target_ms:
                break

        # --- (PPO_V2, 2026-07-11) 2kHz-EXACT BRAKE ONSET ---
        # Throttle off at target+2mph, flip to DETERMINISTIC immediately, then arm the
        # in-Lua slam: abstelemetry.onPhysicsStep checks instSpeed (obj:getVelocity()
        # :length(), the SAME signal the standard brake metric uses) every 0.5ms tick
        # and latches input.brake=1 on the exact tick of the target crossing. The old
        # Python coast loop polled stale electrics at ~50Hz WALL-CLOCK, onset could
        # land tenths of a mph late. The chunked stepping below is just transport;
        # onset precision comes from the 2kHz in-Lua check, not the chunk size.
        # Coast is IN GEAR (drive), no neutral shift here; the car stays in gear
        # through the stop until ~2.5 mph, where step() drops it to neutral once.
        self.vehicle.control(throttle=0.0, steering=0)
        self.bng.control.pause()
        self.bng.settings.set_deterministic(DETERM_HZ)
        self.vehicle.queue_lua_command(
            f"extensions.abstelemetry.armBrakeSlam({target_ms})")
        slam_fired = False
        e = None
        for _ in range(600):                    # ceiling: 600 x 100ms = 60s sim time
            self.bng.step(20)                   # 100ms sim per chunk @ 200Hz
            e = self._poll_electrics()
            self._assert_ext_mode(e)
            if int(e.get('tel_slam_fired', 0)) == 1:
                slam_fired = True
                break
        if not slam_fired:
            raise RuntimeError(
                f"{self._prefix} armed 2kHz brake slam never fired within 60s sim "
                f"(target {target_ms:.2f} m/s, airspeed="
                f"{None if e is None else e.get('airspeed')})")

        # ====================================================================
        # IN-CAR HANDOFF (the divergence from parent): the pedal is already
        # latched in-Lua at the exact crossing; mirror it from the game-input
        # side so beamngpy's input state agrees. Controller engage fires on
        # driverBrake>0.9 && speed>8.0 -> onEngage -> resetAccum + warmup.
        # (PPO_V2) IN GEAR, no gear=0. Car stays in drive until ~2.5 mph.
        # ====================================================================
        self.vehicle.control(brake=1.0, throttle=0.0)
        # Re-assert ext mode AGAIN after the teleport re-init (defensive; also done
        # in the loop below).
        self.vehicle.queue_lua_command(f"{self._ctrl_call}.setExtMode(true)")

        # Ride through engage + warmup until the controller is ACTIVE, PAST warmup,
        # and has published its first REAL obs (tickseq>=1). Hard 2.0s ceiling.
        engaged = False
        e = None
        for _ in range(ENGAGE_RIDE_TICKS):
            self.bng.step(1)
            e = self._poll_electrics()
            self._assert_ext_mode(e)
            active = int(e.get('mlabs_active', 0))
            warmup = int(e.get('mlabs_warmup', 0))
            tickseq = int(e.get('mlabs_tickseq', 0))
            if active == 1 and warmup == 0 and tickseq >= 1:
                engaged = True
                break
        if not engaged:
            raise RuntimeError(
                f"{self._prefix} in-car controller never engaged/exited warmup "
                f"within {ENGAGE_RIDE_TICKS} ticks (2.0s) of brake slam "
                f"(last electrics: active={None if e is None else e.get('mlabs_active')}, "
                f"warmup={None if e is None else e.get('mlabs_warmup')}, "
                f"tickseq={None if e is None else e.get('mlabs_tickseq')}, "
                f"extmode={None if e is None else e.get('mlabs_extmode')})")

        # Start speed + heading from NON-draining sources (no readAll).
        # airspeed is a stock electrics value (not poll-window driven); heading from
        # the controller's published mlabs_heading (obj:getDirection scalar).
        self.start_speed_ms = float(e.get('airspeed', 0.0))
        self.start_heading = float(e.get('mlabs_heading', 0.0))
        self.current_heading = self.start_heading
        self.target_heading = self.start_heading

        # obs0 = the 27 raw obs the controller just published.
        obs0 = self._published_obs_vector(e)

        self._seq = 0
        self.steps_taken = 0
        self.episode_count += 1
        self._ep_wall_start = time.monotonic()
        return obs0, {}

    # -------------------------------------------------------------------- step
    def step(self, action):
        """Mailbox the action -> step physics -> read PUBLISHED obs -> reward v5.0 from
        non-draining sources. The env NEVER drains the poll window (controller is sole
        drainer). Reward computation is BYTE-IDENTICAL to abs_env.step(); only the
        INPUT SOURCE of each term changes (see BUILD_SPEC §4.4/§4.5)."""
        # Map action[0..3] in [0,1] -> brake [0.01,1.0] EXACTLY as parent
        # (abs_env.step: brakes = 0.01 + 0.99*action; np.clip(..,0.01,1.0)).
        brakes = 0.01 + 0.99 * action[:4].astype(np.float64)
        brakes = np.clip(brakes, 0.01, 1.0)
        predicted_speed = 0.0  # speed-guesser removed (parent keeps it as 0)
        fr, fl, rr, rl = brakes
        self._seq += 1

        # --- MAILBOX WRITE (before step; consumed by the tick inside this step, k=0
        #     convention, verified by T1). One atomic call delivers 4 floats + seq. ---
        self.vehicle.queue_lua_command(
            f"{self._ctrl_call}.setExtCmd({fr},{fl},{rr},{rl},{self._seq})")
        # Keep the driver pedal slammed so the controller stays engaged. The deployed
        # controller overwrites input.brake with maxBrake internally anyway, so this
        # is purely "stay engaged", the controller's per-wheel cmd comes from setExtCmd.
        # (PPO_V2) Stay in DRIVE until ~2.5 mph, then drop to NEUTRAL exactly once
        # (the chosen regime, avoids auto-box creep fighting the final stop).
        if not self._neutral_dropped and self._last_gps_speed < 1.118:  # 2.5 mph
            self.vehicle.control(brake=1.0, gear=0)
            self._neutral_dropped = True
        else:
            self.vehicle.control(brake=1.0)

        # --- STEP (FRAME_SKIP=1, sacred) ---
        self.bng.step(FRAME_SKIP)
        self.steps_taken += 1

        # --- READ PUBLISHED OBS (NO readAll / NO buildSensorData from Python) ---
        e = self._poll_electrics()
        obs = self._published_obs_vector(e)  # mlabs_o0..o26 EXACTLY, no recompute

        self.prev_brakes = brakes.astype(np.float32)  # bookkeeping parity (unused for obs)

        # =================================================================
        # REWARD v5.0, BYTE-IDENTICAL arithmetic, non-draining input sources.
        # =================================================================
        # Inputs (see mapping table §4.5):
        #   gy_avg  -> published obs[4] (== controller's gy_avg, mlabs_o4)
        #   yaw_avg -> published obs[6] (== controller's yaw_avg, mlabs_o6)
        braking_g_ms2 = float(obs[4])
        yaw_rate = float(obs[6])
        self.prev_heading_error = 0.0  # placeholder set below for legacy compat

        # speed-prediction tracking (parent keeps spd_rew=0 under Rule 2). Kept so
        # downstream stat refs don't break; spd_error uses the non-draining speed.
        gps_speed_for_spd = float(e.get('tel_fused_speed', self._last_gps_speed))
        spd_error = abs(predicted_speed - gps_speed_for_spd)
        high_yaw = abs(yaw_rate) > 0.3
        if not high_yaw and gps_speed_for_spd > 1.0:
            self._spd_ep_errors.append(spd_error)

        # Per-episode stat tracking (parent semantics; slip unavailable without drain,
        # so ep_peak_slip stays 0, it's a diagnostic-only field, never in reward).
        braking_g_gs = abs(braking_g_ms2) / 9.81
        self.ep_peak_g = max(self.ep_peak_g, braking_g_gs)
        self.ep_max_yaw = max(self.ep_max_yaw, abs(yaw_rate))

        reward = 0.0
        spd_rew = 0.0  # SPEED-PREDICTION REWARD REMOVED per Rule 2 (parent: 0)

        # ─── v5.0 PER-STEP G-FORCE REWARD (BYTE-IDENTICAL to abs_env.step:712-714) ──
        step_g_rew = PER_STEP_K * _terminal_g_shape(braking_g_gs)
        reward += step_g_rew
        self.ep_step_g_sum += step_g_rew

        # heading_error, grading-side ONLY (crash terminal at >CRASH_HEADING).
        # Parent reads heading from readAll()['heading'] (DRAINS). Here: mlabs_heading
        # (controller publishes obj:getDirection() scalar every physics step, no drain).
        self.current_heading = float(e.get('mlabs_heading', self.current_heading))
        heading_error = abs(self.current_heading - self.target_heading)
        self.prev_heading_error = heading_error  # legacy state (parent keeps this)

        # ─── v5.0 YAW (BYTE-IDENTICAL to abs_env.step:727-746) ──
        yaw_error = abs(yaw_rate - self.target_yaw_rate)
        if braking_g_gs > PER_STEP_G_GATE:
            yaw_bonus = YAW_BONUS_K_STEP * math.exp(-yaw_error * YAW_BONUS_ALPHA)
        else:
            yaw_bonus = 0.0
        reward += yaw_bonus
        self.ep_yaw_penalty += yaw_bonus  # field-name kept for CSV-compat
        self.ep_yaw_abs_sum += yaw_error * self._dt
        if yaw_error > YAW_RATE_DEADZONE_RAD_S:
            excess = yaw_error - YAW_RATE_DEADZONE_RAD_S
            self.ep_yaw_sq_sum += excess * excess * self._dt

        yaw_pen = yaw_bonus  # alias parity with parent's audit-log naming

        # --- Audit log (per-step, non-terminal rows), parent format ---
        force_override = 0
        self._audit_step_data = (obs, brakes, predicted_speed, gps_speed_for_spd,
                                 braking_g_gs, force_override,
                                 step_g_rew, yaw_pen, spd_rew, reward)
        if self._audit_writer:
            self._audit_writer.writerow([
                self.env_index, self.episode_count, self.steps_taken, self.target_mph,
                *[round(float(v), 4) for v in obs],
                round(fr, 4), round(fl, 4), round(rr, 4), round(rl, 4),
                round(gps_speed_for_spd, 2), round(gps_speed_for_spd / 0.44704, 1),
                round(predicted_speed, 2),
                round(braking_g_gs, 4), force_override,
                round(step_g_rew, 4), round(yaw_pen, 4), round(spd_rew, 4),
                round(reward, 4),
                '', '', '', '',
            ])

        # ─── stop-detect speed: NON-draining source ──
        # Parent's stop gate (abs_env.py:936) uses data['inst_speed'] with a fallback to
        # data['airspeed'], both are BODY/GPS velocity that reach ~0 at standstill. We
        # MUST match that: use the stock non-draining 'airspeed' electric here.
        #
        # NOTE (validated in-sim, 2026-06-05): tel_fused_speed CANNOT be used for the stop
        # gate. abstelemetry's fused-speed integrator only re-syncs DOWNWARD on brake
        # RELEASE; because this env holds brake=1.0 every step, the integrator stays in its
        # isBraking branch and PLATEAUS (~0.7 m/s) at a true standstill, so it never crosses
        # the 0.05 threshold -> STOP never fires -> every episode TIMEOUTs at 5000 steps with
        # no terminal g-reward. 'airspeed' read 0.002 m/s at the same standstill and stays
        # truthful under lockup (it is not wheel-derived). It is non-draining (stock electric,
        # populated independently of buildSensorData). tel_fused_speed is still read below
        # for the speed-tracking stat only (never gates the episode).
        gps_speed = float(e.get('airspeed', 999.0))
        self._last_gps_speed = gps_speed

        terminated = False

        # --- TIMEOUT (BYTE-IDENTICAL structure to parent) ---
        if self.steps_taken >= MAX_EPISODE_STEPS:
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum
            reward += acc_yaw_pen
            print(f"{self._prefix} [EP {self.episode_count:4d}] TIMEOUT: "
                  f"{self.target_mph:.0f}mph, {self.steps_taken} steps "
                  f"| YawInt: {self.ep_yaw_abs_sum:.3f} AccYaw: {acc_yaw_pen:+.1f}")
            self._log_episode('TIMEOUT', avg_g=0.0, terminal_rew=acc_yaw_pen)
            self._write_audit_terminal(0.0, 0.0, acc_yaw_pen, 'TIMEOUT')
            terminated = True
            return obs, reward, terminated, False, {}

        # --- CRASH (BYTE-IDENTICAL structure to parent) ---
        if heading_error > CRASH_HEADING:
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum
            terminal_rew = CRASH_PENALTY + acc_yaw_pen
            reward += terminal_rew
            print(f"{self._prefix} [EP {self.episode_count:4d}] CRASH (heading): "
                  f"{self.target_mph:.0f}mph, heading={math.degrees(heading_error):.1f}° "
                  f"| YawInt: {self.ep_yaw_abs_sum:.3f} AccYaw: {acc_yaw_pen:+.1f}")
            self._log_episode('CRASH', avg_g=0.0, terminal_rew=terminal_rew)
            self._write_audit_terminal(0.0, 0.0, terminal_rew, 'CRASH')
            terminated = True
            return obs, reward, terminated, False, {}

        # --- STOP detection (BYTE-IDENTICAL structure to parent) ---
        if gps_speed < 0.05:
            self.stop_timer += 1
        else:
            self.stop_timer = 0

        if self.stop_timer >= STOP_FRAMES:
            terminated = True

            # Terminal: disarm the in-Lua pedal latch FIRST (it re-asserts
            # input.brake=1 every physics tick), then release the driver brake so
            # the controller disengages, step, then read the brake-event terminal
            # values. These are published by updateGFX (60Hz) and do NOT drain the
            # poll window: tel_last_brake_avg_g / tel_last_brake_dist.
            self.vehicle.queue_lua_command("extensions.abstelemetry.disarmBrakeSlam()")
            self.vehicle.control(brake=0.0)
            self.bng.step(5)
            e2 = self._poll_electrics()

            avg_g = float(e2.get('tel_last_brake_avg_g', 0.0))
            self.ep_stopping_dist = float(e2.get('tel_last_brake_dist', 0.0))

            # ─── v5.0 TERMINAL G-FORCE REWARD (BYTE-IDENTICAL to abs_env.step:827) ──
            terminal_g_rew = _terminal_g_shape(avg_g)

            # ─── v5.0 TERMINAL YAW BONUS (BYTE-IDENTICAL to abs_env.step:832-836) ──
            if avg_g > PER_STEP_G_GATE:
                yaw_clean_factor = max(0.0, 1.0 - self.ep_yaw_abs_sum / YAW_BONUS_THRESHOLD)
                terminal_yaw_bonus = YAW_BONUS_K_TERMINAL * yaw_clean_factor
            else:
                terminal_yaw_bonus = 0.0

            # ─── catastrophic-yaw accumulator backstop (BYTE-IDENTICAL) ──
            acc_yaw_pen = -YAW_PEN_K_TERMINAL * self.ep_yaw_sq_sum

            terminal_rew = terminal_g_rew + terminal_yaw_bonus + acc_yaw_pen
            reward += terminal_rew

            stop_time = self.steps_taken / DETERM_HZ
            total_ep_reward = (self.ep_step_g_sum + self.ep_yaw_penalty + terminal_rew)

            spd_acc = ""
            if self._spd_ep_errors:
                avg_err = sum(self._spd_ep_errors) / len(self._spd_ep_errors)
                spd_acc = f" | SpdErr: {avg_err:.1f}m/s"
            try:
                self._spd_file.flush()
            except Exception:
                pass

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
