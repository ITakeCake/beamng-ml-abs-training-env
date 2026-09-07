"""The reward, as data instead of hard-coded constants (PLAN_V2.md section 6).

Four presets ship:

  RewardSpec.v5()        , the frozen default. Reproduces the protected
                              abs_env.py's numbers EXACTLY; a parity test
                              asserts this against the real constants and
                              _terminal_g_shape, not a recorded snapshot.
  RewardSpec.normalized(), anchors moved onto the per-config MEASURED
                              references (calibration.py): zero at stock ABS,
                              strongly negative at lockup.
  RewardSpec.v6()        , target-free, G-primary reward. Scores sustained
                              positive deceleration over a short rolling
                              window, with a small terminal stability guard.
  RewardSpec.v7()        , v6's terminal G and stability guard, but the
                              per-step signal is a banded true-slip term
                              instead of dense G: reward at/below 0.50 slip,
                              penalty above it, 3x penalty at/above 0.99.

Why normalized exists, in one measurement (etk800, 60 mph, dry, 2026-08-29):
v5.0 pays +966 for locking the wheels, +987 for this project's best result
(1.043 g) and +5049 for what stock ABS does unaided (1.188 g). Its "exceptional
performance" gatekeeper sits at 1.06 g, BELOW stock. The absolute shape could
not distinguish the brake-slammer local optimum from real progress.

No game imports.
"""
import dataclasses
from collections import deque
import hashlib
import inspect
import json
import math

from calibration import normalized_g


@dataclasses.dataclass
class RewardSpec:
    # --- g-force shape -------------------------------------------------
    ramp_pivot: float          # g (or normalized g) where the ramp crosses zero
    ramp_neg_k: float          # slope below the pivot
    ramp_pos_k: float          # slope above the pivot
    gatekeeper: float          # where the step bonus + quadratic kick in
    step_bonus: float          # instant jump at the gatekeeper
    quad_k: float              # quadratic coefficient above the gatekeeper
    clamp_hi: float            # safety clamp on the input

    # --- per-step ------------------------------------------------------
    per_step_k: float          # per-step reward = k * shape(g_step)
    per_step_g_gate: float     # below this g, no yaw bonus (anti-coast)

    # --- yaw -----------------------------------------------------------
    yaw_bonus_k_step: float
    yaw_bonus_alpha: float
    yaw_bonus_k_terminal: float
    yaw_bonus_threshold: float
    yaw_pen_k_terminal: float
    yaw_rate_deadzone: float

    # --- terminals -----------------------------------------------------
    crash_penalty: float
    timeout_penalty: float = 0.0

    # --- v6 dense reward -----------------------------------------------
    # ``shape`` preserves the v5/normalized one-sample reward.  The rolling
    # mode must be evaluated through make_step_tracker(), because its score
    # depends on recent samples and the real simulation dt.
    step_mode: str = "shape"
    # "inst": per-step G is the raw 2 kHz sample (v6). "window": the 10 ms
    # speed-delta average abstelemetry publishes (tel_win_gy_avg), far cleaner.
    dense_g_source: str = "inst"

    # --- FastTrain v3.3 clone ("v3.3") ---------------------------------
    # shape_mode "v33": g clamped to [0.4, 2.0] then affine -140 .. +2100 with
    # breakeven at 0.5 g, plus a terminal-only log gatekeeper bonus above
    # v33_gate_g. Per-step (step_mode "v33_step") uses the same affine without
    # the gate, scaled by per_step_k. Heading terms are per step: -k*err^2 plus
    # recovery_k * (prev_err - err), err = |integrated yaw rate| in rad.
    shape_mode: str = "ramp"
    v33_gate_g: float = 1.06
    v33_gate_k: float = 2500.0
    v33_gate_scale: float = 10.0
    heading_pen_k: float = 0.0
    heading_recovery_k: float = 0.0
    rolling_window_s: float = 0.0
    dense_g_k: float = 0.0
    consistency_k: float = 0.0

    # --- optional terminal stability guard -----------------------------
    # This is a safety constraint around the G objective, not a per-step yaw
    # target. Cost is zero through the dead zone, rises quadratically, and is
    # capped at yaw_stability_penalty_max.
    yaw_stability_deadzone: float = 0.0
    yaw_stability_full_scale: float = 0.0
    yaw_stability_penalty_max: float = 0.0

    # --- v7 per-step banded slip term ----------------------------------
    # Per wheel, per second: +slip_ok_k while slip <= slip_ok_max (v7 uses a
    # NEGATIVE ok_k: any positive per-step term rewards long coasting stops),
    # -slip_pen_k between the bands, -slip_lock_k at/above slip_lock_min.
    # Averaged over the wheels and scaled by dt.  All zero = term disabled.
    slip_ok_k: float = 0.0
    slip_pen_k: float = 0.0
    slip_lock_k: float = 0.0
    slip_ok_max: float = 0.50
    slip_lock_min: float = 0.99

    # --- v11: confine the distance integral to the scored window -------
    # The env couples ~4.5 m/s above the 80 mph trigger and the metric only
    # opens at the crossing, so a policy that does not brake coasts ~19 s and
    # ~760 m before it is scored at all, fourteen times the ~55 m the metric
    # actually measures. metric_window_only charges the distance integral only
    # while tel_brake_active is set, making the dense sum exactly minus the
    # scored stopping distance; approach_time_k is a flat per-second cost
    # outside that window so the approach cannot be dawdled through instead.
    metric_window_only: bool = False
    approach_time_k: float = 0.0

    # --- v9 distance-integral dense term -------------------------------
    # Per step: -speed_cost_k * ground_speed * dt.  Summed over the episode
    # this is exactly -speed_cost_k * metres travelled, so the dense signal
    # IS the scored quantity. 0 = term disabled.
    speed_cost_k: float = 0.0

    # --- normalization -------------------------------------------------
    # False: `g` is absolute g and the anchors above are absolute (v5.0).
    # True:  `g` is first mapped through calibration.normalized_g() so that
    #        0 = measured lockup and 1 = measured stock ABS on THIS config,
    #        and the anchors above are in those normalized units.
    normalize: bool = False

    name: str = "custom"

    # ------------------------------------------------------------------
    @classmethod
    def v5(cls):
        """The frozen default. Values mirror abs_env.py:50-102 exactly."""
        return cls(
            ramp_pivot=0.5,
            ramp_neg_k=2000.0,
            ramp_pos_k=1818.18,
            gatekeeper=1.06,
            step_bonus=500.0,
            quad_k=200000.0,
            clamp_hi=2.5,
            per_step_k=0.0011,
            per_step_g_gate=0.30,
            yaw_bonus_k_step=1.0,
            yaw_bonus_alpha=20.0,
            yaw_bonus_k_terminal=300.0,
            yaw_bonus_threshold=0.1,
            yaw_pen_k_terminal=5000.0,
            yaw_rate_deadzone=0.05,
            crash_penalty=-2000.0,
            normalize=False,
            name="v5.0",
        )

    @classmethod
    def normalized(cls):
        """Anchors on the measured references instead of absolute g.

        In normalized units: 0 = lockup, 1 = stock ABS, and one unit above 1
        means "beat stock by as much as stock beat locked wheels".

        The MAGNITUDES here are a first cut, chosen to keep v5.0's familiar
        scale (~1000 per component, -2000 crash) while moving the anchors:
          g_n = 0   (lockup)                     -> -1000
          g_n = 1   (stock ABS)                  ->     0
          g_n = 1.1 (beat stock by 10% of its margin) -> gatekeeper fires
          g_n = 1.5 (beat stock by half its margin)   -> ~+1000 + quadratic
        The STRUCTURE (zero at stock, strongly negative at lockup) is the
        principled part; these numbers are sliders and should be tuned against
        real runs rather than treated as derived truth."""
        return cls(
            ramp_pivot=1.0,
            ramp_neg_k=1000.0,
            ramp_pos_k=2000.0,
            gatekeeper=1.1,
            step_bonus=500.0,
            quad_k=12500.0,
            clamp_hi=3.0,
            per_step_k=0.0011,
            per_step_g_gate=0.30,
            yaw_bonus_k_step=1.0,
            yaw_bonus_alpha=20.0,
            yaw_bonus_k_terminal=300.0,
            yaw_bonus_threshold=0.1,
            yaw_pen_k_terminal=5000.0,
            yaw_rate_deadzone=0.05,
            crash_penalty=-2000.0,
            normalize=True,
            name="normalized",
        )

    @classmethod
    def v6(cls):
        """Target-free reward for sustained positive braking G.

        The dense component uses a 100 ms rolling mean minus half of the
        population standard deviation.  This makes steady G preferable to an
        equal-mean high/low oscillation without inventing a desired G target.
        The terminal component remains monotonic and uncapped: a measured
        1.5 g stop scores more than 1.0 g, and 2.0 g scores more than 1.5 g.

        The deliberately compact scale keeps normal episode returns in the
        tens instead of the thousands.  At 100 Hz, a steady 1 g contributes
        about 0.01 per step plus 25 at a successful stop. A terminal-only
        stability guard is free through 0.08 rad of integrated absolute yaw,
        then costs at most 3 points by 0.20 rad. It supplies no per-step yaw
        shaping and cannot overwhelm the G objective.
        """
        return cls(
            ramp_pivot=0.0,
            ramp_neg_k=25.0,
            ramp_pos_k=25.0,
            gatekeeper=0.0,
            step_bonus=0.0,
            quad_k=0.0,
            clamp_hi=0.0,  # <= 0 means deliberately uncapped
            per_step_k=1.0,
            per_step_g_gate=0.0,
            yaw_bonus_k_step=0.0,
            yaw_bonus_alpha=0.0,
            yaw_bonus_k_terminal=0.0,
            yaw_bonus_threshold=0.1,
            yaw_pen_k_terminal=0.0,
            yaw_rate_deadzone=0.05,
            crash_penalty=-50.0,
            timeout_penalty=-30.0,
            step_mode="rolling_consistency",
            rolling_window_s=0.100,
            dense_g_k=1.0,
            consistency_k=0.5,
            yaw_stability_deadzone=0.08,
            yaw_stability_full_scale=0.20,
            yaw_stability_penalty_max=3.0,
            normalize=False,
            name="v6.0",
        )

    @classmethod
    def v6_1(cls):
        """v6.0 scoring, byte-identical, with four independent wheel releases.

        The reward arithmetic is v6's; only the name differs, and the trainer
        maps the name to wheel_mode="independent" (obs 15, act 4) instead of
        v6.0's front/rear axle lock (obs 13, act 2).
        """
        return dataclasses.replace(cls.v6(), name="v6.1")

    @classmethod
    def v7(cls):
        """v6's terminal G and stability guard with a banded slip per-step COST.

        Three per-step bands on TRUE wheel slip (ground-speed referenced,
        privileged information the reward may use and the policy never sees):
        slip <= 0.50 costs 5/s per wheel (a plain time cost while the wheel
        is fine), 0.50 < slip < 0.99 costs 10/s, slip >= 0.99 (locked) costs
        30/s.  Wheels are averaged.

        The per-step term is NEVER positive.  The original +10/s ok-band was a
        survival bonus: PPO-49 learned 24 s coasts at 0.2 g that out-scored
        every real stop.  With a pure cost the cheapest episode is the shortest
        one that avoids lock, which is the objective, and the ordering holds
        for any start speed or surface because the policy cannot change the
        condition it is judged in.  Sizing from 80 mph: 1.1 g at slip 0.2
        ~ -17 + 27.5 terminal = +11; a 6 s 0.6 g coast ~ -30 + 15 = -15;
        a 24 s timeout ~ -120 - 30; a 3.5 s slam ~ -105 + 24.

        Dense G shaping is off; the terminal avg-G reward carries "stop hard"
        exactly as in v6.  Yaw is v6's structure at a weight that cannot be
        overrun: the deadzoned terminal guard is free to 0.08 rad of
        integrated |yaw| and costs up to 25 by 0.20 rad, and the uncapped
        quadratic backstop (200 * integral of yaw^2 dt) keeps growing past
        that, so a spin can never out-score a straight stop.  Actions are
        four independent wheel releases (no axle lock).
        """
        spec = cls.v6()
        return dataclasses.replace(
            spec,
            dense_g_k=0.0,
            consistency_k=0.0,
            yaw_stability_penalty_max=25.0,
            yaw_pen_k_terminal=200.0,
            slip_ok_k=-5.0,   # <= 0 only: a positive ok-band pays for coasting
            slip_pen_k=10.0,
            slip_lock_k=30.0,
            slip_ok_max=0.50,
            slip_lock_min=0.99,
            name="v7.0",
        )

    @classmethod
    def v9(cls):
        """Distance-integral dense reward: the dense term IS the metric.

        Every earlier preset shaped on measured G, which arrives as a +-3 g
        instantaneous sample (or a 10 ms window average of it) at 100 Hz.  The
        episode-to-episode noise in that term, and in v3.3's -1000*err^2
        heading term, is larger than the whole 0.98 g (locked) to 1.20 g
        (stock ABS) span the run is trying to resolve, so the gradient the
        policy actually sees is mostly noise.

        v9 pays -1 per metre travelled, per step: the sum over an episode is
        exactly -(metres), and metres is what avg_g is computed from.  Ground
        speed is privileged reward-only information (never in the obs), and it
        is far cleaner than any accelerometer channel.  There is no way to farm
        duration: standing still earns 0, and every metre costs.

        Sizing on Machine-Trainer-Boy at 80 mph (measured: slam 0.979 g,
        stock ABS 1.199 g):
          1.19 g -> 54 m: -54 dense, +99 terminal  = +45
          0.98 g -> 66 m: -66 dense,   0 terminal  = -66
          0.67 g -> 97 m: -97 dense,  -93 terminal = -190
          24 s coast     : -600 penalty on top of the metres it did travel
          spin-out       : -600 plus the yaw backstop
        Both failures are pinned below the worst legitimate stop. A 0.4 g stop
        scores about -337 (163 m of dense cost plus a -174 terminal), and a
        timeout that crawls to a halt accrues only ~-142 of dense cost because
        speed, and therefore cost, decays as it slows. Leaving the timeout
        unpenalised made NOT stopping the higher-scoring option, which PPO-60
        found within eleven episodes.
        Terminal is a ramp through the measured lockup g, so beating a locked
        wheel is where reward turns positive, with a quadratic kicker above
        1.10 g to sharpen the top of the range where the run has to finish.
        Magnitudes are kept O(100) on purpose: rewards are not normalized
        (norm_reward=False), and the older presets' O(2000) terminals made the
        value target dwarf the dense signal.
        """
        return cls(
            ramp_pivot=0.979, ramp_neg_k=300.0, ramp_pos_k=300.0,
            gatekeeper=1.10, step_bonus=20.0, quad_k=2000.0, clamp_hi=1.6,
            per_step_k=0.0, per_step_g_gate=0.0,
            yaw_bonus_k_step=0.0, yaw_bonus_alpha=0.0,
            yaw_bonus_k_terminal=0.0, yaw_bonus_threshold=0.1,
            yaw_pen_k_terminal=20.0, yaw_rate_deadzone=0.0,
            crash_penalty=-600.0, timeout_penalty=-600.0,
            step_mode="distance", dense_g_source="window",
            yaw_stability_deadzone=0.30, yaw_stability_full_scale=0.60,
            yaw_stability_penalty_max=15.0,
            speed_cost_k=1.0,
            normalize=False, name="v9.0",
        )

    @classmethod
    def v10(cls):
        """v9's distance objective plus ReinVBC's slip constraint, sized for this car.

        v9 alone tells the policy "travel fewer metres" and nothing else, which
        a constant partial brake already answers: every run so far converged to
        a near-constant 0.32 release wobbling at 3-7 Hz (PPO-57 diagnostics) for
        ~1.05 g. Beating that needs the release to track wheel state, and the
        published controller that does it (ReinVBC, arXiv 2604.04401) pays for
        the objective with a speed integral and CONSTRAINS slip with a separate
        indicator cost, rather than making slip the objective.

        The band here is one-sided on purpose. Under-braking is already the most
        expensive thing in v9 (it costs metres), so a lower slip bound would be
        double-counting; only the upper side needs a term. It opens at 0.25,
        just past this car's measured peak-grip slip of ~0.21, and the locked
        band is 8/s per wheel. Sized against v9's ~12 point distance span and
        ~100 point terminal span: four wheels past the band for a whole 5.5 s
        stop costs 11, four locked costs 44. Enough to make lockup lose, not
        enough to make the constraint the objective, which is what killed v7
        (its "ok" band ran to 0.50 slip, deep past the peak, and it had no dense
        objective at all to trade against).
        """
        return dataclasses.replace(
            cls.v9(),
            slip_ok_k=0.0,          # in-band is free; distance already pays for it
            slip_ok_max=0.25,
            slip_pen_k=2.0,
            slip_lock_k=8.0,
            slip_lock_min=0.99,
            name="v10.0",
        )

    @classmethod
    def v11(cls):
        """v9's distance integral, charged only over the window that is scored.

        PPO-62 episode 10 was 23.2 s long and only 3.7 s of it was the brake
        event: the env hands over ~4.5 m/s above the 80 mph trigger, and a
        policy that is not yet braking coasts on drag alone until it crosses.
        Under v9 that approach accrued roughly -760 of distance cost against the
        ~-55 the measured stop is worth, so the reward was mostly grading a
        phase the metric ignores, the same signal-to-noise failure v9 was
        written to fix, arriving through a different door.

        v11 charges the metre-by-metre cost only while tel_brake_active is set,
        which the vehicle Lua clears in the same block that latches avg_g. The
        dense sum is then exactly minus the scored stopping distance. Outside
        that window a flat 3/s applies instead: enough that a 19 s coast costs
        57 and a brisk approach costs 3, so dawdling is never free, and small
        enough that it cannot outweigh the stop itself.
        """
        return dataclasses.replace(
            cls.v9(),
            metric_window_only=True,
            approach_time_k=3.0,
            name="v11.0",
        )

    @classmethod
    def v3_3(cls):
        """Clone of FastTrain reward v3.3, the 1.16 g in-env model (2026-04), under its own name.

        Source: Beamng_AI/ml/PPO/FastTrain/abs_env.py. Terminal: avg_g clamped
        to [0.4, 2.0], -140 at 0.4 g, 0 at 0.5 g, +2100 at 2.0 g, plus
        log1p((g - 1.06) * 10) * 2500 above the 1.06 g gatekeeper. Per step:
        the same affine on the poll-averaged G times PER_STEP_K. FastTrain ran
        at 160 Hz with K = 0.000225; co-sim runs at 100 Hz, so K = 0.00036
        keeps the per-second weight identical. Heading: -1000 * err^2 per
        step plus 500 * (prev_err - err) recovery, err = |integrated yaw|.
        Crash -500, timeout 0. No slip term, no terminal yaw term, no
        consistency term. The speed-prediction auxiliary reward is NOT
        cloned (it was ruled a sensor-honesty violation). Independent wheels,
        as FastTrain's 4-wheel brakes were.
        Note the affine is a time cost (-25/s) plus 50/s per g: the G part
        integrates to the speed change, so duration cannot be farmed.
        """
        return cls(
            ramp_pivot=0.0, ramp_neg_k=0.0, ramp_pos_k=0.0, gatekeeper=0.0,
            step_bonus=0.0, quad_k=0.0, clamp_hi=0.0,
            per_step_k=0.000225 * 160.0 / 100.0,
            per_step_g_gate=0.0,
            yaw_bonus_k_step=0.0, yaw_bonus_alpha=0.0, yaw_bonus_k_terminal=0.0,
            yaw_bonus_threshold=0.1, yaw_pen_k_terminal=0.0, yaw_rate_deadzone=0.0,
            crash_penalty=-500.0, timeout_penalty=0.0,
            step_mode="v33_step", dense_g_source="window",
            shape_mode="v33", v33_gate_g=1.06, v33_gate_k=2500.0, v33_gate_scale=10.0,
            heading_pen_k=1000.0, heading_recovery_k=500.0,
            normalize=False, name="v3.3",
        )

    @classmethod
    def v8(cls):
        """v7's slip COST plus a capped per-step reward on the clean window G.

        PPO-55 (v7.0) sat at its initialisation for 500k steps: the ok band
        costs -5/s whatever the deceleration, so "brake harder now" only shows
        up as a shorter episode, which a 200-step critic horizon cannot see.
        v8 adds dense_g_k * dt * G per step on the 10 ms window average (not
        the +-3 g instantaneous sample v6 used, and with no consistency term).
        3/s per g is deliberately BELOW the 5/s ok-band time cost, so the
        per-step total stays negative up to 1.67 g and coasting can never pay.
        Ordering from 80 mph: 1.1 g ok-band ~ -6.6 + 27.5 = +21; 0.6 g 6 s
        coast ~ -19 + 15 = -4; 24 s timeout ~ -110; locked 3.5 s ~ -95 + 24.
        Yaw guard deadzone widened to 0.30 rad (full 0.45): 7 s stops
        integrate ~0.2 rad of yaw-rate noise, which pinned v7's guard at -25
        from episode 1 with no gradient. The quadratic backstop is unchanged.
        """
        return dataclasses.replace(
            cls.v7(),
            step_mode="window_g",
            dense_g_source="window",
            dense_g_k=3.0,          # per g per second; must stay < -slip_ok_k
            consistency_k=0.0,
            yaw_stability_deadzone=0.30,
            yaw_stability_full_scale=0.45,
            name="v8.0",
        )

    _SLIP_FIELDS = ("slip_ok_k", "slip_pen_k", "slip_lock_k",
                    "slip_ok_max", "slip_lock_min")

    def slip_term_active(self):
        return any(getattr(self, k) != 0.0
                   for k in ("slip_ok_k", "slip_pen_k", "slip_lock_k"))

    # ------------------------------------------------------------------
    def to_scale(self, g, refs=None):
        """Absolute g -> the units this spec's anchors are expressed in."""
        if not self.normalize:
            return g
        if refs is None:
            raise ValueError(
                "this reward spec is normalized but no calibration references "
                "were supplied, refusing to score against absolute anchors "
                "(that is exactly the bug normalization exists to fix).")
        slam_g, stock_g = refs
        return normalized_g(g, slam_g, stock_g)

    def v33_affine(self, g):
        """FastTrain v3.3 base shape: [0.4, 2.0] g -> [-140, 2100], no gate."""
        x = max(0.0, min((max(0.0, min(float(g), 2.0)) - 0.4) / 1.6, 1.0))
        return -140.0 * (1.0 - x) + 2100.0 * x

    def step_heading_penalty(self, heading_err, prev_heading_err):
        """Per-step heading cost with a recovery term (FastTrain v3.3)."""
        if self.heading_pen_k == 0.0 and self.heading_recovery_k == 0.0:
            return 0.0
        err = abs(float(heading_err))
        return (-self.heading_pen_k * err * err
                + self.heading_recovery_k * (float(prev_heading_err) - err))

    def g_shape(self, g, refs=None):
        """The shape both the per-step and terminal g rewards run through."""
        if self.shape_mode == "v33":
            out = self.v33_affine(g)
            gc = max(0.0, min(float(g), 2.0))
            if gc > self.v33_gate_g:
                out += math.log1p((gc - self.v33_gate_g) * self.v33_gate_scale) \
                    * self.v33_gate_k
            return out
        x = self.to_scale(g, refs)
        if self.clamp_hi > 0.0:
            x = (max(0.0, min(x, self.clamp_hi)) if not self.normalize
                 else min(x, self.clamp_hi))
        elif not self.normalize:
            x = max(0.0, x)
        if x < self.ramp_pivot:
            out = -self.ramp_neg_k * (self.ramp_pivot - x)
        else:
            out = self.ramp_pos_k * (x - self.ramp_pivot)
        if x >= self.gatekeeper:
            d = x - self.gatekeeper
            out += self.step_bonus + self.quad_k * d * d
        return out

    # --- the four reward terms the env actually calls -------------------
    def step_g_reward(self, g_step, refs=None):
        if self.step_mode != "shape":
            raise RuntimeError(
                "rolling reward requires RewardSpec.make_step_tracker(dt); "
                "a single G sample has no rolling consistency information")
        return self.per_step_k * self.g_shape(g_step, refs)

    def make_step_tracker(self, dt):
        """Create fresh per-episode state for dense G scoring."""
        return StepGRewardTracker(self, dt)

    def step_slip_reward(self, slips, dt):
        """Banded per-step slip term, averaged over wheels, scaled by dt."""
        if not self.slip_term_active():
            return 0.0
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("slip reward dt must be finite and positive")
        total = 0.0
        n = 0
        for s in slips:
            s = float(s)
            if s <= self.slip_ok_max:
                total += self.slip_ok_k
            elif s >= self.slip_lock_min:
                total -= self.slip_lock_k
            else:
                total -= self.slip_pen_k
            n += 1
        if n == 0:
            return 0.0
        return dt * total / n

    def step_yaw_bonus(self, g_step, yaw_error):
        if g_step > self.per_step_g_gate:
            return self.yaw_bonus_k_step * math.exp(-yaw_error * self.yaw_bonus_alpha)
        return 0.0

    def accumulated_yaw_penalty(self, yaw_sq_sum):
        return -self.yaw_pen_k_terminal * yaw_sq_sum

    def terminal_stability_penalty(self, yaw_abs_sum):
        """Small, dead-zoned terminal cost for a visibly non-straight stop."""
        if self.yaw_stability_penalty_max <= 0.0:
            return 0.0
        span = self.yaw_stability_full_scale - self.yaw_stability_deadzone
        if span <= 0.0:
            raise ValueError(
                "yaw_stability_full_scale must exceed yaw_stability_deadzone")
        yaw = max(0.0, float(yaw_abs_sum))
        if yaw <= self.yaw_stability_deadzone:
            return 0.0
        ratio = (yaw - self.yaw_stability_deadzone) / span
        ratio = max(0.0, min(1.0, ratio))
        return -self.yaw_stability_penalty_max * ratio * ratio

    def terminal_reward(self, avg_g, yaw_abs_sum, yaw_sq_sum, refs=None):
        """STOP terminal: g shape + clean-yaw bonus + catastrophic backstop."""
        return self.terminal_reward_components(
            avg_g, yaw_abs_sum, yaw_sq_sum, refs)["total"]

    def terminal_reward_components(self, avg_g, yaw_abs_sum, yaw_sq_sum,
                                   refs=None):
        """Exact auditable decomposition of :meth:`terminal_reward`."""
        g_rew = self.g_shape(avg_g, refs)
        if avg_g > self.per_step_g_gate:
            clean = max(0.0, 1.0 - yaw_abs_sum / self.yaw_bonus_threshold)
            yaw_bonus = self.yaw_bonus_k_terminal * clean
        else:
            yaw_bonus = 0.0
        accumulated_yaw = self.accumulated_yaw_penalty(yaw_sq_sum)
        stability = self.terminal_stability_penalty(yaw_abs_sum)
        return {
            "terminal_g": g_rew,
            "terminal_clean_yaw_bonus": yaw_bonus,
            "terminal_accumulated_yaw": accumulated_yaw,
            "terminal_stability": stability,
            "total": g_rew + yaw_bonus + accumulated_yaw + stability,
        }

    def crash_reward(self, yaw_sq_sum):
        return self.crash_penalty + self.accumulated_yaw_penalty(yaw_sq_sum)

    def crash_reward_components(self, yaw_sq_sum):
        accumulated_yaw = self.accumulated_yaw_penalty(yaw_sq_sum)
        return {
            "failure_base": self.crash_penalty,
            "failure_accumulated_yaw": accumulated_yaw,
            "total": self.crash_penalty + accumulated_yaw,
        }

    def timeout_reward(self, yaw_sq_sum):
        return self.timeout_penalty + self.accumulated_yaw_penalty(yaw_sq_sum)

    def timeout_reward_components(self, yaw_sq_sum):
        accumulated_yaw = self.accumulated_yaw_penalty(yaw_sq_sum)
        return {
            "failure_base": self.timeout_penalty,
            "failure_accumulated_yaw": accumulated_yaw,
            "total": self.timeout_penalty + accumulated_yaw,
        }

    # ------------------------------------------------------------------
    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def hash(self):
        """Stamped into every run so a result can never be mistaken for a
        default-reward result."""
        payload = self.to_dict()
        if not self.slip_term_active():
            # Slip fields postdate v5/v6/normalized; keep their hashes intact.
            for key in self._SLIP_FIELDS:
                payload.pop(key)
        if not payload.get("metric_window_only"):
            payload.pop("metric_window_only")     # v11 fields; keep older hashes
            if payload.get("approach_time_k") == 0.0:
                payload.pop("approach_time_k")
        if payload.get("speed_cost_k") == 0.0:
            payload.pop("speed_cost_k")       # v9 field; keep every older hash
        if payload.get("dense_g_source") == "inst":
            payload.pop("dense_g_source")     # postdates v5/v6/v7; keep their hashes
        if payload.get("shape_mode") == "ramp":
            for key in ("shape_mode", "v33_gate_g", "v33_gate_k", "v33_gate_scale",
                        "heading_pen_k", "heading_recovery_k"):
                payload.pop(key)              # v3.3 fields; keep every older hash
        if self.step_mode == "shape":
            # Preserve the established v5.0/normalized hashes.  These fields
            # did not exist when those immutable presets were introduced.
            for key in (
                "step_mode", "rolling_window_s", "dense_g_k", "consistency_k",
                "yaw_stability_deadzone", "yaw_stability_full_scale",
                "yaw_stability_penalty_max",
            ):
                payload.pop(key)
            if payload["timeout_penalty"] == 0.0:
                payload.pop("timeout_penalty")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def implementation_hash(self):
        """Hash coefficients and executable reward logic, not just the preset.

        ``hash()`` remains stable for historical v5/v6 records.  New runs also
        store this stronger identity so a logic edit cannot masquerade as the
        same reward merely because its coefficients stayed unchanged.
        """
        methods = (
            RewardSpec.to_scale, RewardSpec.g_shape, RewardSpec.step_g_reward,
            RewardSpec.make_step_tracker, RewardSpec.step_yaw_bonus,
            RewardSpec.accumulated_yaw_penalty,
            RewardSpec.terminal_stability_penalty, RewardSpec.terminal_reward,
            RewardSpec.terminal_reward_components, RewardSpec.crash_reward,
            RewardSpec.crash_reward_components, RewardSpec.timeout_reward,
            RewardSpec.timeout_reward_components, StepGRewardTracker.__init__,
            StepGRewardTracker.reset, StepGRewardTracker.push,
        )
        fields = self.to_dict()
        if self.slip_term_active():
            methods = methods + (RewardSpec.step_slip_reward,)
        else:
            for key in self._SLIP_FIELDS:
                fields.pop(key)
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        source = "\n".join(inspect.getsource(method) for method in methods)
        return hashlib.sha256((payload + "\0" + source).encode("utf-8")).hexdigest()

    def is_default(self):
        return self.to_dict() == RewardSpec.v5().to_dict()


class StepGRewardTracker:
    """Stateful dense G reward for one episode.

    v6 deliberately scores only positive deceleration.  Negative longitudinal
    G (acceleration in the opposite direction) cannot masquerade as braking.
    """

    def __init__(self, spec, dt):
        self.spec = spec
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("reward tracker dt must be finite and positive")
        if spec.step_mode == "rolling_consistency":
            window_steps = max(1, int(math.ceil(spec.rolling_window_s / self.dt)))
        else:
            window_steps = 1
        self._samples = deque(maxlen=window_steps)

    def reset(self):
        self._samples.clear()

    @property
    def rolling_mean(self):
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def rolling_std(self):
        if not self._samples:
            return 0.0
        mean = self.rolling_mean
        variance = (sum((sample - mean) ** 2 for sample in self._samples)
                    / len(self._samples))
        return math.sqrt(max(0.0, variance))

    def push(self, g_step, refs=None, speed=None):
        g_positive = max(0.0, float(g_step))
        if self.spec.step_mode == "distance":
            # v9: the integral of this term is -k * metres, i.e. the metric.
            if speed is None:
                raise ValueError("distance step_mode needs the ground speed")
            return -self.spec.speed_cost_k * max(0.0, float(speed)) * self.dt
        if self.spec.step_mode == "shape":
            return self.spec.step_g_reward(g_positive, refs)
        if self.spec.step_mode == "window_g":
            # v8: plain capped reward on the window-averaged G, no std term.
            return self.spec.dense_g_k * self.dt * g_positive
        if self.spec.step_mode == "v33_step":
            # FastTrain v3.3 per-step hint: affine shape, no gate, per step.
            return self.spec.per_step_k * self.spec.v33_affine(g_positive)
        if self.spec.step_mode != "rolling_consistency":
            raise ValueError(f"unknown reward step_mode: {self.spec.step_mode!r}")

        self._samples.append(g_positive)
        quality_g = max(
            0.0,
            self.rolling_mean - self.spec.consistency_k * self.rolling_std,
        )
        return self.spec.dense_g_k * self.dt * quality_g


PRESETS = {
    "v5.0": RewardSpec.v5,
    "v6.0": RewardSpec.v6,
    "v6.1": RewardSpec.v6_1,
    "v7.0": RewardSpec.v7,
    "v8.0": RewardSpec.v8,
    "v3.3": RewardSpec.v3_3,
    "v9.0": RewardSpec.v9,
    "v10.0": RewardSpec.v10,
    "v11.0": RewardSpec.v11,
    "normalized": RewardSpec.normalized,
}
