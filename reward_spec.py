"""The reward, as data instead of hard-coded constants (PLAN_V2.md section 6).

Two presets ship:

  RewardSpec.v5()          -- the frozen default. Reproduces the protected
                              abs_env.py's numbers EXACTLY; a parity test
                              asserts this against the real constants and
                              _terminal_g_shape, not a recorded snapshot.
  RewardSpec.normalized()  -- anchors moved onto the per-config MEASURED
                              references (calibration.py): zero at stock ABS,
                              strongly negative at lockup.

Why normalized exists, in one measurement (etk800, 60 mph, dry, 2026-08-29):
v5.0 pays +966 for locking the wheels, +987 for this project's best result
(1.043 g) and +5049 for what stock ABS does unaided (1.188 g). Its "exceptional
performance" gatekeeper sits at 1.06 g -- BELOW stock. The absolute shape could
not distinguish the brake-slammer local optimum from real progress.

No game imports.
"""
import dataclasses
import hashlib
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

    # ------------------------------------------------------------------
    def to_scale(self, g, refs=None):
        """Absolute g -> the units this spec's anchors are expressed in."""
        if not self.normalize:
            return g
        if refs is None:
            raise ValueError(
                "this reward spec is normalized but no calibration references "
                "were supplied -- refusing to score against absolute anchors "
                "(that is exactly the bug normalization exists to fix).")
        slam_g, stock_g = refs
        return normalized_g(g, slam_g, stock_g)

    def g_shape(self, g, refs=None):
        """The shape both the per-step and terminal g rewards run through."""
        x = self.to_scale(g, refs)
        x = max(0.0, min(x, self.clamp_hi)) if not self.normalize else min(x, self.clamp_hi)
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
        return self.per_step_k * self.g_shape(g_step, refs)

    def step_yaw_bonus(self, g_step, yaw_error):
        if g_step > self.per_step_g_gate:
            return self.yaw_bonus_k_step * math.exp(-yaw_error * self.yaw_bonus_alpha)
        return 0.0

    def accumulated_yaw_penalty(self, yaw_sq_sum):
        return -self.yaw_pen_k_terminal * yaw_sq_sum

    def terminal_reward(self, avg_g, yaw_abs_sum, yaw_sq_sum, refs=None):
        """STOP terminal: g shape + clean-yaw bonus + catastrophic backstop."""
        g_rew = self.g_shape(avg_g, refs)
        if avg_g > self.per_step_g_gate:
            clean = max(0.0, 1.0 - yaw_abs_sum / self.yaw_bonus_threshold)
            yaw_bonus = self.yaw_bonus_k_terminal * clean
        else:
            yaw_bonus = 0.0
        return g_rew + yaw_bonus + self.accumulated_yaw_penalty(yaw_sq_sum)

    def crash_reward(self, yaw_sq_sum):
        return self.crash_penalty + self.accumulated_yaw_penalty(yaw_sq_sum)

    def timeout_reward(self, yaw_sq_sum):
        return self.accumulated_yaw_penalty(yaw_sq_sum)

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
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def is_default(self):
        return self.to_dict() == RewardSpec.v5().to_dict()


PRESETS = {"v5.0": RewardSpec.v5, "normalized": RewardSpec.normalized}
