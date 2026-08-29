# ResidualABS

Trains PPO/SAC against a **residual (release-from-pedal)** action space instead of
full-authority braking. Design rationale, the training-formulation problem this
solves, and the "why no explicit slip detection is needed" reasoning all live in
the spec: [`docs/TRAINING_GUI_SPEC.md`](../../docs/TRAINING_GUI_SPEC.md).

## The core idea, in one line

Action = per-axle *release* from the driver's pedal, not the brake level itself.
Zero action = full-pedal slam = the known lockup baseline, so training starts at
the anti-lock boundary instead of a few hundred thousand steps away from it.
Reward stays v5.0 (g-force only, imported byte-identical, zero edits) -- this is a
coordinate change on the action, not a reward change.

## Live gate results (2026-08-27, Machine-Trainer-Boy on smallgrid, 60mph)

| Gate | Result | Verdict |
|---|---|---|
| Zero-action reproduces the lockup baseline | 1.028g mean (3 eps: 1.020 / 1.029 / 1.035) | PASS |
| Scripted release beats the slam mean | 1.043g mean (3 eps: 1.041 / 1.039 / 1.047) | PASS, thin margin (+0.015g, inside slam's own episode spread) |

Read honestly: the release mechanism reaches the wheels and the direction is
correct, but a single-speed dry-asphalt straight stop is a weak diagnostic --
this project's own classical-ABS work already found dry tarmac has almost no
headroom above a hard slam. The release advantage should show up more clearly
on lower-friction or transition surfaces; that test wasn't run here.

## Files

- `residual_core.py` -- pure action/brake math + GUI input parsers (no game imports)
- `abs_env.py`, `abs_env_incar.py` -- byte-identical copies of PPO_V3_AxleCurriculum's
  originals; **never edit these** (verify with `git diff --no-index` against the source)
- `abs_env_residual.py` -- the env subclass: 2-float release action, pedal appended to obs
- `abstelemetry.lua` -- telemetry bridge with an added arc-length channel
  (`last_brake_avg_g_arc` / `last_brake_dist_arc`) alongside the existing chord-based
  metric. This is THIS folder's own copy only (auto-deploys to the live .tech
  userpath on the next launch of an env from here; every other training folder's
  copy of this file is untouched); the arc math is unverified against real
  physics until that first launch, and **not yet consumed by the reward** either
  way (see the spec/ledger for why)
- `train_residual.py` -- one trainer, `--algo sac|ppo`
- `gui_train.py` -- tkinter launcher/monitor
- `baseline_probe.py` -- the two live gates above

## CLI examples

```bash
# short smoke test
python train_residual.py --algo sac --speeds "60" --pedal off --total-steps 10000 --run-name smoke_sac

# real run with randomized pedal
python train_residual.py --algo sac --speeds "60,90,120" --pedal "0.4-1.0" --total-steps 500000 --run-name residual_sac_r1

# PPO
python train_residual.py --algo ppo --speeds "60,120" --pedal "0.4-1.0" --total-steps 500000 --run-name residual_ppo_r1

# graceful stop (never taskkill -- can corrupt SAC's replay buffer)
echo stop > STOP_TRAINING.txt
```

Or use `gui_train.py` for the same flow with a live monitor.

**`--total-steps` is an ABSOLUTE target, not an increment, when resuming.**
`--resume <ckpt> --total-steps 500000` trains up to step 500,000 total, counting
steps the checkpoint already has. Resuming a 200k checkpoint with
`--total-steps 200000` does nothing -- `learn()` returns immediately because the
target is already met. To add 300k more steps on top of a 200k checkpoint, pass
`--total-steps 500000`.

## Deferred

**Steering: no code anywhere in this experiment.** The GUI shows a disabled
"Turn angles" switch labeled with the reason -- the reward shape for steering
hasn't been designed yet. Do that design work before adding any steering code.

**V4 (pedal-mixture curriculum: ramp/pump/step patterns, live `input_brake` obs):**
GUI shows a disabled switch; not implemented. Randomized *constant-per-episode*
pedal level is implemented (`--pedal`); varying pedal *within* an episode is V4.

**Arc-G in the reward:** telemetry channel exists, not wired in. See the commit
message on the abstelemetry.lua change and the plan's ledger for the reasoning.
