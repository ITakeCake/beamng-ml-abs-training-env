# Controlled co-sim experiment workflow

The experiment record is append-only at `experiments/ledger.jsonl`. Existing
run directories and reward presets are evidence: never edit or replace them.

## Frozen checkpoint evaluation

Evaluate all paired checkpoints from one run, interleaving their order. The
default performs 20 stops per checkpoint and extends the strongest three to 50
total stops each:

```powershell
py -3.10 evaluate_cosim.py --run-dir runs\PPO-38 --episodes 20 --reevaluate-top 3 --top-total-episodes 50
```

The evaluator deep-validates every model and VecNormalize pair before opening
BeamNG. It uses deterministic policy actions, frozen observation statistics,
and free-running physics. Results are written to a new immutable directory
under `evaluations/`; the selected pair is copied to a new directory under
`best_models/`. `report.md` is the compact checkpoint report,
`checkpoint_summary.csv` contains confidence and block statistics, and
`step_traces.csv.gz` contains actions, slips, torque commands, G, yaw, and exact
reward components. Failed stops count as zero G in the primary aggregate.

## Crash-safe training and resume

Each co-sim run writes `run_state.json` and append-only `run_events.jsonl`.
Periodic and final saves consist of a model, its exact VecNormalize state, and a
`.pair.json` commit manifest. The manifest appears only after both files load,
are finite, match the bounded 13x2 contract, and reach their final names.

The GUI can resume a checkpoint into a new run name. Reward, vehicle, speed,
run-up, PPO settings, and interface versions are inherited from the parent run;
only the new target length and runtime placement may differ. Manual configs can
use:

```json
{
  "run_name": "PPO-39-resume",
  "total_steps": 1000000,
  "resume_checkpoint": "runs/PPO-39/checkpoints/PPO-39_501760_steps.zip",
  "resume_inherit_config": true,
  "seed": 390001
}
```

## Decision discipline

Before every million-step run, append the hypothesis, exact single conceptual
change, expected result, and rejection criterion to the ledger. Evaluate frozen
checkpoints before deciding what changes next. A reward change is permitted only
after measured physical G and PPO-discounted return disagree; it creates the
next immutable preset rather than editing v5.0 or v6.0.

Offline tests do not validate live BeamNG launch, IPC, reset, control timing,
vehicle behavior, or deployment engagement. Those require explicit live checks.
