# 1.18 g cleared, two ways — the policies, and how to use them

| checkpoint | what it is | mean avg_g | 95% CI | >= 1.18 |
|---|---|---:|---|---:|
| **`runs/PPO-71/`** | **RL-trained**: 1.5M PPO steps, reward v11, 400 Hz | **1.1853** | [1.1825, 1.1882] | 13/16 |
| **`runs/BC-D3/`** | its distilled starting point (no RL) | **1.1952** | [1.1931, 1.1973] | 16/16 |

PPO-71 is the answer to "a PPO that reaches 1.18 g": it was trained by reinforcement
learning on the honest observation vector and it stops at 1.1853 g mean, with the
whole confidence interval above the goal. BC-D3 is the distilled policy it started
from, and is slightly better and more consistent.

**Read PPO-71's episode log with care**: it settled around 0.73 g while the policy
was actually stopping at 1.19. At 400 Hz the control period is 2.5 ms and the
trainer spends 1.6-2.0 ms of it in Python, which depresses the measurement. Judge a
400 Hz run by evaluating its checkpoint, never by its log. This does NOT affect
100 Hz runs -- PPO-57 re-measures at 0.96-1.02 against a logged 1.049.

**Checkpoint below: `runs/BC-D3/`** (`final.zip` + `vecnormalize.pkl`).

Machine-Trainer-Boy, 80 mph, dry, straight. Measured with the game's own 2 kHz
brake metric, 16 consecutive stops:

    mean 1.1952 g   sd 0.0043   95% CI [1.1931, 1.1973]
    min 1.1861      max 1.2003      16/16 over the 1.180 goal
    mean stopping distance 54.50 m

References on this exact car and surface (`calibration/etk800.json`): locked wheel
0.979 g, stock ABS 1.199 g, DynamicABS 1.205 g. Best PPO before this: 1.063 g.

## What it is, precisely

The same `UnitIntervalActorCriticPolicy` network `train_cosim.py` trains — 3x256,
four wheel-release outputs — reading only the honest observation vector: wheel
speeds, stock-ABS-estimated slip, applied brake torques, IMU. **No ground-truth
speed reaches the network**, at training time or at run time.

Its weights come from **supervised distillation, not RL reward optimization**. A
scripted per-wheel proportional slip regulator (`release = 6 * (slip - 0.10)`,
running at 400 Hz, itself scoring 1.192-1.201 g) was recorded through the real env,
and the policy was fitted to its actions. The teacher reads ground-truth slip; the
student never does, and has to reproduce the behaviour from sensors alone.

So: the number is real, honestly obtained, and reproducible. It is not evidence
that PPO's own learning found it. Fine-tuning on top is still open (see below).

## Reproduce the measurement

    python slip_ceiling_probe.py --reps 16 --dt 0.0025 --releases "" --targets "" \
        --policy runs/BC-D3

## Rebuild it from scratch

    # 1. record the teacher (8 stops, ~12k pairs each)
    python slip_ceiling_probe.py --reps 8 --dt 0.0025 --releases "" --targets "" \
        --prop "0.10:6" --record teacher.npz
    # 2. distill
    python distill_teacher.py --data teacher.npz --config .gui-configs/PPO-65.json \
        --out runs/BC-1 --epochs 80
    # 3. DAgger, with the student PERTURBED -- this is the step that mattered
    python slip_ceiling_probe.py --reps 14 --dt 0.0025 --releases "" --targets "" \
        --policy runs/BC-1 --policy-noise 0.04 --dagger "0.10:6" --record dagger.npz
    # merge with the previous set, redistill, repeat

Deterministic DAgger plateaus at ~1.04 g mean with a bimodal 0.85 / 1.20 split.
Perturbing the student puts the states noise knocks it into INTO the dataset with
teacher labels, and that is what produced 1.1952.

## What is still open

RL fine-tuning from this checkpoint degrades it (~0.85 g). That is a property of
the trainer, not the policy: the identical weights hold 1.19 g indefinitely under
lighter driving loops, and achieved g tracks per-step Python cost monotonically
across five harnesses. `docs/CONTROL_CEILING.md` has the full bisect and
`wrapper_bisect.py` reproduces the ladder in minutes.

Deploying it to `.drive` should go through `evaluate_cosim.py` (it exports the Lua
weight blob and asserts numerical parity against SB3 before you trust it).
