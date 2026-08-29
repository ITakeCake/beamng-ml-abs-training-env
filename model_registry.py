"""Lists finished training runs (runs/<name>/final.zip + vecnormalize.pkl) as
export candidates for the Output tab, with the metadata export needs: which
algo, which car it trained on, and its best logged avg_g. Pure filesystem +
text parsing -- no game/SB3 imports, so this stays fast and testable offline.
"""
import ast
import csv
import dataclasses
import glob
import os
import re

REFERENCE_VEHICLE_PC = "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc"


@dataclasses.dataclass
class RunInfo:
    run_name: str
    run_dir: str
    algo: str
    vehicle_pc: str
    model: str            # the car folder name, e.g. "etk800" (parsed from vehicle_pc)
    best_avg_g: float      # None if no episode data was found


def _model_from_vehicle_pc(vehicle_pc):
    m = re.match(r"vehicles/([^/]+)/", vehicle_pc)
    return m.group(1) if m else "etk800"


def _parse_args_line(train_log_path):
    """The trainer logs one line starting with 'args: {...}' -- a literal
    Python dict repr (it's logged via %s on the argparse Namespace's vars()).
    ast.literal_eval, never eval -- these are our own log files, but there's
    no reason to use anything but the safe parser for a dict literal."""
    try:
        with open(train_log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "] args: " in line:
                    dict_text = line.split("] args: ", 1)[1].strip()
                    return ast.literal_eval(dict_text)
    except (OSError, ValueError, SyntaxError):
        return None
    return None


def _best_avg_g(episode_csv_path):
    try:
        with open(episode_csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    gs = [float(r["avg_g"]) for r in rows if r.get("avg_g")]
    return max(gs) if gs else None


def list_finished_runs(runs_dir):
    out = []
    for final_zip in sorted(glob.glob(os.path.join(runs_dir, "*", "final.zip"))):
        run_dir = os.path.dirname(final_zip)
        run_name = os.path.basename(run_dir)
        args = _parse_args_line(os.path.join(run_dir, "train.log")) or {}
        vehicle_pc = args.get("vehicle_pc") or REFERENCE_VEHICLE_PC
        out.append(RunInfo(
            run_name=run_name,
            run_dir=run_dir,
            algo=args.get("algo", "?"),
            vehicle_pc=vehicle_pc,
            model=_model_from_vehicle_pc(vehicle_pc),
            best_avg_g=_best_avg_g(os.path.join(run_dir, "episode_log_env0.csv")),
        ))
    return out
