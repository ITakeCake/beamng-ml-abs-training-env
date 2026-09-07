"""Lists finished training runs (runs/<name>/final.zip + vecnormalize.pkl) as
export candidates for the Output tab, with the metadata export needs: which
algo, which car it trained on, and its best logged avg_g. Pure filesystem +
text parsing -- no game/SB3 imports, so this stays fast and testable offline.
"""
import ast
import csv
import dataclasses
import glob
import hashlib
import json
import os
import re
import zipfile

REFERENCE_VEHICLE_PC = "vehicles/etk800/Machine-Trainer-Boy-V2-MLABS.pc"


@dataclasses.dataclass
class RunInfo:
    run_name: str
    run_dir: str
    algo: str
    vehicle_pc: str
    model: str            # the car folder name, e.g. "etk800" (parsed from vehicle_pc)
    best_avg_g: float      # None if no episode data was found
    deployment_interface: str = "legacy_incar_brake4_v1"


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


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_committed_cosim_final(run_dir, config):
    """Require the crash-safe commit record for new bounded co-sim finals.

    Older run families predate pair manifests and remain visible under their
    legacy deployment contracts.  A run explicitly stamped as the bounded
    two-action interface is new enough that a missing or mismatched commit is
    evidence of an interrupted/partial final, not a finished export candidate.
    """
    if config.get("deployment_interface") not in ("cosim_axle_release_v1",
                                                  "cosim_axle_release_v2"):
        return True
    model_path = os.path.join(run_dir, "final.zip")
    vec_path = os.path.join(run_dir, "vecnormalize.pkl")
    manifest = _read_json(os.path.join(run_dir, "final.pair.json"))
    if not manifest or manifest.get("schema") != 1:
        return False
    if (manifest.get("model_path") != "final.zip" or
            manifest.get("vecnormalize_path") != "vecnormalize.pkl"):
        return False
    try:
        if (int(manifest["model_bytes"]) != os.path.getsize(model_path) or
                int(manifest["vecnormalize_bytes"]) != os.path.getsize(vec_path)):
            return False
        if (manifest["model_sha256"] != _sha256(model_path) or
                manifest["vecnormalize_sha256"] != _sha256(vec_path)):
            return False
        if not zipfile.is_zipfile(model_path):
            return False
        with zipfile.ZipFile(model_path) as archive:
            if archive.testzip() is not None:
                return False
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def list_finished_runs(runs_dir):
    out = []
    for final_zip in sorted(glob.glob(os.path.join(runs_dir, "*", "final.zip"))):
        run_dir = os.path.dirname(final_zip)
        if not os.path.isfile(os.path.join(run_dir, "vecnormalize.pkl")):
            continue
        run_name = os.path.basename(run_dir)
        args = _parse_args_line(os.path.join(run_dir, "train.log")) or {}
        config = _read_json(os.path.join(run_dir, "config.json"))
        if config is not None:
            # Co-sim is configured through JSON and has no argparse "args:" log
            # line. Only new bounded-policy runs stamp a deployable interface;
            # older unbounded checkpoints must never be guessed compatible.
            algo = "ppo"
            vehicle_pc = config.get("vehicle_pc") or REFERENCE_VEHICLE_PC
            interface = config.get(
                "deployment_interface", "cosim_legacy_unbounded_v0")
            if not _valid_committed_cosim_final(run_dir, config):
                continue
            episode_path = os.path.join(run_dir, "episode_log.csv")
        else:
            algo = args.get("algo", "?")
            vehicle_pc = args.get("vehicle_pc") or REFERENCE_VEHICLE_PC
            # train_residual.py is a 28x16 observation, two-axle release
            # contract. The existing 27x16/four-wheel controller cannot run it.
            interface = ("residual_axle_release_v1" if args
                         else "legacy_incar_brake4_v1")
            episode_path = os.path.join(run_dir, "episode_log_env0.csv")
        out.append(RunInfo(
            run_name=run_name,
            run_dir=run_dir,
            algo=algo,
            vehicle_pc=vehicle_pc,
            model=_model_from_vehicle_pc(vehicle_pc),
            best_avg_g=_best_avg_g(episode_path),
            deployment_interface=interface,
        ))
    return out
