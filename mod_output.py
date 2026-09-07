"""Writes the actual "ML ABS" mod to a userpath: per-car parent parts (from
jbeam_generator) plus, per exported training run, a merged child part and its
weights lua (via export_policy_weights.py, run as a subprocess so a bad
checkpoint's exception can't take down the GUI process).
"""
import json
import os
import subprocess
import sys
import shutil

from jbeam_generator import find_abs_slot_types, generate_parent_jbeam, generate_model_jbeam
from vehicle_scanner import scan_models

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTER_PATH = os.path.join(HERE, "export_policy_weights.py")
MOD_NAME = "ml_abs"

INFO_JSON = {
    "title": "ML ABS",
    "description": "Machine-learned ABS controllers, trained with the "
                   "BeamNG ML-ABS Training Environment.",
    "author": "ML ABS Training Environment",
}


def mod_dir_for(content_userpath):
    return os.path.join(content_userpath, "mods", "unpacked", MOD_NAME)


def generate_all_cars(vehicles_dir, out_mod_dir):
    """Writes one ml_abs_parent.jbeam per supported car. Returns
    (written_models, skipped_models), skipped = Car-type models with no
    ABS-ish slot in their own jbeam (can't host this mod without overriding a
    stock body/brakes part, out of scope for v1)."""
    all_models = {m.name for m in scan_models(vehicles_dir)}
    slots = find_abs_slot_types(vehicles_dir)

    os.makedirs(out_mod_dir, exist_ok=True)
    with open(os.path.join(out_mod_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(INFO_JSON, fh, indent=2)

    written = []
    for model, (slot_type, _display) in sorted(slots.items()):
        car_dir = os.path.join(out_mod_dir, "vehicles", model)
        os.makedirs(car_dir, exist_ok=True)
        with open(os.path.join(car_dir, "ml_abs_parent.jbeam"), "w", encoding="utf-8") as fh:
            fh.write(generate_parent_jbeam(model, slot_type))
        written.append(model)

    skipped = sorted(all_models - set(written))
    return written, skipped


def _run_exporter(cmd):
    """Real subprocess call, monkeypatched in tests. Returns (ok, message)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, (result.stdout + result.stderr).strip()
    return True, result.stdout.strip()


def _weights_module_name(run_info):
    return f"mlabs_w_{run_info.run_name}"


def _models_jbeam_path(run_info, out_mod_dir):
    return os.path.join(out_mod_dir, "vehicles", run_info.model, "ml_abs_models.jbeam")


def _deployment_spec(run_info):
    interface = getattr(run_info, "deployment_interface",
                        "legacy_incar_brake4_v1")
    if interface in ("cosim_axle_release_v1", "cosim_axle_release_v2"):
        if run_info.algo != "ppo":
            return None, "co-sim deployment currently supports PPO checkpoints only"
        return {
            "head": "ppo_tanh_release01", "controller": "MTB-ML-ABS-CoSim",
            "interface": interface, "control_hz": 100,
        }, None
    if interface == "legacy_incar_brake4_v1":
        return {
            "head": "ppo_clip01" if run_info.algo == "ppo" else "sac_tanh01",
            "controller": "MTB-ML-ABS", "interface": interface,
            "control_hz": 200,
        }, None
    if interface == "residual_axle_release_v1":
        return None, (
            "This residual checkpoint uses 28 observations x 16 frames and two "
            "axle-release actions. The shipped four-wheel controller expects "
            "27 x 16 and four brake actions, so exporting it would produce an "
            "invalid in-game controller. Training is valid; deployment needs a "
            "dedicated residual controller.")
    return None, (
        "This checkpoint predates the bounded PPO action-interface stamp "
        f"({interface}). It cannot be safely inferred as deployable; start a new "
        "bounded PPO run or keep it for offline comparison.")


def export_model_to_game(run_info, out_mod_dir, python_exe=None):
    """Runs export_policy_weights.py against this run's checkpoint, then
    merges a child jbeam part in, read-merge-write, so exporting model B
    never erases model A's entry for the same car."""
    weights_name = _weights_module_name(run_info)
    weights_lua = os.path.join(out_mod_dir, "lua", "vehicle", "controller",
                               weights_name + ".lua")
    os.makedirs(os.path.dirname(weights_lua), exist_ok=True)
    spec, problem = _deployment_spec(run_info)
    if problem:
        return False, problem
    cmd = [python_exe or sys.executable, EXPORTER_PATH,
          "--ckpt", os.path.join(run_info.run_dir, "final.zip"),
          "--vecnorm", os.path.join(run_info.run_dir, "vecnormalize.pkl"),
          "--algo", run_info.algo, "--head", spec["head"],
          "--interface", spec["interface"],
          "--control-hz", str(spec["control_hz"]), "--out", weights_lua]
    ok, msg = _run_exporter(cmd)
    if not ok:
        return False, msg

    # Export is self-contained even even without a prior clicked Install/Update
    # Assets since this controller was added.
    controller_name = spec["controller"] + ".lua"
    controller_src = os.path.join(
        HERE, "assets", "mods", "mtb_ml_abs", "lua", "vehicle",
        "controller", controller_name)
    controller_dst = os.path.join(
        out_mod_dir, "lua", "vehicle", "controller", controller_name)
    if not os.path.isfile(controller_src):
        return False, "required controller asset is missing: " + controller_src
    shutil.copy2(controller_src, controller_dst)

    models_path = _models_jbeam_path(run_info, out_mod_dir)
    os.makedirs(os.path.dirname(models_path), exist_ok=True)
    existing = {}
    if os.path.isfile(models_path):
        with open(models_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    new_part = json.loads(generate_model_jbeam(
        run_info.model, run_info.run_name, weights_name,
        controller=spec["controller"]))
    existing.update(new_part)
    with open(models_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    return True, msg


def remove_model_from_game(run_info, out_mod_dir):
    """Removes this run's entry from its car's ml_abs_models.jbeam and its
    weights lua file. Only ever touches the one entry named for this
    run, never the whole file, never another model's data."""
    models_path = _models_jbeam_path(run_info, out_mod_dir)
    part_name = f"mlabs_model_{run_info.run_name}"
    if os.path.isfile(models_path):
        with open(models_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        existing.pop(part_name, None)
        with open(models_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
    weights_lua = os.path.join(out_mod_dir, "lua", "vehicle", "controller",
                               _weights_module_name(run_info) + ".lua")
    if os.path.isfile(weights_lua):
        os.remove(weights_lua)
