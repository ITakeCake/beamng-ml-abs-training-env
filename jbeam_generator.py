"""Generates the multi-car, multi-model "ML ABS" mod: one PARENT jbeam part
per supported car (declaring a child slot models plug into, defaulting to a
controller-less placeholder so selecting "ML ABS" alone never silently drives
with a stale/random model), plus one CHILD part per exported trained model.

Precedent for the nested-slot pattern: the stock etk800_brakes.jbeam's brake
pad part declares its own child "slots" array (front/rear pad choices); the
per-car mod DynamicABS-E already ships a real ABS-replacement jbeam part per
car (20 cars) using the exact single-part pattern the parent part here also
uses for its ABS slotType.

Pure JSON/string generation + real zip scanning -- no game imports.
"""
import json
import os
import re
import zipfile

MODEL_SLOT_TYPE = "mlabs_model"
PLACEHOLDER_PART_NAME = "mlabs_none"


def find_abs_slot_types(vehicles_dir):
    """{model_name: (slot_type, display_name)} for every Car-type model zip
    in vehicles_dir that declares an ABS-ish slotType somewhere in its jbeam."""
    out = {}
    for zip_path in sorted(__import__("glob").glob(os.path.join(vehicles_dir, "*.zip"))):
        model = os.path.splitext(os.path.basename(zip_path))[0]
        try:
            with zipfile.ZipFile(zip_path) as zf:
                try:
                    info = json.loads(zf.read(f"vehicles/{model}/info.json")
                                      .decode("utf-8", errors="replace"))
                except (KeyError, json.JSONDecodeError):
                    continue
                if info.get("Type") != "Car":
                    continue
                slot_type = None
                for name in zf.namelist():
                    if not name.endswith(".jbeam"):
                        continue
                    text = zf.read(name).decode("utf-8", errors="replace")
                    m = re.search(r'"slotType"\s*:\s*"([^"]*ABS[^"]*)"', text)
                    if m:
                        slot_type = m.group(1)
                        break
                if slot_type:
                    out[model] = (slot_type, info.get("Name", model))
        except (OSError, zipfile.BadZipFile):
            continue
    return out


def generate_parent_jbeam(model, slot_type):
    """The "ML ABS" part occupying the car's real ABS slot. Its only children
    are a controller-less placeholder (the default -- brakes stay at stock
    capacity but nothing scales them, i.e. inert/off) plus, once exported,
    one part per trained model in MODEL_SLOT_TYPE. Emitted fresh each
    generate/export -- callers own writing this alongside any per-model
    children already generated for this car."""
    parent_name = f"{PLACEHOLDER_PART_NAME}_{model}_parent"
    placeholder_name = f"{PLACEHOLDER_PART_NAME}_{model}"
    data = {
        parent_name: {
            "information": {
                "authors": "ML ABS Training Environment",
                "name": "ML ABS",
                "value": 250,
            },
            "slotType": slot_type,
            "slots": [
                ["type", "default", "description"],
                [MODEL_SLOT_TYPE, placeholder_name, "ML ABS Model"],
            ],
        },
        placeholder_name: {
            "information": {"name": "(no model selected)", "value": 0},
            "slotType": MODEL_SLOT_TYPE,
        },
    }
    return json.dumps(data, indent=2)


def generate_model_jbeam(model, run_name, weights_module):
    """One child part for MODEL_SLOT_TYPE: the SAME MTB-ML-ABS controller,
    pointed at this run's exported weights module via jbeamData.weights."""
    part_name = f"mlabs_model_{run_name}"
    data = {
        part_name: {
            "information": {
                "authors": "ML ABS Training Environment",
                "name": f"ML ABS — {run_name}",
                "value": 250,
            },
            "slotType": MODEL_SLOT_TYPE,
            "controller": [
                ["fileName"],
                ["MTB-ML-ABS", {"weights": weights_module}],
            ],
        },
    }
    return json.dumps(data, indent=2)
