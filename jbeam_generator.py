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


def find_abs_slot_types(vehicles_dir, include_common=True):
    """{model_name: (slot_types, display_name)} for every Car-type model zip in
    vehicles_dir, where slot_types is EVERY ABS-ish slot name reachable for that
    car -- its own jbeam plus (include_common) the shared parts files, since the
    whole ETK family gets its socket from common.zip rather than its own zip.

    Plural, and deliberately generous. jbeam silently ignores a part whose
    slotType nothing offers, so naming a socket a car does not have costs
    nothing, while MISSING the one it does have makes the part load and never
    appear -- a failure with no error message anywhere. DynamicABS (this
    project's own 20-car mod, shipped and working) resolves it the same way:
    one identical file per car declaring all 22 slot types."""
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
                found = set()
                for name in zf.namelist():
                    if not name.endswith(".jbeam"):
                        continue
                    text = zf.read(name).decode("utf-8", errors="replace")
                    found.update(_abs_slot_names(text))
                if include_common:
                    found.update(_common_abs_slots(vehicles_dir))
                if found:
                    out[model] = (sorted(found), info.get("Name", model))
        except (OSError, zipfile.BadZipFile):
            continue
    return out


def _abs_slot_names(text):
    """Every ABS-ish slot name in one jbeam file, from BOTH forms it can take:
    a part's own `slotType` declaration, and a socket OFFERED in a `slots` row
    (["<type>", "<default>", "<label>"]). Scanning only the first misses
    sockets that exist purely as an offer, which is how etk_DSE_ABS appears."""
    names = set(re.findall(r'"slotType"\s*:\s*"([^"]*ABS[^"]*)"', text))
    names.update(re.findall(r'\[\s*"(\w*ABS\w*)"\s*,\s*"[^"]*"\s*,', text))
    return names


_COMMON_CACHE = {}


def _common_abs_slots(vehicles_dir):
    """ABS slot names from the shared parts zips rather than any one car's --
    common.zip holds etk_DSE_ABS, which every ETK actually uses. Cached, since
    it would otherwise be re-read once per car."""
    if vehicles_dir in _COMMON_CACHE:
        return _COMMON_CACHE[vehicles_dir]
    found = set()
    path = os.path.join(vehicles_dir, "common.zip")
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".jbeam"):
                    found.update(_abs_slot_names(
                        zf.read(name).decode("utf-8", errors="replace")))
    except (OSError, KeyError, zipfile.BadZipFile):
        pass
    _COMMON_CACHE[vehicles_dir] = found
    return found


def generate_parent_jbeam(model, slot_types):
    """The "ML ABS" part, emitted ONCE PER candidate ABS slot. Its only children
    are a controller-less placeholder (the default -- brakes stay at stock
    capacity but nothing scales them, i.e. inert/off) plus, once exported, one
    part per trained model in MODEL_SLOT_TYPE.

    One variant per slot because a car's real socket cannot be identified
    reliably from its files alone, and guessing wrong is SILENT: the part loads
    and simply never appears in the menu. Unused variants are inert, since
    jbeam drops a part whose slotType nothing offers. Every variant points at
    ONE shared child slot, so a model exported once is offered by whichever
    variant the car actually uses.

    Accepts a bare string too, so single-slot callers keep working."""
    if isinstance(slot_types, str):
        slot_types = [slot_types]
    placeholder_name = f"{PLACEHOLDER_PART_NAME}_{model}"
    data = {
        placeholder_name: {
            "information": {"name": "(no model selected)", "value": 0},
            "slotType": MODEL_SLOT_TYPE,
        },
    }
    for slot_type in sorted(set(slot_types)):
        data[f"{PLACEHOLDER_PART_NAME}_{model}_{slot_type}_parent"] = {
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
