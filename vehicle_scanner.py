"""Reads the BeamNG install's own vehicle zips + a userpath's custom configs
to drive the GUI's Model -> Trim -> Custom car picker.

No game imports -- pure zip/json/filesystem reading, unit-testable offline.
Results are cached by the GUI (vehicles_<version>.json under cache/) since
scanning every zip in content/vehicles/ on each open is slow.
"""
import dataclasses
import glob
import json
import os
import re
import zipfile


# Models the shipped ML-ABS controller mod actually targets (see assets/mods/
# mtb_ml_abs -- its jbeam declares an etk800-only slotType). Selecting any
# other model still works mechanically (you can train/deploy a standard ABS
# replacement), but the reward/gates/controller were only ever validated on
# this one -- the GUI shows a warning rather than a silent wrong assumption.
SUPPORTED_ML_ABS_MODELS = {"etk800"}


@dataclasses.dataclass
class ModelInfo:
    name: str            # zip/folder name, e.g. "etk800" -- the partConfig prefix
    display_name: str    # info.json "Name", e.g. "800-Series"
    zip_path: str


@dataclasses.dataclass
class TrimInfo:
    pc_name: str          # .pc filename without extension -- goes into the
                          # "vehicles/<model>/<pc_name>.pc" partConfig string
    display_name: str     # info_<pc_name>.json "Configuration", else pc_name


def _read_json_from_zip(zf, name):
    try:
        return json.loads(zf.read(name).decode("utf-8", errors="replace"))
    except (KeyError, json.JSONDecodeError):
        return None


def scan_models(vehicles_dir):
    """Every *.zip in vehicles_dir whose info.json says "Type": "Car"."""
    out = []
    for zip_path in sorted(glob.glob(os.path.join(vehicles_dir, "*.zip"))):
        model = os.path.splitext(os.path.basename(zip_path))[0]
        try:
            with zipfile.ZipFile(zip_path) as zf:
                info = _read_json_from_zip(zf, f"vehicles/{model}/info.json")
        except (OSError, zipfile.BadZipFile):
            continue
        if not info or info.get("Type") != "Car":
            continue
        out.append(ModelInfo(name=model, display_name=info.get("Name", model),
                             zip_path=zip_path))
    return out


def scan_trims(zip_path, model):
    """Every vehicles/<model>/*.pc inside the model's zip, display name from
    the matching info_<trim>.json's "Configuration" (fallback: the filename)."""
    if not os.path.isfile(zip_path):
        return []
    out = []
    prefix = f"vehicles/{model}/"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pc_files = [n for n in names
                   if n.startswith(prefix) and n.endswith(".pc")
                   and "/" not in n[len(prefix):]]
        for n in pc_files:
            pc_name = os.path.splitext(os.path.basename(n))[0]
            info = _read_json_from_zip(zf, f"{prefix}info_{pc_name}.json")
            display = (info or {}).get("Configuration", pc_name)
            out.append(TrimInfo(pc_name=pc_name, display_name=display))
    return sorted(out, key=lambda t: t.pc_name)


def scan_custom_configs(userpath, model):
    """Every *.pc a user saved under <userpath>/vehicles/<model>/ -- these have
    no info_<trim>.json (that only ships inside the stock zip), so display_name
    is always just the filename."""
    car_dir = os.path.join(userpath, "vehicles", model)
    if not os.path.isdir(car_dir):
        return []
    out = []
    for f in sorted(os.listdir(car_dir)):
        if f.endswith(".pc"):
            name = os.path.splitext(f)[0]
            out.append(TrimInfo(pc_name=name, display_name=name))
    return out


def resolve_part_config(model, pc_name):
    """The partConfig string beamngpy's Vehicle(...) expects."""
    return f"vehicles/{model}/{pc_name}.pc"


def unique_model_labels(models):
    """{display label: ModelInfo}, disambiguating models that share a display
    name (real-world case: BeamNG's 'midsize' and 'pessima' zips both report
    Name="Pessima") by appending the folder name. A plain dict-by-display-name
    would silently drop one of them."""
    from collections import Counter
    counts = Counter(m.display_name for m in models)
    labels = {}
    for m in models:
        label = (f"{m.display_name} ({m.name})" if counts[m.display_name] > 1
                else m.display_name)
        labels[label] = m
    return labels
