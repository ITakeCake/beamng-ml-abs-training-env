"""Crash-safe experiment artifacts shared by training, evaluation, and the GUI.

The commit record is written *after* both checkpoint files have reached their
final names.  A hard kill can therefore leave an obvious uncommitted partial,
but can never make a half-written pair look resumable.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
import time
import uuid
import zipfile
import numbers
from pathlib import Path

import numpy as np
import torch as th


PAIR_SCHEMA = 1
RUN_STATE_SCHEMA = 1
ATOMIC_REPLACE_ATTEMPTS = 12
ATOMIC_REPLACE_INITIAL_DELAY_S = 0.01
ATOMIC_REPLACE_MAX_DELAY_S = 0.25
SOURCE_SUFFIXES = {".py", ".lua", ".jbeam", ".json", ".md", ".txt"}
SOURCE_EXCLUDES = {
    ".git", ".pytest_cache", ".venv", "__pycache__", "runs", "logs",
    ".gui-configs", "evaluations", "best_models", "experiments",
    "settings.json", "gui_state.json",
}


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def json_safe(value):
    """Convert NumPy scalars and non-finite diagnostics to strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_write_json(path, value):
    """Replace one JSON file atomically without exposing a partial document.

    Windows can transiently reject ``os.replace`` while a GUI or monitor has
    the destination open.  Retrying the same fully flushed temporary file
    preserves atomic-reader semantics without allowing a harmless status read
    to terminate training.
    """
    path = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % os.path.basename(path), suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        delay = ATOMIC_REPLACE_INITIAL_DELAY_S
        for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, ATOMIC_REPLACE_MAX_DELAY_S)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def append_jsonl(path, value):
    """Append one flushed experiment-ledger/status event."""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(value), sort_keys=True,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def update_run_state(path, **updates):
    """Merge fields into the latest atomic run-state snapshot."""
    current = {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            current = loaded
    except (OSError, json.JSONDecodeError):
        pass
    current.update(updates)
    current["schema"] = RUN_STATE_SCHEMA
    current["updated_utc"] = utc_now()
    atomic_write_json(path, current)
    return current


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def paired_vecnormalize_path(model_path):
    """Infer the established VecNormalize filename for final or step models."""
    model_path = os.path.abspath(os.fspath(model_path))
    name = os.path.basename(model_path)
    if name == "final.zip":
        return os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if not name.lower().endswith(".zip"):
        raise ValueError("checkpoint must end in .zip")
    return model_path[:-4] + "_vecnormalize.pkl"


def pair_manifest_path(model_path):
    model_path = os.path.abspath(os.fspath(model_path))
    if not model_path.lower().endswith(".zip"):
        raise ValueError("checkpoint must end in .zip")
    return model_path[:-4] + ".pair.json"


def all_tensors_finite(value):
    if th.is_tensor(value):
        return bool(th.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_tensors_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(all_tensors_finite(item) for item in value)
    return True


def validate_checkpoint_pair(model_path, vec_path=None, *,
                             expected_obs_dim=None, expected_action_dim=None,
                             require_bounded=True):
    """Load and deeply validate a trusted local PPO/VecNormalize pair."""
    from stable_baselines3 import PPO
    from bounded_ppo import has_bounded_action_interface

    model_path = os.path.abspath(os.fspath(model_path))
    vec_path = os.path.abspath(os.fspath(
        vec_path or paired_vecnormalize_path(model_path)))
    if not os.path.isfile(model_path):
        raise ValueError("model checkpoint does not exist: %s" % model_path)
    if not os.path.isfile(vec_path):
        raise ValueError("paired VecNormalize does not exist: %s" % vec_path)
    if not zipfile.is_zipfile(model_path):
        raise ValueError("model checkpoint is not a valid zip: %s" % model_path)
    with zipfile.ZipFile(model_path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError("model checkpoint has corrupt member: %s" % corrupt)

    model = PPO.load(model_path, device="cpu")
    if require_bounded and not has_bounded_action_interface(model):
        raise ValueError("checkpoint does not use the bounded PPO action interface")
    if not all_tensors_finite(model.policy.state_dict()):
        raise ValueError("checkpoint policy contains NaN or Inf")
    if not all_tensors_finite(model.policy.optimizer.state_dict()):
        raise ValueError("checkpoint optimizer contains NaN or Inf")

    try:
        with open(vec_path, "rb") as handle:
            vec = pickle.load(handle)
    except Exception as exc:
        raise ValueError("could not load paired VecNormalize: %s" % exc) from exc
    mean = np.asarray(vec.obs_rms.mean, dtype=np.float64)
    variance = np.asarray(vec.obs_rms.var, dtype=np.float64)
    if (not math.isfinite(float(vec.epsilon)) or float(vec.epsilon) <= 0.0 or
            not math.isfinite(float(vec.clip_obs)) or float(vec.clip_obs) <= 0.0):
        raise ValueError("VecNormalize epsilon/clip_obs is non-finite or invalid")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError("VecNormalize observation statistics contain NaN or Inf")
    if np.any(variance < 0.0):
        raise ValueError("VecNormalize observation variance is negative")

    obs_dim = int(mean.size)
    action_dim = int(np.prod(model.action_space.shape))
    if (not np.allclose(model.action_space.low, 0.0) or
            not np.allclose(model.action_space.high, 1.0)):
        raise ValueError("checkpoint action space is not bounded to [0,1]")
    if expected_obs_dim is not None and obs_dim != int(expected_obs_dim):
        raise ValueError("checkpoint observation dimension %d, expected %d" %
                         (obs_dim, expected_obs_dim))
    if expected_action_dim is not None and action_dim != int(expected_action_dim):
        raise ValueError("checkpoint action dimension %d, expected %d" %
                         (action_dim, expected_action_dim))
    return {
        "model_path": model_path,
        "vecnormalize_path": vec_path,
        "timesteps": int(model.num_timesteps),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "model_sha256": sha256_file(model_path),
        "vecnormalize_sha256": sha256_file(vec_path),
        "model_bytes": os.path.getsize(model_path),
        "vecnormalize_bytes": os.path.getsize(vec_path),
    }


def save_checkpoint_pair(model, vecnormalize, model_path, vec_path=None,
                         metadata=None):
    """Save, validate, publish, and commit one immutable checkpoint pair."""
    model_path = os.path.abspath(os.fspath(model_path))
    vec_path = os.path.abspath(os.fspath(
        vec_path or paired_vecnormalize_path(model_path)))
    manifest_path = pair_manifest_path(model_path)
    for path in (model_path, vec_path, manifest_path):
        if os.path.exists(path):
            raise FileExistsError("refusing to overwrite experiment artifact: %s" % path)
    directory = os.path.dirname(model_path)
    if directory != os.path.dirname(vec_path):
        raise ValueError("model and VecNormalize must be saved in one directory")
    os.makedirs(directory, exist_ok=True)
    token = uuid.uuid4().hex
    partial_model = model_path[:-4] + ".partial-%s.zip" % token
    partial_vec = vec_path + ".partial-%s" % token
    try:
        model.save(partial_model)
        vecnormalize.save(partial_vec)
        expected_obs_dim = int(np.prod(model.observation_space.shape))
        expected_action_dim = int(np.prod(model.action_space.shape))
        validation = validate_checkpoint_pair(
            partial_model, partial_vec, expected_obs_dim=expected_obs_dim,
            expected_action_dim=expected_action_dim, require_bounded=True)
        os.replace(partial_model, model_path)
        os.replace(partial_vec, vec_path)
        committed = dict(validation)
        committed.update({
            "schema": PAIR_SCHEMA,
            "committed_utc": utc_now(),
            "model_path": os.path.basename(model_path),
            "vecnormalize_path": os.path.basename(vec_path),
            "metadata": dict(metadata or {}),
        })
        # Hashes were computed before the atomic rename; bytes are unchanged.
        atomic_write_json(manifest_path, committed)
        return committed
    except BaseException:
        for path in (partial_model, partial_vec):
            try:
                os.remove(path)
            except OSError:
                pass
        raise


def source_provenance(root):
    """Git identity plus a digest of the actual local source tree, dirty or not."""
    root = os.path.abspath(os.fspath(root))
    revision = None
    dirty = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    digest = hashlib.sha256()
    count = 0
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SOURCE_EXCLUDES for part in relative.parts):
            continue
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            digest.update(handle.read())
        digest.update(b"\0")
        count += 1
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "source_files": count,
    }
