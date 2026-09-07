"""Copies this project's shipped assets (assets/mods/mtb_ml_abs, assets/cars/*)
into a chosen BeamNG userpath. Pure filesystem + hashing, no game imports."""
import hashlib
import os
import shutil


def _dest_for(rel_source):
    """assets/-relative source path -> userpath-relative dest path.
    mods/<name>/... -> mods/unpacked/<name>/...
    cars/<model>/X.pc -> vehicles/<model>/X.pc"""
    parts = rel_source.split(os.sep)
    if parts[0] == "mods":
        return os.path.join("mods", "unpacked", *parts[1:])
    if parts[0] == "cars":
        return os.path.join("vehicles", *parts[1:])
    raise ValueError(f"asset path {rel_source!r} is not under mods/ or cars/ "
                     f"-- add a mapping rule in asset_installer._dest_for")


def iter_asset_files(assets_dir):
    """Yields (rel_source_path, rel_dest_path) for every file under assets_dir."""
    for root, _dirs, files in os.walk(assets_dir):
        for f in files:
            abs_src = os.path.join(root, f)
            rel_src = os.path.relpath(abs_src, assets_dir)
            yield rel_src, _dest_for(rel_src)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_status(assets_dir, userpath):
    """{rel_dest_path: "MISSING" | "OK" | "STALE"}."""
    status = {}
    for rel_src, rel_dest in iter_asset_files(assets_dir):
        dest = os.path.join(userpath, rel_dest)
        if not os.path.isfile(dest):
            status[rel_dest] = "MISSING"
        elif _sha256(dest) == _sha256(os.path.join(assets_dir, rel_src)):
            status[rel_dest] = "OK"
        else:
            status[rel_dest] = "STALE"
    return status


def install_assets(assets_dir, userpath):
    """Copies every shipped asset into userpath, overwriting anything that
    differs (these are the tool's own managed files, not user data, an
    intentional 'Install/Update' action, not a silent clobber of unrelated
    work). Returns [(rel_dest_path, "installed"|"updated"|"unchanged")]."""
    report = []
    for rel_src, rel_dest in iter_asset_files(assets_dir):
        src = os.path.join(assets_dir, rel_src)
        dest = os.path.join(userpath, rel_dest)
        if os.path.isfile(dest):
            action = "unchanged" if _sha256(dest) == _sha256(src) else "updated"
        else:
            action = "installed"
        if action != "unchanged":
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
        report.append((rel_dest, action))
    return report
