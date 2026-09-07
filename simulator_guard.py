"""Read-only guard against colliding with somebody else's BeamNG session."""
import os
import re


MAIN_EXE = re.compile(r"^beamng\.(tech|drive).*\.exe$", re.IGNORECASE)


def active_beamng_sessions(processes=None, game=None):
    """Return main BeamNG processes, optionally limited to one product.

    ``game=None`` intentionally retains the conservative legacy behavior and
    reports both BeamNG.tech and BeamNG.drive.  Product-aware launchers should
    pass ``"tech"`` or ``"drive"`` so a separate installation does not create
    a false collision.
    """
    if game is not None:
        game = str(game).strip().lower()
        if game not in ("tech", "drive"):
            raise ValueError("game must be 'tech', 'drive', or None")
    if processes is None:
        import psutil
        processes = psutil.process_iter(["pid", "name", "cmdline"])
    found = []
    for process in processes:
        try:
            info = process.info
            name = str(info.get("name") or "")
            command = [str(part) for part in (info.get("cmdline") or [])]
            pid = int(info.get("pid") or process.pid)
        except (AttributeError, TypeError, ValueError, OSError):
            continue
        match = MAIN_EXE.match(os.path.basename(name))
        if not match:
            continue
        product = match.group(1).lower()
        if game is not None and product != game:
            continue
        if any(part.startswith("--type=") for part in command):
            continue
        found.append({"pid": pid, "name": name, "game": product,
                      "command": " ".join(command)})
    return sorted(found, key=lambda row: row["pid"])


def require_beamng_available(processes=None, game=None):
    sessions = active_beamng_sessions(processes, game=game)
    if sessions:
        description = ", ".join("%s pid=%d" % (row["name"], row["pid"])
                                for row in sessions)
        raise RuntimeError(
            "BeamNG%s is already in use (%s); refusing to launch another "
            "session or interfere with it" % (
                ".%s" % game if game else "", description))
    return True
