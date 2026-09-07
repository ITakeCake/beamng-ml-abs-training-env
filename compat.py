"""beamngpy <-> BeamNG.tech/.drive version compatibility, per BeamNGpy's own
published table (https://github.com/BeamNG/BeamNGpy/blob/master/COMPATIBILITY.md,
fetched 2026-08-29). Replaces a hard version-mismatch refusal with a warning +
fix command, a version pin tuned for one machine has no business hard-failing
on someone else's, especially as new BeamNG/beamngpy releases ship."""
import dataclasses
import re

# game version (major.minor, as BeamNGpy's table keys it) -> required beamngpy
TABLE = {
    "0.39": "1.36",
    "0.38": "1.35.1",
    "0.37": "1.34.1",
    "0.36": "1.33.1",
    "0.35": "1.32",
    "0.34": "1.31",
    "0.33": "1.30",
    "0.32": "1.29",
    "0.31": "1.28",
    "0.30": "1.27.1",
    "0.29": "1.26.1",
    "0.28": "1.26.1",
    "0.27": "1.25.1",
    "0.26": "1.24",
    "0.25": "1.23.1",
    "0.24": "1.22",
    "0.23": "1.21.1",
    "0.22": "1.20",
    "0.21": "1.19.1",
}


@dataclasses.dataclass
class CompatResult:
    ok: object              # True / False / None (None = can't tell, unknown game version)
    required: str = None    # beamngpy version the table says this game needs
    message: str = ""
    fix_command: str = ""


def _major_minor(version):
    m = re.match(r"^(\d+\.\d+)", version or "")
    return m.group(1) if m else None


def required_beamngpy(game_version):
    """None if this game version isn't in the table (newer than our last
    fetch, or a dev build), callers must treat that as 'unknown', not
    'incompatible'."""
    return TABLE.get(_major_minor(game_version))


def check_compat(game_version, beamngpy_version):
    if not beamngpy_version:
        return CompatResult(ok=False, message="beamngpy is not installed.",
                            fix_command="pip install beamngpy")

    required = required_beamngpy(game_version) if game_version else None

    if not game_version or not required:
        return CompatResult(
            ok=None, required=required,
            message=f"Game version unknown or not in BeamNGpy's compatibility "
                    f"table (detected: {game_version!r}), can't confirm "
                    f"beamngpy {beamngpy_version} is right for it. Check "
                    f"https://github.com/BeamNG/BeamNGpy/blob/master/COMPATIBILITY.md "
                    f"and proceed at your own risk.")

    if beamngpy_version == required:
        return CompatResult(ok=True, required=required,
                            message=f"beamngpy {beamngpy_version} matches "
                                    f"BeamNG {game_version}.")

    return CompatResult(
        ok=False, required=required,
        message=f"beamngpy {beamngpy_version} does not match BeamNG "
                f"{game_version} (needs {required}). A mismatched beamngpy "
                f"speaks a different wire protocol and fails the connection "
                f"handshake outright.",
        fix_command=f"pip install beamngpy=={required}")
