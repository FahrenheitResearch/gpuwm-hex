"""One resolution ladder for the Rust engines this package drives.

Nothing in this distribution draws a weather field, interpolates a
meteorological column, or writes an initial condition in Python.  Those
are Rust binaries -- ``rw_mpas_init``, ``rw_mpas_convert``,
``rw_wrfbatch`` -- built from the ``tools/rustwx`` workspace that lives
in the gpuwm repository.  The wheel therefore CANNOT carry them, and a
door that needs one has exactly two states: it found the binary, or it
refuses by name and says what to run.

The reason this module exists rather than a ladder per door: the doors
were each resolving their own way, and one of them was wrong in a way
nobody could see from inside it.  ``pip install gpuwm-hex`` pulls
``gpuwm``; ``gpuwm fetch-bridges`` then stages a release's prebuilt
bundle -- and that bundle carries ``rw_mpas_mesh``, ``rw_mpas_static``,
``rw_mpas_init``, ``rw_mpas_convert`` and ``rw_wrfbatch`` -- into
``~/.gpuwm/bridges``.  The init door read one environment variable and
PATH, so a user who had run the one command that installs the engine
still met a refusal telling them to build it with cargo.  A remedy that
ignores the estate the user already has is not a remedy.

THE LADDER, best first, identical for every engine:

1. the door's own command-line flag;
2. this distribution's environment variable, then any legacy spellings
   it has carried (a rename never silently stops reading a variable);
3. gpuwm's own environment variable and its bridge ladder -- a gpuwm
   checkout's ``tools/rustwx/target/{release,debug}``, ``libexec/bridges``
   beside the installed package, the wheel-bundled directory inside it,
   and ``~/.gpuwm/bridges`` where ``gpuwm fetch-bridges`` stages;
4. ``PATH``.

Rungs 1-3 fail LOUDLY when they name a file that is not there.  An
explicit configuration that falls through to a different binary is how a
box runs the wrong engine and reports success, so a named-but-missing
path is a refusal, never a skipped rung.

Rung 3 is read through :mod:`gpuwm.mpas_mesh` when that module can be
imported, so the two ladders cannot drift; when it cannot -- an older
gpuwm, or none at all -- the two directories that do not depend on
gpuwm's internals (``~/.gpuwm/bridges`` and ``libexec/bridges`` beside
the package) are still probed directly, and the refusal names the
version floor instead of pretending the rung does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from .errors import MpasPortError


class EngineRefusal(MpasPortError):
    """No engine could be resolved: what breaks, and the command that fixes it."""


#: The command that stages a prebuilt bundle, named in every refusal.
FETCH_COMMAND = "gpuwm fetch-bridges"

#: Where that command stages, and the last rung of gpuwm's ladder.
USER_BRIDGE_DIR = Path.home() / ".gpuwm" / "bridges"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


@dataclass(frozen=True)
class EngineSpec:
    """One Rust binary: how it is named, what it owns, how it is built."""

    #: The artifact's basename, without any platform suffix.
    name: str
    #: The door's flag, printed in the resolution order.
    flag: str
    #: Environment variables, preferred first, legacy after.  Every one is
    #: read; none is ever dropped.
    env_names: tuple[str, ...]
    #: gpuwm's own spelling for the same artifact, read after this
    #: distribution's own and before the directory rungs.
    gpuwm_env: str
    #: What the engine is, in one noun phrase.
    subject: str
    #: What is unreachable without it -- the concrete breakage, not "an error".
    what_breaks: str
    #: The cargo package that builds it inside the ``tools/rustwx`` workspace.
    cargo_package: str


#: Meteorological interpolation and the initial-condition write.
INIT = EngineSpec(
    name="rw_mpas_init",
    flag="--engine",
    env_names=("GPUWM_HEX_RW_MPAS_INIT", "RW_MPAS_INIT"),
    gpuwm_env="GPUWM_RW_MPAS_INIT",
    subject="the initial-condition builder",
    what_breaks=(
        "no meteorological interpolation and no initial-condition write can "
        "occur, so the init door has nothing to produce"
    ),
    cargo_package="rw-mpas",
)

#: History onto the renderer's tape.
CONVERT = EngineSpec(
    name="rw_mpas_convert",
    flag="--convert-exe",
    env_names=("GPUWM_HEX_RW_MPAS_CONVERT", "MPAS_PORT_RW_MPAS_CONVERT"),
    gpuwm_env="GPUWM_RW_MPAS_CONVERT",
    subject="the history converter",
    what_breaks=(
        "a native history has no wrfout-shaped frame for the renderer to "
        "import, so nothing can be drawn"
    ),
    cargo_package="rw-mpas",
)

#: The renderer itself.  There is no Python plotter behind this door.
RENDERER = EngineSpec(
    name="rw_wrfbatch",
    flag="--renderer-exe",
    env_names=(
        "GPUWM_HEX_RW_WRFBATCH",
        "MPAS_PORT_RW_WRFBATCH",
    ),
    gpuwm_env="GPUWM_RW_WRFBATCH",
    subject="the product renderer",
    what_breaks=(
        "weather-field products come from this binary only -- this door has "
        "no Python plotter behind it -- so there are no products"
    ),
    cargo_package="rw-wrfbatch",
)

#: Every engine a front door of this distribution drives, best-known name
#: first.  ``mpas_port.doctor`` reads this, so adding an engine is a row
#: here and is reported without a second edit.
ENGINES: tuple[EngineSpec, ...] = (INIT, CONVERT, RENDERER)


# ---------------------------------------------------------------------------
# gpuwm's ladder, read through gpuwm when possible
# ---------------------------------------------------------------------------
def _gpuwm_module_candidates(spec: EngineSpec) -> tuple[Path, ...] | None:
    """gpuwm's own candidate list for this artifact, or ``None``.

    ``None`` means gpuwm could not answer -- not installed, too old to
    know this artifact, or an import that raised.  The caller falls back
    to the two directory rungs that do not need gpuwm's internals, and
    the refusal names the floor.
    """

    try:
        from gpuwm import mpas_mesh  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    bridge = getattr(mpas_mesh, "BRIDGES", {}).get(spec.name)
    if bridge is None:
        return None
    try:
        return tuple(Path(candidate) for candidate in bridge.candidates())
    except Exception:  # pragma: no cover - a gpuwm whose ladder changed shape
        return None


def gpuwm_bundles(spec: EngineSpec) -> bool | None:
    """Does THIS installed gpuwm's ``fetch-bridges`` carry this engine?

    ``True`` yes, ``False`` gpuwm is here but its bundle has no such row,
    ``None`` no gpuwm at all.

    Asked of the INSTALLED distribution, never assumed from a version
    number, because the answer has already differed from the obvious one:
    gpuwm's published 2.5.2 wheel declares ``rw_wrfbatch`` among its
    bundled artifacts and does NOT declare the four MPAS binaries, while
    a gpuwm source checkout of the same era declares all five.  A remedy
    written from the checkout tells a user on the released wheel to run a
    command that cannot give them the file, which is worse than saying
    nothing -- they run it, it succeeds, and the door still refuses.
    """

    try:
        from gpuwm import bridge_assets  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    try:
        return any(
            getattr(artifact, "name", None) == spec.name
            for artifact in bridge_assets.BUNDLED_ARTIFACTS
        )
    except Exception:  # pragma: no cover - a gpuwm whose table changed shape
        return None


def _gpuwm_package_dir() -> Path | None:
    try:
        import gpuwm  # noqa: PLC0415 - lazy on purpose
    except Exception:  # pragma: no cover - depends on the installed estate
        return None
    location = getattr(gpuwm, "__file__", None)
    return Path(location).resolve().parent if location else None


def _directory_candidates(spec: EngineSpec) -> tuple[Path, ...]:
    """The rungs that survive gpuwm being absent or old."""

    filename = executable_name(spec.name)
    candidates: list[Path] = []
    package = _gpuwm_package_dir()
    if package is not None:
        candidates.append(package / "libexec" / "bridges" / filename)
        candidates.append(package.parent / "libexec" / "bridges" / filename)
    candidates.append(USER_BRIDGE_DIR / filename)
    return tuple(candidates)


def gpuwm_candidates(spec: EngineSpec) -> tuple[Path, ...]:
    """Rung 3, whichever way it can be answered."""

    through_gpuwm = _gpuwm_module_candidates(spec)
    if through_gpuwm is not None:
        return through_gpuwm
    return _directory_candidates(spec)


# ---------------------------------------------------------------------------
# the remedy
# ---------------------------------------------------------------------------
def resolution_order(spec: EngineSpec) -> str:
    """The ladder in one line, so a refusal says where it looked."""

    read = [spec.flag]
    read.extend(f"${name}" for name in spec.env_names)
    read.append(f"${spec.gpuwm_env}")
    return (
        f"{', '.join(read)}, gpuwm's bridge directories "
        f"(a gpuwm checkout's tools/rustwx/target/release, libexec/bridges "
        f"beside the installed package, and {USER_BRIDGE_DIR}), then PATH"
    )


def remedy(spec: EngineSpec) -> str:
    """What to run, most likely to work first, ON THIS BOX.

    The order is not fixed text: it is decided by asking the installed
    gpuwm whether its bundle actually carries this engine
    (:func:`gpuwm_bundles`).  Where it does, the one-command staging is
    offered first because it needs no toolchain.  Where it does not, the
    build is offered first and the staging command is NOT offered at
    all -- an offer that cannot deliver the file wastes a user's time
    and costs the next refusal its credibility.

    Every line is a command as typed or a ``#`` comment, never prose
    fused onto a command.
    """

    build = [
        "  # build it from a gpuwm source checkout:",
        "  cargo build --release --locked --offline "
        f"-p {spec.cargo_package} --bin {spec.name}",
        "      # run at tools/rustwx in the checkout, then:",
        f"  # set {spec.env_names[0]} to the built binary",
    ]
    staged = [
        f"  {FETCH_COMMAND}",
        f"      # stages the prebuilt bundle -- which carries {spec.name} --",
        f"      # into {USER_BRIDGE_DIR}, where the ladder above reads it.",
        "      # Requires a published bundle for this platform.",
    ]

    bundled = gpuwm_bundles(spec)
    if bundled is None:
        return "\n".join(
            [
                '  pip install "gpuwm>=2.5.5"',
                "      # no gpuwm is installed in this interpreter, so its",
                f"      # {FETCH_COMMAND} route is unavailable until one is.",
                "",
                *staged,
                "",
                "  # or, on a platform with no published bundle:",
                *build[1:],
            ]
        )
    if bundled:
        return "\n".join([*staged, "", "  # or, to build it yourself:", *build[1:]])
    return "\n".join(
        [
            *build,
            "",
            f"      # NOTE: `{FETCH_COMMAND}` will not supply {spec.name} on the",
            "      # gpuwm installed here -- its bundle carries no such artifact.",
            "      # A later gpuwm release adds it; until then the build above is",
            "      # the route.  Check with:  pip install --upgrade gpuwm",
        ]
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def _named_but_missing(spec: EngineSpec, source: str, path: Path) -> None:
    raise EngineRefusal(
        f"{source} names {path}, which is not a file.  An explicit setting is "
        f"never skipped in favour of a different binary -- that is how a box "
        f"runs the wrong engine and reports success.  Point it at a built "
        f"{spec.name}, or unset it to continue down the ladder.\n"
        f"{remedy(spec)}"
    )


def _usable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name != "posix":
        return True
    return os.access(path, os.X_OK)


def _refuse_not_executable(spec: EngineSpec, path: Path) -> None:
    raise EngineRefusal(
        f"the {spec.subject} at {path} exists but is not executable.  pip does "
        f"not preserve executable bits on package data, so a staged copy can "
        f"arrive present and unrunnable.\n"
        f"  chmod +x {path}"
    )


def resolve(spec: EngineSpec, explicit: str | Path | None = None) -> Path:
    """The engine binary, or a refusal naming the command that supplies it."""

    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            _named_but_missing(spec, spec.flag, candidate)
        if os.name == "posix" and not os.access(candidate, os.X_OK):
            _refuse_not_executable(spec, candidate)
        return candidate

    for name in (*spec.env_names, spec.gpuwm_env):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_file():
            _named_but_missing(spec, f"${name}", candidate)
        if os.name == "posix" and not os.access(candidate, os.X_OK):
            _refuse_not_executable(spec, candidate)
        return candidate

    for candidate in gpuwm_candidates(spec):
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            if os.name == "posix" and not os.access(candidate, os.X_OK):
                _refuse_not_executable(spec, candidate)
            return candidate.resolve()

    found = shutil.which(spec.name)
    if found:
        return Path(found)

    # "<name> not found" is the exact phrase the packaging battery greps
    # for, and it is the phrase a user greps for too.  Keep it.
    raise EngineRefusal(
        f"{spec.name} not found, so {spec.what_breaks}.\n"
        f"Looked at: {resolution_order(spec)}.\n"
        f"This distribution ships no compiled engine: the Rust binaries are "
        f"built from the gpuwm tools/rustwx workspace and are staged onto a "
        f"machine, never carried in this wheel.  Supply it with one of:\n"
        f"{remedy(spec)}"
    )


def locate(spec: EngineSpec) -> tuple[Path | None, str]:
    """Resolve without raising: ``(path, where it came from)``.

    The read-only form :mod:`mpas_port.doctor` reports through, so the
    report can name every gap at once instead of stopping at the first.
    """

    for name in (*spec.env_names, spec.gpuwm_env):
        value = os.environ.get(name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return candidate, f"${name}"
        return None, f"${name} names a missing file: {candidate}"

    for candidate in gpuwm_candidates(spec):
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            return candidate.resolve(), "gpuwm's bridge directories"

    found = shutil.which(spec.name)
    if found:
        return Path(found), "PATH"
    return None, "not found on any rung"


__all__ = [
    "CONVERT",
    "ENGINES",
    "FETCH_COMMAND",
    "INIT",
    "RENDERER",
    "USER_BRIDGE_DIR",
    "EngineRefusal",
    "EngineSpec",
    "executable_name",
    "gpuwm_bundles",
    "gpuwm_candidates",
    "locate",
    "remedy",
    "resolution_order",
    "resolve",
]
