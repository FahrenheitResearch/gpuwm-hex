"""Which published gpuwm this port can run on: measured once, derived after.

THE BREAKAGE THIS MODULE PREVENTS, measured on a real install on 2026-08-27
(``evidence/userwalk-20260827/``): ``pip install gpuwm-hex`` resolved
``gpuwm 2.5.7``, whose bytes the sixteen-file seam manifest this port pins
refused.  The forecast lane stopped at launch with two SHA-256 digests and
no version number, ``gpuwm-hex doctor`` reported the estate healthy and
exited 0, and no document in the distribution named a version that works.
A green install and a dead run, with no route out by reading.

The dependency floor alone cannot prevent that, because a floor with no
ceiling always resolves to the newest engine, and the newest engine is by
definition the one this port has never measured.

So the specifier is DERIVED rather than transcribed:

* :data:`PUBLISHED_ENGINES` is the OUTPUT of a network measurement --
  ``evidence/standalone-20260827/measure_engine_verdicts.py`` -- not a
  restatement of anybody's prose.  Every hand-written statement of these
  numbers in this tree was wrong: the declaration said 5 of 16 files differ
  at 2.5.0 (measured: 9) and 3 at 2.5.4 (measured: 2), and the test guarding
  it said 13 at 2.5.0 and 6 at 2.5.1 (measured: 9 and 8).  Three
  transcriptions, three different wrong answers.  The table is now SPLICED
  from the instrument's JSON by
  ``evidence/repin-258-20260828/render_engine_pin_table.py``, so there is no
  fourth transcription to get wrong.
* :func:`gpuwm_floor`, :func:`gpuwm_ceiling` and :func:`gpuwm_requirement`
  compute the specifier from that table.  ``pyproject.toml`` states the
  result as a literal because a dependency specifier has to be static
  metadata, and ``tests/test_packaging_declaration.py`` fails if the literal
  and the derivation ever disagree.

**The ceiling is EXCLUSIVE and sits at the first engine that is not measured
usable, which is what stops the next engine cut from silently re-opening
this.**  An engine nobody has measured against the manifest is never
resolved onto by default; admitting one is a deliberate act that re-runs the
instrument and moves this table.

WHERE THIS STANDS NOW, re-measured 2026-08-31
(``evidence/repin-260-20260831/``).  The engine cut 2.6.0 and published it
to PyPI, this port re-pinned its manifest to that cut's bytes, and the
instrument was re-run over every published engine.  The derivation now
returns ``gpuwm>=2.6.0,<2.6.1``: 2.6.0 is measured usable and is ADMITTED.
The ceiling did its work again rather than being bypassed -- while the
table said ``<2.5.9``, 2.6.0 was excluded exactly as 2.5.9 would have been,
and letting it in cost what it is meant to cost: re-run the instrument,
move the table, let the literal follow.  The 2026-08-28 re-pin to 2.5.8
(``evidence/repin-258-20260828/``) paid the same toll the same way.

**EXACTLY ONE PUBLISHED ENGINE IS USABLE, and that is a measured finding
rather than a cautious choice.**  Re-pinning the manifest moved the whole
``moved`` column, because that column is relative to the manifest: every
engine below 2.6.0 now fails it, INCLUDING the 2.5.8 that was the floor the
day before.  Three manifest paths -- ``gpuwm/config.py``,
``gpuwm/core/rrtmg_legacy.py`` and ``gpuwm/io/restart.py`` -- moved across
that window, and no engine but 2.6.0 carries all sixteen.  So this port has
no fallback engine at all: if 2.6.0 were yanked from PyPI, the correct
answer is :class:`EnginePinError`, not a loosened bound.  Widening the
range means measuring something that passes, and nothing published does.

Two facts the 2026-08-27 measurement turned up that no document in the tree
carried.  Both are still true, and both are now moot for the FLOOR, because
2.5.5 and 2.5.6 fail the re-pinned manifest on bytes before either one
applies.  They stay in the table and stay written down here, because a row
removed is a reason the derivation can no longer show:

* **gpuwm 2.5.5 is not on PyPI.**  It is a git tag with no release, so the
  floor the declaration once named -- ``gpuwm>=2.5.5`` -- named a version
  pip cannot install.
* **2.5.5's published tree cannot follow its own printed build road.**
  ``tools/rustwx/vendor/crates-io/cc/src/target/generated.rs`` is absent at
  that tag, so the manual's ``cargo build --release --locked --offline``
  stops there; the same command finished in 56.6 s at 2.5.6.  A version that
  satisfies the pin and cannot be built from is not a remedy, so the
  derivation requires both -- which is why :attr:`PublishedEngine.usable`
  keeps all three clauses even now that the byte clause alone decides the
  answer.

This module imports nothing heavy at module scope on purpose: ``doctor``
imports it, and a report about a broken estate must not need the estate to
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

#: The engine distribution, spelled once.
DISTRIBUTION = "gpuwm"

#: The public repository a checkout comes from, named in remedies.
REPOSITORY = "https://github.com/FahrenheitResearch/arwen"


@dataclass(frozen=True)
class PublishedEngine:
    """One published gpuwm, as measured against this port's seam manifest."""

    version: str
    #: Whether pip can resolve it at all.  A tag is not a release.
    on_pypi: bool
    #: Manifest paths whose bytes differ at this version.  Empty means the
    #: whole sixteen-file manifest matches.
    moved: tuple[str, ...]
    #: Whether the vendored crate tree at this version is complete enough for
    #: the ``--offline`` build the manual prints as its remedy.
    offline_build_road: bool

    @property
    def satisfies_manifest(self) -> bool:
        return not self.moved

    @property
    def usable(self) -> bool:
        """Installable, byte-compatible, and buildable from.  All three."""

        return self.on_pypi and self.satisfies_manifest and self.offline_build_road


#: THE OUTPUT OF AN INSTRUMENT, NOT A TRANSCRIPTION.  Re-measured 2026-08-31
#: against the published bytes of every 2.5.x and 2.6.x gpuwm, plus the
#: v2.5.5 tag that has no PyPI release.  The instrument is
#: ``evidence/standalone-20260827/measure_engine_verdicts.py`` (network: it
#: downloads every published wheel); the JSON behind THIS table is
#: ``evidence/repin-260-20260831/engine-verdicts.json``, and the block below
#: is that JSON rendered by
#: ``evidence/repin-258-20260828/render_engine_pin_table.py --splice``.
#:
#: DO NOT HAND-PATCH A ROW.  ``moved`` is measured against THIS port's
#: sixteen-file seam manifest, so re-pinning the manifest moves the whole
#: column at once -- including rows for engines cut long before it.  The
#: 2026-08-28 re-pin to 2.5.8 changed every one of the nine rows: 2.5.7 both
#: GAINED entries (config.py, physics.py) and LOST one (microphysics.py,
#: whose bytes 2.5.7 and 2.5.8 share).  A patched row is a guess wearing a
#: measurement's clothes, and this module's docstring counts what that has
#: cost so far.
# --- BEGIN GENERATED: PUBLISHED_ENGINES ---
PUBLISHED_ENGINES: tuple[PublishedEngine, ...] = (
    PublishedEngine(
        version="2.5.0",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "gpuwm/config.py",
            "gpuwm/core/gf.py",
            "gpuwm/core/kernels/__init__.py",
            "gpuwm/core/kernels/gf.cu",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/mpas_column_batch.py",
            "gpuwm/core/noahmp_runtime.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.1",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "gpuwm/config.py",
            "gpuwm/core/gf.py",
            "gpuwm/core/kernels/__init__.py",
            "gpuwm/core/kernels/gf.cu",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/mpas_column_batch.py",
            "gpuwm/core/noahmp_runtime.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.2",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "gpuwm/config.py",
            "gpuwm/core/gf.py",
            "gpuwm/core/kernels/gf.cu",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/mpas_column_batch.py",
            "gpuwm/core/noahmp_runtime.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.3",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "gpuwm/config.py",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/mpas_column_batch.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.4",
        on_pypi=True,
        moved=(
            "docs/mpas-seam.md",
            "gpuwm/config.py",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/mpas_column_batch.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.5",
        on_pypi=False,
        moved=(
            "gpuwm/config.py",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=False,
    ),
    PublishedEngine(
        version="2.5.6",
        on_pypi=True,
        moved=(
            "gpuwm/config.py",
            "gpuwm/core/microphysics.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.7",
        on_pypi=True,
        moved=(
            "gpuwm/config.py",
            "gpuwm/core/physics.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.5.8",
        on_pypi=True,
        moved=(
            "gpuwm/config.py",
            "gpuwm/core/rrtmg_legacy.py",
            "gpuwm/io/restart.py",
        ),
        offline_build_road=True,
    ),
    PublishedEngine(
        version="2.6.0",
        on_pypi=True,
        moved=(),
        offline_build_road=True,
    ),
)
# --- END GENERATED: PUBLISHED_ENGINES ---


class EnginePinError(RuntimeError):
    """No published engine satisfies the pin, so no specifier can be derived."""


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _next_after(version: str) -> str:
    parts = list(_key(version))
    parts[-1] += 1
    return ".".join(str(part) for part in parts)


def engine(version: str) -> PublishedEngine | None:
    """The measured row for ``version``, or ``None`` if it was never measured."""

    for row in PUBLISHED_ENGINES:
        if row.version == version:
            return row
    return None


def gpuwm_floor() -> str:
    """The lowest engine that is installable, byte-compatible AND buildable."""

    usable = [row for row in PUBLISHED_ENGINES if row.usable]
    if not usable:
        raise EnginePinError(
            "no published gpuwm satisfies this port's seam manifest, is on "
            "PyPI and carries a complete vendored crate tree, so no "
            "dependency specifier can be derived.  Re-run "
            "evidence/standalone-20260827/measure_engine_verdicts.py; if it "
            "agrees, the engine has to publish before this distribution can"
        )
    return min(usable, key=lambda row: _key(row.version)).version


def gpuwm_ceiling() -> str:
    """The EXCLUSIVE upper bound: the first engine above the floor that is
    not measured usable.

    When every measured engine above the floor is usable the ceiling is the
    next patch after the highest measured one, so an engine that has not been
    measured is never resolved onto.  That is the whole point: the defect this
    module exists for was pip taking a newer engine than anybody had checked.
    """

    floor = gpuwm_floor()
    above = sorted(
        (row for row in PUBLISHED_ENGINES if _key(row.version) > _key(floor)),
        key=lambda row: _key(row.version),
    )
    for row in above:
        if not row.usable:
            return row.version
    highest = above[-1].version if above else floor
    return _next_after(highest)


def gpuwm_requirement() -> str:
    """The whole dependency specifier ``pyproject.toml`` must declare."""

    return f"{DISTRIBUTION}>={gpuwm_floor()},<{gpuwm_ceiling()}"


def wanted_version() -> str:
    """The one version to name in a remedy: the lowest engine that works."""

    return gpuwm_floor()


# ---------------------------------------------------------------------------
# what is actually on this machine
# ---------------------------------------------------------------------------
def seam_manifest() -> dict[str, str]:
    """The sixteen pinned paths and their digests.

    Imported here rather than at module scope because the manifest lives in a
    frozen module that pulls numpy, and ``doctor`` has to be able to report a
    numpy-less install rather than fail to import while doing so.
    """

    from .cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST

    return dict(ARWEN_SOURCE_MANIFEST)


@dataclass(frozen=True)
class SeamInspection:
    """What one tree's bytes say about the seam pin."""

    root: Path
    #: Manifest paths that are present and match.
    matched: tuple[str, ...]
    #: Manifest paths that are present and whose bytes have moved.
    moved: tuple[str, ...]
    #: Manifest paths this tree does not carry at all.
    absent: tuple[str, ...]

    @property
    def checked(self) -> int:
        return len(self.matched) + len(self.moved)

    @property
    def drifted(self) -> bool:
        return bool(self.moved)


def inspect_seam(root: Path) -> SeamInspection:
    """Compare every manifest path under ``root`` against the pinned bytes.

    ``absent`` is reported separately from ``moved`` because the two are
    different findings with different remedies: a path this tree does not
    carry at all is a packaging question, while a byte that moved is the
    engine being the wrong version.

    ``absent`` used to be the NORMAL answer for an install.  Engines up to
    2.5.7 shipped no ``docs/mpas-seam.md`` in the wheel, so fifteen of the
    sixteen resolved and the sixteenth could only be reached from a source
    checkout.  2.5.8 places that document in site-packages at the manifest's
    own key: measured 2026-08-28 against the wheel PyPI serves, this returns
    ``checked=16, matched=16, moved=(), absent=()``.  The branch stays,
    because an older engine still answers the old way and the two remedies
    are still different.
    """

    matched: list[str] = []
    moved: list[str] = []
    absent: list[str] = []
    for relative, expected in seam_manifest().items():
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError:
            absent.append(relative)
            continue
        if sha256(payload).hexdigest() == expected:
            matched.append(relative)
        else:
            moved.append(relative)
    return SeamInspection(
        root=root,
        matched=tuple(sorted(matched)),
        moved=tuple(sorted(moved)),
        absent=tuple(sorted(absent)),
    )


def installed_version() -> str | None:
    """The gpuwm distribution version pip has installed, or ``None``."""

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(DISTRIBUTION)
    except PackageNotFoundError:
        return None


def installed_root() -> Path | None:
    """The directory the manifest's ``gpuwm/...`` paths hang off, installed.

    ``find_spec`` rather than an import: importing gpuwm pulls its whole
    physics estate, and a report about that estate must not depend on it
    loading.
    """

    import importlib.util

    try:
        spec = importlib.util.find_spec(DISTRIBUTION)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent


def checkout_version(root: Path) -> str | None:
    """The version a gpuwm SOURCE CHECKOUT declares, read without importing.

    ``pyproject.toml`` states it statically, which is the only place a
    checkout carries its own version: ``gpuwm/__init__.py`` reads it back out
    of installed metadata, so importing it from a checkout answers with
    whatever is installed instead of what is on disk -- the exact confusion a
    refusal naming a version has to avoid.

    An INSTALL root answers ``None`` here, and correctly: site-packages
    carries no ``pyproject.toml``.  ``installed_version()`` is the reader for
    that case, and ``version_from_moved`` the last resort for a tree that
    states nothing.
    """

    declaration = root / "pyproject.toml"
    try:
        text = declaration.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib

        parsed = tomllib.loads(text)
    except Exception:
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else None
    project = parsed.get("project")
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str):
            return version
    return None


def version_from_moved(moved: tuple[str, ...]) -> str | None:
    """Name the published version whose measured drift is exactly ``moved``.

    The last resort when a tree states no version of its own: the table
    records which paths moved at each cut, so an observed drift set that
    matches one row's exactly identifies it.  Ambiguity answers ``None``
    rather than guessing.
    """

    hits = [row.version for row in PUBLISHED_ENGINES if row.moved == tuple(sorted(moved))]
    return hits[0] if len(hits) == 1 else None


def remedy(found: str | None = None) -> str:
    """The commands that put a usable engine on this machine.

    Both routes, because the two halves of this distribution need different
    things: the install is what pip resolves, the checkout is what the
    forecast lane's proof harness reads a git state out of.

    THE REASON MOVED AT 2.5.8 AND THIS TEXT MOVES WITH IT.  Until 2.5.7 the
    second line was owed because the seam pin named ``docs/mpas-seam.md``,
    which no wheel placed in site-packages: an install could not satisfy the
    pin at all.  2.5.8 ships it, all sixteen paths resolve from an install
    (measured 2026-08-28), and the clone is now owed for provenance instead.
    A remedy that keeps a dead reason is a refusal that does not name its
    breakage, which is the law this line is subject to.
    """

    wanted = wanted_version()
    lines = [
        f'  pip install "{gpuwm_requirement()}"',
        f"  git clone --depth 1 --branch v{wanted} {REPOSITORY}",
        "      # the forecast lane needs the GIT CHECKOUT too, and at "
        f"{wanted} the",
        "      # reason is provenance rather than a missing file: the run's",
        "      # proof harness records the checkout's HEAD, tree and dirty",
        "      # paths into every receipt, and an install has no commit to",
        "      # name the executed bytes by.  Pass it as --gpuwm-checkout.",
    ]
    if found is not None:
        lines.insert(0, f"  # found gpuwm {found}; this port pins {wanted}.")
    return "\n".join(lines)


__all__ = [
    "DISTRIBUTION",
    "EnginePinError",
    "PUBLISHED_ENGINES",
    "REPOSITORY",
    "PublishedEngine",
    "SeamInspection",
    "checkout_version",
    "engine",
    "gpuwm_ceiling",
    "gpuwm_floor",
    "gpuwm_requirement",
    "inspect_seam",
    "installed_root",
    "installed_version",
    "remedy",
    "seam_manifest",
    "version_from_moved",
    "wanted_version",
]
