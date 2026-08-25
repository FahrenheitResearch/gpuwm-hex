"""What the distribution declares must match what the package actually is.

Every assertion here exists because the opposite has cost somebody a cut:
a version restated in two files and shipped wrong for four releases, a loose
package glob that swallowed a sibling distribution, a dependency table that
did not name a module the code imports at line one, a data file that never
reached the wheel.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = ROOT / "src" / "mpas_port"

#: Distribution names on PyPI vs the module they install.  Only the entries
#: that actually differ; anything else is assumed to match.
_DISTRIBUTION_OF_MODULE = {"netCDF4": "netCDF4", "cupy": "cupy"}


@pytest.fixture(scope="module")
def declaration() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the version, stated once
# ---------------------------------------------------------------------------
def test_the_version_is_stated_in_exactly_one_place(declaration: dict) -> None:
    declared = declaration["project"]["version"]
    assert declared == "0.1.1", (
        "0.1.1 is the declared cut of the 0.1 line (global variable-resolution "
        "on one consumer GPU, deterministic, ArWen physics); moving it is a "
        "release decision, not an edit"
    )

    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    # "0+unknown" is allowed and is the point: it is what the package says
    # when it is NOT installed, and it says so rather than inventing a number.
    # Anything shaped like a release is a second source of truth.
    literals = [text for text in literals if re.match(r"^\d+\.\d+", text)]
    assert literals == [], (
        "__version__ is assigned a release-shaped string literal in "
        "mpas_port/__init__.py.  "
        "That makes the number a promise to update two files at every cut, "
        "and gpuwm broke exactly that promise: its wheel shipped 0.1.1 for "
        f"four releases.  Read it from installed metadata instead.  Found: {literals}"
    )
    assert "importlib.metadata" in source or "_distribution_version" in source, (
        "the package must read its version back out of installed metadata"
    )


def test_the_runtime_version_agrees_with_the_declaration(declaration: dict) -> None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    import mpas_port

    try:
        installed = distribution_version(mpas_port.DISTRIBUTION_NAME)
    except PackageNotFoundError:
        assert mpas_port.__version__ == "0+unknown", (
            "with no installed distribution the package must say it does not "
            "know its version, not invent one"
        )
        pytest.skip(
            "gpuwm-hex is not installed in this interpreter (a bare "
            "PYTHONPATH=src checkout); the agreement is proven by the "
            "packaging tier, which installs the built wheel first"
        )
    assert installed == declaration["project"]["version"]
    assert mpas_port.__version__ == installed


# ---------------------------------------------------------------------------
# the front doors
# ---------------------------------------------------------------------------
def test_the_console_script_reaches_a_real_entry_point(declaration: dict) -> None:
    scripts = declaration["project"]["scripts"]
    assert scripts == {"gpuwm-hex": "mpas_port.cli:main"}, (
        "one console script with subcommands, mirroring gpuwm's shape; a "
        f"script per door was not the decision.  Found: {scripts}"
    )

    from mpas_port.cli import main

    assert callable(main)


def test_the_user_facing_names_carry_no_mpas_token(declaration: dict) -> None:
    """MPAS may be a trademark, so the distribution is spelled gpuwm-hex.

    The brand and the spellings a user types never use the MPAS name;
    nominative reference in prose documentation is fine and is used freely in
    README.md and docs/.
    """

    project = declaration["project"]
    assert "mpas" not in project["name"].lower()
    for script in project["scripts"]:
        assert "mpas" not in script.lower(), script

    from mpas_port.cli import build_parser

    parser = build_parser()
    assert parser.prog == "gpuwm-hex"
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        for name in getattr(action, "choices", {}):
            assert "mpas" not in name.lower(), name


def test_the_preferred_environment_spellings_carry_the_distribution_name() -> None:
    """...and the older ones still resolve.

    A rename that quietly stops reading a variable is the worst kind: the
    install line still looks right, the door still runs, and it picks up a
    different binary off PATH.  The legacy names stay in the ladder behind the
    preferred ones, permanently.
    """

    from mpas_port import render_door

    assert render_door.CONVERT_ENV == "GPUWM_HEX_RW_MPAS_CONVERT"
    assert render_door.RENDERER_ENV == "GPUWM_HEX_RW_WRFBATCH"
    for preferred in (render_door.CONVERT_ENV, render_door.RENDERER_ENV):
        assert "MPAS_PORT" not in preferred

    assert render_door.CONVERT_ENV_LEGACY == "MPAS_PORT_RW_MPAS_CONVERT"
    assert render_door.RENDERER_ENV_LEGACY == "MPAS_PORT_RW_WRFBATCH"

    source = (PACKAGE / "render_door.py").read_text(encoding="utf-8")
    for legacy in ("CONVERT_ENV_LEGACY", "RENDERER_ENV_LEGACY"):
        # Twice at least: the constant's definition, and its use in the
        # resolution ladder.  A constant that exists but is never consulted
        # is a promise the code does not keep.
        assert source.count(legacy) >= 3, (
            f"{legacy} is defined but does not appear in the resolution "
            "ladder and the refusal message"
        )


def test_the_import_namespace_blocker_is_still_declared() -> None:
    """The import name still carries the MPAS token, and that is a blocker.

    Renaming ``mpas_port`` rewrites the bytes of modules the full-physics
    proof harness pins by SHA-256 -- eleven of the seventeen pinned files
    contain the literal string in a docstring or comment -- so the rename
    costs a re-proof on a 32 GiB card against the native authority.  Nobody
    editing this tree gets to decide that a docstring edit is numerically
    inert; that is the whole point of the pin.  So the gap is recorded here rather than
    quietly closed, and this test is what makes it impossible to forget.
    """

    import mpas_port

    assert mpas_port.__name__ == "mpas_port"
    assert mpas_port.DISTRIBUTION_NAME == "gpuwm-hex"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "import namespace" in readme.lower(), (
        "README must state that the installed import name still carries the "
        "MPAS token while the distribution does not"
    )


# ---------------------------------------------------------------------------
# the dependency table
# ---------------------------------------------------------------------------
def _module_scope_third_party_imports() -> set[str]:
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, always our own package
                    continue
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                if not top or top in sys.stdlib_module_names or top == "mpas_port":
                    continue
                found.add(top)
    return found


def test_every_module_scope_import_is_a_declared_dependency(
    declaration: dict,
) -> None:
    declared = {
        entry.split(">")[0].split("=")[0].split("[")[0].split("<")[0].strip().lower()
        for entry in declaration["project"]["dependencies"]
    }
    for module in sorted(_module_scope_third_party_imports()):
        distribution = _DISTRIBUTION_OF_MODULE.get(module, module).lower()
        assert distribution in declared, (
            f"{module} is imported at module scope somewhere under "
            f"src/mpas_port -- so the package does not import without it -- "
            f"but {distribution} is not in [project].dependencies.  A "
            "dependency that is real at line one and optional in the table "
            "produces a green install and an ImportError on first use"
        )


def test_the_gpuwm_floor_is_above_the_release_that_cannot_satisfy_the_pin(
    declaration: dict,
) -> None:
    """The engine pin and the pip floor must not tell different stories.

    The port pins gpuwm by the SHA-256 of sixteen individual source files
    (``mpas_port.cuda_arwen_physics_v841.ARWEN_SOURCE_MANIFEST``).  Thirteen
    of those sixteen differ at gpuwm's 2.5.0 stamp and six at its 2.5.1
    stamp, because the GF seam-parity work the port pins -- and the GF frame
    cut on top of it -- landed after those cuts.  A ``>=2.5.0`` floor would
    let pip resolve an install the port then refuses at launch: a green
    install and a dead run.  The floor is a coarse filter, not the wall; the
    wall is the sixteen-file manifest, which no published version satisfies
    on its own, which is why the forecast lane needs a source checkout.

    The floor sits at 2.5.5 for two reasons stacked on that one.  First,
    the seam gap recurred at every cut: at gpuwm's published 2.5.4 stamp,
    3 of the 16 manifest files still differ from the pinned bytes
    (``docs/mpas-seam.md``, ``gpuwm/core/mpas_column_batch.py``,
    ``gpuwm/io/restart.py``), because the seam-convergence work this tree
    pins landed after the 2.5.4 cut; 2.5.5 is the first published version
    whose bytes match the manifest.  Second, the four MPAS bridge binaries
    (``rw_mpas_init``, ``rw_mpas_convert``, ``rw_mpas_mesh``,
    ``rw_mpas_static``) entered gpuwm's bundle at 2.5.3, their rows having
    landed the day after the 2.5.2 upload.  Both front doors drive those
    binaries and the gpuwm source tree never publishes, so a user resolved
    onto 2.5.2 can open neither door and cannot build what is missing.  A
    floor is the only place pip can refuse a stranded install.  gpuwm 2.5.5
    publishes before gpuwm-hex 0.1.1 -- the same hard ordering constraint
    the 0.1.0 plan carried -- so this floor is unsatisfiable only in the
    window before that act, by design.
    """

    pins = [
        entry
        for entry in declaration["project"]["dependencies"]
        if entry.split(">")[0].split("=")[0].split("[")[0].strip() == "gpuwm"
    ]
    assert pins == ["gpuwm>=2.5.5"], (
        "the gpuwm floor must be 2.5.5: the first version that both carries "
        "the seam bytes the Arwen manifest pins (3 of 16 still differ at the "
        "published 2.5.4 stamp) AND bundles the four MPAS bridge binaries "
        "both front doors drive.  A lower floor lets pip resolve an engine "
        f"that strands a door or refuses at launch.  Found: {pins}"
    )

    from mpas_port.cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST

    assert len(ARWEN_SOURCE_MANIFEST) == 16
    assert "docs/mpas-seam.md" in ARWEN_SOURCE_MANIFEST, (
        "the manifest pins a repository document that no wheel places in "
        "site-packages, which is why the forecast lane needs a gpuwm source "
        "checkout on top of the installed distribution.  If this file ever "
        "leaves the manifest, re-check whether an installed gpuwm can satisfy "
        "the pin on its own and update README's engine section"
    )


# ---------------------------------------------------------------------------
# what reaches the wheel
# ---------------------------------------------------------------------------
def test_package_discovery_is_anchored(declaration: dict) -> None:
    find = declaration["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert find["include"] == ["mpas_port", "mpas_port.*"], (
        "anchored patterns, never a loose glob: these are fnmatch over "
        "directory names, and gpuwm measured a loose glob swallowing a whole "
        "sibling distribution into its wheel"
    )


def test_no_package_data_file_lands_outside_the_declaration(
    declaration: dict,
) -> None:
    from fnmatch import fnmatch

    patterns = declaration["tool"]["setuptools"]["package-data"]["mpas_port"]
    stray: list[str] = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_dir() or path.suffix == ".py":
            continue
        if "__pycache__" in path.parts:
            continue  # a build product of the sources beside it, never input
        relative = path.relative_to(PACKAGE).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in patterns):
            stray.append(relative)
    assert stray == [], (
        "these files live under the package but match no package-data "
        f"pattern, so the wheel will not carry them: {stray}"
    )


# ---------------------------------------------------------------------------
# the blockers a public cut has to clear
# ---------------------------------------------------------------------------
def test_the_licence_blocker_is_declared_until_a_licence_exists(
    declaration: dict,
) -> None:
    """No LICENSE file, therefore no licence claim in the metadata.

    This port derives from MPAS-Atmosphere v8.4.1 (NCAR/LANL, BSD-3-Clause)
    with source-line citations pinned by tools/check_mpas_citations.py.  What
    licence a derived work ships under is a release decision.  Until a LICENSE and
    NOTICE exist in this tree, the metadata must stay silent rather than
    guess -- a wrong SPDX expression in published metadata is worse than an
    absent one, because it is a claim.
    """

    has_licence_file = (ROOT / "LICENSE").is_file() or (ROOT / "LICENCE").is_file()
    project = declaration["project"]
    if has_licence_file:
        assert "license" in project, (
            "a LICENSE now exists in the tree, so [project].license must "
            "carry its SPDX expression and [project].license-files must ship "
            "it in the wheel"
        )
        return
    assert "license" not in project and "license-files" not in project, (
        "metadata declares a licence that no file in this tree backs"
    )
