"""Where this tree's package lives, in whichever layout the tests run in.

The tests run in this repository, where the package is ``src/hexcore``, and
in a tree that folds the package in under another import name, where the
package sits at that tree's root under its import name, two levels above
the component folder that holds these tests.  A test that opened
``src/hexcore/...`` by a fixed path failed or errored in the folded layout,
so every test that reads one of the package's own files finds the package
by its import name here instead.

Imported by name (``from _layout import PACKAGE_DIR``), not through
``conftest``: a folded tree has more than one ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import hexcore as _package  # noqa: E402

#: True when these tests run folded in under another import name.
FOLDED = _package.__name__ != "hexcore"
#: Why a test of this repository's own declaration skips when folded: the
#: folded tree carries no gpuwm-hex pyproject.toml and no src/hexcore
#: checkout; the folding tree declares its own distribution at its root.
NOT_CARRIED_FOLDED = (
    "folded in under another import name: this tree carries no gpuwm-hex "
    "pyproject.toml and no src/hexcore checkout (the folding tree declares "
    "its own distribution at its root)"
)
#: The package's own directory in the layout these tests run in.
PACKAGE_DIR = (
    ROOT / "src" / "hexcore"
    if _package.__name__ == "hexcore"
    else ROOT.parents[1].joinpath(*_package.__name__.split("."))
)
DRIVERS = PACKAGE_DIR / "drivers"
RUNNER_PATH = DRIVERS / "run_cuda_v841_full_physics_x4.py"
FORECAST_PATH = DRIVERS / "run_cuda_v841_forecast.py"


def package_path(declared: str) -> Path:
    """A source as this tree's tables name it, as a file in this layout.

    Tables key the package's sources by their repository path,
    ``src/hexcore/<module>``, in both layouts (the name is framed into
    digests, so it keeps its bytes).  Anything else is taken from the tree
    root.
    """

    prefix = "src/hexcore/"
    if declared.startswith(prefix):
        return PACKAGE_DIR.joinpath(*declared[len(prefix):].split("/"))
    return ROOT / declared
