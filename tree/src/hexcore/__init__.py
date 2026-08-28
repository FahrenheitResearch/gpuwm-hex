"""GPU-native variable-resolution atmospheric model core.

A CUDA port of the MPAS-Atmosphere v8.4.1 dynamical core, with a NumPy CPU
authority beside it for the operators that are verified against
source-extracted fixtures.

Evidence is component- and configuration-specific; this package does not make
a package-wide numerical-equivalence, production-physics, or GPU claim.
Modules carry their own frozen-source citations and exact refusal boundaries.
"""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from types import MappingProxyType

from .errors import ConfigurationRefusal, EvidenceError, MeshValidationError

__all__ = [
    "COMPONENT_EVIDENCE",
    "DISTRIBUTION_NAME",
    "ConfigurationRefusal",
    "EvidenceError",
    "MeshValidationError",
    "__version__",
]

#: The distribution this package is published as.  It carries no MPAS token,
#: because that name may be a trademark; the import name still does, and that
#: gap is recorded as a pre-public blocker in
#: tests/test_packaging_declaration.py.
DISTRIBUTION_NAME = "gpuwm-hex"

# Read back out of installed metadata rather than restated here.  pyproject's
# [project].version is the single place the number is written; a second
# literal is a promise to update two files at every cut, and gpuwm broke
# exactly that promise -- its wheel shipped 0.1.1 for four releases while a
# refusal message told users which release was speaking.
try:
    __version__ = _distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    # Says so rather than inventing a number.
    __version__ = "0+unknown"

__evidence__ = "component-specific"

# Keep package metadata lightweight: importing ``hexcore`` must not eagerly
# import the full driver, SciPy regridder, or NetCDF output stack.
COMPONENT_EVIDENCE: Mapping[str, str] = MappingProxyType(
    {
        "m1_operators": "source-extracted-fortran-oracle",
        "jw_nomix_whole_step": "frozen-jw-nomix-step-linked",
        "jw_original_mixed_whole_step": "implemented-original-jw-branch",
        "dry_24h": "actual repeated full-step Python-port execution",
        "gfs_initialization": "implemented-unverified",
        "gfs_6h_forecast": (
            "real GFS-initialized repeated full-step Python-port forecast"
        ),
        "history_output": "implemented-unverified",
        "visualization_regrid": "implemented-unverified; non-conservative",
        "rust_renderer_2d": "real unchanged-renderer product-path execution",
        "physics": "seam-only; no production physics",
    }
)
