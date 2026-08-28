"""Manifest-driven MRMS/ASOS observational referee for gpuwm-hex.

The package intentionally does not parse raw MRMS GRIB2 or raw METAR.  Those
boundaries belong to rustwx.  This package validates canonical, checksummed
bundles and performs deterministic matching, metrics, uncertainty, and
reporting.
"""

from .manifest import BASE_COMMIT, SCHEMA, Manifest, load_manifest
from .runner import emit_not_measured, run_suite

__all__ = (
    "BASE_COMMIT",
    "SCHEMA",
    "Manifest",
    "load_manifest",
    "emit_not_measured",
    "run_suite",
)
