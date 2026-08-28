"""Refusal and marker text computes from the admission surface.

Stale-guard audit #347, finding 9: the bigcard tier's refusal strings in
``tests/conftest.py`` and the marker/comment text in ``pyproject.toml``
restated the retired 26.4 GiB figure (and the superseded 08-24 footprint
row) after the constant itself moved to the computed ~20.6 GiB floor.  A
string that restates a number the code no longer uses is a second copy of
the constant with no test on it; these tests make the strings either
compute from ``X4_FULL_PHYSICS_BYTES`` / the admission surface or name the
surface instead of a number.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3

#: Figures retired by the GF frame cut and the L6 floor re-derivation.
#: None of them may be restated as CURRENT by the admission strings.
RETIRED_FIGURES = (
    "26.4",
    "6,296.5",
    "6296.5",
    "9,948",
    "93,474",
    # Retired 2026-08-26 by the merged-tip re-fit: the 08-25 converged row
    # and the two peaks it was fitted from.
    "4,339.1",
    "103,696",
    "8,390",
    "20,542",
)


def test_conftest_refusals_compute_from_the_admission_surface() -> None:
    source = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    for figure in RETIRED_FIGURES:
        assert figure not in source, (
            f"conftest.py restates the retired figure {figure}; refusal "
            "text must format from X4_FULL_PHYSICS_BYTES"
        )
    # The refusal strings interpolate the computed floor rather than
    # restating any literal GiB figure.
    assert source.count("X4_FULL_PHYSICS_BYTES / 1024**3") >= 2, (
        "both bigcard refusal strings must format the computed floor"
    )


def test_conftest_registered_floor_is_the_surface_value() -> None:
    import conftest as tree_conftest
    from hexcore.device_admission import native_device_floor_bytes

    assert tree_conftest.X4_FULL_PHYSICS_BYTES == native_device_floor_bytes()


def test_pyproject_markers_name_the_surface_not_a_number() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for figure in RETIRED_FIGURES:
        assert figure not in text, (
            f"pyproject.toml restates the retired figure {figure}"
        )
    # The static marker table cannot compute, so it names the surface that
    # does; conftest registers the live figure at collection time.
    assert "device_admission" in text
    # The forecast-door comment quotes the merged-tip row of record.
    assert "5,016.5" in text and "98,748" in text
    assert "8,874" in text
