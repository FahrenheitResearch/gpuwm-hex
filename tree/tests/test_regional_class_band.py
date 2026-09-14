"""A same-ladder cull at another point is its class, within the declared band."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.cuda_backend import regional_admission as ra  # noqa: E402


def _key(edge_mm: int, kernel_set: str, dt_ms: int = 5_000) -> ra.RegionalClassKey:
    return ra.RegionalClassKey(
        boundary_zone_width=7, n_vert_levels=55, finest_edge_mm=edge_mm,
        dt_ms=dt_ms, kernel_set_sha256=kernel_set,
    )


def test_the_two_sub_km_classes_declare_the_band_and_nothing_else_does() -> None:
    banded = {cid for cid, row in ra.ADMITTED_CLASSES.items() if row.finest_edge_band_relative > 0.0}
    assert banded == {"graded-869m-dt5-z7", "graded-711m-dt5-z7"}
    for cid in banded:
        assert ra.ADMITTED_CLASSES[cid].finest_edge_band_relative == ra.SAME_LADDER_FINEST_EDGE_BAND
        assert "BAND 2026-09-13" in ra.ADMITTED_CLASSES[cid].basis
    assert ra.SAME_LADDER_FINEST_EDGE_BAND == 0.05


def test_a_relaxation_noise_edge_is_the_class_and_a_far_one_is_not() -> None:
    klass = ra.ADMITTED_CLASSES["graded-869m-dt5-z7"]
    kernel_set = klass.key.kernel_set_sha256
    assert ra.admitted_class_for_key(_key(869_251, kernel_set)).class_id == klass.class_id
    assert ra.admitted_class_for_key(_key(884_000, kernel_set)).class_id == klass.class_id
    assert ra.admitted_class_for_key(_key(855_000, kernel_set)).class_id == klass.class_id
    assert ra.admitted_class_for_key(_key(800_000, kernel_set)) is None
    # An exact class with no band is still exact.
    assert not klass.key.matches(_key(870_000, kernel_set))
    assert klass.key.matches(_key(870_000, kernel_set), band=klass.finest_edge_band_relative)


def test_the_two_bands_never_both_admit_one_edge() -> None:
    fine = ra.ADMITTED_CLASSES["graded-711m-dt5-z7"]
    coarse = ra.ADMITTED_CLASSES["graded-869m-dt5-z7"]
    kernel_set = fine.key.kernel_set_sha256
    for edge_mm in range(600_000, 1_000_000, 1_000):
        key = _key(edge_mm, kernel_set)
        both = (
            fine.key.matches(key, band=fine.finest_edge_band_relative)
            and coarse.key.matches(key, band=coarse.finest_edge_band_relative)
        )
        assert not both, edge_mm


def test_the_band_never_relaxes_the_other_fields_and_the_refusal_names_the_edge() -> None:
    klass = ra.ADMITTED_CLASSES["graded-869m-dt5-z7"]
    kernel_set = klass.key.kernel_set_sha256
    other_dt = _key(869_251, kernel_set, dt_ms=20_000)
    assert not klass.key.matches(other_dt, band=0.5)
    assert klass.key.differing_fields(other_dt, band=0.5) == ["dt_ms"]
    far = _key(700_000, kernel_set)
    text = ra.class_mismatch_refusal("q-row", klass.key, far, band=klass.finest_edge_band_relative)
    assert "finest_edge_mm" in text and "q-row" in text
    assert klass.key.differing_fields(_key(880_000, kernel_set), band=klass.finest_edge_band_relative) == []
    assert klass.as_dict()["finest_edge_band_relative"] == 0.05
