"""The forecast receipt names the engine the run's seam bytes matched.

THE BREAKAGE THIS PREVENTS: through 0.3.1 the receipt's top-level
``arwen_commit`` and glacier digest restated the x4 proof's own sealed
constants, so every forecast receipt named an engine commit older than the
one the run executed.  ``measured_engine_identity`` reads them from the
identity the seam bytes match, the one the adapter itself records.
"""

from __future__ import annotations

from pathlib import Path

from hexcore import engine_pin
from hexcore.drivers import run_cuda_v841_forecast as forecast
from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof


def _inspection(root: Path, **fields) -> engine_pin.SeamInspection:
    manifest = engine_pin.seam_manifest()
    defaults = {
        "root": root,
        "matched": tuple(sorted(manifest)),
        "moved": (),
        "absent": (),
        "engine": engine_pin.wanted_version(),
        "declared": engine_pin.wanted_version(),
    }
    defaults.update(fields)
    return engine_pin.SeamInspection(**defaults)


def test_a_pinned_engine_is_named_by_the_pin_not_by_the_proof(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(engine_pin, "inspect_seam", lambda root: _inspection(root))
    pinned = engine_pin.pinned_engine()

    record = forecast.measured_engine_identity(tmp_path)

    assert record == {
        "arwen_commit": pinned.build_commit,
        "arwen_engine_version": pinned.version,
        "arwen_glacier_composed_tu_sha256": pinned.glacier_composed_tu_sha256,
    }
    assert record["arwen_commit"] != proof.ARWEN_COMMIT


def test_bytes_that_match_no_engine_name_none(monkeypatch, tmp_path):
    manifest = sorted(engine_pin.seam_manifest())
    monkeypatch.setattr(
        engine_pin,
        "inspect_seam",
        lambda root: _inspection(
            root, matched=tuple(manifest[1:]), moved=(manifest[0],), engine=None
        ),
    )

    record = forecast.measured_engine_identity(tmp_path)

    assert set(record.values()) == {None}


def test_the_receipt_no_longer_restates_the_proof_constants():
    source = Path(forecast.__file__).read_text(encoding="utf-8")
    assert '"arwen_commit": proof.ARWEN_COMMIT' not in source
    assert "proof.ARWEN_GLACIER_COMPOSED_TU_SHA256" not in source
    assert "**measured_engine_identity(arwen_checkout)" in source
