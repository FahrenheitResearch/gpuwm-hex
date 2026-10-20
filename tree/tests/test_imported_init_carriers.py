"""Imported initialized carriers retain their values and species order."""
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.drivers import run_cuda_v841_forecast as forecast

NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")


def test_full_model_species_are_selected_in_model_order() -> None:
    variables = {name: object() for name in reversed(NAMES)}
    variables["ni"] = object()
    assert forecast.select_initial_scalar_names(variables, NAMES) == NAMES
    assert forecast.select_initial_scalar_names(dict.fromkeys(NAMES[:3]), NAMES) == NAMES[:3]


def test_partial_model_species_refuse_to_discard_condensate() -> None:
    with pytest.raises(ValueError, match="discard carried condensate"):
        forecast.select_initial_scalar_names(dict.fromkeys(NAMES[:4]), NAMES)


def test_complete_nonzero_scalar_planes_keep_identity_and_hashes() -> None:
    values = np.arange(36, dtype=np.float32).reshape(6, 3, 2)
    before = values.tobytes()
    state = SimpleNamespace(scalars=values)
    receipt = forecast.prepare_initial_scalar_block(
        state, source_names=NAMES, model_names=NAMES, negative_qv={"negative_count": 0},
    )
    assert state.scalars is values
    assert state.scalars.tobytes() == before
    assert receipt["cold_zero_planes_added"] == 0
    assert receipt["source_present"] == list(NAMES)
    assert receipt["source_hashes"]["qi"] == forecast.proof.array_sha256(values[3])


def test_three_source_species_use_the_existing_cold_augmentation(monkeypatch) -> None:
    marker = {"existing_cold_path": True}
    state = SimpleNamespace(scalars=object())
    monkeypatch.setattr(forecast.proof, "augment_exact_wsm6_scalars", lambda value: marker)
    assert forecast.prepare_initial_scalar_block(
        state, source_names=NAMES[:3], model_names=NAMES, negative_qv={},
    ) is marker


def test_supplied_nonzero_legacy_deformation_is_preserved() -> None:
    a = np.full((2, 6), 2.5, dtype=np.float32)
    b = np.full((2, 6), -0.25, dtype=np.float32)
    mesh = SimpleNamespace(arrays={"defc_a": a, "defc_b": b})
    receipt = forecast.prepare_inactive_deformation(mesh)
    assert mesh.arrays["defc_a"] is a
    assert mesh.arrays["defc_b"] is b
    assert receipt["active_mixing_claim"] is False
    assert receipt["defc_a_sha256"] == forecast.proof.array_sha256(a)
    with pytest.raises(ValueError, match="requires both"):
        forecast.prepare_inactive_deformation(SimpleNamespace(arrays={"defc_a": a}))


def test_absent_deformation_uses_the_existing_placeholder_path(monkeypatch) -> None:
    marker = {"existing_zero_path": True}
    monkeypatch.setattr(forecast.proof, "attach_inactive_zero_deformation", lambda mesh: marker)
    assert forecast.prepare_inactive_deformation(SimpleNamespace(arrays={})) is marker
