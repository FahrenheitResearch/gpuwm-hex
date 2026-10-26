"""Replay the failing global sizes against the measured card, before drivers load."""
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hexcore import device_admission as memory, forecast_door as door
from hexcore.drivers import samehost_forecast_v841 as samehost
from test_forecast_door import (_namespace, _registry, _proven_architecture,
                               _reference_card, _satisfied_seam_pin)

CARD = memory.CardProfile("measured 96 GB card", 188, 1536)
FREE = 101_398_020_096
TOTAL = 101_982_404_608


def test_card_shape_reads_the_runtime_attribute(monkeypatch):
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.delenv("GPUWM_HEX_NO_LOCAL_GPU", raising=False)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceProperties=lambda device:
            {"name": b"capacity fixture", "multiProcessorCount": 188},
            cudaDevAttrMaxThreadsPerMultiProcessor=39,
            deviceGetAttribute=lambda attribute, device: 2048))))
    # Undo this file's imported fixture only for this device-contact test.
    assert _READ_CARD_PROFILE().max_threads_per_sm == 2048


_READ_CARD_PROFILE = door.read_card_profile


def test_stage_fits_rederive_from_completed_censuses():
    artifact = Path(__file__).resolve().parents[1] / "tools/device_memory_ledger/admission-stage-peaks-20261017.json"
    receipt = json.loads(artifact.read_text())
    model = memory.model_for_card(CARD)
    assert [p["steps"] for p in receipt["points"]] == [60, 360]
    assert len(model.stage_peak_fits) == len(receipt["stage_fits"])
    for fit, evidence in zip(model.stage_peak_fits, receipt["stage_fits"]):
        assert fit.stage == evidence["stage"]
        assert list(fit.peak_bytes) == evidence["measured_peak_bytes"]
        assert fit.bytes_per_cell == evidence["bytes_per_cell"]
        for cells, peak in zip(fit.cells, fit.peak_bytes):
            assert fit.predict_bytes(cells) == pytest.approx(peak, abs=1)
            assert model.predict_bytes(cells) >= peak - 1
    assert model.margin_terms()["arena_placement"]["bytes"] == receipt["arena_placement"]["bytes"]
    armed = memory.model_for_card(CARD, rollback_armed=True)
    for cells in (40_962, 1_216_922, 1_631_612):
        assert armed.predict_bytes(cells) - model.predict_bytes(cells) == pytest.approx(
            cells * memory.ROLLBACK_ARMED_BYTES_PER_CELL)
    assert not memory.model_for_card(memory.CardProfile("different card", 188, 2048)).stage_peak_fits
    assert not memory.model_for_card(CARD, levels=40).stage_peak_fits


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("cells,admitted", [(1_631_612, False), (1_472_532, False), (1_216_922, True)])
def test_exact_global_cases_decide_before_driver_loading(tmp_path, monkeypatch, cells, admitted, resident):
    from hexcore import init_physics, kernel_prewarm
    cp = SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda device: nullcontext(),
        runtime=SimpleNamespace(getDeviceCount=lambda: 1, memGetInfo=lambda: (FREE, TOTAL),
            cudaDevAttrMaxThreadsPerMultiProcessor=39,
            deviceGetAttribute=lambda attribute, device: 1536,
            getDeviceProperties=lambda device: {"major": 12, "minor": 0,
                "name": b"measured 96 GB card", "multiProcessorCount": 188})))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    for variable in ("GPUWM_NO_LOCAL_GPU", "GPUWM_HEX_NO_LOCAL_GPU"):
        monkeypatch.delenv(variable, raising=False)
    # The ArWen route's admission (MPAS physics is the hex default since 0.4.6, so the ArWen suite is named).
    args = _namespace(tmp_path, samehost_devices="0" if resident else None, physics_suite="arwen_mpas_column")
    request = replace(door.resolve_request(args, registry=_registry()), cells=cells)
    monkeypatch.setattr(door, "resolve_request", lambda arguments: request)
    monkeypatch.setattr(door, "load_registry", lambda repo: _registry())
    monkeypatch.setattr(door, "read_card_profile", lambda: CARD)
    monkeypatch.setattr(door, "measure_device_memory", lambda: (FREE, TOTAL))
    options = {"devices": (0,), "layouts": [SimpleNamespace(n_local_cells=cells)]}
    monkeypatch.setattr(samehost, "request_options", lambda *a: options)
    monkeypatch.setattr(samehost, "automatic_options", lambda *a: None)
    monkeypatch.setattr(init_physics, "prepare_forecast_init", lambda init, *a: (init, None))
    monkeypatch.setattr(kernel_prewarm, "install_recorder", lambda: None)
    monkeypatch.setattr(kernel_prewarm, "start_replay", lambda **kw: None)
    loaded = []

    class DriverReached(Exception):
        pass

    def load(repo):
        loaded.append(True)
        raise DriverReached()

    monkeypatch.setattr(door, "_load_drivers", load)
    if admitted:
        with pytest.raises(DriverReached):
            door.run_forecast(args)
        assert loaded == [True]
    else:
        with pytest.raises((door.ForecastDoorRefusal, ValueError)) as caught:
            door.run_forecast(args)
        assert loaded == []
        message = str(caught.value)
        assert "short" in message and "diagnostics" in message and "CuPy" in message
        assert f"{cells:,}" in message


def test_co_tenancy_charges_the_same_updated_solo_floor():
    model = memory.model_for_card(CARD)
    cells = 1_216_922
    solo = model.required_bytes(cells)
    record = memory.admit_co_tenancy(mesh_kind="global", cells=cells, levels=55,
        members=1, card_bytes=TOTAL, free_bytes=FREE, solo_required_bytes=solo)
    assert record["required_free_bytes"] == solo
    assert record["memory_limited_members"] == 1
    with pytest.raises(MemoryError, match="memory limit 1"):
        memory.admit_co_tenancy(mesh_kind="global", cells=cells, levels=55,
            members=2, card_bytes=TOTAL, free_bytes=FREE, solo_required_bytes=solo)
    small = model.required_bytes(163_842)
    assert memory.admit_co_tenancy(mesh_kind="global", cells=163_842, levels=55,
        members=4, card_bytes=TOTAL, free_bytes=FREE, solo_required_bytes=small)["required_free_bytes"] == 4 * small
    medium = model.required_bytes(655_362)
    record = memory.admit_co_tenancy(mesh_kind="global", cells=655_362, levels=55,
        members=1, card_bytes=TOTAL, free_bytes=FREE, solo_required_bytes=medium)
    assert record["table_maximum_members"] == 4
    assert record["maximum_members"] == min(4, FREE // medium)
