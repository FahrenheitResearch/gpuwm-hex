"""Complete forecast evidence rejects partial and selective byte comparisons."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS, fingerprint_atmosphere

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_samehost_forecasts_v841 as comparator  # noqa: E402


def _hash(value):
    return {"dtype": value.dtype.str, "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _boundary(step, *, native_garbage=False):
    cells, edges = (5, 9) if native_garbage else (4, 8)
    state = SimpleNamespace(time_seconds=step * 60.)
    saved = SimpleNamespace()
    for name in STATE_FIELDS:
        shape = (6, 2, cells) if name == "scalars" else (3 if name == "rho_w" else 2, edges if name == "rho_u" else cells)
        setattr(state, name, np.full(shape, step + 1, dtype=np.float32))
    for name in SIDECAR_FIELDS:
        shape = (3 if name == "vertical_velocity" else 2, edges if name == "normal_velocity" else cells)
        setattr(saved, name, np.full(shape, step + 2, dtype=np.float32))
    arrays = {"seam/arrays/soil": _hash(np.full((cells, 4), step, dtype=np.float32))}
    for name in sorted(comparator._ARTIFACT_PATHS):
        arrays[name] = _hash(np.full((2, 1, cells + 1), step, dtype=np.float32))
    return {"atmosphere": fingerprint_atmosphere(SimpleNamespace(state=state, saved=saved)),
            "backend": {"arrays": arrays,
                        "scalars": {"identity/n_columns": cells,
                                    "identity/constructor_identity_sha256": "whole",
                                    "seam/scalars/step_index": step}}}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_boundaries(output, records):
    (output / "boundary-fingerprints.jsonl").write_text(
        "".join(json.dumps({"step": step, **record}, sort_keys=True) + "\n" for step, record in records.items()),
        encoding="utf-8")


@pytest.fixture
def forecast_pair(tmp_path):
    outputs, measurements = {}, {}
    for arm in ("reference", "candidate"):
        output = tmp_path / arm
        output.mkdir()
        snapshots = {}
        for step in (0, 2):
            path = output / f"cuda-history.{step:04d}.nc"
            payload = bytes((0, step, 128, 255)) * 17
            path.write_bytes(payload)
            snapshots[str(step)] = {"path": str(path), "bytes": len(payload), "sha256": comparator.file_sha256(path)}
        _write_json(output / comparator.DRIVER_NAME,
                    {"status": "passed", "forecast": {"walls": {"steps": 2},
                                                         "refusal": None, "snapshot_files": snapshots}})
        _write_boundaries(output, {step: _boundary(step) for step in range(3)})
        measurement = tmp_path / f"{arm}.measurement.json"
        forcing = {name: _hash(np.arange(8, dtype=np.float32).reshape(2, 4))
                   for name in comparator._FORCING_FIELDS}
        _write_json(measurement, {"cli_exit_code": 0, "driver_status": "passed", "driver_completed": True,
                                 "run_identity_eligible": True,
                                 "protected_bytes_unchanged": True, "ordinary_door_restored": True,
                                 "exception": None, "final_gf_tendency_hashes": forcing})
        outputs[arm], measurements[arm] = output, measurement
    return {"reference_output": outputs["reference"], "candidate_output": outputs["candidate"],
            "reference_measurement": measurements["reference"], "candidate_measurement": measurements["candidate"],
            "expected_steps": 2}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_ordinary_one_card_against_ordinary_one_card_is_complete(forecast_pair):
    result = comparator.compare_forecasts(**forecast_pair)
    assert result["status"] == "PASS"
    assert result["byte_comparison_passed"] is True
    assert result["failures"] == []
    assert result["evidence"]["boundary_comparison"]["boundaries_compared"] == 3
    assert len(result["evidence"]["reference"]["history_files"]) == 2
    assert set(result["evidence"]["candidate"]["final_gf_tendency_hashes"]) == {"rthdynten", "rqvdynten"}


def test_named_partition_exceptions_are_explicit_and_native_garbage_is_retained(forecast_pair):
    reference = {step: _boundary(step, native_garbage=True) for step in range(3)}
    candidate = deepcopy(reference)
    for record in candidate.values():
        record["batch_order_artifacts"] = {}
        for name in sorted(comparator._ARTIFACT_PATHS):
            fingerprint = record["backend"]["arrays"].pop(name)
            record["batch_order_artifacts"][name] = {"rank0_sha256": fingerprint["sha256"]}
        record["backend"]["scalars"]["identity/n_columns"] = 4
        record["backend"]["scalars"]["identity/constructor_identity_sha256"] = "rank0"
    _write_boundaries(forecast_pair["reference_output"], reference)
    _write_boundaries(forecast_pair["candidate_output"], candidate)
    result = comparator.compare_forecasts(**forecast_pair)
    assert result["status"] == "PASS"
    for exceptions in result["evidence"]["declared_boundary_exceptions"].values():
        assert len(exceptions["excluded_batch_order_artifacts"]) == 3
        assert len(exceptions["metadata_diffs"]) == 2
        assert all(row["expected_divergence"] for row in exceptions["metadata_diffs"])
    assert "complete backend restart identity including the excluded artifacts" in result["nonclaims"]


@pytest.mark.parametrize("field,value", [("driver_completed", False), ("driver_status", "truncated_by_model_refusal"),
                                         ("cli_exit_code", 2), ("protected_bytes_unchanged", False),
                                         ("ordinary_door_restored", False), ("exception", {"message": "refused"})])
def test_incomplete_or_changed_real_door_measurement_fails(forecast_pair, field, value):
    path = forecast_pair["candidate_measurement"]
    measurement = _read_json(path)
    measurement[field] = value
    _write_json(path, measurement)
    assert comparator.compare_forecasts(**forecast_pair)["status"] == "FAIL"


@pytest.mark.parametrize("missing_flag", [False, True])
def test_unqualified_runtime_keeps_byte_evidence_but_cannot_pass(forecast_pair, missing_flag):
    path = forecast_pair["candidate_measurement"]
    measurement = _read_json(path)
    if missing_flag:
        del measurement["run_identity_eligible"]
    else:
        measurement["run_identity_eligible"] = False
    _write_json(path, measurement)
    result = comparator.compare_forecasts(**forecast_pair)
    assert result["status"] == "FAIL"
    assert result["byte_comparison_passed"] is True
    assert any(failure["check"] == "candidate/runtime_qualification" for failure in result["failures"])


@pytest.mark.parametrize("mutation", [None, "missing", "disabled", "unqualified", "not_restored", "worker_called"])
def test_no_gil_runtime_requires_serial_extension_policy(forecast_pair, mutation):
    path = forecast_pair["candidate_measurement"]
    measurement = _read_json(path)
    measurement["source_runtime_inputs_after"] = {"runtime": {"interpreter": {
        "Py_GIL_DISABLED": 1, "gil_enabled": False}}}
    serial_io = {"enabled": True, "qualified": True, "module_entrypoints_restored": True, "worker_refusals": 0}
    measurement["serial_extension_io"] = serial_io
    if mutation == "missing":
        del measurement["serial_extension_io"]
    elif mutation == "disabled":
        serial_io["enabled"] = False
    elif mutation == "unqualified":
        serial_io["qualified"] = False
    elif mutation == "not_restored":
        serial_io["module_entrypoints_restored"] = False
    elif mutation == "worker_called":
        serial_io["worker_refusals"] = 1
    _write_json(path, measurement)
    result = comparator.compare_forecasts(**forecast_pair)
    assert result["status"] == ("PASS" if mutation is None else "FAIL")
    assert result["byte_comparison_passed"] is True


@pytest.mark.parametrize("mutation", ["status", "refusal", "steps", "missing_forecast"])
def test_actual_driver_completion_is_required_independently_of_measurement(forecast_pair, mutation):
    path = forecast_pair["reference_output"] / comparator.DRIVER_NAME
    driver = _read_json(path)
    if mutation == "status":
        driver["status"] = "truncated_by_model_refusal"
    elif mutation == "refusal":
        driver["forecast"]["refusal"] = {"step": 2}
    elif mutation == "steps":
        driver["forecast"]["walls"]["steps"] = 1
    else:
        del driver["forecast"]
    _write_json(path, driver)
    assert comparator.compare_forecasts(**forecast_pair)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["missing", "both_missing_final", "extra", "bytes", "self_consistent_bytes", "missing_descriptor"])
def test_history_inventory_and_actual_full_file_bytes_are_required(forecast_pair, mutation):
    output = forecast_pair["candidate_output"]
    path = output / comparator.DRIVER_NAME
    driver = _read_json(path)
    history = output / "cuda-history.0002.nc"
    if mutation == "missing":
        history.unlink()
    elif mutation == "both_missing_final":
        for arm in ("reference", "candidate"):
            root = forecast_pair[f"{arm}_output"]
            (root / "cuda-history.0002.nc").unlink()
            receipt = _read_json(root / comparator.DRIVER_NAME)
            del receipt["forecast"]["snapshot_files"]["2"]
            _write_json(root / comparator.DRIVER_NAME, receipt)
    elif mutation == "extra":
        (output / "cuda-history.extra.nc").write_bytes(b"extra")
    elif mutation in ("bytes", "self_consistent_bytes"):
        history.write_bytes(b"different history bytes")
        if mutation == "self_consistent_bytes":
            driver["forecast"]["snapshot_files"]["2"].update(bytes=history.stat().st_size,
                                                           sha256=comparator.file_sha256(history))
    else:
        del driver["forecast"]["snapshot_files"]["2"]
    if mutation != "both_missing_final":
        _write_json(path, driver)
    assert comparator.compare_forecasts(**forecast_pair)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["missing", "missing_field", "dtype", "shape", "sha256", "driver_consistency"])
def test_final_next_step_forcing_is_required_beyond_backend_boundary_hashes(forecast_pair, mutation):
    path = forecast_pair["candidate_measurement"]
    measurement = _read_json(path)
    if mutation == "missing":
        del measurement["final_gf_tendency_hashes"]
    elif mutation == "missing_field":
        del measurement["final_gf_tendency_hashes"]["rqvdynten"]
    elif mutation in ("dtype", "shape", "sha256"):
        measurement["final_gf_tendency_hashes"]["rthdynten"][mutation] = {
            "dtype": "<f8", "shape": [4, 2], "sha256": "f" * 64}[mutation]
    else:
        driver_path = forecast_pair["candidate_output"] / comparator.DRIVER_NAME
        driver = _read_json(driver_path)
        driver["forecast"]["final_gf_tendency_hashes"] = deepcopy(measurement["final_gf_tendency_hashes"])
        driver["forecast"]["final_gf_tendency_hashes"]["rqvdynten"]["sha256"] = "f" * 64
        _write_json(driver_path, driver)
    _write_json(path, measurement)
    assert comparator.compare_forecasts(**forecast_pair)["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["missing_middle", "both_missing_middle", "wrong_steps", "clock", "both_drifting_clocks", "metadata", "native_garbage", "unnamed_artifact", "missing_state", "rank_metadata"])
def test_full_boundary_clock_state_and_metadata_coverage_is_required(forecast_pair, mutation):
    records = {step: _boundary(step, native_garbage=True) for step in range(3)}
    _write_boundaries(forecast_pair["reference_output"], records)
    candidate = deepcopy(records)
    if mutation in ("missing_middle", "both_missing_middle"):
        del candidate[1]
        if mutation == "both_missing_middle":
            del records[1]
            _write_boundaries(forecast_pair["reference_output"], records)
    elif mutation == "wrong_steps":
        candidate = {step + 3: value for step, value in candidate.items()}
    elif mutation in ("clock", "both_drifting_clocks"):
        candidate[2]["atmosphere"]["model_time_seconds"] = 121.
        if mutation == "both_drifting_clocks":
            records[2]["atmosphere"]["model_time_seconds"] = 121.
            _write_boundaries(forecast_pair["reference_output"], records)
    elif mutation == "metadata":
        candidate[2]["backend"]["scalars"]["seam/scalars/step_index"] = 3
    elif mutation == "native_garbage":
        candidate[2]["atmosphere"]["state"]["fields"]["rho"]["bytes_sha256"] = "f" * 64
    elif mutation == "unnamed_artifact":
        name = "seam/arrays/soil"
        candidate[2]["backend"]["arrays"].pop(name)
        candidate[2]["batch_order_artifacts"] = {name: {"rank0_sha256": "f" * 64}}
    elif mutation == "missing_state":
        for values in (records, candidate):
            del values[2]["atmosphere"]["state"]["fields"]["rho_theta"]
        _write_boundaries(forecast_pair["reference_output"], records)
    else:
        candidate[2]["rank_metadata_divergences"] = [{"rank": 1, "path": "seam/scalars/step_index", "value": 3}]
    _write_boundaries(forecast_pair["candidate_output"], candidate)
    assert comparator.compare_forecasts(**forecast_pair)["status"] == "FAIL"


@pytest.mark.parametrize("corrupt", [False, True])
def test_cli_writes_a_receipt_and_uses_rc2_for_failure(forecast_pair, tmp_path, capsys, corrupt):
    if corrupt:
        forecast_pair["candidate_measurement"].write_text("{bad JSON", encoding="utf-8")
    receipt = tmp_path / "complete-identity.json"
    argv = []
    for option in ("reference_output", "candidate_output", "reference_measurement", "candidate_measurement", "expected_steps"):
        argv.extend(("--" + option.replace("_", "-"), str(forecast_pair[option])))
    argv.extend(("--receipt", str(receipt)))
    assert comparator.main(argv) == (2 if corrupt else 0)
    result = _read_json(receipt)
    assert result["status"] == ("FAIL" if corrupt else "PASS")
    assert capsys.readouterr().out.startswith(result["status"] + " complete forecast identity:")
