from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from mpas_port.obs_referee.bootstrap import paired_case_interval
from mpas_port.obs_referee.bundle import (
    load_grid_bundle,
    write_grid_bundle,
)
from mpas_port.obs_referee.canonical import sha256_file, write_json
from mpas_port.obs_referee.errors import IntegrityError, SchemaError
from mpas_port.obs_referee.manifest import BASE_COMMIT, SCHEMA, load_manifest
from mpas_port.obs_referee.metrics import calculate_metric
from mpas_port.obs_referee.runner import emit_not_measured, run_suite
from mpas_port.obs_referee.treatment import (
    TREATMENT_RECEIPT_SCHEMA,
    compare_output_trees,
    validate_treatment_receipt,
)


TREE = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = TREE / "verification" / "fixtures" / "build_synthetic_suite.py"
PRODUCTION_MANIFEST = (
    TREE / "verification" / "manifests" / "obs-referee-283.production.json"
)


def test_production_manifest_is_strict_and_pins_base() -> None:
    manifest = load_manifest(PRODUCTION_MANIFEST)
    assert manifest.raw["base_commit"] == BASE_COMMIT
    assert manifest.mode == "production"
    assert {case["role"] for case in manifest.cases} == {
        "known_divergence",
        "independent_control",
        "weak_convection_control",
    }
    pending = [case for case in manifest.cases if case["selection_status"] == "pending"]
    assert len(pending) == 2
    assert all(case["cycle_time"] is None for case in pending)


def test_manifest_refuses_automatic_default(tmp_path: Path) -> None:
    value = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    value["policy"]["allow_automatic_default"] = True
    path = tmp_path / "manifest.json"
    write_json(path, value)
    with pytest.raises(SchemaError, match="forbids automatic"):
        load_manifest(path)


def test_categorical_metrics_exact() -> None:
    forecast = np.asarray([0.0, 2.0, 2.0, 0.0])
    observation = np.asarray([0.0, 2.0, 0.0, 2.0])
    result = calculate_metric(
        forecast,
        observation,
        family="categorical",
        statistic="csi",
        threshold=1.0,
    )
    assert result.status == "MEASURED"
    assert result.components["hits"] == 1
    assert result.components["misses"] == 1
    assert result.components["false_alarms"] == 1
    assert result.value == pytest.approx(1.0 / 3.0)


def test_fss_identity_and_displacement() -> None:
    observation = np.zeros((1, 7, 7), dtype=np.float64)
    observation[0, 2:5, 2:5] = 1.0
    identical = calculate_metric(
        observation,
        observation,
        family="fss",
        statistic="fss",
        threshold=0.5,
        neighborhood_radius_cells=1,
    )
    shifted = calculate_metric(
        np.roll(observation, 2, axis=2),
        observation,
        family="fss",
        statistic="fss",
        threshold=0.5,
        neighborhood_radius_cells=1,
    )
    assert identical.value == pytest.approx(1.0)
    assert shifted.value is not None and shifted.value < 1.0


def test_object_matching_reports_centroid_error() -> None:
    lat, lon = np.meshgrid(
        np.linspace(35.0, 35.5, 6),
        np.linspace(-99.5, -99.0, 6),
        indexing="ij",
    )
    observation = np.zeros((1, 6, 6), dtype=np.float64)
    forecast = np.zeros_like(observation)
    observation[0, 2:4, 2:4] = 40.0
    forecast[0, 2:4, 3:5] = 40.0
    result = calculate_metric(
        forecast,
        observation,
        family="objects",
        statistic="median_centroid_error_km",
        threshold=20.0,
        latitude_deg=lat,
        longitude_deg=lon,
        minimum_object_cells=2,
        maximum_object_match_km=100.0,
    )
    assert result.status == "MEASURED"
    assert result.components["matched_objects"] == 1
    assert result.value is not None and 5.0 < result.value < 20.0


def test_case_block_bootstrap_is_reproducible() -> None:
    candidate = {f"case-{i}": float(i + 1) for i in range(4)}
    reference = {f"case-{i}": float(i) for i in range(4)}
    first = paired_case_interval(
        candidate,
        reference,
        direction="higher",
        replicates=500,
        confidence=0.95,
        minimum_cases=3,
        seed=283,
        seed_context=("metric",),
    )
    second = paired_case_interval(
        candidate,
        reference,
        direction="higher",
        replicates=500,
        confidence=0.95,
        minimum_cases=3,
        seed=283,
        seed_context=("metric",),
    )
    assert first == second
    assert first.estimate == pytest.approx(1.0)
    assert first.lower == pytest.approx(1.0)
    assert first.upper == pytest.approx(1.0)


def test_canonical_npz_is_byte_deterministic(tmp_path: Path) -> None:
    times = np.asarray([0, 3600], dtype=np.int64)
    lat, lon = np.meshgrid([35.0, 36.0], [-99.0, -98.0], indexing="ij")
    fields = {"temperature_k": np.ones((2, 2, 2), dtype=np.float64) * 290.0}
    first, _ = write_grid_bundle(
        tmp_path / "first.npz",
        time_unix_s=times,
        latitude_deg=lat,
        longitude_deg=lon,
        fields=fields,
        producer="synthetic-fixture",
        producer_version="test",
    )
    second, _ = write_grid_bundle(
        tmp_path / "second.npz",
        time_unix_s=times,
        latitude_deg=lat,
        longitude_deg=lon,
        fields=fields,
        producer="synthetic-fixture",
        producer_version="test",
    )
    assert sha256_file(first) == sha256_file(second)


def test_treatment_receipt_is_fail_closed(tmp_path: Path) -> None:
    receipt = {
        "schema": TREATMENT_RECEIPT_SCHEMA,
        "treatment_name": "gf-subsidence-scale",
        "mode": "multiply_tendency",
        "value": 0.75,
        "enabled": True,
        "scope": "gf_subsidence_only",
        "call_count": 1,
        "columns_touched": 10,
        "pre_tendency_sha256": "1" * 64,
        "post_tendency_sha256": "2" * 64,
        "producer_commit": "a" * 40,
        "metadata": {},
    }
    path = tmp_path / "receipt.json"
    write_json(path, receipt)
    validate_treatment_receipt(
        path,
        expected_name="gf-subsidence-scale",
        expected_mode="multiply_tendency",
        expected_value=0.75,
    )
    receipt["scope"] = "all_gf_tendencies"
    write_json(path, receipt)
    with pytest.raises(IntegrityError, match="scope"):
        validate_treatment_receipt(
            path,
            expected_name="gf-subsidence-scale",
            expected_mode="multiply_tendency",
            expected_value=0.75,
        )


def test_disabled_output_identity_gate(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "history.nc").write_bytes(b"same")
    (second / "history.nc").write_bytes(b"same")
    result = compare_output_trees(first, second, include=("*.nc",), exclude=())
    assert result["status"] == "IDENTICAL"
    (second / "history.nc").write_bytes(b"different")
    with pytest.raises(IntegrityError, match="not byte-identical"):
        compare_output_trees(first, second, include=("*.nc",), exclude=())


def test_end_to_end_synthetic_evidence_is_byte_identical(tmp_path: Path) -> None:
    builder = _load_fixture_builder()
    fixture_root = tmp_path / "fixture"
    manifest = builder.build(fixture_root)
    first = tmp_path / "evidence-a"
    second = tmp_path / "evidence-b"
    result_a = run_suite(manifest, first)
    result_b = run_suite(manifest, second)
    assert result_a["scorecard"]["scientific_verdict"] == "SYNTHETIC_ONLY"
    assert result_a["scorecard"]["default_on_decision"] == "DO_NOT_ENABLE"
    assert result_a["run_receipt"]["status"] == "MEASURED"
    names_a = sorted(path.name for path in first.iterdir() if path.is_file())
    names_b = sorted(path.name for path in second.iterdir() if path.is_file())
    assert names_a == names_b
    assert {
        name: sha256_file(first / name) for name in names_a
    } == {
        name: sha256_file(second / name) for name in names_b
    }


def test_not_measured_report_contains_no_scientific_values(tmp_path: Path) -> None:
    output = tmp_path / "not-measured"
    result = emit_not_measured(
        PRODUCTION_MANIFEST,
        output,
        reason="Real MRMS/ASOS and model arm artifacts have not been materialized.",
    )
    assert result["scorecard"]["scientific_verdict"] == "NOT_MEASURED"
    metrics = json.loads((output / "case-metrics.json").read_text(encoding="utf-8"))
    assert metrics
    assert all(item["status"] == "NOT_MEASURED" for item in metrics)
    assert all(item["value"] is None for item in metrics)
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "Scientific status: NOT_MEASURED" in report
    assert "DO NOT ENABLE" in report


def test_production_manifest_cannot_allow_synthetic_producer(tmp_path: Path) -> None:
    value = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    value["producer_allowlist"].append("synthetic-fixture")
    path = tmp_path / "manifest.json"
    write_json(path, value)
    with pytest.raises(SchemaError, match="may not allow synthetic"):
        load_manifest(path)


def _load_fixture_builder():
    spec = importlib.util.spec_from_file_location("obs_referee_fixture_builder", FIXTURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
