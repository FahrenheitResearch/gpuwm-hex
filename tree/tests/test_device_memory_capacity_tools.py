"""CPU-only tests for the capacity measurement and claim-boundary tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "device_memory_capacity"


def _load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"capacity_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(module, **overrides):
    values = {
        "variant": "candidate",
        "label": "sample",
        "cells": 100,
        "process_peak_bytes": 2_000,
        "success": True,
        "state_sha256": "a" * 64,
        "isolated_card": True,
        "source_commit": "c" * 40,
    }
    values.update(overrides)
    return module.Sample(**values)



def test_process_probe_parses_rows_and_environment_fail_closed(monkeypatch) -> None:
    module = _load("process_memory_probe")
    assert module._parse_env(("A=1", "B=two=three")) == {
        "A": "1",
        "B": "two=three",
    }
    with pytest.raises(module.ProbeRefusal, match="NAME=VALUE"):
        module._parse_env(("BROKEN",))
    with pytest.raises(module.ProbeRefusal, match="name is empty"):
        module._parse_env(("=value",))

    completed = SimpleNamespace(
        stdout="123, GPU-a, 42\n123, GPU-b, 8\n",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    rows = module._nvidia_rows()
    assert [entry["used_bytes"] for entry in rows[123]] == [42 * module.MIB, 8 * module.MIB]


def test_sample_mapping_refuses_inconsistent_pool_receipt() -> None:
    module = _load("capacity_model")
    with pytest.raises(module.ReceiptError, match="pool live peak exceeds pool total peak"):
        module.Sample.from_mapping(
            {
                "variant": "candidate",
                "label": "bad-pool",
                "cells": 100,
                "process_peak_bytes": 2_000,
                "success": True,
                "pool_live_peak_bytes": 2_000,
                "pool_total_peak_bytes": 1_000,
            }
        )

def test_affine_model_uses_two_distinct_mesh_widths() -> None:
    module = _load("capacity_model")
    samples = (
        _sample(module, label="small", cells=100, process_peak_bytes=2_000),
        _sample(module, label="large", cells=200, process_peak_bytes=3_000),
    )
    model = module.fit_affine_model(samples)
    assert model.fixed_bytes == pytest.approx(1_000.0)
    assert model.bytes_per_cell == pytest.approx(10.0)
    assert model.r_squared == pytest.approx(1.0)
    assert model.predict_bytes(300) == pytest.approx(4_000.0)
    assert model.max_cells(5_000, 1_000) == 300

    with pytest.raises(module.ReceiptError, match="two distinct cell counts"):
        module.fit_affine_model(
            (
                _sample(module, label="same-1", cells=100),
                _sample(module, label="same-2", cells=100),
            )
        )


def test_determinism_distinguishes_pass_fail_and_not_measured() -> None:
    module = _load("capacity_model")
    one = (_sample(module, label="one", cells=100),)
    assert module.determinism_verdict(one, cells=100).status == "NOT MEASURED"

    same = (
        _sample(module, label="same-1", cells=100, state_sha256="1" * 64),
        _sample(module, label="same-2", cells=100, state_sha256="1" * 64),
    )
    assert module.determinism_verdict(same, cells=100).status == "PASS"

    different = (
        _sample(module, label="diff-1", cells=100, state_sha256="1" * 64),
        _sample(module, label="diff-2", cells=100, state_sha256="2" * 64),
    )
    assert module.determinism_verdict(different, cells=100).status == "FAIL"


def test_twelve_gib_gate_cannot_pass_from_projection_or_pool_limit() -> None:
    module = _load("capacity_model")
    target = 163_842
    projection_only = (
        _sample(module, label="x1-a", cells=40_962),
        _sample(module, label="x1-b", cells=40_962),
    )
    assert module.twelve_gib_gate(projection_only, target_cells=target).status == (
        "NOT MEASURED"
    )

    pool_limit_only = tuple(
        _sample(
            module,
            label=f"pool-{index}",
            cells=target,
            process_peak_bytes=11 * module.GIB,
            state_sha256="3" * 64,
            physical_device_total_bytes=32 * module.GIB,
            effective_device_limit_bytes=12 * module.GIB,
            whole_device_limit_enforced=True,
            limit_includes_non_pool=False,
            limit_includes_local_backing_store=False,
        )
        for index in (1, 2)
    )
    pool_verdict = module.twelve_gib_gate(pool_limit_only, target_cells=target)
    assert pool_verdict.status == "NOT MEASURED"
    assert "non-pool" in pool_verdict.detail


def test_twelve_gib_gate_requires_dual_identical_real_runs() -> None:
    module = _load("capacity_model")
    target = 163_842
    qualifying = tuple(
        _sample(
            module,
            label=f"real12-{index}",
            cells=target,
            process_peak_bytes=11 * module.GIB,
            state_sha256="4" * 64,
            physical_device_total_bytes=12 * module.GIB,
        )
        for index in (1, 2)
    )
    verdict = module.twelve_gib_gate(qualifying, target_cells=target)
    assert verdict.status == "PASS"
    assert verdict.qualifying_labels == ("real12-1", "real12-2")


def test_twelve_gib_gate_requires_declared_headroom() -> None:
    module = _load("capacity_model")
    target = 163_842
    too_close = tuple(
        _sample(
            module,
            label=f"tight-{index}",
            cells=target,
            process_peak_bytes=int(11.75 * module.GIB),
            state_sha256="5" * 64,
            physical_device_total_bytes=12 * module.GIB,
        )
        for index in (1, 2)
    )
    verdict = module.twelve_gib_gate(
        too_close,
        target_cells=target,
        headroom_bytes=512 * module.MIB,
    )
    assert verdict.status == "NOT MEASURED"
    assert "required headroom" in verdict.detail


def test_copy_elision_accounting_matches_x4_geometry() -> None:
    module = _load("copy_elision_accounting")
    geometry = module.Geometry(
        cells=163_842,
        edges=491_520,
        vertical_levels=55,
        scalars=6,
        dtype_bytes=4,
    )
    events = {event.name: event for event in module.allocation_events(geometry)}
    assert events["saved_state_scalar_copy"].bytes == 216_271_440
    assert events["candidate_scalar_copy"].bytes == 216_271_440
    assert events["discarded_recovery_pressure_fields"].bytes == 216_271_440
    assert events["cached_tangential_velocity_copy"].bytes == 108_134_400
    assert events["rk1_current_state_and_diagnostics_copy"].bytes == 758_258_136
    assert sum(event.bytes for event in events.values()) == 1_515_206_856


def test_lifetime_contract_refuses_concurrent_writable_alias() -> None:
    module = _load("runtime_ledger")
    read_left = module.LifetimeContract("left", 0, 1, "read", 1000, 100)
    read_right = module.LifetimeContract("right", 0, 1, "read", 1000, 100)
    module.assert_lifetime_contracts((read_left, read_right))

    writer = module.LifetimeContract("writer", 0, 1, "write", 1000, 100)
    with pytest.raises(module.LedgerRefusal, match="concurrent writable"):
        module.assert_lifetime_contracts((read_left, writer))

    later_writer = module.LifetimeContract("later", 2, 3, "write", 1000, 100)
    module.assert_lifetime_contracts((read_left, later_writer))


def test_lifetime_contract_requires_both_sides_to_authorize_writable_alias() -> None:
    module = _load("runtime_ledger")
    left = module.LifetimeContract(
        "left", 0, 1, "write", 1000, 100,
        alias_group="arena-slot", allow_writable_overlap=True,
    )
    right = module.LifetimeContract(
        "right", 0, 1, "write", 1000, 100,
        alias_group="arena-slot", allow_writable_overlap=True,
    )
    module.assert_lifetime_contracts((left, right))

    unapproved = module.LifetimeContract(
        "unapproved", 0, 1, "write", 1000, 100,
        alias_group="arena-slot", allow_writable_overlap=False,
    )
    with pytest.raises(module.LedgerRefusal):
        module.assert_lifetime_contracts((left, unapproved))


def test_array_contract_refuses_wrong_shape_dtype_and_layout() -> None:
    module = _load("runtime_ledger")
    contract = module.ArrayContract("arena-field", (2, 3), "float32")
    valid = SimpleNamespace(
        shape=(2, 3), dtype="float32", flags=SimpleNamespace(c_contiguous=True)
    )
    contract.validate_value(valid)

    wrong_shape = SimpleNamespace(
        shape=(3, 2), dtype="float32", flags=SimpleNamespace(c_contiguous=True)
    )
    with pytest.raises(module.LedgerRefusal, match="wrong shape"):
        contract.validate_value(wrong_shape)

    wrong_dtype = SimpleNamespace(
        shape=(2, 3), dtype="float64", flags=SimpleNamespace(c_contiguous=True)
    )
    with pytest.raises(module.LedgerRefusal, match="wrong dtype"):
        contract.validate_value(wrong_dtype)

    strided = SimpleNamespace(
        shape=(2, 3), dtype="float32", flags=SimpleNamespace(c_contiguous=False)
    )
    with pytest.raises(module.LedgerRefusal, match="C-contiguous"):
        contract.validate_value(strided)


def test_generation_token_refuses_stale_workspace_view() -> None:
    module = _load("runtime_ledger")
    token = module.LeaseToken("horizontal-scratch", 4)
    token.validate_current(4)
    with pytest.raises(module.LedgerRefusal, match="stale workspace view"):
        token.validate_current(5)


def test_parking_refuses_a_live_shared_allocation() -> None:
    module = _load("runtime_ledger")
    owner = module.LifetimeContract("owner", 0, 4, "readwrite", 1000, 400)
    alias = module.LifetimeContract("alias", 1, 3, "read", 1100, 100)
    with pytest.raises(module.LedgerRefusal, match="parking a shared allocation"):
        module.assert_parking_safe(owner, (alias,), phase=2)

    retired = module.LifetimeContract("retired", 0, 1, "read", 1100, 100)
    module.assert_parking_safe(owner, (retired,), phase=2)
