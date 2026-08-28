"""Per-architecture execution admission: the opened pin stays a gate.

The sm_120 execution pin protected one thing: every numerical proof the
port owns was minted on sm_120, so an output from any other architecture
could not be verified against anything.  Opening the pin replaces a flat
refusal with per-architecture admission — an architecture below the proven
floor runs only on its own anchor (contract receipt + frozen-authority
set), every unanchored architecture is still refused by name, and the
sm_120 path is unchanged.

These are the first tests of that gate; before this lane the refusal had
no coverage at all.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from hexcore.cuda_backend import arch_admission
from hexcore.cuda_backend.arch_admission import (
    ADMITTED_BELOW_FLOOR,
    ArchAnchor,
    PROVEN_COMPUTE,
    below_floor_refusal,
)

TREE_ROOT = Path(__file__).resolve().parent.parent


def _fake_cupy(monkeypatch, *, major, minor, name="Fake GPU"):
    """Install a minimal fake cupy stack so require_cuda runs CPU-only."""

    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDeviceProperties=lambda device_id: {
            "major": major,
            "minor": minor,
            "name": name,
            "totalGlobalMem": 10_240 * 1024 * 1024,
            "multiProcessorCount": 68,
        },
        runtimeGetVersion=lambda: 13_000,
        driverGetVersion=lambda: 13_030,
    )

    class _Device:
        def __init__(self, device_id):
            self.device_id = device_id

        def use(self):
            return None

    nvrtc = types.ModuleType("cupy.cuda.nvrtc")
    nvrtc.getSupportedArchs = lambda: (75, 80, 86, 89, 90, 100, 120, 121)
    nvrtc.getVersion = lambda: (13, 0)

    cuda = types.ModuleType("cupy.cuda")
    cuda.runtime = runtime
    cuda.Device = _Device
    cuda.nvrtc = nvrtc

    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    cupy.__version__ = "fake-for-admission-tests"

    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.nvrtc", nvrtc)


def _anchor_sm86(tmp_path: Path) -> ArchAnchor:
    return ArchAnchor(
        compute=(8, 6),
        card="RTX 3080 (test double)",
        admitted_on="2026-08-25",
        contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
        authority_anchor="evidence/sm86-tier-20260825/authority",
        basis="test-registered anchor",
    )


# --- the refusal itself -------------------------------------------------


def test_below_floor_refusal_names_architecture_and_breakage():
    text = below_floor_refusal((7, 5), (12, 0))
    assert "cuda.compute_capability=7.5" in text
    assert "sm_75" in text
    assert "12.0" in text
    # The gate names the concrete breakage it prevents.
    assert "no numerical-contract receipt or frozen-authority set" in text
    assert "could be verified" in text


def test_refusal_reports_the_admitted_roster(monkeypatch):
    assert "anchored below the floor: none" in below_floor_refusal(
        (7, 5), (12, 0)
    ) or arch_admission.ADMITTED_BELOW_FLOOR
    anchor = _anchor_sm86(Path("."))
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): anchor}
    )
    assert "sm_86" in below_floor_refusal((7, 5), (12, 0))


# --- require_cuda behaviour on each class of architecture ----------------


def test_unanchored_below_floor_architecture_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    _fake_cupy(monkeypatch, major=8, minor=6)
    from hexcore.cuda_backend.runtime import CudaRefusal, require_cuda

    with pytest.raises(CudaRefusal) as caught:
        require_cuda(
            min_compute=(12, 0),
            required_compute=(12, 0),
            cache_dir=os.environ.get("TMP", "."),
        )
    message = str(caught.value)
    assert "sm_86" in message
    assert "no per-architecture anchor" in message


def test_other_below_floor_architectures_stay_refused_when_sm86_is_anchored(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): _anchor_sm86(tmp_path)}
    )
    _fake_cupy(monkeypatch, major=7, minor=5)
    from hexcore.cuda_backend.runtime import CudaRefusal, require_cuda

    with pytest.raises(CudaRefusal) as caught:
        require_cuda(min_compute=(12, 0), cache_dir=str(tmp_path))
    message = str(caught.value)
    assert "sm_75" in message
    assert "sm_86" in message  # the roster names what IS anchored


def test_anchored_architecture_is_admitted_at_min_and_required_sites(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): _anchor_sm86(tmp_path)}
    )
    _fake_cupy(monkeypatch, major=8, minor=6, name="NVIDIA GeForce RTX 3080")
    from hexcore.cuda_backend.runtime import require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert capability.compute == (8, 6)
    assert capability.sm == "sm_86"
    assert capability.name == "NVIDIA GeForce RTX 3080"


def test_sm120_path_is_unchanged(monkeypatch, tmp_path):
    consulted = []
    monkeypatch.setattr(
        arch_admission,
        "ADMITTED_BELOW_FLOOR",
        types.MappingProxyType({}) if False else {},
    )

    def _spy(compute):
        consulted.append(compute)
        return None

    monkeypatch.setattr(arch_admission, "admitted_architecture", _spy)
    _fake_cupy(monkeypatch, major=12, minor=0)
    from hexcore.cuda_backend import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "admitted_architecture", _spy)
    capability = runtime_module.require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
    )
    assert capability.compute == (12, 0)
    # At or above the floor the registry is never consulted.
    assert consulted == []


def test_above_floor_mismatch_keeps_the_original_refusal(monkeypatch, tmp_path):
    _fake_cupy(monkeypatch, major=12, minor=1)
    from hexcore.cuda_backend.runtime import CudaRefusal, require_cuda

    with pytest.raises(CudaRefusal) as caught:
        require_cuda(
            min_compute=(12, 0), required_compute=(12, 0), cache_dir=str(tmp_path)
        )
    assert "cuda.compute_capability=12.1 != required 12.0" in str(caught.value)


# --- the contraction pin travels with every architecture -----------------


def test_kernel_cache_contraction_pin_is_architecture_independent():
    from hexcore.cuda_backend.runtime import KernelCache

    # The NVRTC contraction pin is part of the numerical contract on every
    # admitted architecture, not an sm_120 accident.
    defaults = KernelCache.__init__.__kwdefaults__["base_options"]
    assert tuple(defaults) == ("--std=c++17", "--fmad=false")


# --- registry entries are records, not switches --------------------------


def test_every_registered_anchor_carries_its_evidence_in_tree(receipts):
    for compute, anchor in ADMITTED_BELOW_FLOOR.items():
        assert tuple(compute) == tuple(anchor.compute)
        assert tuple(anchor.compute) < PROVEN_COMPUTE, (
            "anchors exist only below the proven floor"
        )
        receipt = TREE_ROOT / anchor.contract_receipt
        authority = TREE_ROOT / anchor.authority_anchor
        assert receipt.exists(), (
            f"{anchor.sm} names a contract receipt that is not in the tree: "
            f"{anchor.contract_receipt}"
        )
        assert authority.exists(), (
            f"{anchor.sm} names an authority anchor that is not in the tree: "
            f"{anchor.authority_anchor}"
        )
        assert anchor.card and anchor.admitted_on and anchor.basis


# --- the real card, when one is present ----------------------------------


def test_real_device_admission_state_is_truthful():
    cupy = pytest.importorskip("cupy")
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        pytest.skip("no CUDA device")
    if count < 1:
        pytest.skip("no CUDA device")
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    compute = (int(properties["major"]), int(properties["minor"]))

    from hexcore.cuda_backend.runtime import CudaRefusal, require_cuda

    if compute >= PROVEN_COMPUTE or arch_admission.admitted_architecture(
        compute
    ):
        capability = require_cuda(min_compute=(12, 0))
        assert capability.compute == compute
    else:
        with pytest.raises(CudaRefusal) as caught:
            require_cuda(min_compute=(12, 0), required_compute=(12, 0))
        message = str(caught.value)
        assert f"sm_{compute[0]}{compute[1]}" in message
        assert "no per-architecture anchor" in message


def test_json_receipt_paths_are_relative(tmp_path):
    anchor = _anchor_sm86(tmp_path)
    record = anchor.as_dict()
    assert record["sm"] == "sm_86"
    assert not Path(record["contract_receipt"]).is_absolute()
    json.dumps(record)  # a receipt row must serialize as-is


# ---------------------------------------------------------------------------
# the FTZ guard-cost timing ceiling is per-architecture (audit #347, finding 8)
# ---------------------------------------------------------------------------
def test_performance_ceiling_registry_carries_both_admitted_architectures():
    """The single global 1.25x ceiling was calibrated when sm_120 was the
    only architecture; on the admitted sm_86 tier the guard cost MEASURES
    1.47-1.57x at transport_edge_values while bitwise identity holds, so a
    global 1.25 hard-refuses a tier the port admits.  Each architecture
    carries its own measured/recorded row."""

    registry = arch_admission.PERFORMANCE_RATIO_CEILINGS
    assert arch_admission.performance_ratio_ceiling("sm_120") == 1.25
    assert arch_admission.performance_ratio_ceiling("sm_86") == 1.75
    sm86 = registry["sm_86"]
    # The row cites the recorded deviation it is set from, not taste.
    assert "transport_edge_values" in sm86["basis"]
    assert "1.471975" in sm86["basis"] and "1.565028" in sm86["basis"]
    assert "perf-control-stability-sm86" in sm86["basis"]
    assert "follow-up" in sm86["basis"], (
        "the sm_86 ceiling is set from the recorded band, not a fresh "
        "calibration; the calibration run must stay named"
    )
    for row in registry.values():
        assert row["basis"], "a ceiling with no basis is an asserted constant"


def test_unregistered_architecture_ceiling_refuses_by_name():
    with pytest.raises(LookupError) as caught:
        arch_admission.performance_ratio_ceiling("sm_99")
    message = str(caught.value)
    assert "sm_99" in message
    assert "sm_120" in message and "sm_86" in message, (
        "the refusal must name the registered roster"
    )


def test_cuda_ftz_constant_is_the_sm120_registry_row():
    from hexcore import cuda_ftz

    assert cuda_ftz.PERFORMANCE_RATIO_CEILING == (
        arch_admission.performance_ratio_ceiling("sm_120")
    )


def test_binding_claim_bytes_are_stable_for_sm120():
    """Saved sm_120 bindings are validated by canonical re-hash: the claim
    string rebuilt for capability 120 must reproduce the pre-change bytes
    exactly, or every archived receipt goes red on a text edit."""

    from hexcore import cuda_ftz

    claim = cuda_ftz._mpas_ftz_claim("120", 1.25)
    assert "declared 1.25x median ceiling" in claim
    assert claim == (
        "The five MPAS RawModule translation units execute under the same "
        "measured terminal -ftz=true route for which gpuwm's "
        "sm_120 probe "
        "observes FP32 DAZ/FTZ. The production transport deck verifies "
        "the guarded subnormal-only FP64 fallback at all 12 transport "
        "kernels and 44 answer-changing non-transport arithmetic classes. "
        "Eight copy/invariant/native-FP64 classes stay green, all 44 "
        "disabled-fallback controls go red. Five named representative "
        "normalized-kernel microbenchmarks remain bitwise identical and "
        "each stays below the declared 1.25x median ceiling; that timing "
        "ceiling is not a whole-step or all-guarded-kernel claim."
    )
