"""Compile compatibility distinguishes stable inputs from capture coverage."""

from copy import deepcopy
import hashlib

import pytest

from hexcore.cuda_backend.compile_contract import CompileManifest, canonical_sha256
from hexcore.partition_samehost_compile_v841 import SameHostCompileMismatch, compare_samehost_compile_manifests


def _manifest(*, kind="cubin", blob=b"same image", captured=True):
    fingerprint = {
        "nvrtc_build": "13.0.48", "nvrtc_build_id": "CL-test",
        "nvrtc_library_sha256": "a" * 64, "cuda_driver_version": "13000",
        "device_compute_capability": "12.0", "cupy_version": "14.1.0",
        "numpy_version": "2.0.0",
    }
    manifest = CompileManifest({"fingerprint": fingerprint, "sha256": canonical_sha256(fingerprint)})
    source = 'extern "C" __global__ void copy_bytes() {}'
    manifest.declare_module("byte_copy", source=source,
        requested_options=("--std=c++17", "--fmad=false"), module_cache_key="b" * 64)
    manifest.record_kernel("byte_copy", "copy_bytes")
    if captured:
        manifest.record_effective_compile("byte_copy", {
            "status": "resolved", "method": "NVRTC entry capture", "observations": [{
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "effective_flags": ["--std=c++17", "--fmad=false", "-ftz=true"],
                "include_path_count": 2,
                "compiled_image": {"status": "resolved", "kind": kind, "sha256": hashlib.sha256(blob).hexdigest()},
            }],
        })
    else:
        manifest.record_effective_compile("byte_copy", {"status": "unavailable", "reason": "disk-cache hit"})
    return manifest.snapshot()


def test_matching_stable_inputs_accept_cache_hit_without_inventing_binary_proof():
    observed, cached = _manifest(), _manifest(captured=False)
    receipt = compare_samehost_compile_manifests({0: observed, 1: cached})
    assert receipt["inputs_compatible"]
    assert not receipt["captured_image_identity_complete"]
    assert not receipt["captured_cubin_identity_complete"]
    assert not receipt["loaded_kernel_binary_identity_proved"]
    assert receipt["modules"]["byte_copy"]["ranks_without_captured_images"] == [1]
    assert receipt["rank_manifests"]["1"] == cached
    cached["modules"].clear()
    assert receipt["rank_manifests"]["1"]["modules"]


@pytest.mark.parametrize("kind", ["ptx", "cubin"])
def test_mutual_capture_records_image_kind_and_keeps_loaded_binary_unproved(kind):
    receipt = compare_samehost_compile_manifests({0: _manifest(kind=kind), 1: _manifest(kind=kind)})
    assert receipt["captured_image_identity_complete"]
    assert receipt["captured_cubin_identity_complete"] is (kind == "cubin")
    assert not receipt["loaded_kernel_binary_identity_proved"]
    assert receipt["modules"]["byte_copy"]["mutually_compared_images"][0]["kind"] == kind


@pytest.mark.parametrize("field", ["source_sha256", "requested_options", "resolved_kernels"])
def test_actual_stable_input_mismatch_refuses(field):
    left, right = _manifest(captured=False), _manifest(captured=False)
    right["modules"]["byte_copy"][field] = {
        "source_sha256": "c" * 64, "requested_options": ["--fmad=true"], "resolved_kernels": ["different_kernel"],
    }[field]
    with pytest.raises(SameHostCompileMismatch, match="stable compile inputs"):
        compare_samehost_compile_manifests({0: left, 1: right})


def test_measured_platform_mismatch_refuses_even_with_valid_individual_digests():
    left, right = _manifest(), _manifest()
    right["compile_platform"]["fingerprint"]["cuda_driver_version"] = "13010"
    digest = canonical_sha256(right["compile_platform"]["fingerprint"])
    right["compile_platform"]["sha256"] = digest
    right["modules"]["byte_copy"]["compile_platform_fingerprint_sha256"] = digest
    with pytest.raises(SameHostCompileMismatch, match="compiler platform differs"):
        compare_samehost_compile_manifests({0: left, 1: right})


def test_cache_bookkeeping_is_preserved_without_treating_it_as_compile_inputs():
    left, right = _manifest(), _manifest()
    right["modules"]["byte_copy"]["module_cache_key"] = "c" * 64
    receipt = compare_samehost_compile_manifests({0: left, 1: right})
    assert receipt["inputs_compatible"]
    assert not receipt["modules"]["byte_copy"]["cache_keys_identical"]


def test_different_mutual_images_refuse():
    with pytest.raises(SameHostCompileMismatch, match="captured cubin image differs"):
        compare_samehost_compile_manifests({0: _manifest(), 1: _manifest(blob=b"different image")})


def test_different_effective_flags_refuse_even_if_images_were_not_exposed():
    left, right = _manifest(), _manifest()
    for manifest in (left, right):
        manifest["modules"]["byte_copy"]["effective_compile"]["observations"][0]["compiled_image"] = {"status": "unavailable", "reason": "no image"}
    right["modules"]["byte_copy"]["effective_compile"]["observations"][0]["effective_flags"][-1] = "-ftz=false"
    with pytest.raises(SameHostCompileMismatch, match="effective NVRTC flags differ"):
        compare_samehost_compile_manifests({0: left, 1: right})


def test_different_capture_representations_are_explicitly_not_comparable():
    receipt = compare_samehost_compile_manifests({0: _manifest(kind="ptx"), 1: _manifest(kind="cubin")})
    assert not receipt["captured_image_identity_complete"]
    assert receipt["modules"]["byte_copy"]["mutually_compared_images"] == []


def test_partial_three_rank_image_coverage_preserves_available_pair():
    receipt = compare_samehost_compile_manifests({0: _manifest(), 1: _manifest(), 2: _manifest(captured=False)})
    assert not receipt["captured_image_identity_complete"]
    assert len(receipt["modules"]["byte_copy"]["mutually_compared_images"]) == 1
    assert receipt["modules"]["byte_copy"]["ranks_without_captured_images"] == [2]


@pytest.mark.parametrize("defect", ["platform_digest", "module_platform", "observed_source", "contradictory_image", "resolved_without_observations"])
def test_contradictory_or_malformed_evidence_refuses(defect):
    manifest = _manifest()
    entry = manifest["modules"]["byte_copy"]
    evidence = entry["effective_compile"]
    if defect == "platform_digest":
        manifest["compile_platform"]["sha256"] = "c" * 64
    elif defect == "module_platform":
        entry["compile_platform_fingerprint_sha256"] = "c" * 64
    elif defect == "observed_source":
        evidence["observations"][0]["source_sha256"] = "c" * 64
    elif defect == "contradictory_image":
        observation = deepcopy(evidence["observations"][0])
        observation["compiled_image"]["sha256"] = "c" * 64
        evidence["observations"].append(observation)
    else:
        evidence["observations"] = []
    with pytest.raises(SameHostCompileMismatch):
        compare_samehost_compile_manifests({0: manifest, 1: _manifest()})


def test_repeated_observation_counts_and_kernel_order_do_not_change_identity():
    left, right = _manifest(), _manifest()
    for manifest in (left, right):
        manifest["modules"]["byte_copy"]["resolved_kernels"].append("another_kernel")
    right["modules"]["byte_copy"]["resolved_kernels"].reverse()
    evidence = right["modules"]["byte_copy"]["effective_compile"]
    evidence["observations"].append(deepcopy(evidence["observations"][0]))
    assert compare_samehost_compile_manifests({0: left, 1: right})["inputs_compatible"]


@pytest.mark.parametrize("manifests", [{}, {0: {}}, {"0": {}}])
def test_missing_manifests_or_invalid_rank_keys_refuse(manifests):
    with pytest.raises(SameHostCompileMismatch):
        compare_samehost_compile_manifests(manifests)
