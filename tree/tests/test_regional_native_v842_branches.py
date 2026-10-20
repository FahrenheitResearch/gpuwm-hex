"""Additional branch decks against the unchanged strict-native routines.

Uses the same direct-source artifact and GPU slot requirements as the primary
replay. These synthetic decks supplement, and do not replace, forecast proof.
"""
import pytest
import test_regional_native_v842 as replay


@pytest.mark.parametrize('case', range(17, 25))
def test_native_cpu_branches(case):
    replay.test_native_masks(case)
    replay.test_native_mesh_scaling(case)
    replay.test_native_cpu_boundary_stages(case)


@pytest.mark.parametrize('case', range(17, 25))
def test_native_cuda_branches(case):
    replay.test_native_cuda_stages(case)
    replay.test_native_cuda_driving_pool(case)
