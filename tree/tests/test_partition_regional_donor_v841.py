"""Padded regional mesh metadata keeps a canonical global donor index."""

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.cuda_regional_forecast_v841 import PaddedRegionalHostMesh
from hexcore.cuda_physics_prep_v841 import _regional_donor_for_extent
from test_regional_physics_seam import _SyntheticCull, _masks


def _layout(inverse):
    return SimpleNamespace(
        _global_cells=3, partition=1,
        cell_g2l=lambda: np.asarray(inverse, dtype=np.int32),
        cell_row_complete=np.array([True, False, True]),
        edge_row_complete=np.array([True, False, True, False]),
        vertex_row_complete=np.array([True, False]),
    )


def test_global_ordering_provenance_does_not_create_a_regional_padding_donor():
    mesh = SimpleNamespace(attrs={"regional_garbage_donor_cell": 900000})
    assert _regional_donor_for_extent(mesh, None) is None


def test_padded_extent_keeps_the_exact_declared_donor():
    mesh = SimpleNamespace(regional_garbage_donor_cell=np.int32(2))
    assert _regional_donor_for_extent(mesh, 4) == 2


def test_padded_extent_without_a_declared_donor_retains_the_legacy_default():
    assert _regional_donor_for_extent(SimpleNamespace(attrs={}), 4) is None


def test_donor_uses_the_global_last_cells_local_index_and_keeps_real_masks():
    mesh = _SyntheticCull()
    layout = _layout([1, 2, 0])
    mesh.partition_layout_v841 = layout
    source = object()
    mesh.partition_source_mesh = source
    padded = PaddedRegionalHostMesh(mesh, _masks())
    assert padded.regional_garbage_donor_cell == 0
    assert padded.partition_source_mesh is source
    assert padded.partition_layout_v841 is layout
    for kind in ("cell", "edge", "vertex"):
        original = getattr(layout, f"{kind}_row_complete")
        mask = getattr(padded.partition_layout, f"{kind}_row_complete")
        np.testing.assert_array_equal(mask[:-1], original)
        assert mask.size == original.size + 1 and not mask[-1]
        assert not np.shares_memory(mask, original)


def test_whole_mesh_keeps_its_last_real_column_as_the_donor():
    padded = PaddedRegionalHostMesh(_SyntheticCull(), _masks())
    assert padded.regional_garbage_donor_cell == padded.garbage_cell - 1


def test_partition_without_canonical_donor_refuses_wrong_sounding():
    mesh = _SyntheticCull()
    mesh.partition_layout_v841 = _layout([0, 1, -1])
    with pytest.raises(ValueError, match="canonical regional garbage donor"):
        PaddedRegionalHostMesh(mesh, _masks())


def test_ordered_partition_uses_the_declared_source_donor_instead_of_last_index():
    mesh = _SyntheticCull()
    layout = _layout([1, 2, 0])
    mesh.partition_layout_v841 = layout
    mesh.partition_source_mesh = SimpleNamespace(attrs={"regional_garbage_donor_cell": 0})
    padded = PaddedRegionalHostMesh(mesh, _masks())
    assert padded.regional_garbage_donor_cell == 1
    assert padded.regional_garbage_donor_cell != int(layout.cell_g2l()[2])


def test_ordered_partition_refuses_a_missing_declared_donor_even_when_last_is_present():
    mesh = _SyntheticCull()
    mesh.partition_layout_v841 = _layout([-1, 1, 0])
    mesh.partition_source_mesh = SimpleNamespace(attrs={"regional_garbage_donor_cell": 0})
    with pytest.raises(ValueError, match="declared global donor"):
        PaddedRegionalHostMesh(mesh, _masks())