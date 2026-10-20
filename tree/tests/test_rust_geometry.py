"""Exact bits against the retired scalar implementation through the built bridge."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import _geometry_authority as authority
from hexcore import rust_geometry, vector
from hexcore.engines import EngineRefusal


def mesh(dtype, *, sphere=True, periodic=False, ragged=False):
    points = np.asarray([[-1.1, -1.2, 1.3], [1.0, 1.1, 1.3],
                         [-1.2, 1.4, -1.0], [1.5, -1.1, -1.3]], dtype=dtype)
    if not sphere:
        points = np.asarray([[-1.1, -1.2, 0], [1.0, -1.1, 0],
                             [-0.7, 1.4, 0], [1.5, 1.2, 0]], dtype=dtype)
    coe = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    edges = (points[coe[:, 0]] + points[coe[:, 1]]) * dtype(0.5)
    eoc = np.asarray([[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]], dtype=np.int64)
    counts = np.asarray([2, 3, 3, 2] if ragged else [3, 3, 3, 3])
    return SimpleNamespace(**{**{f"{c}Cell": points[:, i] for i, c in enumerate("xyz")},
                             **{f"{c}Edge": edges[:, i] for i, c in enumerate("xyz")},
                             "cellsOnEdge": coe, "edgesOnCell": eoc,
                             "nEdgesOnCell": counts, "on_a_sphere": sphere,
                             "is_periodic": periodic, "x_period": 3.0,
                             "y_period": 3.0, "nCellsSolve": 2})


@pytest.fixture(autouse=True)
def bridge_required():
    try:
        rust_geometry.geometry_engine()
    except EngineRefusal as error:
        if os.environ.get("GPUWM_HEX_REQUIRE_GEOMETRY_TESTS") == "1":
            raise
        pytest.skip(f"built geometry bridge required for numeric bit proof: {error}")


def assert_bits(first, second):
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert first.tobytes(order="C") == second.tobytes(order="C")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sphere,periodic,ragged", [(True, False, False),
                         (True, False, True), (False, False, False),
                         (False, True, False)])
@pytest.mark.parametrize("halos", [False, True])
def test_geometry_matches_every_scalar_bit(dtype, sphere, periodic, ragged, halos):
    source = mesh(dtype, sphere=sphere, periodic=periodic, ragged=ragged)
    for old, new in zip(authority.initialize_vector_geometry(source),
                        vector.initialize_vector_geometry(source)):
        assert_bits(old, new)
    expected = authority.initialize_reconstruction_coefficients(source, include_halos=halos)
    assert_bits(expected, vector.initialize_reconstruction_coefficients(source, include_halos=halos))
    if halos:
        computed = rust_geometry.derived_geometry(source)
        for name, old in zip(("edgeNormalVectors", "localVerticalUnitVectors", "cellTangentPlane"),
                             authority.initialize_vector_geometry(source)):
            assert_bits(old, computed[name])
        assert_bits(expected, computed["coeffs_reconstruct"])


def test_mixed_coordinate_precision_keeps_geometry_promotion_and_coefficient_kind():
    source = mesh(np.float32)
    for component in "xyz":
        name = f"{component}Edge"
        setattr(source, name, getattr(source, name).astype(np.float64))
    expected = authority.initialize_vector_geometry(source)
    for old, new in zip(expected, vector.initialize_vector_geometry(source)):
        assert_bits(old, new)
    assert_bits(authority.initialize_reconstruction_coefficients(source),
                vector.initialize_reconstruction_coefficients(source))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_boundary_cell_and_geometry_switch_overrides(dtype):
    source = mesh(dtype)
    source.cellsOnEdge = source.cellsOnEdge.copy()
    source.cellsOnEdge[0, 0] = -1
    source.cellsOnEdge[-1, 1] = -1
    for options in ({}, {"on_a_sphere": False, "is_periodic": True,
                         "x_period": 4.0, "y_period": 4.0}):
        for old, new in zip(authority.initialize_vector_geometry(source, **options),
                            vector.initialize_vector_geometry(source, **options)):
            assert_bits(old, new)
    assert_bits(authority.initialize_reconstruction_coefficients(source),
                vector.initialize_reconstruction_coefficients(source))


@pytest.mark.parametrize("fault,message", [("zero", "cell position"),
                         ("missing", "two missing cells"),
                         ("count", "nEdgesOnCell"), ("index", "out of range")])
def test_degenerate_geometry_still_refuses(fault, message):
    source = mesh(np.float32)
    if fault == "zero":
        for component in "xyz":
            getattr(source, f"{component}Cell")[0] = 0
    elif fault == "missing":
        source.cellsOnEdge[0] = -1
    elif fault == "count":
        source.nEdgesOnCell[0] = 4
    else:
        source.edgesOnCell[0, 1] = 100
    with pytest.raises(ValueError, match=message):
        vector.initialize_reconstruction_coefficients(source)


def test_stale_executable_has_no_python_fallback(tmp_path, monkeypatch):
    path = tmp_path / "bridge"
    path.write_bytes(b"old bridge")
    path.chmod(0o755)
    monkeypatch.setenv("GPUWM_HEX_RW_MPAS_GEOMETRY", str(path))
    with pytest.raises(EngineRefusal, match="wrong ABI"):
        vector.initialize_reconstruction_coefficients(mesh(np.float32))


def test_init_geometry_cache_and_netcdf_completion_keep_exact_bits(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    from hexcore.vertical_spec import _complete_derived_geometry, _derived_geometry_arrays

    source = mesh(np.float64)
    old_geometry = authority.initialize_vector_geometry(source)
    old_coefficients = authority.initialize_reconstruction_coefficients(source)
    expected = dict(zip(("edgeNormalVectors", "localVerticalUnitVectors", "cellTangentPlane"),
                        old_geometry))
    expected["coeffs_reconstruct"] = old_coefficients
    cache = tmp_path / "cache.npz"
    first, origin = _derived_geometry_arrays(source, cache, "input-one")
    assert origin == "computed"
    second, origin = _derived_geometry_arrays(source, cache, "input-one")
    assert origin == "cache"
    third, origin = _derived_geometry_arrays(source, cache, "input-two")
    assert origin == "computed"
    for name, old in expected.items():
        for current in (first, second, third):
            assert_bits(old, current[name])
    path = tmp_path / "static.nc"
    with netcdf.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in (("nCells", 4), ("nEdges", 6), ("maxEdges", 3), ("TWO", 2), ("THREE", 3)):
            ds.createDimension(name, size)
        dimensions = {"edgeNormalVectors": ("nEdges", "THREE"),
                      "localVerticalUnitVectors": ("nCells", "THREE"),
                      "cellTangentPlane": ("nCells", "TWO", "THREE"),
                      "coeffs_reconstruct": ("nCells", "maxEdges", "THREE")}
        for name, dims in dimensions.items():
            ds.createVariable(name, "f4", dims)[:] = 0
        completed, origin = _complete_derived_geometry(ds, source)
        assert completed == list(expected)
        assert origin == "computed"
        for name, old in expected.items():
            assert_bits(old.astype(np.float32), np.asarray(ds.variables[name][:]))
