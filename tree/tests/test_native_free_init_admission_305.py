"""CPU/no-assets gates for native-free initialization and timestep admission #305."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mpas_port.init_door import InitDoorRefusal, select_vertical_mode
from mpas_port.timestep_admission import (
    CourantPolicy,
    TimestepAdmissionError,
    admit_timestep,
    edge_length_authority,
)
from mpas_port.vertical import (
    build_edge_vertical_metrics,
    build_vertical_grid,
    runtime_vertical_vectors,
    smooth_terrain,
    validate_vertical_grid,
)
from mpas_port.vertical_spec import VerticalSpec, VerticalSpecError


class ClosedFourCellMesh:
    """A deterministic K4 closed mesh sufficient to exercise every scalar loop."""

    def __init__(self) -> None:
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.cellsOnEdge = np.asarray(pairs, dtype=np.int64)
        self.dcEdge = np.asarray([1000, 1100, 1200, 1300, 1400, 1500], dtype=np.float64)
        self.dvEdge = np.asarray([900, 950, 1000, 1050, 1100, 1150], dtype=np.float64)
        self.areaCell = np.asarray([1.0e6, 1.1e6, 1.2e6, 1.3e6], dtype=np.float64)
        incident: list[list[tuple[int, int]]] = [[] for _ in range(4)]
        for edge, (left, right) in enumerate(pairs):
            incident[left].append((edge, right))
            incident[right].append((edge, left))
        self.nEdgesOnCell = np.full(4, 3, dtype=np.int64)
        self.edgesOnCell = np.asarray(
            [[edge for edge, _ in row] for row in incident], dtype=np.int64
        )
        self.cellsOnCell = np.asarray(
            [[neighbor for _, neighbor in row] for row in incident], dtype=np.int64
        )
        # Native orientation (nEdges, TWO, nCoeff); all-zero second derivative
        # reduces the order-3 edge value to the arithmetic average while still
        # exercising orientation and topology traversal.
        self.deriv_two = np.zeros((6, 2, 4), dtype=np.float64)
        self.ter = np.asarray([0.0, 100.0, 220.0, 40.0], dtype=np.float64)


@pytest.fixture()
def mesh() -> ClosedFourCellMesh:
    return ClosedFourCellMesh()


def test_terrain_zero_guard_matches_native_branch(mesh: ClosedFourCellMesh) -> None:
    smoothed = smooth_terrain(mesh, mesh.ter, passes=2)
    assert smoothed[0] == 0.0
    assert np.all(np.isfinite(smoothed))
    assert not np.array_equal(smoothed[1:], mesh.ter[1:])


@pytest.mark.parametrize("scheme", ["tc", "legacy"])
def test_vertical_grid_invariants_two_native_branches(
    mesh: ClosedFourCellMesh, scheme: str
) -> None:
    vertical = build_vertical_grid(
        mesh,
        mesh.ter,
        n_vert_levels=5,
        ztop=10_000.0,
        scheme=scheme,
        terrain_smoothing_passes=1,
        surface_smoothing_passes=2,
        hybrid_transition_height=8_000.0,
    )
    report = validate_vertical_grid(vertical, n_cells=4, n_edges=6)
    assert report["minimum_physical_layer_m"] > 0.0
    assert vertical.zgrid.shape == (6, 4)
    assert vertical.zxu.shape == (5, 6)
    assert vertical.hx[0, 0] == 0.0


def test_nonhybrid_and_specified_interfaces(mesh: ClosedFourCellMesh) -> None:
    interfaces = np.asarray([0.0, 200.0, 700.0, 1800.0, 4200.0, 10_000.0])
    vertical = build_vertical_grid(
        mesh,
        mesh.ter,
        n_vert_levels=5,
        ztop=10_000.0,
        scheme="specified",
        specified_zw=interfaces,
        hybrid_coordinate=False,
        smooth_surfaces=False,
    )
    np.testing.assert_array_equal(vertical.zw, interfaces)
    np.testing.assert_allclose(vertical.ah, 1.0 - interfaces / 10_000.0)
    assert vertical.first_height_level == 6


def test_edge_metrics_have_native_orientation_and_zero_top(mesh: ClosedFourCellMesh) -> None:
    vertical = build_vertical_grid(
        mesh,
        mesh.ter,
        n_vert_levels=5,
        ztop=10_000.0,
        smooth_surfaces=False,
    )
    metrics = build_edge_vertical_metrics(mesh, vertical, theta_adv_order=3)
    assert metrics.zb.shape == (6, 2, 6)
    assert metrics.zb3.shape == (6, 2, 6)
    assert np.all(metrics.zb[-1] == 0.0)
    assert np.all(metrics.zb3 == 0.0)
    assert np.any(metrics.zb[:-1] != 0.0)


def test_runtime_vectors_materialize_only_unused_lower_slots(mesh: ClosedFourCellMesh) -> None:
    vertical = build_vertical_grid(mesh, mesh.ter, n_vert_levels=5, ztop=10_000.0)
    runtime = runtime_vertical_vectors(vertical)
    for name in ("dzu", "rdzu", "rdzwp", "rdzwm", "fzm", "fzp"):
        assert runtime[name][0] == 0.0
        np.testing.assert_array_equal(runtime[name][1:], getattr(vertical, name)[1:])


def test_vertical_spec_rejects_unknown_and_nonmonotonic() -> None:
    with pytest.raises(VerticalSpecError, match="unknown key"):
        VerticalSpec.from_mapping({"schema": "gpuwm-hex.vertical-spec/v1", "mystery": 4})
    with pytest.raises(VerticalSpecError, match="strictly increasing"):
        VerticalSpec.from_mapping(
            {
                "schema": "gpuwm-hex.vertical-spec/v1",
                "n_vert_levels": 3,
                "ztop_m": 1000.0,
                "scheme": "specified",
                "specified_interfaces_m": [0.0, 200.0, 150.0, 1000.0],
            }
        )


def test_vertical_spec_canonical_hash_is_order_independent() -> None:
    first = VerticalSpec.from_mapping(
        {"schema": "gpuwm-hex.vertical-spec/v1", "n_vert_levels": 5, "ztop_m": 10_000.0}
    )
    second = VerticalSpec.from_mapping(
        {"ztop_m": 10_000.0, "n_vert_levels": 5, "schema": "gpuwm-hex.vertical-spec/v1"}
    )
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.sha256() == second.sha256()


def test_edge_length_authority_uses_real_minimum_not_nominal() -> None:
    authority = edge_length_authority(
        np.asarray([15_000.0, 12_500.0, 14_000.0, 9_500.0], dtype=np.float32)
    )
    assert authority.minimum_m == 9_500.0
    assert authority.median_m != 15_000.0
    assert len(authority.raw_sha256) == 64


def test_safe_timestep_is_unchanged_and_recorded() -> None:
    authority = edge_length_authority(np.asarray([10_000.0, 11_000.0]))
    result = admit_timestep(
        60.0,
        authority,
        policy=CourantPolicy(max_characteristic_speed_m_s=125.0, safety_factor=0.9),
    )
    assert result.resolved_dt_seconds == 60.0
    assert result.auto_shrunk is False
    assert result.maximum_admitted_dt_seconds == pytest.approx(72.0)


def test_unsafe_timestep_refuses_with_numbers_and_no_auto_shrink() -> None:
    authority = edge_length_authority(np.asarray([8_000.0, 12_000.0]))
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(
            120.0,
            authority,
            policy=CourantPolicy(max_characteristic_speed_m_s=125.0, safety_factor=0.9),
        )
    message = str(caught.value)
    assert "min(dcEdge)=8000" in message
    assert "requested dt=120" in message
    assert "maximum dt=57.6" in message
    assert "will not auto-shrink" in message


def test_bad_dc_edge_is_not_filtered() -> None:
    with pytest.raises(TimestepAdmissionError, match="corrupted geometry"):
        edge_length_authority(np.asarray([10_000.0, 0.0, np.nan]))


def _mode_namespace(**values: object) -> Namespace:
    defaults = {
        "vertical_spec": None,
        "vertical_artifact": None,
        "capsule": None,
        "reference": None,
        "grid": None,
    }
    defaults.update(values)
    return Namespace(**defaults)


def test_vertical_modes_are_explicit_and_mutually_exclusive() -> None:
    assert select_vertical_mode(
        _mode_namespace(vertical_spec=SimpleNamespace(), grid=SimpleNamespace())
    ) == "constructed"
    assert select_vertical_mode(
        _mode_namespace(capsule=SimpleNamespace(), reference=SimpleNamespace())
    ) == "compatibility-capsule"
    with pytest.raises(InitDoorRefusal, match="both declared"):
        select_vertical_mode(
            _mode_namespace(
                vertical_spec=SimpleNamespace(),
                grid=SimpleNamespace(),
                capsule=SimpleNamespace(),
                reference=SimpleNamespace(),
            )
        )
    with pytest.raises(InitDoorRefusal, match="no vertical source"):
        select_vertical_mode(_mode_namespace())


# ---------------------------------------------------------------------------
# The edge-length authority entered the constants fingerprint in this lane.
# Every baseline compared against a bind therefore has to carry it too, or the
# x4 frozen no-op self-check compares two digests that can never be equal.
# ---------------------------------------------------------------------------
TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load_tool(key: str, filename: str) -> object:
    name = f"_test_305_{key}"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def binding_mod() -> object:
    return _load_tool("mesh_binding", "mpas_mesh_binding.py")


@pytest.fixture(scope="module")
def mesh_run_tool() -> object:
    return _load_tool("mesh_run", "run_cuda_v841_forecast_mesh.py")


@pytest.fixture(scope="module")
def native_mesh_pair(tmp_path_factory, binding_mod) -> SimpleNamespace:
    """A synthetic grid/static pair carrying the native mesh's real shape.

    Dimensions, nominal resolution, sphere radius and a finite positive
    ``dcEdge`` are all genuine, so the whole cross-examination -- geometry,
    unit-sphere/Earth-scale agreement, edge authority, Courant admission,
    fingerprint -- runs for real.  Only the registered byte count and digest
    cannot be met: those name 523 MB of files that ship with no fetch path.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    binding = binding_mod.MESH_BINDINGS[binding_mod.NATIVE_MESH_NAME]
    directory = tmp_path_factory.mktemp("native-mesh-pair")
    grid = directory / "native.grid.nc"
    static = directory / "native.static.nc"
    radius = 6_371_229.0

    with netCDF4.Dataset(grid, "w", format="NETCDF4") as dataset:
        dataset.setncattr("sphere_radius", 1.0)
        dataset.createDimension("nCells", binding.n_cells)
        dataset.createDimension("nEdges", binding.n_edges)
        dataset.createDimension("TWO", 2)
        nominal = dataset.createVariable("nominalMinDc", "f8", ())
        nominal[...] = binding.nominal_dx_m / radius
        dataset.createVariable("nEdgesOnCell", "i4", ("nCells",))[:] = np.full(
            binding.n_cells, 6, dtype=np.int32
        )
        pairs = np.empty((binding.n_edges, 2), dtype=np.int32)
        edges = np.arange(binding.n_edges, dtype=np.int32)
        pairs[:, 0] = edges % binding.n_cells + 1
        pairs[:, 1] = (edges + 1) % binding.n_cells + 1
        dataset.createVariable("cellsOnEdge", "i4", ("nEdges", "TWO"))[:] = pairs

    # min(dcEdge)=20 km admits the registry's 120 s at 125 m/s and 0.90 safety
    # (bound 144 s), so admission passes on geometry rather than on a waiver.
    dc_edge = 20_000.0 + (np.arange(binding.n_edges, dtype=np.float64) % 5_000.0)
    with netCDF4.Dataset(static, "w", format="NETCDF4") as dataset:
        dataset.setncattr("sphere_radius", radius)
        dataset.createDimension("nEdges", binding.n_edges)
        dataset.createDimension("nVertLevels", binding.n_levels)
        dataset.createDimension("nSoilLevels", binding.n_soil_levels)
        nominal = dataset.createVariable("nominalMinDc", "f8", ())
        nominal[...] = binding.nominal_dx_m
        dataset.createVariable("dcEdge", "f8", ("nEdges",))[:] = dc_edge

    return SimpleNamespace(grid=grid, static=static, dc_edge=dc_edge)


def _unpinned_require_file(binding_mod):
    """Stand in for the byte-pin gate ONLY.

    Existence still refuses by name; the declared byte count and SHA-256 are
    the one thing a fixture cannot satisfy.  Everything the bind does after
    this point runs against the real module.
    """

    def _require(role, path, want_bytes, want_sha, mesh):
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise binding_mod.MeshBindingMismatch(
                f"mesh {mesh!r}: {role} authority is missing: {resolved}"
            )
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }

    return _require


def _frozen_native_proof(binding_mod) -> SimpleNamespace:
    """A module whose constants already hold the frozen native values."""

    binding = binding_mod.MESH_BINDINGS[binding_mod.NATIVE_MESH_NAME]
    return SimpleNamespace(
        N_CELLS=binding.n_cells,
        N_EDGES=binding.n_edges,
        N_LEVELS=binding.n_levels,
        N_INTERFACES=binding.n_interfaces,
        N_SOIL_LEVELS=binding.n_soil_levels,
        DT_SECONDS=binding.dt_seconds,
        NOMINAL_DX_M=np.float32(binding.nominal_dx_m),
        MIN_FREE_DEVICE_BYTES=binding_mod.NATIVE_DEVICE_FLOOR,
        RESTART_WORKER_MIN_FREE_DEVICE_BYTES=binding_mod.NATIVE_RESTART_FLOOR,
        INIT_RECONSTRUCTION_COEFFICIENTS_PIN={"role": "coefficients"},
        INIT_EDGE_NORMAL_VECTORS_PIN={"role": "edge-normals"},
        PHYSICS_GEOMETRY_CARRIER_PIN={"role": "carrier"},
        LANDMASK_CONSTRUCTOR_CAST_PIN={"role": "landmask"},
        AUTHORITY_PINS={"grid": {"role": "grid"}, "static": {"role": "static"}},
        require_frozen_execution_sources=lambda: {"files": {}, "sha256": "0" * 64},
    )


def test_frozen_native_selftest_reproduces_the_bound_fingerprint(
    binding_mod, mesh_run_tool, native_mesh_pair, monkeypatch, capsys
) -> None:
    """The x4 no-op self-check must be able to match the digest the bind reports.

    The baseline the door computes and the fingerprint ``bind_mesh`` returns
    have to be the same call with the same inputs.  A baseline that omits the
    edge-length authority digests ``EDGE_LENGTH_AUTHORITY_SHA256=None`` and can
    never equal a bound digest, so a rebind-free run reports a moved
    fingerprint every time and the frozen no-op stops being provable.
    """

    monkeypatch.setattr(
        binding_mod, "_require_file", _unpinned_require_file(binding_mod)
    )
    proof = _frozen_native_proof(binding_mod)

    rc = mesh_run_tool._selftest(
        binding_mod,
        proof,
        None,
        binding_mod.NATIVE_MESH_NAME,
        str(native_mesh_pair.grid),
        str(native_mesh_pair.static),
    )

    printed = capsys.readouterr().out
    assert "native no-op (unchanged)" in printed, printed
    assert "[selftest] FAIL" not in printed, printed
    assert rc == 0, printed


def test_a_bare_constants_fingerprint_is_never_a_valid_baseline(
    binding_mod, native_mesh_pair
) -> None:
    """The one call that supplies the fingerprint's authority keywords."""

    proof = _frozen_native_proof(binding_mod)
    authority = edge_length_authority(
        native_mesh_pair.dc_edge, source="static.dcEdge"
    )

    bound = binding_mod.binding_fingerprint(proof, native_mesh_pair.static)

    assert bound["fields"]["EDGE_LENGTH_AUTHORITY_SHA256"] == authority.raw_sha256
    assert bound["fields"]["EDGE_LENGTH_MINIMUM_M"] == repr(float(authority.minimum_m))
    assert bound["sha256"] != binding_mod.constants_fingerprint(proof)["sha256"]
