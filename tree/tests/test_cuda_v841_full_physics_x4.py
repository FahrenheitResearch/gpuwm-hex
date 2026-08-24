from __future__ import annotations

import dataclasses
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_cuda_v841_full_physics_x4.py"


def _load_runner() -> object:
    name = "_test_run_cuda_v841_full_physics_x4"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_exact_release_scope_and_all_execution_sources_are_frozen() -> None:
    runner = _load_runner()
    assert str(runner.SRC) in sys.path
    assert runner.SOURCE_RELEASE == "v8.4.1"
    assert runner.ARWEN_COMMIT == "0d04db71298d010a61fee3267c07277da3b8b64f"
    assert (runner.N_CELLS, runner.N_EDGES, runner.N_LEVELS) == (
        163_842,
        491_520,
        55,
    )
    assert runner.SCALAR_NAMES == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert runner.DT_SECONDS == 120.0
    assert runner.FULL_STEPS == 30
    assert runner.CHECKPOINT_STEP == 15
    assert runner.ARWEN_CONTRACT_SURFACE_SHA256 == (
        "823af4a55018a71ad630144fae7b21a459095249cedb1180bc9f3e1a2fbfe511"
    )
    assert runner.ARWEN_GLACIER_COMPOSED_TU_SHA256 == (
        "edafcac585d4786c0cdfddf07f8e767b64d0d40b6db0e4da3dc3b2fa8c21fb59"
    )
    assert float(runner.ARWEN_XICE_THRESHOLD) == float(np.float32(0.02))
    assert runner.NEGATIVE_QV_PIN["negative_count"] == 215
    assert runner.NEGATIVE_QV_PIN["full_qv_sha256"] == (
        "c0180afa0fa99253f414472d8f767421c85b1324f67e161fdef9f6244b565099"
    )
    assert runner.NEGATIVE_QV_PIN["negative_indices_sha256"] == (
        "b622daead41e76d988e5419d92bac9b28a2390e8cfdb619d38ff19c1188a10a4"
    )
    assert runner.NEGATIVE_QV_PIN["negative_values_sha256"] == (
        "192e06fa6a1302eb06f7148e0cb85483ca61a21caf0eed4bc95ccfcb3452eb4b"
    )
    assert runner.unresolved_source_pins() == ()
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in runner.EXECUTION_SOURCE_PINS.values()
    )
    receipt = runner.require_frozen_execution_sources()
    assert set(receipt["files"]) == set(runner.EXECUTION_SOURCE_PINS)


def test_frozen_source_hashes_and_adapter_contract_are_exact() -> None:
    runner = _load_runner()
    assert runner.EXECUTION_SOURCE_PINS == {
        "src/mpas_port/cuda_physics_prep_v841.py": (
            "29fb9bb7c6f37f90e1f66fabd576810fa89db902ad7e4495eaf21a57610cbccf"
        ),
        "src/mpas_port/cuda_gwdo_v841.py": (
            "11e038bc2365964b6c8b8db36d3dd99ed200edc3f40e7795e208af3af08bd316"
        ),
        "src/mpas_port/cuda_physics_v841.py": (
            "ea6afd713883530e317936f93285b4d4ffe22c2fecf25d76f3f1b6af4041529f"
        ),
        "src/mpas_port/cuda_driver.py": (
            "9daf917a89b3b9dd6f013be3d971c76d255bcfbbb9c1027b9de0c8823cb49e66"
        ),
        "src/mpas_port/cuda_backend/recovery.py": (
            "40635e20e4de9f1cf49c2590dcc14f262fa03667dd4b547f0dd61fb47892dac3"
        ),
        "src/mpas_port/config_v841.py": (
            "2bc878868e41ffc71491479059d3bd9165ce980a38360ed683e2717f54a8111a"
        ),
        "src/mpas_port/cuda_arwen_physics_v841.py": (
            "20c4b22dcd36fa165d15642e45e3fac5cbe7b8de01dbffdfd0a84361c222d13b"
        ),
        "src/mpas_port/mixing.py": (
            "864f0686325108100afc10a8804ea4e2dd6de81e3269ee4cbc2747be82b09e2e"
        ),
        "src/mpas_port/mixing_v841.py": (
            "f82e9f5c64547b6763db37ada8ba79a966e9ef8f310cf84fc71375f8380e3a73"
        ),
        "src/mpas_port/cuda_horizontal.py": (
            "97faf0869a0a5ea9ebbc4c67b3c2d6c68cefdfa10dece73cd204d818962efde4"
        ),
        "src/mpas_port/cuda_horizontal_v841.py": (
            "3fc0b860ebd67dfed453617c348810964ea1110e782fe85db10283afb406e2fe"
        ),
        "src/mpas_port/cuda_transport_v841.py": (
            "55c66759d9c81f65ed71ce77570897c102fd64661da6ad6c37b438b27771ab23"
        ),
        "src/mpas_port/partition_assets_v841.py": (
            "dc5f2cb3f7bdadeca28854a15644273f7a94cdb710a36df18f4f91bdba70450e"
        ),
        "src/mpas_port/partition_local_mesh_v841.py": (
            "609955b3db527528f1e2ffd949483099a8d19dd0bd23f724d7a711fbba08e150"
        ),
        "src/mpas_port/partition_state_v841.py": (
            "a504e6f5c5abc2014d40e4a8e3e89885f97d6456ff428c585fbaf57910eccbe8"
        ),
        "src/mpas_port/partition_device_scheduler_v841.py": (
            "fcf2f94fc368e71b6b87ddb5c1d3b1b68a4cfc2bccfcf1e50e57f0f2d3432276"
        ),
        "src/mpas_port/partition_executor_v841.py": (
            "6343fe0f89f39d81b3ef0d61343330c2ad09f59e00295ceedc090c3e4a61879c"
        ),
        "src/mpas_port/partition_net_v841.py": (
            "30b0988b2d40bbda8d68be1ec236564ea87bb6690bfdcd08370eea12ca11753b"
        ),
    }
    assert runner.KNOWN_CONTRACT_PINS["coupling_contract_sha256"] == (
        "63d9edb9ea4a12b78ccdeec64c2424de2ddbc10ff3a8c58361aa943f19c517db"
    )
    assert runner.KNOWN_CONTRACT_PINS["coupling_kernel_sha256"] == (
        "70d2006d4687b67fe087fd4a5c9e69a76e4a39c648703913f4d79903249bdcab"
    )
    assert runner.KNOWN_CONTRACT_PINS["adapter_contract_sha256"] == (
        "6c3ca3bae5f92a7ffa3f9cf27db0f2329ab0506bbf5d76e362f010676a0c78e1"
    )


def test_landmask_cast_and_exhaustive_sealed_constructor_audit_are_exact() -> None:
    from netCDF4 import Dataset
    from mpas_port import cuda_arwen_physics_v841 as adapter

    runner = _load_runner()
    pin = runner.LANDMASK_CONSTRUCTOR_CAST_PIN
    assert pin["source_dimensions"] == ("nCells",)
    assert pin["shape"] == (163_842,)
    assert pin["source_dtype"] == "<i4"
    assert pin["source_array_sha256"] == (
        "6aee8da961b605b9aa21362840aec214beaa384d42041e232ff8f8a9f60b22b5"
    )
    assert pin["source_unique_values"] == (0, 1)
    assert pin["target_dtype"] == "<f4"
    assert pin["target_array_sha256"] == (
        "31a7e78b7f0a6b10d719df443b0e2d629e60028c8ac628c5ad8556a87c5eae65"
    )
    assert pin["target_uint32_values"] == (0, 1_065_353_216)
    assert "landmask" in adapter._SURFACE_FLOAT_FIELDS
    assert adapter._SURFACE_INT_FIELDS == ("ivgtyp", "isltyp")
    assert adapter._SOIL_FIELDS == ("soil_temperature", "soil_moisture")
    assert set(adapter._CONSTRUCTOR_ARRAY_FIELDS) == (
        set(adapter._SURFACE_FLOAT_FIELDS)
        | set(adapter._SURFACE_INT_FIELDS)
        | set(adapter._SOIL_FIELDS)
        | {"z_interface_nominal_m", "dx_column_m"}
    )
    # GF's native geometry/shallow boundary is part of the sealed mapping:
    # a per-cell dx vector and the hardwired ishallow=1.
    assert "dx_column_m" in adapter._CONSTRUCTOR_KEYS
    assert "gf_ishallow" in adapter._CONSTRUCTOR_KEYS

    with Dataset(runner.default_authority_paths()["init"], "r") as dataset:
        variable = dataset.variables["landmask"]
        variable.set_auto_maskandscale(False)
        source = np.ascontiguousarray(variable[...])
    target = np.ascontiguousarray(source, dtype=np.float32)
    assert source.dtype.str == pin["source_dtype"]
    assert runner.array_sha256(source) == pin["source_array_sha256"]
    assert tuple(int(value) for value in np.unique(source)) == (0, 1)
    assert target.dtype.str == pin["target_dtype"]
    assert runner.array_sha256(target) == pin["target_array_sha256"]
    assert tuple(int(value) for value in np.unique(target.view(np.uint32))) == (
        0,
        1_065_353_216,
    )
    assert np.array_equal(target.astype(np.int32), source)

    builder = inspect.getsource(runner.build_arwen_constructor_values)
    for token in (
        "source_landmask_sha256",
        "source_unique_values",
        "target_uint32_values",
        "value_preserving_exact_fp32_cast",
        '"landmask": landmask',
        '"xland": xland',
        '"xice_threshold": float(ARWEN_XICE_THRESHOLD)',
        "surface_classification_receipt",
    ):
        assert token in builder
    preparation = inspect.getsource(runner._prepare_host_execution)
    mapping = preparation.index("build_arwen_constructor_values")
    audit = preparation.index("SealedArwenConstructorV841.from_mapping")
    returned = preparation.index('"constructor_values": constructor_values')
    assert mapping < audit < returned
    assert "all_required_keys_dtypes_shapes_validated" in preparation

def test_real_x4_native_xland_classification_and_cuda_census_are_exact() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    with Dataset(runner.default_authority_paths()["init"], "r") as dataset:
        for name in ("xland", "xice", "ivgtyp", "landmask"):
            dataset.variables[name].set_auto_maskandscale(False)
        xland_source = np.ascontiguousarray(dataset.variables["xland"][...])
        xice_source = np.ascontiguousarray(dataset.variables["xice"][...])
        ivgtyp = np.ascontiguousarray(dataset.variables["ivgtyp"][...])
        landmask = np.ascontiguousarray(dataset.variables["landmask"][...])
    assert xland_source.dtype == np.dtype(np.float32)
    assert xland_source.shape == (1, runner.N_CELLS)
    assert runner.array_sha256(xland_source) == runner.NATIVE_XLAND_SOURCE_SHA256
    xland = np.ascontiguousarray(xland_source[0])
    xice = np.ascontiguousarray(xice_source[0])
    assert runner.array_sha256(xland) == runner.NATIVE_XLAND_FLAT_SHA256
    sea_ice = xice >= runner.ARWEN_XICE_THRESHOLD
    open_water = (xland >= np.float32(1.5)) & ~sea_ice
    land = ~(sea_ice | open_water)
    glacier = land & (ivgtyp == np.int32(15))
    sflx_land = land & ~glacier
    observed = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(runner.ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": int(np.count_nonzero(sea_ice)),
        "open_water_columns": int(np.count_nonzero(open_water)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land)),
        "glacier_columns": int(np.count_nonzero(glacier)),
    }
    assert observed == dict(runner.EXPECTED_SURFACE_CLASSIFICATION)
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice))
    delta_indices = np.ascontiguousarray(
        np.flatnonzero(
            (xice >= runner.ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5))
        )
    )
    assert int(glacier_indices[0]) == 30
    assert runner.array_sha256(glacier_indices) == runner.GLACIER_INDEX_SHA256
    assert runner.array_sha256(sea_ice_indices) == runner.SEA_ICE_INDEX_SHA256
    assert runner.array_sha256(delta_indices) == runner.THRESHOLD_DELTA_INDEX_SHA256
    assert np.all(ivgtyp[sea_ice] == 15)
    assert np.all(landmask[sea_ice] == 0)
    assert np.all(xland[sea_ice] == np.float32(1.0))
    assert np.all(landmask[glacier] == 1)
    assert np.all(xland[glacier] == np.float32(1.0))
    f000 = runner.require_arwen_v2_surface_execution(
        {"surface_classification": observed, "last_noahmp_census": None},
        executed=False,
        label="F000",
    )
    assert f000["last_noahmp_census"] is None
    executed = runner.require_arwen_v2_surface_execution(
        {
            "surface_classification": observed,
            "last_noahmp_census": dict(runner.EXPECTED_NOAHMP_CENSUS),
        },
        executed=True,
        label="F001",
    )
    assert executed["last_noahmp_census"]["glacier_path"] == (
        runner.ARWEN_GLACIER_CUDA_PROVENANCE
    )
    bad = dict(runner.EXPECTED_NOAHMP_CENSUS)
    bad["glacier_path"] = "noahmp-glacier/host"
    with pytest.raises(ValueError, match="census/provenance"):
        runner.require_arwen_v2_surface_execution(
            {"surface_classification": observed, "last_noahmp_census": bad},
            executed=True,
            label="mutant",
        )


def test_p_top_is_exact_f000_area_weighted_native_derivation() -> None:
    runner = _load_runner()
    expected = np.float32(1_159.38818359375)
    assert runner.EXPECTED_ARWEN_P_TOP_PA_F32.view(np.uint32) == expected.view(
        np.uint32
    )
    assert runner.EXPECTED_TOP_PRESSURE_RANGE_PA == (
        592.24884,
        1_233.00952,
        1_342.08362,
    )
    source = inspect.getsource(runner.derive_area_weighted_p_top_v841)
    for token in (
        "pressure_base",
        "pressure_perturbation",
        "zgrid",
        "area_cell",
        "np.float32",
        "weighted_mean64",
        "cavallo_buffer_layers",
        "claimed_native_identity",
    ):
        assert token in source
    assert "ARWEN_P_TOP_PA = 5_000" not in RUNNER_PATH.read_text(encoding="utf-8")


def test_exact_init_reconstruction_overlay_replaces_only_static_placeholder() -> None:
    runner = _load_runner()
    pin = runner.INIT_RECONSTRUCTION_COEFFICIENTS_PIN
    assert pin["dimensions"] == ("nCells", "maxEdges", "R3")
    assert pin["shape"] == (163_842, 10, 3)
    assert pin["static_placeholder_raw_sha256"] == (
        "b3b09d26538fe509096884906c93ff8b8d2c794300bafcab91449eee1c7bd31c"
    )
    assert pin["init_carrier_raw_sha256"] == (
        "1d25d2439a6cdcc3cc4a3cabfb5b6720730bb548f4cd88208340cca9df883350"
    )
    assert pin["active_slots"] == 983_040
    assert pin["active_components"] == 2_949_120
    assert pin["nonzero_components"] == 2_949_120

    overlay = inspect.getsource(
        runner.overlay_exact_init_reconstruction_coefficients
    )
    for token in (
        'prior_source != "static"',
        "init/static nEdgesOnCell topology differs",
        "init/static edgesOnCell topology differs",
        "active_every_component_nonzero",
        "bitwise_positive_zero",
        "source_files_mutated",
        "dynamics mesh in memory only",
    ):
        assert token in overlay
    preparation = inspect.getsource(runner._prepare_host_execution)
    overlay_call = preparation.index(
        "overlay_exact_init_reconstruction_coefficients"
    )
    prepared_seal = preparation.index("PreparedCudaInputs.validated")
    assert overlay_call < prepared_seal
    assert '"reconstruction_coefficients": reconstruction_overlay' in preparation


def test_exact_init_edge_normal_overlay_closes_all_geometry_carriers() -> None:
    runner = _load_runner()
    pin = runner.INIT_EDGE_NORMAL_VECTORS_PIN
    assert pin["dimensions"] == ("nEdges", "R3")
    assert pin["shape"] == (491_520, 3)
    assert pin["dtype"] == "<f4"
    assert pin["static_placeholder_raw_sha256"] == (
        "91e0c31eb2d6776a903dc5456c5e72e1e447c2d835cebbcde87581738cac735b"
    )
    assert pin["init_carrier_raw_sha256"] == (
        "25d9ef5c70b38a2e7d2c9c60456d9835a1e0fc790b60a3c98d1dbb44482d41da"
    )
    assert pin["nonzero_components"] == 1_474_550
    assert pin["exact_zero_components"] == 10
    assert pin["zero_rows"] == 0
    assert pin["float64_norm_min"] == 0.999999867111912
    assert pin["float64_norm_max"] == 1.000000135860109

    carriers = runner.PHYSICS_GEOMETRY_CARRIER_PIN
    assert carriers["cellsOnEdge"]["raw_sha256"] == (
        "a53b1c9bf9e5c0e026c7253b9026f4711bc973acd376f70a6e996adfa584b2d3"
    )
    assert carriers["cellsOnEdge"]["source_roles"] == ("grid", "static", "init")
    assert carriers["east_north"]["absent_source_roles"] == (
        "grid",
        "static",
        "init",
    )
    assert carriers["east_north"]["fallback_source_fields"] == (
        "lonCell",
        "latCell",
    )
    assert carriers["edgeNormalVectors"] == {
        "grid_present": False,
        "static_present": True,
        "init_present": True,
    }

    overlay = inspect.getsource(runner.overlay_exact_init_edge_normal_vectors)
    for token in (
        'prior_source != "static"',
        "grid/{role} raw cellsOnEdge topology differs",
        "absent_in_grid_static_init_and_prepared_mesh",
        "zonal_meridional_vectors(lonCell, latCell)",
        "only_zero_placeholder_trap",
        "float64_norm_min",
        "source_files_mutated",
        "dynamics mesh in memory only",
    ):
        assert token in overlay
    preparation = inspect.getsource(runner._prepare_host_execution)
    reconstruction_call = preparation.index(
        "overlay_exact_init_reconstruction_coefficients"
    )
    edge_normal_call = preparation.index("overlay_exact_init_edge_normal_vectors")
    prepared_seal = preparation.index("PreparedCudaInputs.validated")
    assert reconstruction_call < edge_normal_call < prepared_seal
    assert '"edge_normal_vectors": edge_normal_overlay' in preparation


def test_real_edge_normal_overlay_uses_sealed_init_bytes_without_source_mutation() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    paths = runner.default_authority_paths()
    before = {
        role: (path.stat().st_size, path.stat().st_mtime_ns)
        for role, path in paths.items()
        if role in {"grid", "static", "init"}
    }
    with Dataset(paths["static"], "r") as dataset:
        normal_variable = dataset.variables["edgeNormalVectors"]
        normal_variable.set_auto_maskandscale(False)
        static_normals = np.ascontiguousarray(normal_variable[...])
        cells_variable = dataset.variables["cellsOnEdge"]
        cells_variable.set_auto_maskandscale(False)
        raw_cells = np.ascontiguousarray(cells_variable[...])
    with Dataset(paths["grid"], "r") as dataset:
        lon_variable = dataset.variables["lonCell"]
        lat_variable = dataset.variables["latCell"]
        lon_variable.set_auto_maskandscale(False)
        lat_variable.set_auto_maskandscale(False)
        lon = np.ascontiguousarray(lon_variable[...])
        lat = np.ascontiguousarray(lat_variable[...])

    mesh = SimpleNamespace(
        arrays={
            "edgeNormalVectors": static_normals,
            "cellsOnEdge": np.ascontiguousarray(raw_cells.astype(np.int64) - 1),
            "lonCell": lon,
            "latCell": lat,
        },
        variable_sources={
            "edgeNormalVectors": "static",
            "cellsOnEdge": "static",
            "lonCell": "grid_binary64_test_carrier",
            "latCell": "grid_binary64",
        },
        variable_dimensions={},
        variable_attrs={},
        provenance={},
    )
    receipt = runner.overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    assert receipt["static_placeholder"]["bitwise_positive_zero"] is True
    assert receipt["init_carrier"]["raw_c_sha256"] == (
        runner.INIT_EDGE_NORMAL_VECTORS_PIN["init_carrier_raw_sha256"]
    )
    assert receipt["init_carrier"]["nonzero_components"] == 1_474_550
    assert receipt["init_carrier"]["exact_positive_zero_components"] == 10
    assert receipt["init_carrier"]["zero_rows"] == 0
    audit = receipt["cuda_physics_geometry_carrier_audit"]
    assert audit["cellsOnEdge"]["grid_static_init_raw_bit_identical"] is True
    assert audit["east_north"][
        "absent_in_grid_static_init_and_prepared_mesh"
    ] is True
    assert audit["edgeNormalVectors"]["only_zero_placeholder_trap"] is True
    assert mesh.variable_sources["edgeNormalVectors"] == (
        "init_exact_in_memory_physics_edge_normal_overlay"
    )
    assert np.count_nonzero(mesh.arrays["edgeNormalVectors"]) == 1_474_550
    after = {
        role: (path.stat().st_size, path.stat().st_mtime_ns)
        for role, path in paths.items()
        if role in {"grid", "static", "init"}
    }
    assert after == before

def test_arwen_v2_git_and_source_pin_precede_cuda_backend_and_kernel_cache() -> None:
    runner = _load_runner()
    main_source = inspect.getsource(runner.main)
    # The source pins verify FIRST: the checkout guard imports the seam
    # manifest from a pinned module, so the module's bytes must be proven
    # before any of its constants are trusted.
    assert main_source.index("require_frozen_execution_sources") < main_source.index(
        "verify_arwen_checkout_git"
    )
    source = inspect.getsource(runner._execute_full_proof)
    pin_import = source.index("pin_arwen_physics_v841")
    pin_call = source.index("arwen_pin = dict(pin_arwen_physics_v841")
    backend_import = source.index("from mpas_port.cuda_backend import")
    cache = source.index("cache = KernelCache(")
    assert pin_import < pin_call < backend_import < cache


class _FakeBackend:
    def __init__(
        self,
        log: list[str],
        *,
        commit_fails: bool = False,
        finish_auto_rolls_back: bool = False,
        rollback_boundary_valid: bool = True,
    ) -> None:
        self.log = log
        self.phase = "boundary"
        self.commit_fails = commit_fails
        self.finish_auto_rolls_back = finish_auto_rolls_back
        self.rollback_boundary_valid = rollback_boundary_valid
        self._at_boundary = True
        self._start_time = 0.0

    def begin_step(self, **kwargs: object) -> object:
        self.log.append("begin")
        self._start_time = float(kwargs["atmosphere"].state.time_seconds)
        self.phase = "begun"
        self._at_boundary = False
        return object()

    def finish_step(self, **_: object) -> object:
        self.log.append("phase2")
        if self.finish_auto_rolls_back:
            self.phase = "automatic_rollback"
            self._at_boundary = self.rollback_boundary_valid
            raise FloatingPointError("phase-two numeric refusal")
        self.phase = "finished_unpublished"
        return object()

    def commit_step(self) -> None:
        self.log.append("adapter_commit")
        if self.commit_fails:
            raise RuntimeError("adapter commit failure")
        self.phase = "complete"
        self._at_boundary = True

    def abort_step(self) -> None:
        if self.phase not in ("begun", "finished_unpublished"):
            raise RuntimeError("abort requires an active transaction")
        self.log.append("adapter_abort")
        self.phase = "rolled_back"
        self._at_boundary = True

    def step_receipt(self) -> dict[str, object]:
        return {
            "schema": "fake-transaction/v1",
            "adapter_contract_sha256": "fake-adapter",
            "constructor": {"identity_sha256": "fake-constructor"},
            "phase": self.phase,
            "start_time_seconds": self._start_time,
        }

    def restart_state(self) -> dict[str, object]:
        if not self._at_boundary:
            raise RuntimeError("restart state is not at a boundary")
        return {
            "schema": "fake-transaction/v1",
            "identity": {
                "adapter_contract_sha256": "fake-adapter",
                "constructor_identity_sha256": "fake-constructor",
            },
            "seam": {
                "identity": {"fake": True},
                "arrays": {},
                "scalars": {"elapsed_seconds": self._start_time},
            },
            "adapter": {},
        }


class _FakeDriver:
    def __init__(self, log: list[str], *, rollback_changes_state: bool = False) -> None:
        self.log = log
        self.rollback_changes_state = rollback_changes_state
        start = SimpleNamespace(
            time_seconds=0.0,
            rho=np.ones((1, 1), dtype=np.float32),
            rho_u=np.ones((1, 1), dtype=np.float32),
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        self.atmosphere = SimpleNamespace(state=start)
        self.horizontal = SimpleNamespace(
            recover_edge_fields=lambda *_: SimpleNamespace(
                rho_edge=np.ones((1, 1), dtype=np.float32)
            )
        )

    def step_device_with_physics(self, _: object) -> object:
        self.log.append("moist_rk")
        endpoint = SimpleNamespace(
            time_seconds=120.0,
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        return SimpleNamespace(atmosphere=SimpleNamespace(state=endpoint))

    def commit_post_wsm6_candidate(self, candidate: object, recovery: object) -> object:
        self.log.append("driver_commit")
        return SimpleNamespace(
            atmosphere=candidate.atmosphere,
            recovery=recovery,
            surface_updates={},
        )

    def abort_post_wsm6_candidate(self, _: object) -> None:
        self.log.append("driver_abort")
        if self.rollback_changes_state:
            self.atmosphere = SimpleNamespace(
                state=SimpleNamespace(time_seconds=0.0)
            )


def _transaction_callables(log: list[str], *, recovery_fails: bool = False):
    def couple(*_: object, **__: object) -> object:
        log.append("couple")
        return object()

    def clamp(*_: object, **__: object) -> object:
        log.append("clamp")
        return object()

    def recover(*_: object, **kwargs: object) -> object:
        log.append("recover")
        assert kwargs["phase2_dt_seconds"] == 120.0
        if recovery_fails:
            raise RuntimeError("recovery failure")
        return object()

    return couple, clamp, recover


def test_composite_transaction_order_is_atomic_and_staged() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log)
    result = runner.execute_composite_step(
        driver=_FakeDriver(log),
        backend=_FakeBackend(log),
        scalar_names=runner.SCALAR_NAMES,
        physics_geometry=object(),
        kernel_cache=object(),
        previous_surface_updates=None,
        couple=couple,
        clamp=clamp,
        recover=recover,
    )
    assert log == [
        "begin",
        "couple",
        "moist_rk",
        "clamp",
        "phase2",
        "recover",
        "driver_commit",
        "adapter_commit",
    ]
    assert result.committed.atmosphere.state.time_seconds == 120.0


def test_composite_transaction_aborts_both_owners_before_publication() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log, recovery_fails=True)
    with pytest.raises(runner.CompositeTransactionError, match="aborted without publication"):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=_FakeBackend(log),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert log[-2:] == ["driver_abort", "adapter_abort"]
    assert "driver_commit" not in log


def test_composite_transaction_accepts_verified_finish_automatic_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    backend = _FakeBackend(log, finish_auto_rolls_back=True)
    couple, clamp, recover = _transaction_callables(log)
    with pytest.raises(runner.CompositeTransactionError, match="aborted without publication"):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=backend,
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert backend.phase == "automatic_rollback"
    assert log == ["begin", "couple", "moist_rk", "clamp", "phase2", "driver_abort"]
    assert "adapter_abort" not in log


def test_composite_transaction_rejects_unverified_automatic_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log)
    with pytest.raises(
        runner.CompositeTransactionError,
        match="rollback was incomplete.*restart state is not at a boundary",
    ):
        runner.execute_composite_step(
            driver=_FakeDriver(log),
            backend=_FakeBackend(
                log,
                finish_auto_rolls_back=True,
                rollback_boundary_valid=False,
            ),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert "adapter_abort" not in log


def test_composite_transaction_rejects_incomplete_driver_rollback() -> None:
    runner = _load_runner()
    log: list[str] = []
    couple, clamp, recover = _transaction_callables(log, recovery_fails=True)
    with pytest.raises(
        runner.CompositeTransactionError,
        match="rollback was incomplete.*exact start-state identity",
    ):
        runner.execute_composite_step(
            driver=_FakeDriver(log, rollback_changes_state=True),
            backend=_FakeBackend(log),
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=couple,
            clamp=clamp,
            recover=recover,
        )
    assert log[-2:] == ["driver_abort", "adapter_abort"]


def test_native_gate_includes_physical_edge_u_interface_w_and_gwdo() -> None:
    runner = _load_runner()
    assert runner.NATIVE_FIELD_MAP["normal_u"] == ("u", "level_edge")
    assert runner.NATIVE_FIELD_MAP["w"] == ("w", "interface_cell")
    for name in (
        *runner.SCALAR_NAMES,
        "rho",
        "theta",
        "pressure",
        "surface_pressure",
        "normal_u",
        "w",
        "smois",
        "tslb",
        "rainc",
        "rainnc",
        "dusfcg",
        "dvsfcg",
        "dtaux3d",
        "dtauy3d",
        "rubldiff",
        "rvbldiff",
    ):
        assert name in runner.NATIVE_RMSE_LIMITS


def test_diagnostic_dataclass_and_restart_hash_projection_are_exact() -> None:
    runner = _load_runner()
    groups = {
        name: {"x": np.array([1.0], dtype=np.float32)}
        for name in ("surface", "soil", "precipitation", "gwdo")
    }
    snapshot = SimpleNamespace(
        **groups,
        metadata={"time_seconds": 0.0},
        receipt={"phase": "boundary"},
    )
    mapped = runner.backend_diagnostic_mapping(snapshot)
    assert set(mapped) == {
        "surface",
        "soil",
        "precipitation",
        "gwdo",
        "metadata",
        "receipt",
    }
    direct = {"arrays": {"qv": np.array([1.0], dtype=np.float32), "q2": np.array([-1.0], dtype=np.float32)}}
    resumed = {"arrays": {"qv": np.array([1.0], dtype=np.float32), "q2": np.array([9.0], dtype=np.float32)}}
    identity = runner.require_bitwise_restart_identity(direct, resumed)
    assert identity["bitwise_identical"] is True
    assert identity["q2_excluded_from_publication_projection"] is True


def test_restart_fingerprint_identity_names_exact_numerical_leaf() -> None:
    runner = _load_runner()
    direct = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 2.0], dtype=np.float32)},
            "step": 16,
        }
    )
    identical = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 2.0], dtype=np.float32)},
            "step": 16,
        }
    )
    receipt = runner.require_fingerprint_identity(
        "step 16 atmosphere", direct, identical
    )
    assert receipt["bitwise_identical"] is True
    assert receipt["sha256"] == direct["sha256"]

    changed = runner.fingerprint_nested_arrays(
        {
            "state": {"rho": np.array([1.0, 3.0], dtype=np.float32)},
            "step": 16,
        }
    )
    with pytest.raises(
        RuntimeError,
        match=r"step 16 atmosphere differs.*arrays/state/rho/sha256",
    ):
        runner.require_fingerprint_identity("step 16 atmosphere", direct, changed)


def test_restart_localization_gates_precede_first_resumed_step() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner._execute_full_proof)
    worker_spawn = source.index("worker = _spawn_restart_worker(")
    restored_f030 = source.index('restored_f030 = worker["restored_f030"]')
    step16 = source.index('"first resumed step 16 MPAS atmosphere"')
    f001 = source.index("identity = require_bitwise_restart_identity(")
    assert worker_spawn < restored_f030 < step16 < f001
    assert '"F030 restored MPAS atmosphere"' in source
    assert '"F030 restored Arwen backend"' in source
    assert '"F030 restored GF advective forcing"' in source
    assert '"first resumed step 16 Arwen backend"' in source
    assert '"restart_arm_fresh_process": True' in source

    worker_source = inspect.getsource(runner._execute_restart_worker)
    construct = worker_source.index("stack = _construct_device_stack(")
    rehydrate = worker_source.index(
        "restored_f030 = fingerprint_execution_boundary(stack)"
    )
    identity = worker_source.index(
        '"F030 restored MPAS atmosphere (fresh restart process)"'
    )
    forcing_identity = worker_source.index(
        '"F030 restored GF advective forcing (fresh restart process)"'
    )
    steps = worker_source.index(
        "restart_snapshots, _, restart_receipts = _run_steps("
    )
    assert construct < rehydrate < identity < forcing_identity < steps
    assert "boundary_observer=capture_step16" in worker_source

    loop_source = inspect.getsource(runner._run_steps)
    receipt = loop_source.index("receipts.append(")
    observer = loop_source.index("boundary_observer(step, stack)")
    capture = loop_source.index("if step in capture_steps:", receipt)
    assert receipt < observer < capture


def test_checkpoint_carries_and_restores_the_gf_advective_forcing() -> None:
    """The #327 mechanism, pinned: GF's rthdynten/rqvdynten pair is per-step
    carried state formed by each step's dynamics and consumed by the NEXT
    begin_step -- outside both the MPAS atmosphere and the Arwen backend
    restart payload.  The v2 checkpoint never carried it, so every restored
    arm re-entered step 16 on zero forcing lanes while the unbroken run fed
    the real step-15 pair: a deterministic all-fields divergence at exactly
    step 16, on every arm.  Schema v3 captures the pair at F030, refuses a
    checkpoint written without it, and re-seeds it on restore under its own
    fingerprint identity."""

    runner = _load_runner()
    assert runner.CHECKPOINT_SCHEMA.endswith("-checkpoint/v3")
    fields = {field.name for field in dataclasses.fields(runner.HostDriverCheckpoint)}
    assert {"gf_dynamics_tendencies", "gf_forcing_fingerprint"} <= fields

    # Writing a checkpoint without the carrier refuses by name, before any
    # device access.
    with pytest.raises(RuntimeError, match="GF advective-forcing"):
        runner.download_driver_checkpoint(
            object(), object(), dynamics_tendencies=None
        )

    # The step loop seeds the carrier from the stack, so a restored stack
    # that carries it continues exactly where the unbroken run is.
    loop_source = inspect.getsource(runner._run_steps)
    assert 'stack.get("gf_dynamics_tendencies")' in loop_source
    proof_source = inspect.getsource(runner._execute_full_proof)
    assert 'dynamics_tendencies=stack.get("gf_dynamics_tendencies")' in proof_source

    # The restore path uploads the pair back onto the stack.
    construct_source = inspect.getsource(runner._construct_device_stack)
    assert "CudaV841GfDynamicsTendencies(" in construct_source
    assert 'stack["gf_dynamics_tendencies"] = carrier' in construct_source

    # The worker refuses a pre-v3 checkpoint by name and returns the
    # restored-forcing fingerprint the parent gates on.
    worker_source = inspect.getsource(runner._execute_restart_worker)
    assert "predates the GF advective-forcing capture" in worker_source
    assert "gf_dynamics_tendencies=forcing" in worker_source
    spawn_source = inspect.getsource(runner._spawn_restart_worker)
    assert '"restored_gf_forcing"' in spawn_source

    # A stale pickle (pre-v3 HostDriverCheckpoint with the slot unset) at the
    # correct F030 time is refused by the worker before any CUDA probing.
    import pickle

    stale = object.__new__(runner.HostDriverCheckpoint)
    for name, value in {
        "state": None,
        "saved_diagnostics": None,
        "backend_state": {},
        "atmosphere_fingerprint": {},
        "backend_fingerprint": {},
        "model_time_seconds": runner.CHECKPOINT_STEP * runner.DT_SECONDS,
        # A v2 pickle leaves these slots unset; the closest picklable
        # simulation is the carrier explicitly absent.
        "gf_dynamics_tendencies": None,
        "gf_forcing_fingerprint": None,
    }.items():
        object.__setattr__(stale, name, value)
    job = {
        "schema": runner.RESTART_WORKER_SCHEMA,
        "arwen_commit": runner.ARWEN_COMMIT,
        "checkpoint": stale,
    }
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as scratch:
        stale_path = _Path(scratch) / "stale.pkl"
        with stale_path.open("wb") as stream:
            pickle.dump(job, stream, protocol=4)
        with pytest.raises(
            RuntimeError, match="predates the GF advective-forcing capture"
        ):
            runner._execute_restart_worker(
                stale_path, _Path(scratch) / "never.pkl"
            )


def test_restart_arm_is_a_genuinely_fresh_worker_process(tmp_path: Path) -> None:
    """The measured defect: a seam aggregate reconstructed inside the live
    baseline process reproduces frozen legacy-RRTMG longwave heating only to
    within 1 ULP (raw phase-1 dtheta, carrier rthratenlw), while the same
    restore in a fresh process continues bit-identically.  The restart arm
    must therefore execute in a fresh worker process."""

    import pickle

    runner = _load_runner()
    # Worker CLI flags travel only as a pair.
    with pytest.raises(SystemExit):
        runner.parse_args(["--restart-worker-input", str(tmp_path / "in.pkl")])
    with pytest.raises(SystemExit):
        runner.parse_args(["--restart-worker-output", str(tmp_path / "out.pkl")])
    args = runner.parse_args(
        [
            "--restart-worker-input", str(tmp_path / "in.pkl"),
            "--restart-worker-output", str(tmp_path / "out.pkl"),
        ]
    )
    assert args.restart_worker_input == tmp_path / "in.pkl"
    assert args.restart_worker_output == tmp_path / "out.pkl"
    # The spawn launches a separate interpreter on this exact tool file.
    spawn_source = inspect.getsource(runner._spawn_restart_worker)
    assert "sys.executable" in spawn_source
    assert "subprocess.run" in spawn_source
    assert '"--restart-worker-input"' in spawn_source
    assert '"--restart-worker-output"' in spawn_source
    # The spawn refuses non-fresh or foreign worker payloads.
    assert 'results.get("fresh_process") is not True' in spawn_source
    assert 'results.get("arwen_commit") != ARWEN_COMMIT' in spawn_source
    # The worker refuses foreign payload schemas before any CUDA probing.
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as stream:
        pickle.dump({"schema": "not-the-restart-worker-schema"}, stream, protocol=4)
    with pytest.raises(RuntimeError, match="schema"):
        runner._execute_restart_worker(bad, tmp_path / "never.pkl")
    # The worker refuses a non-F030 checkpoint before any CUDA probing.
    wrong_time = SimpleNamespace(model_time_seconds=16 * 120.0)
    job = {
        "schema": runner.RESTART_WORKER_SCHEMA,
        "arwen_commit": runner.ARWEN_COMMIT,
        "checkpoint": wrong_time,
    }
    wrong = tmp_path / "wrong-time.pkl"
    with wrong.open("wb") as stream:
        pickle.dump(job, stream, protocol=4)
    with pytest.raises(TypeError, match="HostDriverCheckpoint"):
        runner._execute_restart_worker(wrong, tmp_path / "never2.pkl")


def test_gf_deviation_is_explicit_and_never_claims_native_parity() -> None:
    runner = _load_runner()
    assert any("GF" in item and "zero" in item for item in runner.NONCLAIMS)
    source = inspect.getsource(runner._execute_full_proof)
    assert '"fa35_rthften_rqvften": "zero"' in source
    assert '"native_gf_parity_claim": False' in source
    assert '"f001_restart_snapshot_receipt"' in source


def test_baseline_diagnostic_cli_is_isolated_and_precedes_restart(tmp_path: Path) -> None:
    runner = _load_runner()
    cache = tmp_path / "cache"
    output = tmp_path / "release-output"
    diagnostic = tmp_path / "baseline-output"
    args = runner.parse_args(
        [
            "--cache-root", str(cache),
            "--output", str(output),
            "--baseline-diagnostic-output", str(diagnostic),
        ]
    )
    assert args.baseline_diagnostic_output == diagnostic
    admitted_cache, admitted_output = runner.validate_destination(cache, output, ())
    admitted_diagnostic = runner.validate_baseline_diagnostic_destination(
        diagnostic,
        cache_root=admitted_cache,
        output_root=admitted_output,
        protected=(),
    )
    assert admitted_diagnostic == diagnostic.absolute()
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_baseline_diagnostic_destination(
            output / "nested",
            cache_root=admitted_cache,
            output_root=admitted_output,
            protected=(),
        )
    source = inspect.getsource(runner._execute_full_proof)
    diagnostic_write = source.index("_write_baseline_diagnostic_output(")
    release_uninterrupted = source.index("# Release the uninterrupted device graph")
    restart_worker = source.index("worker = _spawn_restart_worker(")
    assert diagnostic_write < release_uninterrupted < restart_worker
    assert runner.BASELINE_DIAGNOSTIC_WARNING == (
        "ENGINEERING BASELINE; restart proof pending; NOT RELEASE"
    )
    assert runner.BASELINE_DIAGNOSTIC_STATUS == (
        "uninterrupted_baseline_passed_restart_not_evaluated_non_release"
    )


def test_f000_optional_surface_diagnostics_are_exact_init_snapshot_only() -> None:
    from netCDF4 import Dataset

    runner = _load_runner()
    paths = runner.default_authority_paths()
    carrier = runner.load_f000_initialized_surface_diagnostics(paths["init"])
    assert set(carrier) == {"arrays", "receipt"}
    assert set(carrier["arrays"]) == {"t2", "u10", "v10"}
    with Dataset(paths["native_f000"], "r") as native:
        for target, pin in runner.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            source = carrier["arrays"][target]
            authority = np.ascontiguousarray(
                np.asarray(native.variables[str(pin["source"])][0]), dtype=np.float32
            )
            np.testing.assert_array_equal(source, authority)
            assert runner.array_sha256(source) == pin["sha256"]

    placeholders = {
        name: np.zeros(runner.N_CELLS, dtype=np.float32)
        for name in carrier["arrays"]
    }
    receipt = runner.overlay_f000_initialized_surface_diagnostics(
        placeholders, carrier["arrays"]
    )
    assert receipt["applied"] is True
    assert "snapshot only" in receipt["scope"]
    for name, source in carrier["arrays"].items():
        np.testing.assert_array_equal(placeholders[name], source)
        assert receipt["fields"][name]["source_correct"] is True

    nonzero = {
        name: np.zeros(runner.N_CELLS, dtype=np.float32)
        for name in carrier["arrays"]
    }
    nonzero["t2"][0] = np.float32(1.0)
    with pytest.raises(ValueError, match=r"not the exact \+0 FP32 placeholder"):
        runner.overlay_f000_initialized_surface_diagnostics(
            nonzero, carrier["arrays"]
        )

    capture_source = inspect.getsource(runner.capture_snapshot)
    assert capture_source.index("arrays.update(diagnostics)") < capture_source.index(
        "overlay_f000_initialized_surface_diagnostics("
    )
    assert "if step == 0:" in capture_source


def test_baseline_writer_validates_nested_exact_capture_receipt_schema() -> None:
    runner = _load_runner()
    for label, step in (
        ("F000", 0),
        ("F030", runner.CHECKPOINT_STEP),
        ("F001", runner.FULL_STEPS),
    ):
        nested = {
            "surface_classification": dict(
                runner.EXPECTED_SURFACE_CLASSIFICATION
            ),
            "last_noahmp_census": (
                None if step == 0 else dict(runner.EXPECTED_NOAHMP_CENSUS)
            ),
        }
        capture_receipt = {
            "schema": runner.SNAPSHOT_SCHEMA,
            "label": label,
            "step": step,
            "time_seconds": step * runner.DT_SECONDS,
            "arwen_v2_surface_execution": nested,
            "audit_d2h_outside_step_receipt": True,
            "prep": {},
            "backend": {},
            "backend_diagnostic_metadata": {},
            "f000_initialized_surface_diagnostics": (
                {} if step == 0 else None
            ),
            "arrays": {},
        }
        assert set(capture_receipt) == {
            "schema", "label", "step", "time_seconds",
            "arwen_v2_surface_execution", "audit_d2h_outside_step_receipt",
            "prep", "backend", "backend_diagnostic_metadata",
            "f000_initialized_surface_diagnostics", "arrays",
        }
        observed = runner.require_snapshot_receipt_surface_execution(
            capture_receipt, label=label
        )
        assert observed == nested

        flattened = dict(capture_receipt)
        flattened.pop("arwen_v2_surface_execution")
        flattened.update(nested)
        with pytest.raises(ValueError, match="lacks nested"):
            runner.require_snapshot_receipt_surface_execution(
                flattened, label=label
            )

    writer = inspect.getsource(runner._write_baseline_diagnostic_output)
    assert "require_snapshot_receipt_surface_execution(" in writer
    assert "snapshot_receipts[label], label=label" in writer
