"""Regional row fields in the mesh-binding registry: table work, not code.

Registering a regional cull is a :class:`MeshBinding` row whose three
regional slots are filled -- ``boundary_zone_width`` (7 on every measured
native cull), ``bdy_mask_sha256`` (the digest of the grid's
``bdyMaskCell/Edge/Vertex`` triple), and ``lbc_source`` (nullable today).
``admit_regional_row`` cross-examines every row against the inspected grid
before any constant is rebound, and each refusal names the breakage.

When ``GPUWM_HEX_REGIONAL_REFERENCE_DIR`` points at the native-culled
reference set, the digest and inspection legs run against the real bytes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"

REFERENCE_DIR_VARIABLE = "GPUWM_HEX_REGIONAL_REFERENCE_DIR"


def _load_binding() -> object:
    name = "_test_regional_mesh_binding_module"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, TOOLS / "mpas_mesh_binding.py")
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
def binding_module() -> object:
    return _load_binding()


def _row(binding_module: object, **overrides: object) -> object:
    fields = dict(
        name="regional-test-row",
        n_cells=2_971,
        n_edges=9_116,
        n_levels=55,
        n_interfaces=56,
        n_soil_levels=4,
        nominal_dx_m=120_000.0,
        dt_seconds=120.0,
        grid_bytes=1,
        grid_sha256="0" * 64,
        static_bytes=1,
        static_sha256="0" * 64,
    )
    fields.update(overrides)
    return binding_module.MeshBinding(**fields)


_REGIONAL_OBSERVED = {
    "regional_masks": {"present": True, "zone_width": 7, "sha256": "a" * 64}
}
_GLOBAL_OBSERVED = {"regional_masks": {"present": False}}


def test_a_regional_registration_is_a_row(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source=None,
    )
    assert row.regional
    assert not _row(binding_module).regional


def test_global_row_with_regional_grid_is_refused_by_name(
    binding_module: object,
) -> None:
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(_row(binding_module), _REGIONAL_OBSERVED)
    assert "declares no boundary zone" in str(refusal.value)
    assert "unforced boundary" in str(refusal.value)


def test_regional_row_with_global_grid_is_refused_by_name(
    binding_module: object,
) -> None:
    row = _row(
        binding_module, boundary_zone_width=7, bdy_mask_sha256="a" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _GLOBAL_OBSERVED)
    assert "carries no bdyMask triple" in str(refusal.value)


def test_mask_digest_mismatch_is_refused_by_name(binding_module: object) -> None:
    row = _row(
        binding_module, boundary_zone_width=7, bdy_mask_sha256="b" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "bdyMask triple SHA-256" in str(refusal.value)
    assert "wrong cells" in str(refusal.value)


def test_zone_width_mismatch_is_refused_by_name(binding_module: object) -> None:
    row = _row(
        binding_module, boundary_zone_width=5, bdy_mask_sha256="a" * 64
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "outermost ring" in str(refusal.value)


def test_half_declared_regional_row_is_refused_by_name(
    binding_module: object,
) -> None:
    row = _row(binding_module, boundary_zone_width=7)
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "all-or-nothing" in str(refusal.value)


def test_empty_lbc_slot_refuses_execution_by_name(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source=None,
    )
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert "lbc_source slot is empty" in str(refusal.value)
    assert "unforced boundary" in str(refusal.value)


def test_filled_regional_row_admits(binding_module: object) -> None:
    row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256="a" * 64,
        lbc_source="registered-boundary-stream",
    )
    receipt = binding_module.admit_regional_row(row, _REGIONAL_OBSERVED)
    assert receipt["regional"] is True
    assert receipt["lbc_source"] == "registered-boundary-stream"


def test_global_row_with_global_grid_is_untouched(binding_module: object) -> None:
    receipt = binding_module.admit_regional_row(
        _row(binding_module), _GLOBAL_OBSERVED
    )
    assert receipt == {"regional": False}


# ---------------------------------------------------------------------------
# the real bytes: digest and inspection against the native-culled reference
# ---------------------------------------------------------------------------
def _reference_grid() -> Path:
    root = os.environ.get(REFERENCE_DIR_VARIABLE, "")
    if not root or not Path(root).is_dir():
        pytest.skip(
            f"{REFERENCE_DIR_VARIABLE} does not point at the native-culled "
            "regional reference set"
        )
    candidates = sorted((Path(root) / "cull-x1").glob("*.nc"))
    if len(candidates) != 1:
        pytest.skip("reference set does not hold exactly one quick-cull grid")
    return candidates[0]


def test_regional_mask_digest_is_computable_from_the_real_grid(
    binding_module: object,
) -> None:
    digest = binding_module.regional_mask_digest(_reference_grid())
    assert len(digest) == 64 and digest == digest.lower()
    # deterministic: the digest a row pins is reproducible read-over-read
    assert binding_module.regional_mask_digest(_reference_grid()) == digest


def test_inspect_grid_observes_the_real_mask_triple(binding_module: object) -> None:
    grid = _reference_grid()
    row = _row(binding_module)
    observed = binding_module._inspect_grid(grid, row)
    masks = observed["regional_masks"]
    assert masks["present"] is True
    assert masks["zone_width"] == 7
    assert masks["sha256"] == binding_module.regional_mask_digest(grid)
    # and the admission path sees exactly what a registered row would verify
    regional_row = _row(
        binding_module,
        boundary_zone_width=7,
        bdy_mask_sha256=masks["sha256"],
        lbc_source="registered-boundary-stream",
    )
    receipt = binding_module.admit_regional_row(regional_row, observed)
    assert receipt["bdy_mask_sha256"] == masks["sha256"]


def test_global_digest_request_is_refused_by_name(
    binding_module: object, tmp_path: Path
) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "not-regional.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", 3)
        variable = dataset.createVariable("nEdgesOnCell", "i4", ("nCells",))
        variable[:] = [6, 6, 6]
    with pytest.raises(binding_module.MeshBindingMismatch) as refusal:
        binding_module.regional_mask_digest(path)
    assert "not a regional cull" in str(refusal.value)
