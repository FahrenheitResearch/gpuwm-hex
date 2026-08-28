"""The regional CUDA translation unit and its earned-anchor admission.

These tests are CPU-testable by design: they hold the source, its declared
entrypoints, its numerical hazards and its admission gate to properties that
do not need a card.  The numbers themselves are proved on hardware by
``tools/run_cuda_regional_contract.py``, whose receipt is the evidence the
anchor registry names.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from hexcore import regional_v841
from hexcore.cuda_backend import regional_admission
from hexcore.cuda_regional_v841 import (
    CUDA_REGIONAL_SOURCE,
    MODULE_KEY,
    REGIONAL_KERNELS,
)


ROOT = Path(__file__).resolve().parents[1]


def _kernel_names(source: str) -> list[str]:
    return re.findall(r"__global__ void (\w+)", source)


def test_every_defined_kernel_is_declared_and_vice_versa():
    """A kernel with no declaration has no deck, and a deck proves nothing
    about a kernel that does not exist."""

    defined = set(_kernel_names(CUDA_REGIONAL_SOURCE))
    declared = set(REGIONAL_KERNELS)
    assert defined == declared
    assert len(REGIONAL_KERNELS) == len(set(REGIONAL_KERNELS))


def test_module_key_is_its_own_translation_unit():
    """The regional kernels must not be folded into a shared module key.

    Folding them in would change the source SHA-256 of a translation unit
    that archived receipts, FTZ audit counts and compile manifests pin, and
    every one of those proofs would have to be re-minted to say the same
    thing about the global lane.
    """

    assert MODULE_KEY == "hexcore.cuda_regional_v841"
    for other in ("cuda_acoustic_v841", "cuda_transport_v841", "cuda_driver"):
        assert other not in MODULE_KEY


def test_zone_constants_come_from_the_cpu_authority():
    """The device zone geometry cannot drift from the host authority's."""

    assert f"#define REGIONAL_N_SPEC_ZONE {regional_v841.N_SPEC_ZONE}\n" in (
        CUDA_REGIONAL_SOURCE
    )
    assert f"#define REGIONAL_N_RELAX_ZONE {regional_v841.N_RELAX_ZONE}\n" in (
        CUDA_REGIONAL_SOURCE
    )


def test_no_literal_float_divisor_anywhere_in_the_source():
    """MEASURED 2026-08-26 on the RTX 5070 Ti through CuPy 14.2.0: NVRTC
    rewrites ``x / <float literal>`` as ``x * (1/<literal>)``, which is a
    different float32 result whenever the reciprocal is inexact.  It cost
    four kernels a one-ulp divergence from the CPU authority at once, and
    the contract deck is the only reason it was caught.  A literal divisor
    reintroduces exactly that defect, so the guard is a grep.

    Since then the boundary has been located (#355,
    ``tests/test_nvrtc_reciprocal_rewrite.py``): the rewrite is a property of
    the TARGET, not of one card or one CuPy build -- NVRTC divides for every
    target up to ``compute_90`` and multiplies by the reciprocal from
    ``compute_100`` up.  Either remedy holds, because both give the host a
    way to write the divisor and so forbid folding: a runtime kernel
    argument, which is what this translation unit uses, or a ``__constant__``
    translation-unit symbol, which is what the shared units use."""

    def divisors(source: str) -> list[str]:
        """Top-level second argument of every ``mpas_div`` call."""

        found = []
        for start in (m.end() for m in re.finditer(r"mpas_div\(", source)):
            depth = 1
            comma = None
            index = start
            while index < len(source) and depth:
                character = source[index]
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                elif character == "," and depth == 1 and comma is None:
                    comma = index
                index += 1
            if comma is not None:
                found.append(source[comma + 1 : index - 1].strip())
        return found

    offenders = [
        divisor
        for divisor in divisors(CUDA_REGIONAL_SOURCE)
        if re.fullmatch(r"-?\d+(\.\d*)?f", divisor)
    ]
    assert offenders == [], (
        "literal float divisors found; give the host a way to write the "
        "divisor -- a runtime kernel argument or a __constant__ "
        f"translation-unit symbol -- instead: {offenders}"
    )
    # The plain operator is equally exposed and is not used for division in
    # this translation unit.
    assert not re.search(r"/\s*-?\d+(\.\d*)?f", CUDA_REGIONAL_SOURCE)

def test_an_unminted_configuration_class_refuses_by_name():
    """The gate it was ruled on 2026-08-25, re-keyed 2026-08-27.

    An anchor has two halves.  This is the FORECAST half: a configuration
    nobody has ever run twice and compared has no mint, and the refusal has
    to say so rather than saying "your cull is new".
    """

    summary = regional_admission.admitted_summary()
    assert "graded-4457m-dt20-z7" in summary
    assert "conus-x1-120km-dt120-z7" in summary
    with pytest.raises(RuntimeError) as excinfo:
        regional_admission.require_regional_anchor(
            "unregistered-region.2971",
            bdy_mask_sha256="00" * 32,
            n_cells=2971,
            boundary_zone_width=7,
            n_vert_levels=55,
            # A timestep and a resolution no class was ever minted at.
            finest_edge_m=1234.5,
            dt_seconds=7.0,
        )
    message = str(excinfo.value)
    # The gate law: a gate names the concrete breakage it prevents.
    assert "holds no forecast mint" in message
    assert "reproduces itself" in message
    assert "--mint-anchor" in message
    assert "unregistered-region.2971" in message


def test_a_new_cull_of_a_minted_class_refuses_for_its_deck_and_nothing_else():
    """THE RE-KEYING, in one test, in both directions.

    A cull the cascade placed this cycle belongs to a class that IS minted --
    same parent finest edge, same seven rings, same 55 levels, same timestep,
    same kernel bytes.  It must NOT be told to mint a second forecast pair,
    which cost 5.5 to 8.7 minutes of card per re-placed swath.  It must be
    told to run its own contract deck, because the 22 regional kernels are
    indexed by ring and no deck has ever looked at these rings.
    """

    with pytest.raises(RuntimeError) as excinfo:
        regional_admission.require_regional_anchor(
            None,
            bdy_mask_sha256="1a" * 32,
            n_cells=13_402,
            boundary_zone_width=7,
            n_vert_levels=55,
            finest_edge_m=4457.233,
            dt_seconds=20.0,
        )
    message = str(excinfo.value)
    assert "no contract deck has been run" in message
    assert "run_cuda_regional_contract.py" in message
    # And it names the class as MINTED, so nobody reads this as "mint again".
    assert "'graded-4457m-dt20-z7' IS minted" in message
    assert "--mint-anchor" not in message

    # The other direction: the SAME class key on a geometry that has a deck
    # is admitted, and the anchor it returns names both halves.
    anchor = regional_admission.require_regional_anchor(
        "r4.75.14050",
        bdy_mask_sha256=(
            "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
        ),
        n_cells=14_050,
        boundary_zone_width=7,
        n_vert_levels=55,
        finest_edge_m=4457.233,
        dt_seconds=20.0,
    )
    assert anchor.class_id == "graded-4457m-dt20-z7"
    assert anchor.contract_route == "shipped"
    assert "CLASS graded-4457m-dt20-z7" in anchor.basis
    assert "GEOMETRY" in anchor.basis


def test_the_class_key_is_measured_from_the_tree_not_declared():
    """A mint is a statement about bytes, and this is what re-reads them.

    Nothing checked the mint's recorded source digests at admission before
    2026-08-27: the receipt wrote them down and no gate ever looked.  If a
    regional translation unit moves, every class lapses until it is re-minted,
    and this test is what makes the pinned constant a measurement rather than
    a declaration.
    """

    measured = regional_admission.kernel_set_sha256()
    assert measured == regional_admission.MINTED_KERNEL_SET_SHA256, (
        "a regional translation unit's bytes have moved since the shipped "
        "classes were minted.  Re-mint each class in ADMITTED_CLASSES and "
        "update MINTED_KERNEL_SET_SHA256 to " + measured
    )
    for name in regional_admission.REGIONAL_KERNEL_SOURCES:
        assert (ROOT / name).is_file(), name
    for row in regional_admission.ADMITTED_CLASSES.values():
        assert row.key.kernel_set_sha256 == measured, row.class_id


def test_a_moved_kernel_set_lapses_every_class():
    """The teeth of the previous test, exercised rather than asserted."""

    key = regional_admission.RegionalClassKey.build(
        boundary_zone_width=7,
        n_vert_levels=55,
        finest_edge_m=4457.233,
        dt_seconds=20.0,
        kernel_set="ff" * 32,
    )
    assert regional_admission.admitted_class_for_key(key) is None
    with pytest.raises(RuntimeError, match="those bytes have moved"):
        regional_admission.require_regional_anchor(
            "r4.75.14050",
            bdy_mask_sha256=(
                "a8e66046452db881bb4a9da08952610207ee5aa2e0a58d48b1d2348b48f84088"
            ),
            n_cells=14_050,
            kernel_set="ff" * 32,
        )


def test_unanchored_refusal_names_a_remedy_that_exists():
    """A refusal that names a tool nobody can run is the defect L4 fixed in
    the transport sentinel pair.  Every path this message names must be in
    the tree."""

    message = regional_admission.uncontracted_geometry_refusal(
        "ab" * 32, 4_440, "graded-4457m-dt20-z7"
    )
    assert "SHIPPED_CONTRACTS" in message
    assert "run_cuda_regional_contract.py" in message
    assert (ROOT / "tools/run_cuda_regional_contract.py").exists()

    minted = regional_admission.unanchored_class_refusal(
        regional_admission.RegionalClassKey.build(
            boundary_zone_width=7,
            n_vert_levels=55,
            finest_edge_m=1.0,
            dt_seconds=1.0,
            kernel_set="00" * 32,
        ),
        None,
    )
    assert "ADMITTED_CLASSES" in minted
    assert "run_cuda_regional_forecast.py" in minted
    assert (ROOT / "tools/run_cuda_regional_forecast.py").exists()


def test_every_registered_anchor_names_evidence_that_exists(receipts):
    """An anchor is earned, not declared: the evidence it names must be in
    this repository -- both halves, and separately, because they are now
    earned separately."""

    assert regional_admission.SHIPPED_CONTRACTS, (
        "the registry lost its contracted geometries; a vacuous pass here "
        "would let an anchor name evidence that is not in the tree"
    )
    assert regional_admission.ADMITTED_CLASSES
    for digest, contract in regional_admission.SHIPPED_CONTRACTS.items():
        assert contract.bdy_mask_sha256 == digest
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert (ROOT / contract.contract_receipt).exists(), contract.contract_receipt
        assert contract.n_cells > 0
        assert contract.boundary_zone_width == regional_v841.N_BDY_ZONE
        assert contract.class_id in regional_admission.ADMITTED_CLASSES
    for row in regional_admission.ADMITTED_CLASSES.values():
        assert len(row.mint_receipts) >= 2, row.class_id
        for receipt in row.mint_receipts:
            assert (ROOT / receipt).exists(), receipt
        assert row.key.boundary_zone_width == regional_v841.N_BDY_ZONE
        # Every class states what it does NOT admit, on the row.
        assert "DRY regional dycore" in row.basis


def test_digest_and_cell_count_mismatches_refuse_separately():
    """A contract deck speaks for one zone geometry and one cull.

    This half of the anchor did NOT move on 2026-08-27 and must not: a deck
    run on another cull's rings measured another cull's zones.
    """

    contract = regional_admission.RegionalContract(
        bdy_mask_sha256="ab" * 32,
        n_cells=10,
        boundary_zone_width=regional_v841.N_BDY_ZONE,
        class_id="graded-4457m-dt20-z7",
        mesh_row="test-row",
        card="test",
        admitted_on="2026-08-27",
        contract_receipt="evidence/nowhere",
        basis="unit test",
    )
    original = regional_admission.SHIPPED_CONTRACTS
    try:
        regional_admission.SHIPPED_CONTRACTS = {  # type: ignore[misc]
            "ab" * 32: contract
        }
        with pytest.raises(RuntimeError, match="zone geometry"):
            regional_admission.require_regional_anchor(
                "test-row", bdy_mask_sha256="cd" * 32, n_cells=10
            )
        with pytest.raises(RuntimeError, match="different domain"):
            regional_admission.require_regional_anchor(
                "test-row", bdy_mask_sha256="ab" * 32, n_cells=11
            )
        admitted = regional_admission.require_regional_anchor(
            "test-row", bdy_mask_sha256="ab" * 32, n_cells=10
        )
        assert admitted.contract_receipt == "evidence/nowhere"
        assert admitted.class_id == "graded-4457m-dt20-z7"
    finally:
        regional_admission.SHIPPED_CONTRACTS = original  # type: ignore[misc]


def test_a_presented_receipt_is_checked_by_content_not_by_name(tmp_path):
    """The route a cycling cascade needs, and every way it is refused.

    A cull placed this cycle cannot be a row in a file written before it
    existed.  So its own contract deck's receipt admits it -- but only if the
    receipt says, in its own recorded numbers, everything a shipped row's
    basis sentence says.
    """

    import json

    kernel_set = regional_admission.kernel_set_sha256()
    good = {
        "instrument": "run_cuda_regional_contract",
        "all_decks_bitwise": True,
        "all_kernels_covered": True,
        "all_controls_have_teeth": True,
        "dual_run_identical": True,
        "decks_selected": False,
        "bdy_mask_sha256": "1a" * 32,
        "n_cells": 13_402,
        "boundary_zone_width": 7,
        "kernel_set_sha256": kernel_set,
        "class_id": "graded-4457m-dt20-z7",
        "date_utc": "2026-08-27T00:00:00Z",
        "card": "RTX 5090",
    }
    (tmp_path / "contract.json").write_text(json.dumps(good), encoding="utf-8")

    anchor = regional_admission.require_regional_anchor(
        None,
        bdy_mask_sha256="1a" * 32,
        n_cells=13_402,
        boundary_zone_width=7,
        n_vert_levels=55,
        finest_edge_m=4457.233,
        dt_seconds=20.0,
        contract_directories=(tmp_path,),
    )
    assert anchor.contract_route == "presented"
    assert anchor.class_id == "graded-4457m-dt20-z7"

    # Each defect refuses on its own, and the refusal says which one.
    for key, broken, expected in (
        ("all_decks_bitwise", False, "8 decks bitwise"),
        ("all_kernels_covered", False, "22 declared regional kernels"),
        ("all_controls_have_teeth", False, "deliberately wrong zone geometry"),
        ("dual_run_identical", False, "reproducing across two runs"),
        ("decks_selected", True, "subset of the deck set"),
        ("instrument", "hand-written", "run_cuda_regional_contract.py"),
        ("kernel_set_sha256", "ff" * 32, "different bytes"),
        ("bdy_mask_sha256", "cc" * 32, "different zone geometry"),
        ("n_cells", 99, "cells"),
    ):
        document = dict(good)
        document[key] = broken
        (tmp_path / "contract.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        with pytest.raises(RuntimeError) as excinfo:
            regional_admission.require_regional_anchor(
                None,
                bdy_mask_sha256="1a" * 32,
                n_cells=13_402,
                boundary_zone_width=7,
                n_vert_levels=55,
                finest_edge_m=4457.233,
                dt_seconds=20.0,
                contract_directories=(tmp_path,),
            )
        assert expected in str(excinfo.value), key


def test_an_unmeasured_finest_edge_cannot_grow_its_class(tmp_path):
    """The one class that does not key on a resolution pays for it.

    The 120 km reference cull's grid publishes unit-sphere ``dcEdge`` and its
    Earth-scaled metrics are in a static no copy of this repository holds, so
    the class declares its finest edge NOT MEASURED.  That looseness is
    bought back by refusing every geometry that is not already a shipped row.
    """

    import json

    row = regional_admission.ADMITTED_CLASSES["conus-x1-120km-dt120-z7"]
    assert not row.key.finest_edge_measured
    assert row.key.as_dict()["finest_edge_m"] is None
    assert "NOT MEASURED" in row.basis

    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "instrument": "run_cuda_regional_contract",
                "all_decks_bitwise": True,
                "all_kernels_covered": True,
                "all_controls_have_teeth": True,
                "dual_run_identical": True,
                "decks_selected": False,
                "bdy_mask_sha256": "2b" * 32,
                "n_cells": 4_100,
                "boundary_zone_width": 7,
                "kernel_set_sha256": regional_admission.kernel_set_sha256(),
                "class_id": "conus-x1-120km-dt120-z7",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="never measured"):
        regional_admission.require_regional_anchor(
            None,
            bdy_mask_sha256="2b" * 32,
            n_cells=4_100,
            boundary_zone_width=7,
            n_vert_levels=55,
            finest_edge_m=113_137.085,
            dt_seconds=120.0,
            contract_directories=(tmp_path,),
        )


def test_cuda_host_validation_routes_a_regional_mesh_through_the_gate():
    """A culled regional mesh must refuse at the CUDA door by name, not die
    in an unnamed range check or an out-of-bounds device read."""

    import numpy as np

    from hexcore.cuda_driver import (
        _validate_v841_host_mesh,
        regional_bdy_mask_digest,
    )
    from hexcore.errors import ConfigurationRefusal

    class _Mesh:
        def __init__(self) -> None:
            self.arrays = {
                "areaCell": np.ones(3, dtype=np.float64),
                "dcEdge": np.ones(2, dtype=np.float64),
                "areaTriangle": np.ones(2, dtype=np.float64),
                # One ring-7 edge carrying the culler's stored-0 sentinel.
                "cellsOnEdge": np.array([[0, 1], [1, -1]], dtype=np.int64),
                "edgesOnCell": np.zeros((3, 3), dtype=np.int64),
                "bdyMaskCell": np.array([0, 6, 7], dtype=np.int32),
                "bdyMaskEdge": np.array([5, 7], dtype=np.int32),
                "bdyMaskVertex": np.array([6, 7], dtype=np.int32),
            }
            self.dimensions = {"nCells": 3, "nEdges": 2, "nVertices": 2}

    mesh = _Mesh()
    assert regional_bdy_mask_digest(mesh) is not None
    with pytest.raises(ConfigurationRefusal) as excinfo:
        _validate_v841_host_mesh(mesh)
    message = str(excinfo.value)
    assert "regional CUDA execution is refused" in message
    assert "no contract deck has been run" in message
    assert "closed/global" not in message
