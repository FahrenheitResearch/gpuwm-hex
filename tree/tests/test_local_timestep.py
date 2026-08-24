"""Local time stepping: classing, derivation fidelity, and the default-off shape.

These are CPU-only checks.  The bit-identity and conservation gates are run
against the exe on a real mesh; nothing here substitutes for those.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpas_port.config_lts import (
    DEFAULT_LOCAL_TIMESTEP_RATES,
    V841LocalTimestepDryConfig,
    V841LocalTimestepGwdoConfig,
    V841LocalTimestepSmagorinskyGwdoConfig,
    local_timestep_enabled,
)
from mpas_port.config_v841 import (
    V841DryDycoreConfig,
    V841MpasColumnPhysicsGwdoConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from mpas_port.errors import ConfigurationRefusal
from mpas_port.lts_v841 import (
    admissible_rates,
    cell_min_spacing,
    classify_local_timestep,
)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------
def test_released_schedule_admits_one_and_three_only():
    # (1, 3, 6) is the released schedule from integration.RKSchedule.from_mpas.
    # 2 and 4 do not divide the RK2 stage's three sub-steps, so a rate-2 or
    # rate-4 class would have to change that stage's sub-step count.
    assert admissible_rates((1, 3, 6)) == (1, 3)


def test_single_substep_stage_does_not_collapse_the_ladder():
    # RK1 runs one sub-step; requiring r | 1 would leave only rate 1 and turn
    # the whole feature into a silent no-op.
    assert admissible_rates((1, 1, 4)) == (1, 2, 4)


def test_ladder_refuses_a_zero_substep_stage():
    with pytest.raises(ConfigurationRefusal):
        admissible_rates((1, 0, 6))


# --------------------------------------------------------------------------
# h_c
# --------------------------------------------------------------------------
def test_cell_min_spacing_hand_case():
    dc = np.array([3.0, 7.0, 11.0])
    edges_on_cell = np.array([[1, 2, 0], [2, 3, 0]])  # 1-based, 0 padded
    counts = np.array([2, 2])
    h = cell_min_spacing(dc, edges_on_cell, counts)
    assert h.tolist() == [3.0, 7.0]


def test_cell_min_spacing_ignores_padding_beyond_the_count():
    dc = np.array([3.0, 7.0, 1.0])
    edges_on_cell = np.array([[1, 2, 3]])
    counts = np.array([2])  # slot 3 (dcEdge 1.0) is past the count
    assert cell_min_spacing(dc, edges_on_cell, counts).tolist() == [3.0]


# --------------------------------------------------------------------------
# classing
# --------------------------------------------------------------------------
def _chain(spacings):
    """A 1-D chain of cells: cell i shares edge i with cell i+1."""

    n_cells = len(spacings)
    dc = np.asarray(spacings, dtype=float)
    n_edges = n_cells - 1
    cells_on_edge = np.array(
        [[i + 1, i + 2] for i in range(n_edges)], dtype=np.int64
    )
    edges_on_cell = np.zeros((n_cells, 2), dtype=np.int64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for edge in range(n_edges):
        for cell in (edge, edge + 1):
            edges_on_cell[cell, counts[cell]] = edge + 1
            counts[cell] += 1
    return dc[:n_edges], edges_on_cell, counts, cells_on_edge


def test_uniform_mesh_is_a_single_class_with_the_identity_permutation():
    dc, eoc, counts, coe = _chain([100.0] * 9)
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
    )
    assert classing.is_single_class
    assert classing.identity_permutation
    assert classing.interface_edges.size == 0
    assert classing.arithmetic_acoustic_saving((1, 3, 6)) == 0.0


def test_a_ratio_below_the_next_rung_stays_in_the_fine_class():
    # 2.9x is not admissible for a rate-3 sub-step.  Rounding it up would hand
    # a cell a step its own spacing does not permit.
    dc, eoc, counts, coe = _chain([10.0, 29.0, 29.0, 29.0])
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
    )
    assert classing.is_single_class


def test_a_coarse_run_earns_a_coarse_class_and_an_interface():
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    classing = classify_local_timestep(
        dc_edge=dc,
        edges_on_cell=eoc,
        n_edges_on_cell=counts,
        cells_on_edge=coe,
        rates=(1, 3),
        buffer_rings=1,
    )
    assert classing.rates == (1, 3)
    assert classing.interface_edges.size >= 1
    # An edge always runs at the finer of its two cells' rates, so the fine
    # side's flux is sampled at the fine rate and the reflux has something to
    # accumulate.
    c0 = coe[:, 0] - 1
    c1 = coe[:, 1] - 1
    expected = np.minimum(classing.cell_rate[c0], classing.cell_rate[c1])
    assert np.array_equal(classing.edge_rate, expected)
    assert classing.arithmetic_acoustic_saving((1, 3, 6)) > 0.0


def test_the_buffer_ring_widens_the_fine_class():
    dc, eoc, counts, coe = _chain([10.0] + [50.0] * 8)
    narrow = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    wide = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=3,
    )
    assert int((wide.cell_rate == 1).sum()) >= int((narrow.cell_rate == 1).sum())


def test_no_cell_is_handed_a_step_its_own_spacing_forbids():
    rng = np.random.default_rng(20260821)
    spacings = rng.uniform(10.0, 300.0, size=200)
    dc, eoc, counts, coe = _chain(spacings.tolist())
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    ratio = classing.h_cell / classing.h_min
    assert np.all(classing.cell_rate <= ratio + 1e-9)


def test_classing_refuses_a_ladder_that_does_not_start_at_one():
    dc, eoc, counts, coe = _chain([10.0] * 5)
    with pytest.raises(ConfigurationRefusal):
        classify_local_timestep(
            dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
            cells_on_edge=coe, rates=(2, 4),
        )


def test_classing_refuses_a_zero_buffer():
    dc, eoc, counts, coe = _chain([10.0] * 5)
    with pytest.raises(ConfigurationRefusal):
        classify_local_timestep(
            dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
            cells_on_edge=coe, rates=(1, 3), buffer_rings=0,
        )


# --------------------------------------------------------------------------
# the option shape
# --------------------------------------------------------------------------
def test_the_switch_is_off_by_default():
    for config in (
        V841LocalTimestepDryConfig(),
        V841LocalTimestepGwdoConfig(),
        V841LocalTimestepSmagorinskyGwdoConfig(),
    ):
        assert config.config_local_timestep is False
        assert local_timestep_enabled(config) is False
        config.validate()


def test_the_switch_off_leaves_every_pinned_knob_where_it_was():
    for lts_type, parent_type in (
        (V841LocalTimestepDryConfig, V841DryDycoreConfig),
        (V841LocalTimestepGwdoConfig, V841MpasColumnPhysicsGwdoConfig),
        (
            V841LocalTimestepSmagorinskyGwdoConfig,
            V841MpasColumnPhysicsSmagorinskyGwdoConfig,
        ),
    ):
        lts = lts_type()
        parent = parent_type()
        for field in parent.__dataclass_fields__:
            assert getattr(lts, field) == getattr(parent, field), field
        # config_dt in particular: local time stepping does not move the model
        # timestep, it changes how many acoustic sub-steps a column runs.
        assert lts.config_dt == parent.config_dt

    # The column-physics lane's dt is pinned at 120 s and the subtype inherits
    # it unchanged; the dry lane leaves dt free, which is what lets the referee
    # arm run the same mesh at a globally smaller acoustic step.
    assert V841LocalTimestepSmagorinskyGwdoConfig().config_dt == 120.0
    assert V841LocalTimestepDryConfig(config_dt=7.5).config_dt == 7.5


def test_the_default_ladder_is_two_classes():
    assert DEFAULT_LOCAL_TIMESTEP_RATES == (1, 3)
    assert len(DEFAULT_LOCAL_TIMESTEP_RATES) == 2


def test_the_dry_lane_carries_the_block_and_stays_dry():
    # The conservation gate is defined for a no-physics run with qv as a
    # passive scalar; a subtype that quietly turned physics back on would be
    # measuring the microphysics budget instead of the transport.
    config = V841LocalTimestepDryConfig(config_local_timestep=True)
    config.validate()
    assert config.config_physics_suite == "none"
    assert config.config_moist_physics is False
    assert local_timestep_enabled(config) is True


def test_the_declared_off_arm_is_the_subtype_with_the_switch_off():
    # GATE 2's arm: the local-timestep configuration TYPE with the switch off.
    # Selecting the parent type instead would prove nothing, because then the
    # option's own code never entered the run at all.
    arm = V841LocalTimestepDryConfig(config_local_timestep=False)
    arm.validate()
    assert type(arm) is not V841DryDycoreConfig
    assert isinstance(arm, V841DryDycoreConfig)
    assert local_timestep_enabled(arm) is False


def test_a_ladder_that_skips_rate_one_is_refused():
    with pytest.raises(ConfigurationRefusal):
        V841LocalTimestepGwdoConfig(
            config_local_timestep=True, config_local_timestep_rates=(3, 6)
        ).validate()


def test_a_safety_factor_above_one_is_refused():
    with pytest.raises(ConfigurationRefusal):
        V841LocalTimestepGwdoConfig(
            config_local_timestep=True,
            config_local_timestep_safety_factor=1.5,
        ).validate()


# --------------------------------------------------------------------------
# derivation fidelity: the LTS kernel is its ancestor plus the gather, nothing else
# --------------------------------------------------------------------------
_GATHERED = (
    ("acoustic_coefficients_v841", "cell", "ncells"),
    ("acoustic_ru_v841", "edge", "nedges"),
    ("acoustic_prepare_v841", "cell", "ncells"),
    ("acoustic_rs_ts_v841", "cell", "ncells"),
    ("acoustic_column_solve_v841", "cell", "ncells"),
)


def test_every_derived_acoustic_kernel_is_its_ancestor_plus_the_gather():
    from mpas_port import cuda_acoustic_lts as lts
    from mpas_port import cuda_acoustic_v841

    source = cuda_acoustic_v841._CUDA_SOURCE
    for name, index, bound in _GATHERED:
        original = lts._extract_kernel(source, name)
        derived = lts._gather(
            original,
            name=name,
            index=index,
            bound=bound,
            new_name=name.replace("_v841", "_lts"),
        )
        # Undo exactly the three declared edits and require the ancestor back.
        restored = derived.replace(
            "const int lts_slot = blockDim.x * blockIdx.x + threadIdx.x;\n"
            "    if (lts_slot >= lts_n_active) return;\n"
            f"    const int {index} = lts_active[lts_slot];",
            f"const int {index} = blockDim.x * blockIdx.x + threadIdx.x;\n"
            f"    if ({index} >= {bound}) return;",
        )
        restored = restored.replace(
            ",\n    const int *lts_active, const int lts_n_active", ""
        )
        restored = restored.replace(name.replace("_v841", "_lts"), name)
        assert restored == original, name


def test_the_damping_kernel_is_its_ancestor_plus_the_gather():
    from mpas_port import cuda_acoustic_lts as lts
    from mpas_port import cuda_horizontal

    original = lts._extract_kernel(cuda_horizontal._CUDA_SOURCE, "divergence_damping_f32")
    derived = lts._gather(
        original,
        name="divergence_damping_f32",
        index="edge",
        bound="nedges",
        new_name="divergence_damping_lts",
    )
    restored = derived.replace(
        "const int lts_slot = blockDim.x * blockIdx.x + threadIdx.x;\n"
        "    if (lts_slot >= lts_n_active) return;\n"
        "    const int edge = lts_active[lts_slot];",
        "const int edge = blockDim.x * blockIdx.x + threadIdx.x;\n"
        "    if (edge >= nedges) return;",
    )
    restored = restored.replace(
        ",\n    const int *lts_active, const int lts_n_active", ""
    )
    restored = restored.replace("divergence_damping_lts", "divergence_damping_f32")
    assert restored == original


def test_a_moved_anchor_refuses_instead_of_silently_deriving_a_whole_domain_launch():
    from mpas_port import cuda_acoustic_lts as lts

    drifted = (
        'extern "C" __global__ void made_up(const int ncells)\n'
        "{\n"
        "    const int cell = threadIdx.x;\n"
        "    if (cell >= ncells) return;\n"
        "}\n"
    )
    with pytest.raises(lts.LocalTimestepSourceDrift):
        lts._gather(
            lts._extract_kernel(drifted, "made_up"),
            name="made_up",
            index="cell",
            bound="ncells",
            new_name="made_up_lts",
        )


def test_a_flux_corrected_limiter_is_refused_alongside_the_option():
    # Not reachable today -- the v8.4.1 CUDA lane refuses both knobs outright --
    # but admitting the FCT branch later must not let the option ride along on
    # an interface that was never proved with a limiter crossing it.
    import types

    from mpas_port import cuda_driver_lts

    for knob in ("config_monotonic", "config_positive_definite"):
        config = types.SimpleNamespace(
            config_local_timestep=True,
            config_monotonic=False,
            config_positive_definite=False,
            config_time_integration_order=3,
            config_number_of_sub_steps=6,
            config_local_timestep_rates=(1, 3),
            config_local_timestep_buffer_rings=1,
            config_local_timestep_safety_factor=1.0,
        )
        setattr(config, knob, True)
        driver = types.SimpleNamespace(
            config=config, v841_context=object(), halo_exchanger_v841=None
        )
        with pytest.raises(ConfigurationRefusal) as caught:
            cuda_driver_lts.attach_local_timestep(driver)
        assert knob in str(caught.value)


def test_the_reflux_settle_carries_no_atomics():
    # A coarse cell can own more than one interface edge -- 404 of 529 do on
    # the published x4.163842 mesh -- so a per-edge settle has to atomicAdd
    # into the shared cell.  Binary32 addition is not associative, and two
    # identical runs then differ: measured over 120 steps that seed reached
    # 59-99% of the domain.  The port's corruption screen on cards with no ECC
    # is the dual-run byte comparison, which a nondeterministic path defeats.
    from mpas_port import cuda_acoustic_lts as lts

    import re

    source = lts.local_timestep_cuda_source()
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    # The stripper has to work, or the test passes for the wrong reason.
    assert "atomicAdd" in source, "the explanatory comment names the hazard"
    assert "ONE THREAD PER COARSE CELL" not in code, "comments were not stripped"
    # No accumulating atomic anywhere in the translation unit.  The one atomic
    # that remains is the ancestor's ``atomicExch(singular, 1)``, which writes
    # the same constant from every racing thread and so has one outcome
    # whatever the order; it is pinned text and is not touched here.
    assert "atomicAdd" not in code
    assert "atomicSub" not in code
    assert code.count("atomicExch") == 1
    assert "atomicExch(singular, 1)" in code

    settle = lts._extract_kernel(source, "lts_reflux_settle")
    accumulate = lts._extract_kernel(source, "lts_reflux_accumulate")
    for name, body in (("settle", settle), ("accumulate", accumulate)):
        assert "atomic" not in body, name


def test_the_settle_grouping_keeps_every_interface_edge_exactly_once():
    import numpy as _np

    from mpas_port.lts_v841 import classify_local_timestep

    dc, eoc, counts, coe = _chain([10.0] * 3 + [50.0] * 6)
    classing = classify_local_timestep(
        dc_edge=dc, edges_on_cell=eoc, n_edges_on_cell=counts,
        cells_on_edge=coe, rates=(1, 3), buffer_rings=1,
    )
    iface = _np.asarray(classing.interface_edges)
    assert iface.size >= 1
    # Every interface edge has exactly one coarse side, so a grouping by coarse
    # cell partitions them.  Losing or duplicating one would silently drop or
    # double count mass at the boundary.
    c0 = coe[iface, 0] - 1
    c1 = coe[iface, 1] - 1
    coarse = _np.where(
        classing.cell_rate[c1] > classing.cell_rate[c0], c1, c0
    )
    order = _np.argsort(coarse, kind="stable")
    cells, starts = _np.unique(coarse[order], return_index=True)
    offsets = _np.append(starts, coarse.size)
    assert int(offsets[-1]) == int(iface.size)
    assert int(_np.diff(offsets).sum()) == int(iface.size)
    assert cells.size == _np.unique(coarse).size


def test_the_derived_translation_unit_declares_every_kernel_once():
    from mpas_port import cuda_acoustic_lts as lts

    source = lts.local_timestep_cuda_source()
    for name in (
        "acoustic_cofrz_v841",
        "acoustic_coefficients_lts",
        "acoustic_ru_lts",
        "acoustic_prepare_lts",
        "acoustic_rs_ts_lts",
        "acoustic_column_solve_lts",
        "divergence_damping_lts",
        "lts_reflux_accumulate",
        "lts_reflux_settle",
        "lts_recover_edges",
        "lts_recover_interfaces",
    ):
        assert source.count(f"void {name}(") == 1, name
