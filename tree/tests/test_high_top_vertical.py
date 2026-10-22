"""High-top grid, runtime controls and registry must describe the same column."""
import hashlib
import json
from pathlib import Path

import pytest

from hexcore.cli import build_parser
from hexcore import global_door, s2s_door, mesh_rows
from hexcore.vertical_spec import VerticalSpec, VerticalSpecError


SPEC = global_door.DATA / "countries/s2s80-v1.json"


def arguments(tmp_path, *extra):
    return build_parser().parse_args(["s2s", "--row", "global-120km-high-top",
        "--source", "era5", "--cycle", "2018-02-05T00", "--days", "30", "--members", "1",
        "--work-dir", str(tmp_path / "run"), "--cache-dir", str(tmp_path / "cache"), *extra])


def test_high_top_row_binds_grid_sponge_radiation_and_each_member(tmp_path):
    value = s2s_door.plan(arguments(tmp_path))
    spec = VerticalSpec.from_file(SPEC)
    assert spec.n_vert_levels == 80 and spec.ztop_m == 50000
    assert len(spec.specified_interfaces_m) == 81
    heights = spec.specified_interfaces_m
    assert all(b - a == 750 for a, b in zip(heights[40:], heights[41:]))
    assert spec.rayleigh_damping_start_m == 40000
    assert spec.rayleigh_xnutr == .2
    assert value["vertical"]["sha256"] == spec.sha256()
    assert value["wall_estimate"] is None and value["cards"]["selected_cards"] is None
    argv = value["commands"]["forecast"]
    parsed = build_parser().parse_args(argv)
    assert parsed.cam_coefficient == .2 and parsed.cam_damping_levels == 8
    assert parsed.rayleigh_u and parsed.damping_start_m == 40000
    assert parsed.radiation_seconds == 1800
    assert "config_radt_lw_scheme=rrtmg_lw" in parsed.physics
    assert "config_radt_sw_scheme=rrtmg_sw" in parsed.physics
    assert "config_xnutr=0.2" in parsed.dynamics
    assert "config_number_rayleigh_damp_u_levels=6" in parsed.dynamics
    member = value["members"][0]
    assert "-v" + spec.sha256()[:12] in member["init_dir"]
    assert member["init_argv"][member["init_argv"].index("--vertical-spec") + 1] == str(SPEC)
    assert value["paths"]["mesh_rows"] == str(tmp_path / "run/mesh-rows.json")
    assert not list(tmp_path.iterdir())


def test_runtime_controls_reach_the_production_config(tmp_path):
    from hexcore.drivers.run_cuda_v841_forecast import build_forecast_config, parse_dynamics_overrides
    args = build_parser().parse_args(s2s_door.plan(arguments(tmp_path))["commands"]["forecast"])
    config = build_forecast_config(dt_seconds=300, cam_coefficient=args.cam_coefficient,
        cam_damping_levels=args.cam_damping_levels, rayleigh_u=args.rayleigh_u,
        damping_start_m=args.damping_start_m, radiation_seconds=args.radiation_seconds,
        dynamics_overrides=parse_dynamics_overrides(args.dynamics))
    config.validate()
    assert config.config_zd == 40000 and config.config_xnutr == .2
    assert config.config_number_rayleigh_damp_u_levels == 6
    assert config.config_rayleigh_damp_u_timescale_days == 5


def test_v1_canonical_hash_and_default_plan_stay_unchanged(tmp_path):
    path = global_door.DATA / "countries/tc55-v1.json"
    raw = json.loads(path.read_text())
    old_bytes = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    spec = VerticalSpec.from_file(path)
    assert spec.sha256() == hashlib.sha256(old_bytes).hexdigest()
    assert spec.forecast_arguments() == []
    value = s2s_door.plan(arguments(tmp_path, "--row", "global-120km"))
    assert value["vertical"]["n_vert_levels"] == 55
    assert value["vertical"]["mesh_name"] == "hex-x1.40962-native"


def test_explicit_vertical_changes_restart_identity_and_init_cache(tmp_path):
    low = s2s_door.plan(arguments(tmp_path, "--row", "global-120km"))
    high = s2s_door.plan(arguments(tmp_path, "--row", "global-120km", "--vertical-spec", str(SPEC)))
    assert low["signature"] != high["signature"]
    assert low["members"][0]["init_dir"] != high["members"][0]["init_dir"]
    assert low["vertical"]["mesh_name"] != high["vertical"]["mesh_name"]


def test_registry_uses_selected_level_count_and_identity(tmp_path, monkeypatch):
    value = s2s_door.plan(arguments(tmp_path))
    grid, static = tmp_path / "grid.nc", tmp_path / "static.nc"
    grid.write_bytes(b"grid")
    static.write_bytes(b"static")
    monkeypatch.setattr(mesh_rows, "_dimensions", lambda _: {"nCells": 40962, "nEdges": 122880})
    captured = []
    monkeypatch.setattr(mesh_rows, "write_rows", lambda path, rows: captured.extend(rows))
    global_door._register(value, grid, static, Path(value["paths"]["mesh_rows"]), tmp_path / "static.json")
    row = captured[0]
    assert row.n_levels == 80 and row.n_interfaces == 81
    assert row.name == value["vertical"]["mesh_name"]


def test_high_top_does_not_inherit_55_level_resident_member_limits(tmp_path, monkeypatch):
    from hexcore.global_ensemble import group_limit
    from hexcore import global_ensemble
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [96 * 2**30])
    # The planner also prices the card's free memory (a child-process probe); an empty 96 GB card makes
    # the co-tenancy table the binding limit, so the result does not depend on the test host's card.
    monkeypatch.setattr(global_ensemble, "_probe_cards", lambda devices: [])
    args = arguments(tmp_path)
    assert group_limit(s2s_door.plan(args), args) == 1
    low = s2s_door.plan(arguments(tmp_path, "--row", "global-120km"))
    assert group_limit(low, args) == 4


def test_prepared_init_cannot_bypass_selected_vertical_binding(tmp_path):
    init = tmp_path / "init.nc"
    init.write_bytes(b"existing init")
    (tmp_path / "init-source-receipt.json").write_text(json.dumps({
        "source": "era5", "cycle": "2018-02-05T00", "member": None}))
    args = arguments(tmp_path, "--prepared-init", f"2018-02-05T00={init}")
    value = s2s_door.plan(args)
    provenance = init.with_name("init.nc.provenance.json")
    with pytest.raises(s2s_door.S2SError, match="requires vertical provenance"):
        s2s_door._inits(value, args, {}, None)
    provenance.write_text(json.dumps({"vertical_source": {"summary": {
        "vertical_spec_canonical_sha256": "low-top"}}}))
    with pytest.raises(s2s_door.S2SError, match="vertical spec mismatch"):
        s2s_door._inits(value, args, {}, None)
    provenance.write_text(json.dumps({"vertical_source": {"summary": {
        "vertical_spec_canonical_sha256": value["vertical"]["sha256"]}}}))
    class Status:
        def stage(self, *args, **kwargs):
            pass
    result = s2s_door._inits(value, args, {}, Status())
    assert result["built"] == 0


@pytest.mark.parametrize("key,value", [("cam_damping_levels", 81), ("rayleigh_u_levels", True),
    ("radiation_seconds", 0), ("ozone", "constant"), ("radiation_lw", "off")])
def test_invalid_runtime_cannot_enter_a_plan(key, value):
    raw = json.loads(SPEC.read_text())
    raw["runtime"][key] = value
    with pytest.raises(VerticalSpecError):
        VerticalSpec.from_mapping(raw)


def test_runtime_requires_versioned_schema():
    raw = json.loads(SPEC.read_text())
    raw["schema"] = "gpuwm-hex.vertical-spec/v1"
    with pytest.raises(VerticalSpecError, match="require vertical-spec/v2"):
        VerticalSpec.from_mapping(raw)


def test_high_top_memory_floor_and_roster_use_the_initialized_column(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from netCDF4 import Dataset
    from hexcore import forecast_door, device_admission
    init = tmp_path / "init.nc"
    with Dataset(init, "w") as ds:
        ds.createDimension("nVertLevels", 80)
        ds.createDimension("nVertLevelsP1", 81)
    card = device_admission.KNOWN_CARDS["32gib-170sm"]
    monkeypatch.setattr(forecast_door, "read_card_profile", lambda: card)
    base = device_admission.model_for_card(card)
    model = forecast_door.resolve_admission_model(SimpleNamespace(
        init=init, model=None, lbc_dir=None, stop_on_refusal=False))
    assert model.levels == 80 and not model.measured
    assert model.bytes_per_cell == pytest.approx(base.bytes_per_cell * 80 / 55)
    assert model.required_bytes(40962) > base.required_bytes(40962)
    extra = forecast_door.stochastic_array_bytes(40962, '{"sppt":true}', levels=model.levels)
    verdict = forecast_door.admission_verdict(mesh="high", cells=40962,
        free_bytes=96 * 1024**3, total_bytes=96 * 1024**3, headroom_bytes=None,
        registry={}, model=model, stochastic='{"sppt":true}')
    assert verdict.stochastic_array_bytes == extra
    args = dict(mesh_kind="global", cells=40962, levels=model.levels,
        card_bytes=96 * 1024**3, free_bytes=96 * 1024**3,
        solo_required_bytes=model.required_bytes(40962))
    assert device_admission.admit_co_tenancy(members=1, **args)["levels"] == 80
    with pytest.raises(MemoryError, match="table limit 1"):
        device_admission.admit_co_tenancy(members=2, **args)
