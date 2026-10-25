"""Either physics route's history frames are found by the doors that count them."""

from __future__ import annotations

import json

from hexcore.forecast_door import history_files


def _touch(path):
    path.write_bytes(b"")
    return path


def test_arwen_folder_reads_cuda_history_only(tmp_path):
    frames = [_touch(tmp_path / f"cuda-history.2025-05-19_1{h}.00.00.nc") for h in (2, 3)]
    _touch(tmp_path / "history.2025-05-19_12.00.00.nc")  # not the ArWen route's: never counted there
    assert history_files(tmp_path) == frames


def test_mpas_suite_folder_reads_its_output_stream(tmp_path):
    (tmp_path / "native-physics-request.json").write_text(json.dumps({"streams": [
        {"name": "output", "filename": "history.$Y-$M-$D_$h.$m.$s.nc", "directions": ["output"]},
        {"name": "diagnostics", "filename": "diag.$Y-$M-$D_$h.$m.$s.nc", "directions": ["output"]}]}))
    frames = [_touch(tmp_path / f"history.2025-05-19_1{h}.00.00.nc") for h in (2, 3, 4)]
    _touch(tmp_path / "diag.2025-05-19_12.00.00.nc")
    _touch(tmp_path / ".history.2025-05-19_15.00.00.nc.0123.tmp")
    assert history_files(tmp_path) == frames


def test_s2s_resume_finds_either_routes_checkpoint(tmp_path):
    from hexcore.s2s_door import latest_checkpoint

    member = {"run_id": "m1", "start_time": "2025-05-19_12:00:00", "hours": 48}
    root = tmp_path / "a0001" / "m1"
    root.mkdir(parents=True)
    (root / "checkpoint.f024.pkl").write_bytes(b"x")          # MPAS suite route
    (root / "checkpoint.f048.pkl").write_bytes(b"x")          # at the end: never a resume point
    assert latest_checkpoint(member, [{"out": str(tmp_path / "a0001")}]) == root / "checkpoint.f024.pkl"
    (root / "checkpoint.2025-05-20_18.00.00.pkl").write_bytes(b"x")  # ArWen route, later
    assert latest_checkpoint(member, [{"out": str(tmp_path / "a0001")}]) == root / "checkpoint.2025-05-20_18.00.00.pkl"


def test_export_reads_either_routes_wind_names():
    from types import SimpleNamespace

    from hexcore.ensemble import history_name

    arwen = SimpleNamespace(variables={"normal_u": 0, "u_zonal": 0, "v_meridional": 0, "u": 0})
    mpas = SimpleNamespace(variables={"u": 0, "uReconstructZonal": 0, "uReconstructMeridional": 0})
    assert [history_name(arwen, t) for t in ("u", "uReconstructZonal", "uReconstructMeridional")] == \
        ["normal_u", "u_zonal", "v_meridional"]
    assert [history_name(mpas, t) for t in ("u", "uReconstructZonal", "uReconstructMeridional")] == \
        ["u", "uReconstructZonal", "uReconstructMeridional"]
    assert history_name(SimpleNamespace(variables={}), "uReconstructZonal") == "u_zonal"  # refusal names the alias
