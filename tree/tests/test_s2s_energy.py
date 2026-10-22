"""Energy route configuration and fail-closed dated object coverage."""
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest
from hexcore import s2s_energy as energy
from hexcore.s2s_door import S2SError, plan
from hexcore.cli import build_parser


def test_route_preserves_explicit_thread_budget(monkeypatch):
    from hexcore import s2s_door
    monkeypatch.setattr(s2s_door.os, "sched_getaffinity", lambda _: set(range(256)))
    monkeypatch.setenv("RAYON_NUM_THREADS", "24")
    assert s2s_door.cores() == 24
    monkeypatch.setenv("RAYON_NUM_THREADS", "512")
    assert s2s_door.cores() == 256
    for value in ("0", "-1", "invalid"):
        monkeypatch.setenv("RAYON_NUM_THREADS", value)
        with pytest.raises(S2SError, match="positive integer"):
            s2s_door.cores()


def args(tmp_path, *extra):
    return build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs",
        "--cycle", "2023-03-31T12", "--members", "1", "--lagged", "0", "--days", "30",
        "--work-dir", str(tmp_path), *extra])


def test_default_energy_and_arbitrary_heights_are_in_restart_identity(tmp_path):
    a = args(tmp_path)
    assert a.history_hours == 6
    first = plan(a)
    assert first["energy"]["heights_m_agl"] == [80, 100]
    second = plan(args(tmp_path, "--hub-heights", "87.5,153"))
    assert second["signature"] != first["signature"]
    assert second["energy"]["heights_m_agl"] == [87.5, 153]
    with pytest.raises(S2SError, match="alias"):
        plan(args(tmp_path, "--history-hours", "24"))


@pytest.mark.parametrize("value", ["nan", "inf", "0", "10", "1001", "80,80", "junk"])
def test_invalid_heights_refuse(tmp_path, value):
    with pytest.raises(S2SError, match="height"):
        energy.energy_config(args(tmp_path, "--hub-heights", value))


def test_rows_validate_curve_and_units_and_content_identity(tmp_path):
    row = json.loads(energy.TABLE.read_text())
    row["rows"][2]["curve"][2][0] = 2
    path = tmp_path / "energy.json"
    path.write_text(json.dumps(row))
    with pytest.raises(S2SError, match="ordered"):
        energy.energy_config(args(tmp_path, "--energy-table", str(path)))
    row = json.loads(energy.TABLE.read_text())
    row["rows"][1]["units"] = "kW"
    path.write_text(json.dumps(row))
    with pytest.raises(S2SError, match="units"):
        energy.energy_config(args(tmp_path, "--energy-table", str(path)))


def test_every_dated_object_includes_endpoint_and_crosses_calendar():
    surface = {"source": "cfs", "forecast_cycle": "2024-02-28T18", "start": "2024-02-29_00:00:00",
               "end": "2024-03-30_00:00:00"}
    objects = energy.forcing_objects(surface)
    assert len(objects) == 121
    assert objects[-1]["valid"] == surface["end"]
    assert objects[-1]["lead_hours"] == 726
    assert "/cfs.20240228/18/" in objects[-1]["urls"][0]
    assert "flxf2024033000.01.2024022818.grb2" in objects[-1]["urls"][0]
    surface["forecast_cycle"] = "2024-03-01T00"
    with pytest.raises(S2SError, match="cover"):
        energy.forcing_objects(surface)


class Response:
    status = 206
    headers = {"Content-Range": "bytes 0-15/1000"}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, size): return b"GRIB" + b"\0" * 12


def test_probe_needs_real_object_range_and_grib_and_uses_mirrors(monkeypatch):
    obj = {"valid": "2024-02-29_00:00:00", "urls": ["https://one/object", "https://two/object"]}
    calls = []
    def open_url(request, timeout):
        calls.append(request.full_url)
        assert request.get_header("Range") == "bytes=0-15"
        if request.full_url.startswith("https://one/"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        return Response()
    monkeypatch.setattr(energy.urllib.request, "urlopen", open_url)
    actual = energy.probe_object(obj)
    assert actual["status"] == "available" and actual["bytes"] == 1000
    assert len(actual["attempts"]) == 1 and len(calls) == 2
    monkeypatch.setattr(Response, "status", 200)
    assert energy.probe_object(obj)["status"] == "missing"


def test_missing_middle_object_receipt_refuses_whole_window(tmp_path, monkeypatch):
    value = {"paths": {"work": str(tmp_path)}, "surface": {"source": "cfs", "forecast_cycle": "2023-03-29T12",
              "start": "2023-03-31_12:00:00", "end": "2023-04-30_12:00:00"}}
    def probe(obj):
        return {**obj, "status": "missing" if obj["valid"] == "2023-04-15_12:00:00" else "available"}
    monkeypatch.setattr(energy, "probe_object", probe)
    with pytest.raises(S2SError, match="missing 1/121"):
        energy.check_forcing(value)
    receipt = json.loads((tmp_path / "s2s-forcing-coverage.json").read_text())
    assert len(receipt["objects"]) == 121 and receipt["status"] == "missing"
