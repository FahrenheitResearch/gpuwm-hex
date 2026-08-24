"""Refusal-surface tests for the ``mpas-port init`` front door.

These exercise the door's own contract — engine discovery, argument
completeness, met-source resolution — with no engine binary and no real
data.  The numeric path is proven against the artifact (the built
rw_mpas_init exe on real factory inputs), not mocked here.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from mpas_port.cli import main
from mpas_port.init_door import (
    InitDoorRefusal,
    _probe_wps_header,
    resolve_engine,
    resolve_met_source,
)


def _wps_probe_bytes(version: int = 5, endian: str = ">") -> bytes:
    return struct.pack(f"{endian}ii", 4, version)


COMPLETE_SWITCHES = [
    "--start-time", "2025-01-01_00:00:00",
    "--nfglevels", "38",
    "--nfgsoillevels", "4",
    "--extrap-airtemp", "lapse-rate",
    "--use-spechumd", "no",
    "--theta-adv-order", "3",
    "--coef-3rd-order", "0.25",
    "--virtual-factor", "reproduce-fortran",
    "--deep-soil-moisture", "reproduce-fortran",
    "--landuse-table", "MODIFIED_IGBP_MODIS_NOAH",
    "--frac-seaice", "yes",
    "--tsk-seaice-threshold", "100.0",
    "--oned-underflow", "preserve",
]


def _paths(tmp_path: Path) -> list[str]:
    return [
        "--met", str(tmp_path / "met"),
        "--static", str(tmp_path / "static.nc"),
        "--capsule", str(tmp_path / "capsule.nc"),
        "--reference", str(tmp_path / "reference.nc"),
        "--out", str(tmp_path / "init.nc"),
    ]


def test_missing_switch_refusal_names_the_switch_and_the_breakage(tmp_path, capsys):
    argv = ["init", *_paths(tmp_path), *COMPLETE_SWITCHES]
    index = argv.index("--extrap-airtemp")
    del argv[index : index + 2]
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2
    assert "--extrap-airtemp was not given" in err
    assert "no default" in err
    assert "config_extrap_airtemp" in err


def _no_engines_anywhere(monkeypatch):
    """A machine with no rw_mpas_init on any rung.

    Deleting the environment variables is not enough: the ladder also reads
    gpuwm's bridge directories, and a box that has ever run
    ``gpuwm fetch-bridges`` has the binary there.  That is the point of the
    rung, and it means a refusal test has to remove it deliberately.
    """

    from mpas_port import engines

    for name in ("GPUWM_HEX_RW_MPAS_INIT", "RW_MPAS_INIT", "GPUWM_RW_MPAS_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())


def test_engine_absent_refusal_names_env_var_and_build_line(monkeypatch):
    _no_engines_anywhere(monkeypatch)
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_engine(None)
    message = str(caught.value)
    assert "RW_MPAS_INIT" in message
    assert "cargo build" in message
    # One command stages this binary; naming it first is the difference
    # between a remedy and a research project.
    assert "gpuwm fetch-bridges" in message


def test_engine_path_must_exist(tmp_path):
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_engine(tmp_path / "no-such-exe")
    message = str(caught.value)
    assert "which is not a file" in message
    assert "--engine" in message


def test_a_staged_engine_resolves_with_no_environment_variable(
    monkeypatch, tmp_path
):
    """``gpuwm fetch-bridges`` alone is enough to open this door."""

    from mpas_port import engines

    staged = tmp_path / engines.executable_name("rw_mpas_init")
    staged.write_bytes(b"")
    staged.chmod(0o755)
    for name in ("GPUWM_HEX_RW_MPAS_INIT", "RW_MPAS_INIT", "GPUWM_RW_MPAS_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

    assert resolve_engine(None) == staged.resolve()


def test_met_source_missing_is_refused_by_name(tmp_path):
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path / "absent")
    assert "does not exist" in str(caught.value)
    assert "ungrib" in str(caught.value)


def test_met_directory_with_no_intermediate_is_refused(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"not an intermediate file")
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path)
    assert "no WPS intermediate file" in str(caught.value)


def test_met_directory_with_one_candidate_resolves_to_it(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"prose")
    candidate = tmp_path / "MET_2025-01-01_00"
    candidate.write_bytes(_wps_probe_bytes())
    assert resolve_met_source(tmp_path) == candidate


def test_met_directory_with_two_candidates_demands_an_explicit_file(tmp_path):
    for name in ("MET_2025-01-01_00", "MET_2025-01-02_00"):
        (tmp_path / name).write_bytes(_wps_probe_bytes())
    with pytest.raises(InitDoorRefusal) as caught:
        resolve_met_source(tmp_path)
    message = str(caught.value)
    assert "exactly one valid time" in message
    assert "MET_2025-01-01_00" in message and "MET_2025-01-02_00" in message


@pytest.mark.parametrize("endian", [">", "<"])
@pytest.mark.parametrize("version", [3, 4, 5])
def test_probe_accepts_every_wps_version_both_endians(tmp_path, endian, version):
    path = tmp_path / "candidate"
    path.write_bytes(_wps_probe_bytes(version, endian))
    assert _probe_wps_header(path)


def test_probe_rejects_non_wps_bytes(tmp_path):
    path = tmp_path / "candidate"
    path.write_bytes(b"\x00" * 8)
    assert not _probe_wps_header(path)
    short = tmp_path / "short"
    short.write_bytes(b"\x00")
    assert not _probe_wps_header(short)


def test_missing_out_dir_is_refused_before_any_engine_run(tmp_path, monkeypatch, capsys):
    engine = tmp_path / "rw_mpas_init"
    engine.write_bytes(b"#!/bin/sh\n")
    engine.chmod(0o755)
    monkeypatch.setenv("RW_MPAS_INIT", str(engine))
    argv = [
        "init",
        "--met", str(tmp_path / "met"),
        "--static", str(tmp_path / "static.nc"),
        "--capsule", str(tmp_path / "capsule.nc"),
        "--reference", str(tmp_path / "reference.nc"),
        "--out", str(tmp_path / "no-such-dir" / "init.nc"),
        *COMPLETE_SWITCHES,
    ]
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2
    assert "output directory" in err
    assert "does not exist" in err
