"""The boundary door's --physics-mode (IN4, IN5).

exact passes the argument vector the door always passed; correct is the default.  An engine that would
silently ignore the flag is not handed it: the series is native case 9 and the door receipt names IN4
and IN5 as gaps, printed loudly (hexcore.correct_mode: a named gap, never a refusal).
"""
from __future__ import annotations

import argparse
import stat
import sys

import pytest

from hexcore.hrrr_intermediate import IntermediateRefusal, add_lbc_parser
from hexcore import correct_mode
from hexcore.lbc_producer import lbc_physics_mode, lbc_physics_mode_args


def _engine(tmp_path, abi: str):
    """A stand-in rw_mpas_lbc that answers --abi with the given contract line."""
    body = tmp_path / "fake_rw_mpas_lbc.py"
    body.write_text("import sys\nif '--abi' in sys.argv:\n    print(%r)\n" % abi)
    if sys.platform == "win32":
        script = tmp_path / "rw_mpas_lbc.bat"
        script.write_text('@"%s" "%s" %%*\n' % (sys.executable, body))
        return script
    script = tmp_path / "rw_mpas_lbc"
    script.write_text("#!/bin/sh\nexec '%s' '%s' \"$@\"\n" % (sys.executable, body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


OLD_ABI = ("rw_mpas_lbc --grid INIT.nc --out-dir DIR --start-time YYYY-MM-DD_HH:MM:SS "
           "--oned-underflow preserve|reproduce-ifx-ftz [--config-attrs JSON] [--receipt JSON]")
NEW_ABI = OLD_ABI.replace("[--config-attrs", "[--physics-mode exact|correct] [--config-attrs")


def test_exact_adds_nothing_to_the_engine_arguments(tmp_path):
    assert lbc_physics_mode_args(_engine(tmp_path, OLD_ABI), "exact") == []


def test_correct_asks_an_engine_that_speaks_it(tmp_path):
    assert lbc_physics_mode_args(_engine(tmp_path, NEW_ABI), "correct") == ["--physics-mode", "correct"]


def test_correct_on_an_engine_that_would_ignore_it_is_a_named_gap(tmp_path, capsys):
    with pytest.warns(correct_mode.CorrectModeGapWarning, match="IN4, IN5"):
        argv, record = lbc_physics_mode(_engine(tmp_path, OLD_ABI), "correct")
    assert argv == []
    assert record["mode"] == "correct" and record["applied"] == []
    assert [row["finding"] for row in record["gaps"]] == ["IN4", "IN5"]
    err = capsys.readouterr().err
    assert "CORRECT MODE: named gap IN4 NOT APPLIED" in err and "CORRECT MODE: named gap IN5 NOT APPLIED" in err
    assert [row["slot"] for row in record["gaps"]] == ["lateral_boundary", "lateral_boundary"]


def test_correct_on_an_engine_that_speaks_it_records_the_findings(tmp_path):
    argv, record = lbc_physics_mode(_engine(tmp_path, NEW_ABI), "correct")
    assert argv == ["--physics-mode", "correct"]
    assert record["applied"] == ["IN4", "IN5"] and record["gaps"] == []
    assert lbc_physics_mode(_engine(tmp_path, OLD_ABI), "exact") == ([], None)


def test_an_unknown_mode_is_refused():
    with pytest.raises(IntermediateRefusal, match="not one of"):
        lbc_physics_mode_args("unused", "corect")


def test_the_door_defaults_to_correct_and_accepts_exact():
    parser = argparse.ArgumentParser()
    add_lbc_parser(parser.add_subparsers(dest="command"))
    base = ["lbc", "--grid", "g", "--met-dir", "m", "--out-dir", "o", "--start-time", "s",
            "--stop-time", "t", "--nfglevels", "34", "--extrap-airtemp", "lapse-rate", "--use-spechumd", "no"]
    assert parser.parse_args(base).physics_mode == "correct"
    assert parser.parse_args(base + ["--physics-mode", "exact"]).physics_mode == "exact"
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--physics-mode", "corect"])
