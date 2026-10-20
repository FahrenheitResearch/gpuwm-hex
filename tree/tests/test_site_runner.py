"""hexcore.site_runner: a public checkout runs directly; a shared host's queues come from its site file.

THE BREAKAGE THESE PREVENT: the gate spelled one host's queue programs into its
command line, so off that host it failed on a missing program before running
anything; and on that host, dropping the queue would put a proof forecast on a
card other jobs share.  Both halves are checked here without either program.
"""

from __future__ import annotations

import sys

from hexcore import site_runner


def _clean(monkeypatch, tmp_path, site_text=None):
    for key in site_runner.KEYS:
        monkeypatch.delenv(key, raising=False)
    site = tmp_path / "runners.env"
    if site_text is not None:
        site.write_text(site_text)
    monkeypatch.setenv("HEX_SITE_FILE", str(site))
    return site


def test_no_site_and_no_environment_runs_the_command_directly(monkeypatch, tmp_path):
    _clean(monkeypatch, tmp_path)
    command = [sys.executable, "-m", "hexcore.cli", "verify-mpas"]
    assert site_runner.gpu_command(command, 120, proof=True, build=True) == command
    assert site_runner.build_command(["make", "-j4"]) == ["make", "-j4"]
    assert site_runner.gate_cpuset() == ""


def test_site_file_names_the_shared_hosts_queues(monkeypatch, tmp_path):
    _clean(monkeypatch, tmp_path, "# comment\nHEX_GPU_QUEUE=gq\nHEX_BUILD_QUEUE=bq\nHEX_GATE_CPUSET=8-23\nOTHER=x\n")
    assert site_runner.gpu_command(["cmd"], 120, proof=True, build=True) == [
        "gq", "run", "--proof", "--minutes", "120", "--", "bq", "cmd"]
    assert site_runner.gpu_command(["cmd"], 10, tag="t") == ["gq", "run", "--tag", "t", "--minutes", "10", "--", "cmd"]
    assert site_runner.build_command(["make"]) == ["bq", "make"]
    assert site_runner.gate_cpuset() == "8-23"


def test_environment_overrides_the_site_file(monkeypatch, tmp_path):
    _clean(monkeypatch, tmp_path, "HEX_GPU_QUEUE=gq\nHEX_BUILD_QUEUE=bq\n")
    monkeypatch.setenv("HEX_GPU_QUEUE", "")
    monkeypatch.setenv("HEX_BUILD_QUEUE", "nice -n 10")
    assert site_runner.gpu_command(["cmd"], 5, build=True) == ["nice", "-n", "10", "cmd"]


def test_missing_named_site_file_is_no_site(monkeypatch, tmp_path):
    _clean(monkeypatch, tmp_path)
    assert site_runner.site_file() is None
    assert site_runner.read_site() == {}
