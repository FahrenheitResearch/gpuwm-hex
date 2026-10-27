"""The container CPU quota bounds hex's child thread pools.

Breakage this guards: a box listing 256 CPUs under a 61.44-core cgroup v1
quota ran the init's Rust vertical build on 257 threads and stalled for an
hour (2026-10-26, uniform 15 km global init).
"""

import sys

import pytest

from hexcore import cpu_budget


def _v1_container(tmp_path, quota="6143999", period="100000"):
    group = tmp_path / "cpu,cpuacct"
    group.mkdir()
    (group / "cpu.cfs_quota_us").write_text(quota + "\n")
    (group / "cpu.cfs_period_us").write_text(period + "\n")
    return tmp_path


@pytest.fixture
def linux_box(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cpu_budget, "affinity_cpus", lambda: 256)


def test_v1_quota_is_read_through_the_controller_folder(tmp_path):
    root = _v1_container(tmp_path)
    assert cpu_budget.cgroup_cpus(root, "4:cpu,cpuacct:/\n") == 61


def test_v2_ancestor_quota_binds(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "cpu.max").write_text("400000 100000\n")
    (tmp_path / "box" / "job").mkdir()
    (tmp_path / "box" / "job" / "cpu.max").write_text("max 100000\n")
    assert cpu_budget.cgroup_cpus(tmp_path, "0::/box/job\n") == 4


def test_unset_and_inherited_counts_are_held_to_the_quota(tmp_path, monkeypatch, linux_box):
    root = _v1_container(tmp_path)
    real = cpu_budget.cgroup_cpus
    monkeypatch.setattr(cpu_budget, "cgroup_cpus", lambda: real(root, "4:cpu,cpuacct:/\n"))
    environ = {"OMP_NUM_THREADS": "256"}
    record = cpu_budget.bind_thread_defaults(environ)
    assert environ["RAYON_NUM_THREADS"] == "61"
    assert environ["OMP_NUM_THREADS"] == "61"
    assert record["changed"]["OMP_NUM_THREADS"] == {"from": 256, "to": 61}
    assert "RAYON_NUM_THREADS:unset->61" in environ[cpu_budget.BOUND_RECORD]


def test_no_quota_changes_nothing(monkeypatch, linux_box):
    monkeypatch.setattr(cpu_budget, "cgroup_cpus", lambda: None)
    environ = {"RAYON_NUM_THREADS": "32"}
    assert cpu_budget.bind_thread_defaults(environ) is None
    assert environ == {"RAYON_NUM_THREADS": "32"}


def test_smaller_explicit_count_and_opt_out_are_kept(monkeypatch, linux_box):
    monkeypatch.setattr(cpu_budget, "cgroup_cpus", lambda: 61)
    environ = {"RAYON_NUM_THREADS": "32", "OMP_NUM_THREADS": "8"}
    assert cpu_budget.bind_thread_defaults(environ) is None
    environ = {cpu_budget.OPT_OUT: "off"}
    assert cpu_budget.bind_thread_defaults(environ) is None
    assert "RAYON_NUM_THREADS" not in environ
