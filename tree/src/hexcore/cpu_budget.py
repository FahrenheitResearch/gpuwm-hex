"""The CPUs this process may really use: affinity AND the container's CPU quota.

THE BREAKAGE THIS PREVENTS, MEASURED (rented box, 2026-10-26, uniform 15 km
global init, x1.2621442, 55 levels): the box listed 256 CPUs while its
container's cgroup quota granted 61.44.  The init's Rust vertical build ran
257 Rayon threads, the cgroup was throttled in 56,647 of 123,855 periods,
and the first vertical call never returned (Round 2 stopped it unfinished
after 3,511 s).  The same binary on the same input with 32 workers finished
the whole init in 242.9 s.  ``os.cpu_count()`` and ``sched_getaffinity`` both
said 256; only the quota files said 61.

So the quota is a hard limit, like affinity: at import (``hexcore/__init__``)
the thread-count variables every Rust and OpenMP child of hex reads
(``RAYON_NUM_THREADS``, ``OMP_NUM_THREADS``) are set to this budget when
unset and lowered to it when an inherited value asks for more.  Nothing is
changed on a host whose quota does not bind below its affinity, so home
nodes and whole-machine boxes keep their thread counts exactly.  Thread count
changes scheduling only: Rayon collects and the vertical arrays are the same
bytes at any worker count (32 vs 256 workers, 10 arrays, identical SHA-256).

The quota is read on cgroup v2 (``cpu.max``) and v1 (``cpu.cfs_quota_us`` /
``cpu.cfs_period_us``) through every ancestor of this process's group that is
visible, and the smallest one wins, the same reading gpuwm's preparation
workers use.  ``GPUWM_HEX_CPU_QUOTA=off`` leaves the variables alone.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

THREAD_VARIABLES = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS")
OPT_OUT = "GPUWM_HEX_CPU_QUOTA"
#: Set when hex lowered or supplied a thread count, so receipts can say why.
BOUND_RECORD = "GPUWM_HEX_CPU_QUOTA_BOUND"


def _quota(quota: str, period: str) -> int | None:
    try:
        quota_us, period_us = int(quota), int(period)
    except (TypeError, ValueError):
        return None  # "max" or "-1": no quota at this level
    if quota_us <= 0 or period_us <= 0:
        return None
    return max(1, quota_us // period_us)


def _ancestors(root: Path, relative: str) -> tuple[Path, ...]:
    if ".." in Path(relative).parts:
        return (root,)
    current = root / relative.lstrip("/")
    found = [current]
    while current != root and root in current.parents:
        current = current.parent
        found.append(current)
    return tuple(found)


def cgroup_cpus(root: Path = Path("/sys/fs/cgroup"), membership: str | None = None) -> int | None:
    """Whole CPUs the tightest visible CPU quota grants, else ``None``."""

    root = Path(root)
    if membership is None:
        try:
            membership = Path("/proc/self/cgroup").read_text()
        except OSError:
            membership = ""
    groups: list[tuple[Path, bool]] = [(root, True)]
    for controller in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
        groups.append((root / controller, False))
    groups.append((root, False))
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        if not controllers:
            groups.extend((path, True) for path in _ancestors(root, relative))
        elif "cpu" in controllers.split(","):
            for folder in (controllers, "cpu", "cpu,cpuacct"):
                groups.extend((path, False) for path in _ancestors(root / folder, relative))
    limits = []
    seen = set()
    for group, v2 in groups:
        if (group, v2) in seen:
            continue
        seen.add((group, v2))
        try:
            if v2:
                words = (group / "cpu.max").read_text().split()
            else:
                words = [(group / "cpu.cfs_quota_us").read_text().strip(),
                         (group / "cpu.cfs_period_us").read_text().strip()]
        except OSError:
            continue
        count = _quota(*words) if len(words) == 2 else None
        if count is not None:
            limits.append(count)
    return min(limits) if limits else None


def affinity_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def available_cpus() -> int:
    """min(affinity, cgroup quota): the most threads that can run at once."""

    affinity = affinity_cpus()
    quota = cgroup_cpus() if sys.platform.startswith("linux") else None
    return min(affinity, quota) if quota else affinity


def bind_thread_defaults(environ=None) -> dict | None:
    """Hold child thread pools to the CPU quota when the quota binds; return what changed."""

    environ = os.environ if environ is None else environ
    if environ.get(OPT_OUT, "").strip().lower() in {"off", "0", "no", "false"}:
        return None
    if not sys.platform.startswith("linux"):
        return None
    affinity = affinity_cpus()
    quota = cgroup_cpus()
    budget = min(affinity, quota) if quota else affinity
    changed = {}
    for name in THREAD_VARIABLES:
        text = environ.get(name, "").strip()
        try:
            asked = int(text) if text else None
        except ValueError:
            continue  # not ours to interpret; the runtime reports it
        if asked is None:
            if quota is not None and quota < affinity:
                environ[name] = str(budget)
                changed[name] = {"from": None, "to": budget}
        elif asked > budget:
            environ[name] = str(budget)
            changed[name] = {"from": asked, "to": budget}
    if changed:
        environ[BOUND_RECORD] = ",".join(
            f"{name}:{entry['from'] or 'unset'}->{entry['to']}" for name, entry in changed.items())
        return {"affinity_cpus": affinity, "cgroup_cpus": quota, "budget": budget, "changed": changed}
    return None
