"""How a command reaches a shared host's GPU and build queues.

A public install runs every command directly: the gate, the native-reference
tools and the oracle builds call the program they name and nothing else.  A
host whose card or compilers are shared by several jobs names its queue
programs, and only then does a command go through them:

``HEX_GPU_QUEUE=<program>``
    the GPU queue, invoked as
    ``<program> run [--proof] [--priority] [--tag TAG] --minutes N -- <command>``
``HEX_BUILD_QUEUE=<program>``
    the build/compile queue, invoked as ``<program> <command>``
``HEX_GATE_CPUSET=<cpu list>``
    CPU affinity for the verification gate (``taskset -c <list>``)
``HEX_ARCHIVE_HOST=<label>``
    the host label a native cold-archive receipt must carry (unset: any recorded host)

The same names may be written, one ``KEY=VALUE`` per line, into a site file:
``$HEX_SITE_FILE`` if set, else ``site/runners.env`` beside ``src/`` in a
source checkout.  The environment overrides the file.  An installed wheel has
no site file, so it runs directly unless the environment says otherwise.

THE BREAKAGE THIS PREVENTS: the gate and the reference tools spelled one
host's queue programs into their command lines, so on every other machine the
G0-G2 gate and the oracle builds failed on a program that does not exist
before running anything (public gate f: the CPU tier must be green off that
host).  Dropping the queue instead would let a proof forecast on the shared
host collide with the other jobs on its card; the site file keeps that host's
behaviour byte for byte.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Mapping, Sequence

KEYS = ("HEX_GPU_QUEUE", "HEX_BUILD_QUEUE", "HEX_GATE_CPUSET", "HEX_ARCHIVE_HOST")


def site_file() -> Path | None:
    """The site file in effect, or None."""
    explicit = os.environ.get("HEX_SITE_FILE")
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    # src/hexcore/site_runner.py -> <tree>/site/runners.env (source checkouts only)
    candidate = Path(__file__).resolve().parents[2] / "site" / "runners.env"
    return candidate if candidate.is_file() else None


def read_site(path: Path | None = None) -> dict[str, str]:
    path = site_file() if path is None else path
    values: dict[str, str] = {}
    if path is None:
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in KEYS:
            values[key] = value.strip().strip("'\"")
    return values


def setting(key: str, environ: Mapping[str, str] | None = None) -> str:
    """One setting: the environment first, then the site file, else empty."""
    environ = os.environ if environ is None else environ
    if key in environ:
        return environ[key].strip()
    return read_site().get(key, "")


def build_prefix(environ: Mapping[str, str] | None = None) -> list[str]:
    queue = setting("HEX_BUILD_QUEUE", environ)
    return shlex.split(queue) if queue else []


def gpu_prefix(minutes: int, *, proof: bool = False, priority: bool = False, tag: str | None = None,
               environ: Mapping[str, str] | None = None) -> list[str]:
    queue = setting("HEX_GPU_QUEUE", environ)
    if not queue:
        return []
    prefix = [*shlex.split(queue), "run"]
    if proof:
        prefix.append("--proof")
    if priority:
        prefix.append("--priority")
    if tag:
        prefix += ["--tag", tag]
    return prefix + ["--minutes", str(int(minutes)), "--"]


def build_command(command: Sequence[str], environ: Mapping[str, str] | None = None) -> list[str]:
    """``command`` through the build queue when one is configured."""
    return [*build_prefix(environ), *command]


def gpu_command(command: Sequence[str], minutes: int, *, proof: bool = False, priority: bool = False,
                tag: str | None = None, build: bool = False,
                environ: Mapping[str, str] | None = None) -> list[str]:
    """``command`` through the GPU queue (and, with ``build``, the build queue inside it)."""
    inner = build_command(command, environ) if build else list(command)
    return [*gpu_prefix(minutes, proof=proof, priority=priority, tag=tag, environ=environ), *inner]


def gate_cpuset(environ: Mapping[str, str] | None = None) -> str:
    return setting("HEX_GATE_CPUSET", environ)
