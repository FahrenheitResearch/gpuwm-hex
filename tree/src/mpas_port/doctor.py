"""``gpuwm-hex doctor``: what this install can actually reach, and what to run.

A wheel for this project is deliberately partial, and saying so is the
point.  The Python driver, the front doors and the tables ship in it.
The things that do the work do not:

* the Rust engines (``rw_mpas_init``, ``rw_mpas_convert``,
  ``rw_wrfbatch``) are built from the gpuwm ``tools/rustwx`` workspace
  and staged onto a machine -- ``gpuwm fetch-bridges`` is the one
  command that does it;
* the CUDA lane needs a CuPy wheel matching the box's CUDA major, which
  no pip extra can detect for the user;
* the meshes and their static fields are external assets with no fetch
  path in this distribution;
* the forecast lane pins gpuwm by the sha256 of individual source files,
  one of which no wheel places in site-packages -- so it needs a gpuwm
  SOURCE CHECKOUT on top of the installed distribution.

Every one of those is a place a fresh install meets a wall.  This module
exists so the wall has a sign on it: each gap prints THE command that
closes it, on this platform, spelled as it is typed.  A bare
``ImportError`` three commands later is the failure mode being replaced.

Statuses.  ``verified`` means the deep check ran and passed -- the module
imported in a short-lived subprocess, the binary was found on a named
rung.  ``present`` is for what can only be checked by existence.
``missing`` is a gap, and a gap always carries a remedy.  ``info`` is
context that is never a gap.

Exit status is 1 when any REQUIRED finding is missing, 0 otherwise.
Required means: without it, no front door of this distribution opens.
The three Python dependencies and the three engines are required in that
sense; CuPy, a gpuwm source checkout and the mesh assets are reported and
do not fail the process, because a user who only wants the render door
should not be told the install is broken by the absence of a mesh.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from . import DISTRIBUTION_NAME
from . import engines


#: How long a single import probe may take before it is called wedged.
_PROBE_TIMEOUT_S = 90

VERIFIED = "verified"
PRESENT = "present"
MISSING = "missing"
INFO = "info"


@dataclass
class Finding:
    """One checked thing: what it is, what was found, what to run."""

    subject: str
    status: str
    detail: str
    #: Commands that close the gap.  Every line is a command as typed or a
    #: ``#`` comment -- never prose fused onto a command.
    remedy: str = ""
    #: Whether a gap here closes every front door.
    required: bool = False
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
            "required": self.required,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def _import_probe(module: str) -> tuple[bool, str]:
    """Import ``module`` in a short-lived subprocess and read its version.

    A subprocess rather than an import here for two reasons that have
    both been measured on this stack: a broken native dependency (a
    netCDF4 or CuPy whose shared libraries do not load) can take the
    interpreter down rather than raise, and a partially initialised
    module poisons every later check in the same process.  The report
    must survive the thing it is reporting on.
    """

    code = (
        "import importlib, json, sys\n"
        f"m = importlib.import_module({module!r})\n"
        "print(json.dumps({'version': getattr(m, '__version__', None),"
        " 'file': getattr(m, '__file__', None)}))\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"the import probe could not run: {error}"
    if probe.returncode != 0:
        tail = (probe.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else f"import exited {probe.returncode}"
    try:
        facts = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return True, "imported (version not reported)"
    version = facts.get("version") or "version not reported"
    return True, f"imported, {version}"


def _distribution_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as distribution_version

    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def check_interpreter() -> list[Finding]:
    from . import __version__

    return [
        Finding(
            subject="distribution",
            status=INFO,
            detail=f"{DISTRIBUTION_NAME} {__version__}",
            evidence={
                "import_package": __package__ or "mpas_port",
                "location": str(Path(__file__).resolve().parent),
            },
        ),
        Finding(
            subject="interpreter",
            status=INFO,
            detail=(
                f"Python {platform.python_version()} on "
                f"{platform.system()} {platform.machine()}"
            ),
            evidence={"executable": sys.executable},
        ),
    ]


#: Module -> the distribution that installs it, where they differ.
_REQUIRED_MODULES = (
    ("numpy", "numpy", "arrays; 48 modules import it at line one"),
    ("netCDF4", "netCDF4", "reads and writes every mesh, static and history file"),
    ("scipy", "scipy", "the regridder's spatial index"),
)


def check_python_dependencies() -> list[Finding]:
    findings: list[Finding] = []
    for module, distribution, why in _REQUIRED_MODULES:
        ok, detail = _import_probe(module)
        findings.append(
            Finding(
                subject=f"{module} ({why})",
                status=VERIFIED if ok else MISSING,
                detail=detail,
                remedy="" if ok else f"  pip install {distribution}",
                required=True,
            )
        )
    return findings


def check_gpu_runtime() -> list[Finding]:
    """CuPy, and the CUDA-major pairing no pip extra can choose."""

    ok, detail = _import_probe("cupy")
    if not ok:
        return [
            Finding(
                subject="cupy (the CUDA lane)",
                status=MISSING,
                detail=detail,
                remedy=(
                    "  # match the CUDA major your driver reports, then:\n"
                    f'  pip install "{DISTRIBUTION_NAME}[gpu-cu12]"\n'
                    f'  pip install "{DISTRIBUTION_NAME}[gpu-cu13]"\n'
                    "      # exactly one of the two.  A CuPy wheel built for a\n"
                    "      # different CUDA major imports, probes clean, and\n"
                    "      # then fails on the first real device call."
                ),
            )
        ]
    return [
        Finding(
            subject="cupy (the CUDA lane)",
            status=VERIFIED,
            detail=detail,
        )
    ]


#: The floor that pip enforces.  Stated once; the wall is the source
#: manifest, which is a different and stricter thing (see below).
#:
#: 2.5.5 is the first published version whose bytes match the Arwen seam
#: manifest (3 of the 16 pinned files still differ at the 2.5.4 stamp), and
#: it keeps the bundle guarantee that arrived at 2.5.3: the four MPAS bridge
#: binaries the front doors drive.  On 2.5.2 `gpuwm fetch-bridges` can stage
#: none of the four, so a user resolved onto it reaches neither door.
GPUWM_FLOOR = "2.5.5"


def check_physics_seam() -> list[Finding]:
    """gpuwm: the installed distribution, and the checkout the pin needs.

    Two findings, because they are two different facts with two different
    remedies, and folding them into one is how "gpuwm is installed" got
    read as "the forecast lane will run".
    """

    findings: list[Finding] = []
    installed = _distribution_version("gpuwm")
    if installed is None:
        findings.append(
            Finding(
                subject="gpuwm (the physics seam)",
                status=MISSING,
                detail="no gpuwm distribution is installed in this interpreter",
                remedy=f'  pip install "gpuwm>={GPUWM_FLOOR}"',
                required=True,
            )
        )
    else:
        findings.append(
            Finding(
                subject="gpuwm (the physics seam)",
                status=PRESENT,
                detail=f"gpuwm {installed} is installed",
                evidence={"version": installed, "floor": GPUWM_FLOOR},
            )
        )
    findings.append(
        Finding(
            subject="gpuwm source checkout (the forecast lane only)",
            status=INFO,
            detail=(
                "the forecast lane pins gpuwm by the sha256 of sixteen "
                "individual source files, one of which is a repository "
                "document that no wheel places in site-packages.  An "
                "installed gpuwm satisfies pip and does not satisfy the pin, "
                "so that lane needs a source checkout at the pinned commit.  "
                "The init and render doors do not: they import no gpuwm and "
                "drive Rust binaries instead."
            ),
        )
    )
    return findings


def check_engines() -> list[Finding]:
    """The three Rust binaries, and the one command that stages all of them."""

    findings: list[Finding] = []
    # Asked once, of the INSTALLED gpuwm, and reported plainly: whether the
    # one-command staging route can supply these engines at all on this box.
    # gpuwm's published 2.5.2 bundles rw_wrfbatch and none of the MPAS
    # binaries, so "run gpuwm fetch-bridges" is a complete answer for one of
    # the three engines here and no answer at all for the other two.  A
    # report that does not distinguish those sends a user round a loop.
    supplied = {spec.name: engines.gpuwm_bundles(spec) for spec in engines.ENGINES}
    unsupplied = sorted(name for name, ok in supplied.items() if ok is False)
    if unsupplied:
        findings.append(
            Finding(
                subject="gpuwm fetch-bridges coverage",
                status=INFO,
                detail=(
                    "the gpuwm installed here bundles no "
                    f"{', '.join(unsupplied)}, so `{engines.FETCH_COMMAND}` "
                    "cannot supply "
                    f"{'them' if len(unsupplied) > 1 else 'it'}.  A later "
                    "gpuwm release adds these artifacts; until then they come "
                    "from a source build."
                ),
                evidence={"not_bundled": unsupplied},
            )
        )
    for spec in engines.ENGINES:
        path, source = engines.locate(spec)
        if path is not None:
            findings.append(
                Finding(
                    subject=f"{spec.name} ({spec.subject})",
                    status=VERIFIED,
                    detail=f"found via {source}: {path}",
                    required=True,
                    evidence={"path": str(path), "resolved_from": source},
                )
            )
            continue
        findings.append(
            Finding(
                subject=f"{spec.name} ({spec.subject})",
                status=MISSING,
                detail=(
                    f"{source}.  Without it {spec.what_breaks}.  "
                    f"Looked at: {engines.resolution_order(spec)}"
                ),
                remedy=engines.remedy(spec),
                required=True,
                evidence={"resolution_order": engines.resolution_order(spec)},
            )
        )
    return findings


def check_assets() -> list[Finding]:
    """The mesh pair, which this distribution ships and fetches neither."""

    return [
        Finding(
            subject="mesh grid + static pair",
            status=INFO,
            detail=(
                "external assets.  This distribution carries no mesh and has "
                "no fetch path for one, so every door that needs a mesh takes "
                "--grid and --static explicitly and refuses rather than guess "
                "a default.  Two routes exist: generate a mesh of any "
                "resolution with the staged rw_mpas_mesh and rw_mpas_static "
                "binaries, or supply an existing grid/static pair."
            ),
            evidence={"bridge_directory": str(engines.USER_BRIDGE_DIR)},
        )
    ]


def collect() -> list[Finding]:
    """Every finding, in the order a reader should meet them."""

    findings: list[Finding] = []
    findings.extend(check_interpreter())
    findings.extend(check_python_dependencies())
    findings.extend(check_physics_seam())
    findings.extend(check_gpu_runtime())
    findings.extend(check_engines())
    findings.extend(check_assets())
    return findings


def blocking_gaps(findings: list[Finding]) -> list[Finding]:
    """The findings that make the exit status 1.  The rule, stated once."""

    return [f for f in findings if f.status == MISSING and f.required]


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
_LABEL = {
    VERIFIED: "OK      ",
    PRESENT: "PRESENT ",
    MISSING: "MISSING ",
    INFO: "INFO    ",
}


def render(findings: list[Finding], *, explain: bool) -> str:
    """One line per finding, or the whole evidence and remedy block."""

    lines: list[str] = []
    for finding in findings:
        label = _LABEL.get(finding.status, finding.status)
        if explain:
            lines.append(f"{label}{finding.subject}")
            lines.append(f"          {finding.detail}")
            if finding.evidence:
                for key, value in sorted(finding.evidence.items()):
                    lines.append(f"          {key}: {value}")
            if finding.remedy:
                lines.append("")
                lines.extend(finding.remedy.splitlines())
            lines.append("")
            continue
        headline = finding.detail.splitlines()[0] if finding.detail else ""
        if len(headline) > 96:
            headline = headline[:93] + "..."
        lines.append(f"{label}{finding.subject}: {headline}")
        if finding.remedy:
            first = [
                line
                for line in finding.remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if first:
                lines.append(f"          {first[0].strip()}")
    return "\n".join(lines)


def summarise(findings: list[Finding]) -> str:
    blocking = blocking_gaps(findings)
    if not blocking:
        gaps = [f for f in findings if f.status == MISSING]
        if gaps:
            return (
                f"\nEvery required check passed.  {len(gaps)} optional gap(s) "
                f"remain; each prints its own command above."
            )
        return "\nEvery check passed."
    names = ", ".join(f.subject.split(" (")[0] for f in blocking)
    return (
        f"\n{len(blocking)} required item(s) missing: {names}.\n"
        f"Run `gpuwm-hex doctor --explain` for the full remedy for each."
    )


def add_doctor_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "doctor",
        help="report what this install can reach, and the command for each gap",
        description=(
            "Check the runtime estate this distribution needs and cannot "
            "carry: the Python dependencies, the physics seam, the CUDA "
            "lane, and the Rust engines the front doors drive.  Every gap "
            "prints the command that closes it."
        ),
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print the evidence and the whole pasteable remedy for each finding",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the findings as JSON instead of the text report",
    )
    parser.set_defaults(handler=run_doctor)


def run_doctor(arguments: argparse.Namespace) -> int:
    findings = collect()
    if getattr(arguments, "json", False):
        print(
            json.dumps(
                {
                    "distribution": DISTRIBUTION_NAME,
                    "findings": [finding.as_dict() for finding in findings],
                    "blocking": [
                        finding.subject for finding in blocking_gaps(findings)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render(findings, explain=getattr(arguments, "explain", False)))
        print(summarise(findings))
    return 1 if blocking_gaps(findings) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpuwm-hex doctor")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--json", action="store_true")
    return run_doctor(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "Finding",
    "GPUWM_FLOOR",
    "INFO",
    "MISSING",
    "PRESENT",
    "VERIFIED",
    "add_doctor_parser",
    "blocking_gaps",
    "collect",
    "main",
    "render",
    "run_doctor",
    "summarise",
]
