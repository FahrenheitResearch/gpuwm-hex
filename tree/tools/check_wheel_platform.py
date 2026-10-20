"""Refuse a gpuwm-hex wheel whose platform tag an older Linux cannot install.

THE BREAKAGE THIS PREVENTS (verify-final-hex, QA H3, 2026-10-12): the 0.4.0
candidate wheel was built on an ubuntu-24.04 class host, so setup.py measured
the bundled helpers' glibc floor from that host's toolchain and tagged the
wheel ``manylinux_2_35_x86_64``.  pip on any host with glibc older than 2.35
(Ubuntu 20.04/22.04, RHEL 8/9, most HPC login nodes) then skips the wheel and
falls back to the sdist, which needs cargo and an engine source checkout to
build the helpers.  The publish workflow now builds inside the
manylinux_2_28 image and repairs with auditwheel; this check is what fails
the publish if the artefact that comes out still claims a newer glibc than
the policy, claims an older one than its own executables need, or lost one of
the executables the doors look up in ``hexcore/libexec``.

Stdlib only, so it runs in the bare manylinux image, on the publish runner
before the upload, and in the CPU battery.

    python tools/check_wheel_platform.py dist/*.whl --policy manylinux_2_28_x86_64 \
        --require-libexec hex_contract_hostprep --require-libexec hex_obs_scorecard

Exit 0 when every wheel passes, 1 with one line per finding otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

#: PEP 600 tag; the legacy aliases map onto it (PEP 600, "Legacy manylinux tags").
_PEP600 = re.compile(r"^manylinux_(\d+)_(\d+)_([A-Za-z0-9_]+)$")
_LEGACY = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}
_GLIBC_SYMBOL = re.compile(rb"GLIBC_(\d+)\.(\d+)")
_ELF_MAGIC = b"\x7fELF"


def parse_platform_tag(tag: str) -> tuple[tuple[int, int], str] | None:
    """``((major, minor), arch)`` for a manylinux tag, ``None`` for anything else."""

    match = _PEP600.match(tag)
    if match:
        return (int(match.group(1)), int(match.group(2))), match.group(3)
    for legacy, glibc in _LEGACY.items():
        if tag.startswith(legacy + "_"):
            return glibc, tag[len(legacy) + 1:]
    return None


def wheel_filename_tags(name: str) -> tuple[str, str, list[str]]:
    """``(python, abi, [platforms])`` from a wheel file name (PEP 427)."""

    stem = name[:-len(".whl")] if name.endswith(".whl") else name
    parts = stem.split("-")
    if len(parts) not in (5, 6):
        raise ValueError(f"{name}: not a PEP 427 wheel file name")
    python, abi, platforms = parts[-3], parts[-2], parts[-1]
    return python, abi, platforms.split(".")


def glibc_floor(payload: bytes) -> tuple[int, int] | None:
    """The newest GLIBC symbol version an ELF file names, or ``None``.

    The same reading setup.py's ``glibc_floor`` makes: the linker writes the
    version-requirement names (``GLIBC_2.34`` and the like) into the file.
    """

    found = [(int(a), int(b)) for a, b in _GLIBC_SYMBOL.findall(payload)]
    return max(found) if found else None


@dataclass
class WheelReport:
    wheel: str
    platforms: list[str] = field(default_factory=list)
    elf: dict[str, str | None] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings


def check_wheel(path: Path, policy: str, required_libexec: list[str],
                package: str = "hexcore", allow_grafted: bool = False) -> WheelReport:
    report = WheelReport(wheel=path.name)
    wanted = parse_platform_tag(policy)
    if wanted is None:
        raise ValueError(f"--policy {policy!r} is not a manylinux tag")
    (policy_glibc, policy_arch) = wanted
    try:
        python, abi, platforms = wheel_filename_tags(path.name)
    except ValueError as error:
        report.findings.append(str(error))
        return report
    report.platforms = platforms
    if python != "py3" or abi != "none":
        report.findings.append(
            f"tag {python}-{abi}: the helpers are subprocess executables that link no libpython, "
            "so the wheel must be py3-none (one wheel for every supported Python)")
    claimed: list[tuple[int, int]] = []
    for platform in platforms:
        parsed = parse_platform_tag(platform)
        if parsed is None:
            report.findings.append(
                f"platform tag {platform!r} is not manylinux: PyPI refuses a bare linux tag, "
                "and pip on another distribution cannot tell whether it can load the helpers")
            continue
        glibc, arch = parsed
        if arch != policy_arch:
            report.findings.append(f"platform tag {platform!r} is for {arch}, the policy is {policy_arch}")
        if glibc > policy_glibc:
            report.findings.append(
                f"platform tag {platform!r} needs glibc {glibc[0]}.{glibc[1]}, newer than the "
                f"{policy} policy: hosts between glibc {policy_glibc[0]}.{policy_glibc[1]} and "
                f"{glibc[0]}.{glibc[1]} would skip this wheel and build the sdist")
        claimed.append(glibc)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        report.findings.append(f"not a zip archive: {error}")
        return report
    with archive:
        names = archive.namelist()
        wheel_files = [n for n in names if PurePosixPath(n).name == "WHEEL"
                       and PurePosixPath(n).parent.name.endswith(".dist-info")]
        if len(wheel_files) != 1:
            report.findings.append(f"expected one .dist-info/WHEEL, found {len(wheel_files)}")
        else:
            metadata = archive.read(wheel_files[0]).decode("utf-8", "replace")
            tags = sorted(line.split(":", 1)[1].strip() for line in metadata.splitlines()
                          if line.startswith("Tag:"))
            expected = sorted(f"{python}-{abi}-{platform}" for platform in platforms)
            if tags != expected:
                report.findings.append(f"WHEEL Tag lines {tags} disagree with the file name's {expected}")
            if "Root-Is-Purelib: false" not in metadata:
                report.findings.append("WHEEL says Root-Is-Purelib is not false for a wheel carrying executables")
        oldest_claim = min(claimed) if claimed else None
        for name in names:
            info = archive.getinfo(name)
            if info.is_dir():
                continue
            with archive.open(info) as handle:
                head = handle.read(4)
            if head != _ELF_MAGIC:
                continue
            floor = glibc_floor(archive.read(info))
            report.elf[name] = f"{floor[0]}.{floor[1]}" if floor else None
            if floor is not None and oldest_claim is not None and floor > oldest_claim:
                report.findings.append(
                    f"{name} needs glibc {floor[0]}.{floor[1]} but the wheel claims "
                    f"{oldest_claim[0]}.{oldest_claim[1]}: pip would install it where the helper cannot load")
        grafted = sorted(n for n in names if ".libs/" in n and not n.endswith("/"))
        if grafted and not allow_grafted:
            report.findings.append(
                f"{len(grafted)} grafted shared libraries ({', '.join(PurePosixPath(n).name for n in grafted[:6])}"
                f"{' ...' if len(grafted) > 6 else ''}): every helper must be self-contained, or each "
                "third-party binary rides in the public wheel without its licence notice")
        for executable in required_libexec:
            member = f"{package}/libexec/{executable}"
            if member not in names:
                report.findings.append(
                    f"{member} is missing: the door that looks it up refuses on every install of this wheel")
                continue
            mode = (archive.getinfo(member).external_attr >> 16) & 0o777
            if not mode & 0o111:
                report.findings.append(f"{member} is not executable in the archive (mode {mode:o})")
            if member not in report.elf:
                report.findings.append(f"{member} is not an ELF executable")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--policy", default="manylinux_2_28_x86_64",
                        help="the newest manylinux tag the wheel may claim (default %(default)s)")
    parser.add_argument("--require-libexec", action="append", default=[], metavar="NAME",
                        help="an executable that must ship in hexcore/libexec (repeatable)")
    parser.add_argument("--allow-grafted", action="store_true",
                        help="accept shared libraries auditwheel grafted into the wheel (default: refuse)")
    parser.add_argument("--json", type=Path, help="write the reports here as well")
    args = parser.parse_args(argv)
    reports = []
    for wheel in args.wheels:
        if wheel.suffix != ".whl":
            print(f"SKIP {wheel.name}: not a wheel")
            continue
        report = check_wheel(wheel, args.policy, args.require_libexec, allow_grafted=args.allow_grafted)
        reports.append(report)
        if report.passed:
            elf = ", ".join(f"{PurePosixPath(k).name} glibc {v or 'none'}" for k, v in sorted(report.elf.items()))
            print(f"PASS {report.wheel}: {'.'.join(report.platforms)} within {args.policy}; {elf or 'no ELF members'}")
        for finding in report.findings:
            print(f"FAIL {report.wheel}: {finding}")
    if not reports:
        print("FAIL no wheel was given: a check over nothing cannot pass")
    if args.json:
        args.json.write_text(json.dumps([report.__dict__ | {"passed": report.passed} for report in reports],
                                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if reports and all(report.passed for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
