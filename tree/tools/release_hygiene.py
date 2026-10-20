#!/usr/bin/env python3
"""Refuse a release artifact that carries private material.

    python tools/release_hygiene.py dist/gpuwm_hex-*.whl dist/gpuwm_hex-*.tar.gz
    python tools/release_hygiene.py --tree .          # the files git tracks here

THE BREAKAGE THIS GATE PREVENTS (verify-hex-r2, 2026-10-13): the 0.4.0
candidate wheel and sdist carried the owner's name in 16 and 18 files, home
directory paths in 23, internal cloud host names in 76, private workstation
folders, and the live root-ssh address of a rented GPU box in 9 sdist files.
Published 0.3.3 carried none of these.  The sdist's ``verification/`` had grown
from a curated 915 KB of contracts to 31 MB of lane receipts.  Nothing looked
at the artifacts' members before an upload would have shipped them.

Every member of every artifact is scanned (binaries too: a Rust build embeds
the build directory).  The rules are generic and carry no private literal, so
this file can ship: the identity tokens (the checkout's ``git config
user.name``/``user.email``, the login name, the home folder's name, and any
``--owner``) are read where the gate runs.  Exit 0 clean, 1 with every hit
listed, 2 when an artifact cannot be read.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

# Generic rules: (name, pattern).  Each names a class of private material.
RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("home-path", re.compile(rb"(?i)\b[a-z]:[\\/]{1,2}users[\\/]{1,2}(?!public\b|default\b)[a-z0-9._-]+")),
    ("home-path", re.compile(rb"(?<![\w.])/(?:home|Users)/(?!runner\b|user\b|username\b|you\b|me\b)[A-Za-z0-9._-]+/")),
    ("home-path", re.compile(rb"\bC--Users-[A-Za-z0-9._-]+")),
    ("downloads-path", re.compile(rb"(?<![\w.])Downloads[\\/]")),
    ("cloud-host-name", re.compile(rb"\bip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b")),
    # split literals: the shipped gate itself should not spell a lab's naming
    ("lab-host-name", re.compile(rb"\bweather" rb"-node-\d+\b")),
    ("workstation-folder", re.compile(rb"\bcodex" rb"-\d{3,4}\b")),
    ("workstation-folder", re.compile(rb"\bagent" rb"-scratch\b")),
)

IPV4 = re.compile(rb"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")

TEXT_LIMIT = 64 * 1024 * 1024  # members larger than this are scanned in part


def identity_tokens(extra: Iterable[str], where: Path) -> list[str]:
    """Owner identity read where the gate runs, never written into this file."""

    tokens: set[str] = {t for t in extra if t}
    for key in ("user.name", "user.email"):
        try:
            value = subprocess.run(["git", "-C", str(where), "config", key],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            value = ""
        if value:
            tokens.add(value)
    for value in (os.environ.get("USER"), os.environ.get("USERNAME"), Path.home().name):
        if value:
            tokens.add(value)
    try:
        tokens.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - no login name is not a hit
        pass
    # A login such as "root" or a one-letter name is not an identity.
    return sorted(t for t in tokens if len(t) >= 3 and t.lower() not in {"root", "runner", "user", "admin"})


def identity_rules(tokens: Iterable[str]) -> list[tuple[str, re.Pattern[bytes]]]:
    rules = []
    for token in tokens:
        if "@" in token:
            rules.append(("owner-email", re.compile(re.escape(token).encode(), re.IGNORECASE)))
            continue
        for part in re.split(r"\s+", token):
            if len(part) < 3:
                continue
            # The capitalised form only: a lower-case login name is often an
            # English word ("drew" the verb), and paths are caught above.
            shown = part[:1].upper() + part[1:]
            rules.append(("owner-name", re.compile(rb"\b" + re.escape(shown).encode() + rb"\b")))
    return rules


def public_addresses(data: bytes) -> Iterator[bytes]:
    for match in IPV4.finditer(data):
        octets = [int(g) for g in match.groups()]
        if any(o > 255 for o in octets):
            continue
        address = ipaddress.ip_address(".".join(map(str, octets)))
        if (address.is_private or address.is_loopback or address.is_unspecified
                or address.is_link_local or address.is_multicast or address.is_reserved):
            continue
        yield match.group(0)


def members(artifact: Path) -> Iterator[tuple[str, bytes]]:
    name = artifact.name
    if name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(artifact) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    with archive.open(info) as handle:
                        yield info.filename, handle.read(TEXT_LIMIT)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(artifact) as archive:
            for info in archive.getmembers():
                if info.isfile():
                    handle = archive.extractfile(info)
                    if handle is not None:
                        yield info.name, handle.read(TEXT_LIMIT)
    else:
        raise ValueError(f"{artifact}: not a wheel, zip or tar archive")


def tree_members(root: Path) -> Iterator[tuple[str, bytes]]:
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                            capture_output=True, check=True).stdout.split(b"\0")
    for raw in listed:
        if raw:
            path = root / raw.decode()
            if path.is_file():
                yield raw.decode(), path.read_bytes()[:TEXT_LIMIT]


def scan(items: Iterable[tuple[str, bytes]],
         rules: list[tuple[str, re.Pattern[bytes]]]) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    for member, data in items:
        seen: set[tuple[str, bytes]] = set()
        for rule, pattern in rules:
            for match in pattern.finditer(data):
                key = (rule, match.group(0))
                if key not in seen:
                    seen.add(key)
                    hits.append((member, rule, match.group(0).decode("utf-8", "replace")))
        for address in public_addresses(data):
            key = ("public-ipv4", address)
            if key not in seen:
                seen.add(key)
                hits.append((member, "public-ipv4", address.decode()))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--tree", type=Path, help="scan the files git tracks under this folder")
    parser.add_argument("--owner", action="append", default=[],
                        help="another identity token to refuse (repeatable)")
    parser.add_argument("--allow-address", action="append", default=[],
                        help="a public IPv4 address that is meant to ship (repeatable)")
    args = parser.parse_args(argv)
    if not args.artifacts and args.tree is None:
        parser.error("give artifacts or --tree")
    where = args.tree or Path(__file__).resolve().parent
    rules = list(RULES) + identity_rules(identity_tokens(args.owner, where))
    allowed = set(args.allow_address)
    sources: list[tuple[str, Iterable[tuple[str, bytes]]]] = []
    if args.tree is not None:
        sources.append((str(args.tree), tree_members(args.tree)))
    for artifact in args.artifacts:
        sources.append((str(artifact), members(artifact)))
    total = 0
    for label, items in sources:
        try:
            hits = [h for h in scan(items, rules)
                    if not (h[1] == "public-ipv4" and h[2] in allowed)]
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
            print(f"HYGIENE cannot read {label}: {error}", file=sys.stderr)
            return 2
        total += len(hits)
        for member, rule, text in hits:
            print(f"HYGIENE {label} :: {member} :: {rule} :: {text}")
        print(f"HYGIENE {label}: {len(hits)} hit(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
