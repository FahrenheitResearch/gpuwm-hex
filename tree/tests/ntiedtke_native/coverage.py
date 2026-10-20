"""Inventory unvisited native gcov branches without treating them as covered."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from hexcore.mpasphys.ntiedtke_v842 import MPAS_ROUTINES


def inventory(directory: Path, output: Path, certificate: Path | None = None):
    files = {}
    for path in sorted(directory.glob("*.gcov")):
        if path.name not in {"cu_ntiedtke.F90.gcov", "cu_ntiedtke_pre.F.gcov",
                             "cu_ntiedtke_post.F.gcov", "module_cu_ntiedtke.F.gcov"}:
            continue  # Auxiliary accessibility/trace builds are separate receipts.
        gaps, branches, visited, taken = [], 0, 0, 0
        line, text = 0, ""
        for row in path.read_text().splitlines():
            match = re.match(r"\s*([^:]+):\s*(\d+):(.*)", row)
            if match:
                line, text = int(match[2]), match[3].strip()
            elif row.startswith("branch "):
                branches += 1
                executed = "never executed" not in row
                count = re.search(r"\btaken\s+(\d+)(?:%)?", row)
                covered = executed and count is not None and int(count[1]) > 0
                visited += executed
                taken += covered
                if not covered:
                    routines = MPAS_ROUTINES if path.name == "cu_ntiedtke.F90.gcov" else {}
                    routine = next((n for n, (lo, hi) in routines.items()
                                    if lo <= line <= hi), "driver wrapper")
                    gaps.append(dict(line=line, routine=routine, source=text,
                                     gcov=row.strip(), executed=executed))
        files[path.name] = dict(branches=branches, executed=visited,
                               taken_at_least_once=taken, uncovered=gaps)
    result = dict(schema="mpas-ntiedtke-native-coverage/v1", native_host="reference-host",
                  native_host_identity_certified=False,
                  diagnostic_branch_inventory=True, files=files)
    if certificate is not None:
        raw = certificate.read_bytes()
        ruling = json.loads(raw)
        if not (ruling.get("all_fields_identical") is True
                and ruling.get("all_floating_words_identical") is True):
            raise ValueError("native host certificate lacks exact-word proof")
        # This inventory combines synthetic profiles and several real meshes
        # and dates. A scoped full-forecast certificate cannot certify that
        # combined matrix. Retain its authority without extending its scope.
        result["native_host_identity_certificate_available"] = True
        result["native_host_identity_certificate"] = dict(
            sha256=hashlib.sha256(raw).hexdigest(), ruling=ruling)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    inventory(Path(sys.argv[1]), Path(sys.argv[2]),
              Path(sys.argv[3]) if len(sys.argv) > 3 else None)
