#!/usr/bin/env python3
"""Re-derive every table in this repository that pins THIS repository's own
sources by path and SHA-256.

WHY THIS FILE EXISTS.  Eight tables in three directories carry the digest of a
file that also lives in this tree.  Any change that moves those bytes -- a
provenance scrub, an import rewrite, the 0.2.0 package rename from
``mpas_port`` to ``hexcore`` -- lapses every one of them at once, and the
failure mode is not a clean red.  Several of these tables are read only by
gates that need a CUDA card, so a stale row there is SILENT on a developer box
and surfaces hours later, on the node, as a refusal to launch the proof the
run was booked for.  That happened three times in this tree before this file
existed:

* ``8aeb582`` edited a docstring in ``cuda_arwen_physics_v841.py`` and left
  ``EXECUTION_SOURCE_PINS`` naming the pre-edit digest, so
  ``require_frozen_execution_sources`` refused with *"frozen execution source
  changed d10eaeef... != ..."* and the x4 full-physics proof could not start.
* ``AUTHORITY_PINS_SHA256`` (``8fb3d656...``) and ``EXPECTED_RUNNER_SHA256``
  (``4d38c89a...``) arrived with the base import at ``8a34759`` and were
  killed by ``36802f8`` and ``0911c88``.  Both sat dead for weeks, because
  nothing that runs without a card reads either one.

HAND-PATCHING IS THE BUG, NOT THE FIX.  A person re-deriving forty-odd rows
with a shell loop gets forty of them.  So the derivation is a program, it
lives beside the tables it writes, and the next rename is a re-run of

    python tools/repin_source_tables.py --write

THE FIXED POINT IS REAL.  ``cuda_ftz_v841_authority_pins.json`` carries the
digest of ``run_cuda_ftz_contract.py`` INSIDE it, and ``AUTHORITY_PINS_SHA256``
is the digest of that json, so writing the first value moves the second's
input.  ``--write`` therefore iterates until a whole pass changes nothing, and
refuses to claim success if it does not converge.

WHAT THIS PROGRAM MUST NEVER TOUCH, and does not:

* ``ARWEN_SOURCE_MANIFEST`` and ``ARWEN_BUILD_COMMIT`` in
  ``hexcore.cuda_arwen_physics_v841`` -- sixteen files of the gpuwm ENGINE at
  a named engine commit.  The engine is a separate repository and did not
  change; an engine pin that appears to need moving means the port is wrong,
  not the pin.
* ``FROZEN_BATCH_CROSS_PIN`` in ``hexcore.mod.manifest`` -- engine files too.
* ``_GPUWM_STARTUP_CLOSURE_PINS`` and the ``gpuwm_sources`` / ``gpuwm_receipt``
  blocks of the ftz authority pins -- engine again.
* the ``recorded`` half of ``SOURCE_DRIFT_SINCE_CAMPAIGN`` -- the bytes an
  ARCHIVED receipt was produced by.  Only ``current`` moves; the pair is the
  gate.
* every digest of a MESH, a snapshot, a namelist or a compile manifest.  Those
  are measurements of data or of a card, not of this repository's source, and
  re-deriving one from the tree would be a fabrication.

Usage:  ``--check`` (the default) exits 1 on any drift and prints it;
        ``--write`` rewrites in place and re-checks until it converges.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

TREE = Path(__file__).resolve().parents[1]

MAX_PASSES = 8


class Drift(NamedTuple):
    """One pinned constant that disagrees with the tree."""

    table: str
    file: Path
    line: int
    label: str
    found: str
    expected: str


# ---------------------------------------------------------------------------
# digests
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def framed_set_digest(names: Iterable[str]) -> str:
    """``regional_admission.kernel_set_sha256``'s framing, re-implemented.

    Deliberately re-implemented rather than imported.  This program is the
    instrument that WRITES the minted constant; if it computed the value by
    calling the module under test it could only ever agree with itself, and a
    framing bug would be re-pinned rather than caught.  The two definitions
    are held equal by ``tests/test_cuda_regional_kernels.py``, which computes
    the digest THROUGH the module and compares it to what is written here.
    """

    digest = hashlib.sha256()
    for name in names:
        payload = (TREE / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# locating a pinned constant structurally
# ---------------------------------------------------------------------------


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def joined_constant(node: ast.expr) -> ast.Constant | None:
    """The string Constant a pin value is, seen through its parentheses."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node
    return None


def module_assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                assert node.value is not None
                return node.value
    raise LookupError(f"{name} is not assigned at module level")


def dict_rows(node: ast.expr) -> Iterator[tuple[ast.expr, ast.expr]]:
    assert isinstance(node, ast.Dict), f"expected a dict literal, got {type(node)}"
    for key, value in zip(node.keys, node.values):
        assert key is not None
        yield key, value


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------


def _string_key(key: ast.expr) -> str | None:
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _pins_by_declared_path(
    table: str, path: Path, node: ast.expr, key_of: Callable[[ast.expr], str | None]
) -> list[Drift]:
    """Rows keyed by a repo-relative source path, valued by that file's digest."""

    out: list[Drift] = []
    for key, value in dict_rows(node):
        relative = key_of(key)
        if relative is None:
            continue
        constant = joined_constant(value)
        if constant is None:  # a None pin is a deliberate hard pre-CUDA refusal
            continue
        expected = sha256_file(TREE / relative)
        if constant.value != expected:
            out.append(
                Drift(table, path, constant.lineno, relative, constant.value, expected)
            )
    return out


def table_execution_source_pins() -> list[Drift]:
    """``tools/run_cuda_v841_full_physics_x4.py::EXECUTION_SOURCE_PINS``.

    Twenty-three execution boundaries of the x4 full-physics proof.
    ``require_frozen_execution_sources`` refuses the whole run on any
    mismatch, before a device is touched.
    """

    path = TREE / "tools" / "run_cuda_v841_full_physics_x4.py"
    node = module_assignment(parse(path), "EXECUTION_SOURCE_PINS")
    return _pins_by_declared_path("EXECUTION_SOURCE_PINS", path, node, _string_key)


def table_execution_source_pins_mirror() -> list[Drift]:
    """The same twenty-three rows, asserted literally by the test suite.

    ``tests/test_cuda_v841_full_physics_x4.py`` holds a second copy so the
    table cannot move without a reviewer seeing the diff twice.  A mirror
    derived FROM its original would be worthless, so both sides are derived
    from the tree here and never from each other.
    """

    path = TREE / "tests" / "test_cuda_v841_full_physics_x4.py"
    for node in ast.walk(parse(path)):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and node.left.attr == "EXECUTION_SOURCE_PINS"
            and isinstance(node.comparators[0], ast.Dict)
        ):
            return _pins_by_declared_path(
                "EXECUTION_SOURCE_PINS (test mirror)",
                path,
                node.comparators[0],
                _string_key,
            )
    raise LookupError("the mirrored EXECUTION_SOURCE_PINS comparison is gone")


def table_capacity_copy_elisions() -> list[Drift]:
    """``tests/test_capacity_copy_elisions.py::EXPECTED_SOURCE_SHA256``.

    Three modules the bit-preserving copy-elision patch touched, keyed by
    ``SRC / "<module>.py"`` rather than by a string.
    """

    path = TREE / "tests" / "test_capacity_copy_elisions.py"
    node = module_assignment(parse(path), "EXPECTED_SOURCE_SHA256")

    def key_of(key: ast.expr) -> str | None:
        # SRC / "name.py", where SRC = ROOT / "src" / "hexcore"
        if (
            isinstance(key, ast.BinOp)
            and isinstance(key.op, ast.Div)
            and isinstance(key.left, ast.Name)
            and key.left.id == "SRC"
            and isinstance(key.right, ast.Constant)
        ):
            return f"src/hexcore/{key.right.value}"
        return None

    return _pins_by_declared_path("EXPECTED_SOURCE_SHA256", path, node, key_of)


def table_source_drift_current() -> list[Drift]:
    """``tests/test_regional_forecast_anchor.py::SOURCE_DRIFT_SINCE_CAMPAIGN``.

    Five sources that moved since the regional campaign ran.  ONLY the
    ``current`` half is derived: ``recorded`` is the byte state an archived
    receipt was produced by, which is evidence and is never rewritten.
    """

    path = TREE / "tests" / "test_regional_forecast_anchor.py"
    node = module_assignment(parse(path), "SOURCE_DRIFT_SINCE_CAMPAIGN")
    out: list[Drift] = []
    for key, value in dict_rows(node):
        relative = _string_key(key)
        assert relative is not None and isinstance(value, ast.Dict)
        for inner_key, inner_value in dict_rows(value):
            if _string_key(inner_key) != "current":
                continue
            constant = joined_constant(inner_value)
            assert constant is not None
            expected = sha256_file(TREE / relative)
            if constant.value != expected:
                out.append(
                    Drift(
                        "SOURCE_DRIFT_SINCE_CAMPAIGN.current",
                        path,
                        constant.lineno,
                        relative,
                        constant.value,
                        expected,
                    )
                )
    return out


def table_minted_kernel_set() -> list[Drift]:
    """``hexcore.cuda_backend.regional_admission.MINTED_KERNEL_SET_SHA256``.

    ONE digest over the fourteen translation units the regional step launches
    through, framed name-then-payload.  Because the NAME is framed in, the
    0.2.0 package rename moved it with no arithmetic having changed -- the
    cost a name-framed digest is designed to charge.  It is read at ADMISSION
    time, so a stale value here refuses every limited-area run at the door of
    the shipped package rather than failing a test.
    """

    path = TREE / "src" / "hexcore" / "cuda_backend" / "regional_admission.py"
    tree = parse(path)
    names = ast.literal_eval(module_assignment(tree, "REGIONAL_KERNEL_SOURCES"))
    expected = framed_set_digest(names)
    constant = joined_constant(module_assignment(tree, "MINTED_KERNEL_SET_SHA256"))
    assert constant is not None
    if constant.value == expected:
        return []
    return [
        Drift(
            "MINTED_KERNEL_SET_SHA256",
            path,
            constant.lineno,
            f"framed digest over {len(names)} regional sources",
            constant.value,
            expected,
        )
    ]


def _named_file_pin(
    table: str, path: Path, constant_name: str, relative: str
) -> list[Drift]:
    constant = joined_constant(module_assignment(parse(path), constant_name))
    assert constant is not None, f"{constant_name} is not a plain string constant"
    expected = sha256_file(TREE / relative)
    if constant.value == expected:
        return []
    return [Drift(table, path, constant.lineno, relative, constant.value, expected)]


def table_ftz_trust_pins() -> list[Drift]:
    """``tools/run_cuda_ftz_v841_trust_measurement.py``'s four instrument pins.

    ``AUTHORITY_PINS_SHA256`` is the second half of the fixed point: the json
    it digests carries ``run_cuda_ftz_contract.py``'s own digest, so it can
    only settle after that json has.
    """

    path = TREE / "tools" / "run_cuda_ftz_v841_trust_measurement.py"
    out: list[Drift] = []
    for constant_name, relative in (
        ("FROZEN_TOOL_SHA256", "tools/run_cuda_ftz_contract.py"),
        ("FROZEN_VALIDATOR_SHA256", "tools/cuda_ftz_v841_binding_validator.py"),
        ("ISOLATED_BOOTSTRAP_SHA256", "tools/cuda_ftz_v841_isolated_bootstrap.py"),
        ("AUTHORITY_PINS_SHA256", "tools/cuda_ftz_v841_authority_pins.json"),
    ):
        out += _named_file_pin(
            "ftz trust instrument pins", path, constant_name, relative
        )
    return out


def table_stabilized_products_pins() -> list[Drift]:
    """``tools/run_cuda_x1_163842_stabilized_products.py``'s two runner pins.

    The tool re-reads both files BEFORE and AFTER a run and refuses on either
    side, so a stale value here cannot be worked around at the call site.
    """

    path = TREE / "tools" / "run_cuda_x1_163842_stabilized_products.py"
    out: list[Drift] = []
    for constant_name, relative in (
        ("EXPECTED_RUNNER_SHA256", "tools/run_real_gfs_cuda_x1_163842.py"),
        (
            "EXPECTED_REMOTE_EVIDENCE_VALIDATOR_SHA256",
            "tools/validate_cuda_x1_163842_remote_evidence.py",
        ),
    ):
        out += _named_file_pin(
            "stabilized-products pins", path, constant_name, relative
        )
    return out


AUTHORITY_PINS_JSON = TREE / "tools" / "cuda_ftz_v841_authority_pins.json"


def table_authority_pins_json() -> list[Drift]:
    """``tools/cuda_ftz_v841_authority_pins.json::frozen_tool``.

    The ONLY block of this document describing a file of THIS repository.
    ``gpuwm_sources``, ``gpuwm_receipt`` and ``expected_measurement`` describe
    the engine and a card measurement and are left alone.
    ``_validate_pre_pins`` compares this block field-for-field against a fresh
    stat and digest of the tool, so ``bytes`` has to move with ``sha256`` or
    the comparison fails anyway.
    """

    document = json.loads(AUTHORITY_PINS_JSON.read_text(encoding="utf-8"))
    relative = document["frozen_tool"]["path"]
    target = TREE / relative
    expected_sha = sha256_file(target)
    expected_bytes = target.stat().st_size
    row = document["frozen_tool"]
    if row["sha256"] == expected_sha and row["bytes"] == expected_bytes:
        return []
    return [
        Drift(
            "authority pins json frozen_tool",
            AUTHORITY_PINS_JSON,
            0,
            relative,
            f"{row['sha256']} ({row['bytes']} B)",
            f"{expected_sha} ({expected_bytes} B)",
        )
    ]


#: The four contract documents this package hashes AT IMPORT, and the module
#: attribute each computation is bound to.  Every one of them frames a module
#: NAME into the document -- ``"module_key": "hexcore.cuda_gwdo_v841"`` and its
#: siblings -- so the 0.2.0 package rename moved all four with no arithmetic
#: having changed, exactly as it moved ``MINTED_KERNEL_SET_SHA256``.
CONTRACT_AUTHORITIES: tuple[tuple[str, str], ...] = (
    ("hexcore.cuda_gwdo_v841", "CUDA_GWDO_V841_CONTRACT_SHA256"),
    ("hexcore.cuda_gwdo_v841", "CUDA_GWDO_V841_KERNEL_SHA256"),
    ("hexcore.cuda_physics_v841", "CUDA_PHYSICS_V841_CONTRACT_SHA256"),
    ("hexcore.cuda_physics_v841", "CUDA_PHYSICS_V841_KERNEL_SHA256"),
    ("hexcore.cuda_physics_prep_v841", "CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256"),
    ("hexcore.cuda_physics_prep_v841", "CUDA_PHYSICS_PREP_V841_KERNEL_SHA256"),
    (
        "hexcore.cuda_arwen_physics_v841",
        "CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256",
    ),
)

#: Every place in this tree that RESTATES one of those computed digests as a
#: hand-written literal, and which authority each restatement belongs to.
#: ``(file, container, key-or-constant) -> authority attribute``.
RESTATED_CONTRACT_DIGESTS: tuple[tuple[str, str | None, str, str], ...] = (
    (
        "src/hexcore/config_v841.py",
        None,
        "V841_GWDO_CONTRACT_SHA256",
        "CUDA_GWDO_V841_CONTRACT_SHA256",
    ),
    (
        "src/hexcore/config_v841.py",
        None,
        "V841_GWDO_KERNEL_SHA256",
        "CUDA_GWDO_V841_KERNEL_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "prep_contract_sha256",
        "CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "prep_kernel_sha256",
        "CUDA_PHYSICS_PREP_V841_KERNEL_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "gwdo_contract_sha256",
        "CUDA_GWDO_V841_CONTRACT_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "gwdo_kernel_sha256",
        "CUDA_GWDO_V841_KERNEL_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "coupling_contract_sha256",
        "CUDA_PHYSICS_V841_CONTRACT_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "coupling_kernel_sha256",
        "CUDA_PHYSICS_V841_KERNEL_SHA256",
    ),
    (
        "tools/run_cuda_v841_full_physics_x4.py",
        "KNOWN_CONTRACT_PINS",
        "adapter_contract_sha256",
        "CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256",
    ),
)


def contract_authority_values() -> dict[str, str]:
    """The four computed contract digests, read from the modules themselves.

    IMPORTED, not re-implemented, and that is the opposite of the choice
    ``framed_set_digest`` makes one screen up.  The difference is which side
    this program writes.  There it WRITES the minted constant, so computing it
    through the module under test could only ever agree with itself.  Here it
    writes the MIRRORS, and the authority is by definition whatever the module
    computes at import -- that value is what the running adapter compares
    against, so a re-implementation that drifted from it would re-pin the
    mirrors to a number no run will ever produce.

    None of these four modules imports cupy at module scope, so this costs no
    card and works on a box that has none.
    """

    import importlib

    source = str(TREE / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    values: dict[str, str] = {}
    for module_name, attribute in CONTRACT_AUTHORITIES:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
        assert isinstance(value, str) and len(value) == 64, (
            f"{module_name}.{attribute} is not a sha256 hex digest"
        )
        values[attribute] = value
    return values


def table_restated_contract_digests() -> list[Drift]:
    """Hand-written copies of the four import-time contract digests.

    THE BREAKAGE THIS PREVENTS, measured on the card on 2026-08-28 and not
    inferred: ``hexcore.config_v841.V841_GWDO_CONTRACT_SHA256`` restated the
    GWDO contract digest, the rename moved the digest and not the restatement,
    and ``CudaPhaseOneExecutionProvenanceV841.validate`` then refused EVERY
    full-physics forecast at the first composite step --

        ValueError: external GWDO provenance gwdo_contract_sha256 changed:
        <what the module computed> != <what config_v841 restated>

    -- with the whole battery green, because the only test that read the
    sibling table restated the table's own values instead of measuring them.
    That is the same silent shape the module docstring above records twice:
    a pin nothing checks without a card is a pin that surfaces on the node.

    ``tests/test_cuda_v841_full_physics_x4.py`` now compares every row here to
    its authority, so this table has a CPU-tier gate as well as a re-derivation.
    """

    authorities = contract_authority_values()
    out: list[Drift] = []
    for relative, container, key, attribute in RESTATED_CONTRACT_DIGESTS:
        path = TREE / relative
        tree = parse(path)
        if container is None:
            node = module_assignment(tree, key)
        else:
            holder = module_assignment(tree, container)
            if isinstance(holder, ast.Call):  # MappingProxyType({...})
                holder = holder.args[0]
            node = None
            for row_key, row_value in dict_rows(holder):
                if _string_key(row_key) == key:
                    node = row_value
                    break
            assert node is not None, f"{relative}::{container} has no row {key!r}"
        constant = joined_constant(node)
        assert constant is not None, f"{relative}::{key} is not a plain string"
        expected = authorities[attribute]
        if constant.value != expected:
            out.append(
                Drift(
                    "restated contract digests",
                    path,
                    constant.lineno,
                    f"{container + '::' if container else ''}{key} -> {attribute}",
                    constant.value,
                    expected,
                )
            )
    return out


GATED_FILES = TREE / "evidence" / "assembly-rehearsal-20260828" / "gated-files.txt"


def gated_file_list() -> list[str]:
    """Every source of this tree that sits under a digest gate, sorted.

    The union of the x4 frozen execution set and the regional kernel set.  It
    is the input ``evidence/assembly-rehearsal-20260828/check_gated_files.py``
    takes, and the public assembly's section 4a asks it one question: would
    the provenance scrub move bytes under a pin?  A list naming retired paths
    answers "no" for every file, which is the wrong answer delivered quietly
    -- the check prints ``MISSING FROM ASSEMBLY`` per line and reports no hit.
    """

    runner = parse(TREE / "tools" / "run_cuda_v841_full_physics_x4.py")
    pins = ast.literal_eval(module_assignment(runner, "EXECUTION_SOURCE_PINS"))
    admission = parse(
        TREE / "src" / "hexcore" / "cuda_backend" / "regional_admission.py"
    )
    kernels = ast.literal_eval(module_assignment(admission, "REGIONAL_KERNEL_SOURCES"))
    return sorted(set(pins) | set(kernels))


def table_gated_files() -> list[Drift]:
    declared = [line for line in GATED_FILES.read_text(encoding="utf-8").split() if line]
    expected = gated_file_list()
    if declared == expected:
        return []
    return [
        Drift(
            "gated-files.txt",
            GATED_FILES,
            0,
            f"{len(expected)} gated sources",
            f"{len(declared)} rows, first {declared[0] if declared else '<empty>'}",
            f"{len(expected)} rows, first {expected[0]}",
        )
    ]


TABLES: tuple[tuple[str, Callable[[], list[Drift]]], ...] = (
    ("EXECUTION_SOURCE_PINS", table_execution_source_pins),
    ("EXECUTION_SOURCE_PINS mirror", table_execution_source_pins_mirror),
    ("EXPECTED_SOURCE_SHA256", table_capacity_copy_elisions),
    ("SOURCE_DRIFT_SINCE_CAMPAIGN.current", table_source_drift_current),
    ("MINTED_KERNEL_SET_SHA256", table_minted_kernel_set),
    ("ftz authority pins json", table_authority_pins_json),
    ("ftz trust instrument pins", table_ftz_trust_pins),
    ("stabilized-products pins", table_stabilized_products_pins),
    ("gated-files.txt", table_gated_files),
    ("restated contract digests", table_restated_contract_digests),
)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def rewrite_line(path: Path, line: int, found: str, expected: str) -> None:
    """Replace one 64-hex literal on one line, refusing anything ambiguous."""

    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    index = line - 1
    if lines[index].count(found) != 1:
        raise SystemExit(
            f"{path}:{line} does not hold exactly one copy of {found[:16]}...; "
            "refusing to guess which occurrence is the pin"
        )
    lines[index] = lines[index].replace(found, expected)
    path.write_text("\n".join(lines), encoding="utf-8", newline="")


def apply(drifts: list[Drift]) -> None:
    for drift in drifts:
        if drift.file == AUTHORITY_PINS_JSON:
            document = json.loads(AUTHORITY_PINS_JSON.read_text(encoding="utf-8"))
            target = TREE / document["frozen_tool"]["path"]
            document["frozen_tool"]["sha256"] = sha256_file(target)
            document["frozen_tool"]["bytes"] = target.stat().st_size
            AUTHORITY_PINS_JSON.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="",
            )
        elif drift.file == GATED_FILES:
            GATED_FILES.write_text(
                "\n".join(gated_file_list()) + "\n", encoding="utf-8", newline=""
            )
        else:
            rewrite_line(drift.file, drift.line, drift.found, drift.expected)


def collect() -> list[Drift]:
    out: list[Drift] = []
    for _, table in TABLES:
        out += table()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="re-derive the self-referential pins")
    parser.add_argument("--write", action="store_true", help="rewrite in place")
    arguments = parser.parse_args()

    if not arguments.write:
        drifts = collect()
        for drift in drifts:
            where = f"{drift.file.relative_to(TREE).as_posix()}:{drift.line}"
            print(
                f"DRIFT {drift.table} {where} {drift.label}\n"
                f"      found    {drift.found}\n"
                f"      expected {drift.expected}"
            )
        print(f"{len(drifts)} pinned value(s) disagree with the tree")
        return 1 if drifts else 0

    total = 0
    for attempt in range(1, MAX_PASSES + 1):
        drifts = collect()
        if not drifts:
            print(
                f"converged after {attempt - 1} writing pass(es); "
                f"{total} value(s) moved"
            )
            return 0
        for drift in drifts:
            print(
                f"pass {attempt}: {drift.table} "
                f"{drift.file.relative_to(TREE).as_posix()} {drift.label} "
                f"{drift.found[:16]}... -> {drift.expected[:16]}..."
            )
        apply(drifts)
        total += len(drifts)
    raise SystemExit(
        f"did not converge in {MAX_PASSES} passes; a pinned file's digest "
        "depends on itself in a way this program cannot settle"
    )


if __name__ == "__main__":
    sys.exit(main())
