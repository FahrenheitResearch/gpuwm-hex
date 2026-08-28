"""Per-architecture FTZ/subnormal characterization of the port's own kernels.

Runs the numerical-contract decks the project already owns — the scalar
transport subnormal deck, the guarded-kernel subnormal audit, the normalized
fallback performance control, and the v8.4.1 95-entrypoint audit — on the
live card, and writes one JSON receipt recording, per deck, either the deck's
own result or the named refusal it raised.

This driver exists for architecture characterization: unlike
``tools/run_cuda_ftz_contract.py`` it does not require a gpuwm FTZ probe
receipt, so it can measure the port's kernels on a card even while the
engine-side probe has an open drift.  It builds the compile manifests exactly
the way the contract tool does (KernelCache + raw_kernels over the frozen TU
inventories) and validates them through the same relations.  It makes no
authority claim; it measures.

Usage:
    python tools/run_cuda_arch_ftz_decks.py --output <receipt.json> \
        --cache-root <fresh-dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.cuda_backend import KernelCache, require_cuda  # noqa: E402
from hexcore.cuda_ftz import (  # noqa: E402
    canonical_sha256,
    production_translation_units,
    run_guarded_kernel_subnormal_audit,
    run_normalized_fallback_performance_control,
    run_scalar_transport_subnormal_deck,
    v841_reached_translation_units,
    validate_compile_manifest_relation,
    validate_v841_compile_manifest_relation,
)


def _attempt(label: str, work: Callable[[], Any]) -> dict[str, Any]:
    """Run one deck; a named refusal is a recorded outcome, not a crash."""

    try:
        result = work()
    except Exception as error:  # noqa: BLE001 - every refusal is evidence here
        return {
            "status": "refused",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-4:],
        }
    return {"status": "measured", "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    arguments = parser.parse_args()

    cache_root = Path(arguments.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_root)

    receipt: dict[str, Any] = {
        "schema": "mpas-port.cuda-arch-ftz-decks/v1",
        "capability": capability.as_dict(),
        "decks": {},
    }

    def compile_manifest(inventory: dict[str, Any]) -> dict[str, Any]:
        cache = KernelCache(capability=capability, cache_dir=cache_root)
        for module_key, (source, names) in sorted(inventory.items()):
            cache.raw_kernels(names, source, module_key=module_key)
        return cache.compile_manifest()

    manifest_v823 = _attempt(
        "compile-manifest-v8.2.3",
        lambda: compile_manifest(dict(production_translation_units())),
    )
    receipt["decks"]["compile_manifest_v823"] = (
        {
            "status": "measured",
            "compile_manifest_sha256": canonical_sha256(
                manifest_v823["result"]
            ),
            "relation": _attempt(
                "relation",
                lambda: validate_compile_manifest_relation(
                    manifest_v823["result"]
                )["compile_platform"]["fingerprint"],
            ),
        }
        if manifest_v823["status"] == "measured"
        else manifest_v823
    )

    receipt["decks"]["transport_deck"] = _attempt(
        "transport-deck", run_scalar_transport_subnormal_deck
    )
    receipt["decks"]["guarded_kernel_audit"] = _attempt(
        "guarded-kernel-audit", run_guarded_kernel_subnormal_audit
    )
    receipt["decks"]["normalized_performance_control"] = _attempt(
        "normalized-performance-control",
        run_normalized_fallback_performance_control,
    )

    def v841_audit() -> dict[str, Any]:
        from hexcore.cuda_ftz_v841 import (
            run_v841_guarded_kernel_subnormal_audit,
        )

        manifest = compile_manifest(dict(v841_reached_translation_units()))
        validate_v841_compile_manifest_relation(manifest)
        return run_v841_guarded_kernel_subnormal_audit(
            compile_manifest=manifest
        )

    receipt["decks"]["v841_kernel_audit"] = _attempt(
        "v841-kernel-audit", v841_audit
    )

    statuses = {
        name: deck.get("status") for name, deck in receipt["decks"].items()
    }
    receipt["statuses"] = statuses
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=repr) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "statuses": statuses}, indent=2))
    return 0 if all(value == "measured" for value in statuses.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
