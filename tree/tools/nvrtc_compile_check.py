#!/usr/bin/env python3
"""Compile every element-parallel translation unit under NVRTC and resolve
every kernel it declares, on the card this process can see.

The check a merge of CUDA source strings needs before any card time is
spent on it: two lanes editing the same translation unit can each compile
and the union not, and the first thing that would notice is the forecast
door refusing at its first launch, seven minutes into a contract deck.
Each unit named by ``tools/kernel_element_ab.py``'s ``SOURCE_ATTRIBUTES``
is compiled with the instrument's NVRTC options and every ``__global__``
symbol it declares -- directly or through a ``DECLARE_*_KERNEL`` macro
instantiation -- is resolved through ``get_function``, which is where an
undefined symbol or a bad launch bound surfaces.

Prints one line per unit and a final ``compile check PASSED`` or
``compile check FAILED``; the exit code follows the verdict.  Needs a card
(CuPy loads the module), so it is a chain step and not a test.
"""

from __future__ import annotations

import importlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def declared_kernels(source: str) -> list[str]:
    names = [n for n in re.findall(r"__global__ void (\w+)", source) if n != "NAME"]
    names += [n for n in re.findall(r"DECLARE_\w+_KERNEL\((\w+), ", source) if n != "NAME"]
    return sorted(set(names))


def main() -> int:
    import cupy as cp

    from kernel_element_ab import NVRTC_OPTIONS, SOURCE_ATTRIBUTES

    failed = 0
    for key, (module_name, attribute) in SOURCE_ATTRIBUTES.items():
        source = getattr(importlib.import_module(module_name), attribute)
        started = time.perf_counter()
        try:
            module = cp.RawModule(code=source, options=NVRTC_OPTIONS, backend="nvrtc")
            names = declared_kernels(source)
            for name in names:
                module.get_function(name).attributes
            print(f"OK   {key:36s} {len(names):3d} kernels  {time.perf_counter() - started:5.1f} s", flush=True)
        except Exception as error:  # noqa: BLE001 - the verdict is the point
            failed += 1
            print(f"FAIL {key}: {str(error)[:1500]}", flush=True)
    print("compile check", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
