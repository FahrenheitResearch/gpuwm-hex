#!/usr/bin/env python3
"""Compare each executable cubin section against the real CuPy JIT route."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile

from elftools.elf.elffile import ELFFile


def executable(blob):
    elf = ELFFile(io.BytesIO(blob))
    return {section.name: section.data() for section in elf.iter_sections()
            if section.name.startswith(".text.")}


def main():
    import cupy as cp
    from cupy.cuda import compiler
    from hexcore.kernel_pack import read
    from hexcore.cuda_backend.aot_build import inventory
    from hexcore.cuda_backend.aot import compiler_identity

    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    original = compiler._compile_using_nvrtc_no_warning
    current = "sm_" + cp.cuda.Device().compute_capability
    sources = {entry["source_sha256"]: entry["source"] for entry in inventory()}
    for row in read(args.pack, compiler_identity()):
        row = {**row, "source": row.get("source") or sources[row["source_sha256"]]}
        from hexcore.cuda_backend.aot import local_options
        row["options"] = local_options(row["options"])
        captured = []
        def capture(*a, **kw):
            result = original(*a, **kw)
            captured.append(result[0])
            return result
        compiler._compile_using_nvrtc_no_warning = capture
        try:
            # Each entry gets a truly empty disk cache. Explicit targets retain
            # CuPy's unmodified target selector and complete JIT option route.
            with tempfile.TemporaryDirectory() as directory:
                import os
                previous = os.environ.get("CUPY_CACHE_DIR")
                os.environ["CUPY_CACHE_DIR"] = directory
                try:
                    try:
                        module = compiler._compile_with_cache_cuda(
                            row["source"], tuple(row["options"]),
                            row["arch"].removeprefix("sm_"), cache_in_memory=True,
                            **row.get("compile_request", {}))
                        loaded = True
                    except cp.cuda.driver.CUDADriverError:
                        if row["arch"] == current or not captured:
                            raise
                        # A different-SM cubin is compiled, but this card cannot
                        # execute it. This is explicitly a codegen-only check.
                        loaded = False
                finally:
                    if previous is None:
                        os.environ.pop("CUPY_CACHE_DIR", None)
                    else:
                        os.environ["CUPY_CACHE_DIR"] = previous
        finally:
            compiler._compile_using_nvrtc_no_warning = original
        if not captured:
            raise ValueError("JIT returned no newly compiled image")
        import lzma
        aot = executable(lzma.decompress(base64.b64decode(row["cubin_xz_base64"]))
                         if "cubin_xz_base64" in row else base64.b64decode(row["cubin_base64"]))
        jit = executable(captured[-1])
        if not aot or aot != jit:
            raise ValueError(f"SASS differs: {row['module_key']} {row['arch']}")
        records.append({"module_key": row["module_key"], "arch": row["arch"],
                        "loaded_on_current_device": loaded,
                        "kernels": {name: {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
                                    for name, blob in sorted(aot.items())}, "sass_identical": True})
    args.output.write_text(json.dumps({"verdict": "PASS", "device": current,
                                      "entries": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
