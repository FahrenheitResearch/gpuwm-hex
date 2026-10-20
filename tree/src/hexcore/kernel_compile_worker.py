"""CPU-only NVRTC replay in an isolated process, never a CUDA context.

The explicit target selector is the same one the AOT builder uses. All
compiler options, including the engine's final numerical options, come from
the parent request. Only compilation runs here; the parent loads its image.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache


@lru_cache(maxsize=1)
def _identity():
    # Each spawned worker keeps the same loaded compiler for its lifetime.
    from .cuda_backend.aot import compiler_identity

    return compiler_identity()


def compile_request(arguments, arch, identity, warm=None):
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from .cuda_backend.aot import preserve_final_options

    if _identity() != identity:
        raise ValueError("kernel-plan worker compiler identity changed")
    if warm is not None:
        from . import kernel_warm
        kernel_warm.install(**warm)
    target = str(arch).removeprefix("sm_")
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError("kernel-plan worker requires the explicit supported cubin target")
    selector = compiler._get_arch_for_options_for_nvrtc

    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError("kernel-plan worker target changed")
        return "-arch=sm_" + target, "cubin"

    wall, cpu = time.perf_counter(), time.process_time()
    compiler._get_arch_for_options_for_nvrtc = explicit
    try:
        with preserve_final_options():
            result = compiler._compile_using_nvrtc_no_warning(
                arguments["source"], tuple(arguments["options"]), target,
                arguments["filename"], arguments["name_expressions"], None,
                arguments["cache_in_memory"], None, arguments["method"])
    finally:
        compiler._get_arch_for_options_for_nvrtc = selector
    return result, {"pid": os.getpid(), "seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu}
