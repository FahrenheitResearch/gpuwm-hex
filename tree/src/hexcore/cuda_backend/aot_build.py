"""Wheel-time CUDA driver orchestration. No CUDA arithmetic lives here."""
from __future__ import annotations
from hexcore import numerics as _hex_numerics

import ast
import base64
import hashlib
import importlib
import json
import lzma
from pathlib import Path

from .aot import (ARCHITECTURES, compiler_identity, local_options, portable_options,
                  preserve_final_options, recipe, source_hash)


def inventory(package_root=None, numerics=None, compiler_family=None, *, compiler=None):
    """Read production KernelCache declarations, using their source bindings.

    ``numerics`` names the kernel numerics the rows compile with (``fast`` or ``strict``,
    :mod:`hexcore.numerics`); ``None`` is this process's mode.
    ``compiler`` binds the recorded CuPy header layout to its CUDA family.
    ``compiler_family`` selects that family's recorded include paths; ``None``
    retains all catalogs for inspection. The legacy catalog belongs to CUDA 13.

    Proof instruments have local source variables and are excluded. Production
    declarations that cannot be resolved fail the build rather than disappear.
    """
    root = Path(package_root or Path(__file__).resolve().parents[1])
    result = {}
    excluded = {"cuda_ftz.py", "cuda_ftz_v841.py", "cuda_dualrun.py"}
    for path in sorted(root.rglob("*.py")):
        if path.name in excluded or "drivers" in path.parts:
            continue
        # Shape resolution forwards runtime source/key arguments. Its generic
        # carriers remain declared at the callers; concrete shapes are recorded
        # by first-use discovery, rather than evaluated from this forwarding ABI.
        if path.name in {"runtime.py", "aot.py", "aot_build.py", "cuda_shape.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("raw_kernel", "raw_kernels") and len(n.args) >= 2]
        if not calls:
            continue
        module_name = "hexcore." + ".".join(path.relative_to(root).with_suffix("").parts)
        namespace = vars(importlib.import_module(module_name))
        # The resident launcher takes a unit and source as arguments. Resolve
        # its two production bindings using the same source providers, without
        # constructing a session or querying a GPU during wheel discovery.
        bindings = [namespace]
        if module_name == "hexcore.mpasphys.resident_v842":
            bindings = [dict(namespace, unit="latitude", source=namespace["_LATITUDE"]),
                        dict(namespace, unit="preparation", source=namespace["preparation_source"]())]
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            if "module_key" not in keywords:
                raise ValueError(f"{path}:{call.lineno}: missing module_key")
            for binding in bindings:
                def value(node):
                    return eval(compile(ast.Expression(node), str(path), "eval"), binding)
                try:
                    key = value(keywords["module_key"])
                    source = value(call.args[1])
                    extra = value(keywords["options"]) if "options" in keywords else ()
                except Exception as error:
                    raise ValueError(f"{path}:{call.lineno}: unresolved kernel declaration: {error}") from error
                row = {"module_key": key, "source": source,
                       "options": [*_hex_numerics.kernel_base(numerics), *extra],
                       "source_sha256": source_hash(source)}
                if module_name == "hexcore.mpasphys.resident_v842":
                    # These calls use MpasPhysicsKernelCache, whose carrier
                    # route appends these flags and bypasses RawModule's FTZ
                    # override. Record that actual compile request.
                    carrier = (*row["options"], "--fmad=false", "--ftz=false",
                               "--prec-div=true", "--prec-sqrt=true")
                    row = recipe({"source": source,
                                  "options": _hex_numerics.options(carrier, numerics=numerics),
                                  "route": "direct-nvrtc", "filename": "kern.cu"})
                digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                result[digest] = row
    # Direct-NVRTC regional declarations retain native subnormal words.
    from hexcore.cuda_regional_v841 import (
        MODULE_KEY, CUDA_REGIONAL_SOURCE, REGIONAL_CUDA_OPTIONS,
    )
    regional = {"module_key": MODULE_KEY, "source": CUDA_REGIONAL_SOURCE,
                "options": list(_hex_numerics.options(REGIONAL_CUDA_OPTIONS, numerics=numerics)),
                "source_sha256": source_hash(CUDA_REGIONAL_SOURCE),
                "compiler_route": "direct-nvrtc"}
    digest = hashlib.sha256(json.dumps(regional, sort_keys=True).encode()).hexdigest()
    result[digest] = regional
    for row in physics_inventory(numerics):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        result[digest] = row
    from . import arch_admission
    from hexcore.cuda_fp32 import CUDA_FTZ_HELPERS
    probe = recipe({"source": CUDA_FTZ_HELPERS + arch_admission._PROBE_KERNEL,
                    "options": list(_hex_numerics.kernel_base(numerics))})
    result[probe["module_key"]] = probe
    # Preserve each compiler family's exact include layout. A CUDA 13
    # recording has nvidia/cu13, while CUDA 12 requests cuda_runtime; treating
    # those options as aliases would weaken the loader's compile contract.
    # Only source-bound KernelCache declarations suppress recorded copies.
    # Earlier catalogs must not suppress a later family's different options.
    declared = {row["source_sha256"] for row in result.values()}
    catalogs = sorted((root / "data").glob("compiler-recipes*.json"))
    if compiler_family is not None:
        family_catalog = root / "data" / f"compiler-recipes-cuda{int(compiler_family)}.json"
        if not family_catalog.is_file() and int(compiler_family) == 13:
            family_catalog = root / "data" / "compiler-recipes.json"
        if not family_catalog.is_file():
            raise ValueError(f"no compiler recipe catalog for CUDA {compiler_family}: {family_catalog}")
        catalogs = [family_catalog]
    for catalog in catalogs:
        entries = json.loads(catalog.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or len(entries) > 4000:
            raise ValueError("compiler recipe inventory is malformed or oversized")
        # A KernelCache module is served under its own module key before
        # CuPy is asked, so the same source recorded through CuPy is skipped.
        for entry in entries:
            row = catalog_recipe(entry, compiler)
            if row["source_sha256"] in declared:
                continue
            digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            result[digest] = row
    if not result:
        raise ValueError("empty CUDA module inventory cannot produce an AOT wheel")
    return list(result.values())


_PIP_CUDA_INCLUDE = "-I@SITE@/nvidia/"


def toolkit_includes():
    """The CUDA header ``-I`` options this CuPy appends, when they name a toolkit outside site-packages.

    CuPy adds its CUDA include directory from the NVRTC it loaded: the pip
    headers (``<site-packages>/nvidia/cu13/include``) or, when a system toolkit
    shadows the pip wheel, ``/usr/local/cuda/include``.  ``None`` for the pip
    layout or without CuPy, so the recorded layout stands.
    """
    try:
        from cupy._core import core
        assembled = core.assemble_cupy_compiler_options(())
    except Exception:
        return None
    portable = portable_options([option for option in assembled if str(option).startswith("-I")])
    toolkit = [option for option in portable if not option.startswith("-I@SITE@/")]
    return toolkit or None


def catalog_recipe(entry, compiler=None):
    """Bind recorded CuPy include roots to the compiler the rows are formed for.

    CUDA 12 and 13 pip runtimes put headers in different directories. Keep
    those paths exact in the pack so production selection needs no aliases.
    With ``compiler`` (this process's compiler) and a system toolkit's headers
    in CuPy's own options, the recorded pip include becomes that toolkit's.
    THE BREAKAGE THIS ENDS (CUDA 13.0 driver 580, 2026-10-24): CuPy
    loaded the system libnvrtc and compiled with ``-I/usr/local/cuda/include``,
    so 150 of 217 first-run requests matched neither the pack nor the
    ahead-of-time pool and compiled in the foreground on one core.
    """
    options = portable_options(entry.get("options", ()))
    version = (compiler or {}).get("nvrtc") or ()
    if version and version[0] == 12:
        options = ["-I@SITE@/nvidia/cuda_runtime/include"
                   if option == "-I@SITE@/nvidia/cu13/include" else option
                   for option in options]
    toolkit = toolkit_includes() if compiler is not None else None
    if toolkit and any(option.startswith(_PIP_CUDA_INCLUDE) for option in options):
        bound = []
        for option in options:
            if option.startswith(_PIP_CUDA_INCLUDE):
                bound.extend(toolkit)
                toolkit = []
            else:
                bound.append(option)
        options = bound
    return recipe({**entry, "options": options})


def physics_inventory(numerics=None):
    """Direct MYNN, MYNN-EDMF and sfclay images, in exact and corrected forms.

    Raw NVRTC call sites are not KernelCache declarations and were absent from
    every wheel inventory, leaving minutes of serial compilation on a new box.
    Assemble these with the production source providers and the same flags.
    """
    from hexcore.mpasphys import bl_mynn_v842 as bl, sf_mynn_v842 as sf, sfclay_mm5_v842 as mm5
    from hexcore.mpasphys import resources_build as resources
    from hexcore.mpasphys import noah_v842 as noah, rrtmg_lw_v842 as lw
    from hexcore.mpasphys import gf_mpas_v842 as gf, thompson_v842 as thompson
    from hexcore.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    from hexcore.cuda_fp32 import CUDA_FTZ_HELPERS
    from hexcore.mynn_glibc_math import mynn_glibc_source

    declarations = [
        ("bl_mynn_adapter", bl.ADAPTER_CUDA_SOURCE, bl.MYNN_CUDA_OPTIONS, "kern.cu"),
        ("bl_mynn", CUDA_FTZ_HELPERS + resources.mynn_maths(resources._vendored_maths())
         + bl.MYNN_ARRAY_SUPPORT + bl.MYNN_MODULE_CUDA, bl.MYNN_CUDA_OPTIONS, "kern.cu"),
        ("bl_mynnedmf", CUDA_FTZ_HELPERS + mynn_glibc_source() + edmf.ARRAY_SUPPORT
         + edmf.MODULE_CUDA, edmf.CUDA_OPTIONS, "rrfs_mynnedmf.cu"),
    ]
    for corrections in ((), ("RP3",)):
        declarations.extend([
            ("sf_mynn", sf._cuda_source(frozenset(corrections)), sf.CUDA_OPTIONS, "kern.cu"),
            ("sfclay_mm5", mm5._shared_math_source() + mm5._cuda_text(corrections),
             mm5.COMPILE_OPTIONS, "sfclay_mm5_v842.cu"),
        ])
    noah_source = (noah._device_math_source() + noah._CORE_CUDA + noah._ENTRY_CUDA
                   + noah._LANDUSE_CUDA + noah._MPAS_IO_CUDA + noah._STATE_CUDA)
    noah_source = noah._shared_tables_source(noah_source)
    lw_source = (lw._solver_source() + "\n" + lw._RANDOM_SOURCE + "\n"
                 + lw._DRIVER_SOURCE + "\n" + lw._INIT_SOURCE)
    lw_source = lw_source.replace('#define RLW_MAXLAY 128',
                                  f'#define RLW_MAXLAY {lw.MAX_RADIATION_LAYERS}')
    declarations.append(("rrtmg_lw", lw_source,
                         ("-std=c++17", "--ftz=false", "--fmad=false",
                          "--prec-div=true", "--prec-sqrt=true"), "mpas_rrtmg_lw_v842.cu"))
    rows = [{**recipe({"source": source,
                       "options": list(_hex_numerics.options(options, numerics=numerics)),
                       "route": "direct-nvrtc", "filename": filename}), "physics_family": family}
            for family, source, options, filename in declarations]
    rows.append({**recipe({"source": noah_source,
                           "options": list(_hex_numerics.options(noah.CUDA_OPTIONS,
                                           keep_ftz=True, numerics=numerics)),
                           "route": "direct-nvrtc", "filename": "kern.cu"}),
                 "physics_family": "noah_shared_tables"})
    rows.append({**recipe({"source": gf.cuda_source(),
                           "options": list(_hex_numerics.options(gf.COMPILER_OPTIONS, numerics=numerics)),
                           "route": "raw-nvrtc", "filename": "gf_mpas_v842.cu"}),
                 "physics_family": "gf_mpas"})
    # MPAS's standard 55-level atmosphere, with the production workspace storage.
    for corrections in ((), ("WB3",)):
        source = thompson.cuda_source(device_math_source=resources.thompson_maths(resources._vendored_maths()),
                                      max_levels=55, corrections=corrections, workspace=True, numerics=numerics)
        rows.append({**recipe({"source": source,
                               "options": list(_hex_numerics.options(thompson.COMPILE_OPTIONS, numerics=numerics)),
                               "route": "direct-nvrtc", "filename": "kern.cu"}),
                     "physics_family": "thompson_workspace"})
    return rows


def pack_inventory(package_root=None, compiler_family=None, *, compiler=None):
    """Every row the shipped pack holds: each declared module under strict AND fast numerics.

    THE BREAKAGE THIS PREVENTS: the pack selects on the exact options, so a pack of strict rows
    alone misses every hex kernel of a default (fast) run, and a fresh box JITs them all on one
    core before step 1. Rows both modes share (recorded CuPy recipes) are held once."""
    rows = {}
    for mode in ("strict", "fast"):
        for row in inventory(package_root, numerics=mode, compiler_family=compiler_family, compiler=compiler):
            rows.setdefault(hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(), row)
    return list(rows.values())


def cached_program_filename(row, options, target):
    """Reproduce CuPy's program name, including headers used only in its key.

    CUB embeds assertion filenames in its images. A placeholder filename
    produces different bytes despite identical source and arithmetic flags.
    """
    from hexcore.kernel_plan import _cupy_entry, _cupy_extra_sources
    metadata = row.get("compile_request") or {}
    expected = metadata.get("extra_source")
    extra = None
    if expected is not None:
        candidates = [value for value in _cupy_extra_sources()
                      if value is not None and "sha256:" + source_hash(value) == expected]
        if not candidates:
            raise ValueError("compiler recipe headers differ from the installed CuPy headers")
        extra = candidates[0]
    return _cupy_entry(dict(source=row["source"], options=options,
                            name_expressions=metadata.get("name_expressions")), target, extra) + ".cu"


def compile_entry(row, arch):
    """Use CuPy's exact RawModule NVRTC route with an explicit Blackwell SM.

    CuPy chooses a cubin for both certified SMs. The driver is unnecessary for
    this step. --ftz and execution-space options match _compile_with_cache_cuda.
    """
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from .compile_contract import NvrtcCompileCapture
    target = arch.removeprefix("sm_")
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError(f"{arch}: compiler cannot reproduce the default cubin route")
    options = tuple(local_options(row["options"]))
    raw = row.get("compiler_route") == "raw-nvrtc"
    if row.get("compiler_route") not in ("direct-nvrtc", "raw-nvrtc"):
        options += ("-ftz=true",)
    if not raw and compiler._get_bool_env_variable("CUPY_CUDA_COMPILE_WITH_DEBUG", False):
        options += ("--device-debug", "--generate-line-info")
    # CuPy's target selector is decorated with per-device memoization. The
    # decorator queries a GPU even for an explicit target. Supply the explicit
    # admitted target during this CPU-only build, keeping the actual compiler
    # entry and all option processing intact. Never install this in a forecast.
    selector = compiler._get_arch_for_options_for_nvrtc
    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError("AOT compiler changed the explicitly requested SM")
        return "-arch=" + arch, "cubin"
    raw_evidence = None
    with NvrtcCompileCapture() as capture, preserve_final_options():
        compiler._get_arch_for_options_for_nvrtc = explicit
        try:
            # The direct route keeps the unit's own file name, as gpuwm passes it.
            filename = (row.get("compile_request") or {}).get("filename") or "hex-aot.cu"
            if row.get("compile_request") and row.get("compiler_route") not in ("direct-nvrtc", "raw-nvrtc"):
                filename = cached_program_filename(row, options, target)
            if raw:
                program = nvrtc.createProgram(row["source"], filename, (), ())
                try:
                    raw_options = options + ("--gpu-architecture=" + arch,)
                    nvrtc.compileProgram(program, raw_options)
                    blob, mapping = nvrtc.getCUBIN(program), None
                    from .compile_contract import NvrtcObservation, _object_record
                    raw_evidence = {
                        "status": "resolved", "method": "raw nvrtc.compileProgram with explicit options",
                        "observations": [NvrtcObservation(row["source_sha256"], raw_options, 0,
                                                           _object_record(blob)).as_dict()]}
                except Exception as error:
                    raise RuntimeError(nvrtc.getProgramLog(program)) from error
                finally:
                    nvrtc.destroyProgram(program)
            else:
                blob, mapping = compiler._compile_using_nvrtc_no_warning(
                    row["source"], options, target, filename,
                    name_expressions=row.get("compile_request", {}).get("name_expressions"),
                    **({"cache_in_memory": True} if row.get("compile_request", {}).get("cache_in_memory") else {}))
        finally:
            compiler._get_arch_for_options_for_nvrtc = selector
    if not isinstance(blob, bytes) or blob[:4] != b"\x7fELF":
        raise ValueError(f"{row['module_key']} {arch}: expected cubin, received non-ELF compiler output")
    shipped = {key: value for key, value in row.items() if key != "source"}
    return {**shipped, "arch": arch, "cubin_sha256": hashlib.sha256(blob).hexdigest(),
            # Cubins are mostly zero padding and relocation tables: LZMA
            # stores the four architectures' images in about a twentieth.
            "cubin_xz_base64": base64.b64encode(
                lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME)).decode("ascii"),
            "name_mapping": mapping,
            "effective_compile": raw_evidence or capture.evidence(row["source_sha256"])}


def _compile_job(job):
    import sys
    import time
    started = time.perf_counter()
    result = compile_entry(*job)
    print(json.dumps({"module_key": job[0]["module_key"], "arch": job[1],
                      "compile_seconds": time.perf_counter() - started}),
          file=sys.stderr, flush=True)
    return result


def build(output, *, entries=None, architectures=ARCHITECTURES, base_pack=None, workers=None):
    """Build requested images, optionally extending a verified shared pack.

    Existing selections are reused only with the exact compiler, source,
    options and compile request. Unrelated images remain in the shared pack.
    """
    from .aot import Pack
    from hexcore.kernel_pack import write
    if tuple(architectures) != ARCHITECTURES:
        raise ValueError(f"the shipped pack covers exactly {', '.join(ARCHITECTURES)}")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")
    identity = compiler_identity()
    rows = list(pack_inventory(compiler_family=identity["nvrtc"][0], compiler=identity) if entries is None else entries)
    retained = []
    selected = None
    if base_pack is not None:
        selected = Pack(base_pack, identity=identity)
        if selected.reason:
            raise ValueError(f"base pack cannot be reused: {selected.reason}")
        retained = list(selected.rows)
    # One NVRTC compile per (module, architecture), across worker processes: the
    # explicit-target hook in compile_entry is process-global, so each worker
    # is a process running one compile at a time.
    import concurrent.futures
    import multiprocessing
    import os
    jobs = []
    reused = 0
    for row in rows:
        for arch in architectures:
            existing = None
            if selected is not None:
                existing, _ = selected.select(row["module_key"], row["source"],
                                              row["options"], arch,
                                              metadata=row.get("compile_request"))
            if existing is None:
                jobs.append((row, arch))
            else:
                if selected.image(existing) is None:
                    raise ValueError(f"base pack has a corrupt image: {row['module_key']} {arch}")
                reused += 1
    context = multiprocessing.get_context("fork")
    compiled = []
    if jobs:
        count = min(len(jobs), workers or max(1, os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(count, mp_context=context) as pool:
            compiled = list(pool.map(_compile_job, jobs, chunksize=1))
    from gpuwm.certify.compile_platform import nvrtc_banner, nvrtc_build, nvrtc_build_id
    from .compile_contract import build_id_accepted
    banner = nvrtc_banner()
    if nvrtc_build(banner) == "unavailable" or not build_id_accepted(nvrtc_build_id(banner)):
        raise ValueError("NVRTC build banner could not be measured for the wheel")
    compiled = [{**row, "nvrtc_banner": banner} for row in compiled]
    write(output, identity, [*retained, *compiled])
    return {"modules": len(rows), "entries": len(retained) + len(compiled),
            "compiled": len(compiled), "reused": reused, "retained": len(retained),
            "architectures": list(architectures)}
