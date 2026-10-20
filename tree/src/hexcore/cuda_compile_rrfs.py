"""Strict NVRTC loader for additive RRFS translation units.

CuPy RawModule forces FTZ; the lane native-defect policy requires direct NVRTC.
This is orchestration only and does not change the existing kernel cache.
"""
from hexcore import numerics as _hex_numerics
from typing import Any
OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true')
_MODULES: dict[tuple[int, str, str], Any] = {}

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
def module(source: str, name: str) -> Any:
    import hashlib
    import cupy as cp
    from cupy.cuda import compiler, function
    key = (cp.cuda.Device().id, name, hashlib.sha256(source.encode()).hexdigest(), _hex_numerics.options(OPTIONS))
    if key not in _MODULES:
        image, _ = compiler.compile_using_nvrtc(source, options=_hex_numerics.options(OPTIONS), cache_in_memory=True)
        compiled = function.Module()
        compiled.load(image)
        _MODULES[key] = compiled
    return _MODULES[key]

# Shared stencils retain their source bytes while using the same strict route.
from .cuda_backend.runtime import KernelCache

class RRFSKernelCache(KernelCache):
    """Direct NVRTC cache with the existing compile manifest and launch observer."""
    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def __init__(self,**kwargs):
        kwargs['base_options']=_hex_numerics.options(OPTIONS)
        super().__init__(**kwargs)

    # MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
    def raw_kernel(self,name,source,*,module_key,options=()):
        import hashlib
        import time
        import cupy as cp
        from cupy.cuda import compiler,function
        from .cuda_backend.compile_contract import NvrtcCompileCapture,validate_module_key
        stable=validate_module_key(module_key)
        requested=(*_hex_numerics.options(OPTIONS),*tuple(options))
        key=self._module_key(source,tuple(options));kernel_key=key+':'+name
        self._compile_manifest.declare_module(stable,source=source,requested_options=requested,module_cache_key=key)
        if key not in self._modules:
            started=time.perf_counter()
            with NvrtcCompileCapture() as capture:
                blob,_=compiler.compile_using_nvrtc(source,options=requested,cache_in_memory=True)
            compiled=function.Module();compiled.load(blob)
            self._modules[key]=compiled
            self._compile_manifest.record_effective_compile(stable,capture.evidence(hashlib.sha256(source.encode()).hexdigest()))
            self._load_timings.append(dict(module_key=stable,route='direct_nvrtc_rrfs',module_load_seconds=time.perf_counter()-started))
        if kernel_key not in self._kernels:
            self._kernels[kernel_key]=self._modules[key].get_function(name)
        self._compile_manifest.record_kernel(stable,name)
        return self._observe(kernel_key,self._kernels[kernel_key],name,stable)


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1928-7759 at 1d6e370e.
def kernel(source: str, module_key: str, name: str, cache=None):
    """Use the forecast cache when supplied, preserving its launch observer."""
    if cache is None:return module(source,module_key).get_function(name)
    if not isinstance(cache,RRFSKernelCache):raise TypeError('RRFS kernels require the strict RRFS cache')
    # This generic adapter receives its source/key from the caller. Binding
    # the method distinguishes delegation from a static kernel declaration
    # to the existing AOT inventory; native launch manifests still record
    # every actual source/key/options through RRFSKernelCache.raw_kernel.
    compile_kernel = cache.raw_kernel
    return compile_kernel(name,source,module_key=module_key)
