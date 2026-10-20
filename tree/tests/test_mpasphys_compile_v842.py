"""Native carrier math stays scoped to the MPAS physics cache."""
from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
from hexcore.cuda_backend.runtime import KernelCache


def test_native_carrier_binding_does_not_change_other_kernel_sources(monkeypatch):
    captured = []
    def compile(self, name, source, *, module_key, options=(), mpas_strict=False):
        captured.append((source, module_key, mpas_strict))
        return name
    monkeypatch.setattr(KernelCache, 'raw_kernel', compile)
    cache = MpasPhysicsKernelCache.__new__(MpasPhysicsKernelCache)
    source = 'float f(float x) {return sinf(x) + cosf(x) + logf(x) + expf(x);}'
    cache.raw_kernel('f', source, module_key='hexcore.cuda_physics_prep_v841')
    native = captured[-1][0]
    for name in ('glibc_sinf(x)', 'glibc_cosf(x)', 'gfk_log(x)', 'gfk_exp_fma(x)'):
        assert name in native
    assert captured[-1][2] is True
    cache.raw_kernel('f', source, module_key='hexcore.mpasphys.other')
    assert captured[-1] == (source, 'hexcore.mpasphys.other', True)
    # The default compiler itself receives no changed source or strict default.
    KernelCache.raw_kernel(None, 'f', source, module_key='hexcore.cuda_physics_prep_v841')
    assert captured[-1] == (source, 'hexcore.cuda_physics_prep_v841', False)
