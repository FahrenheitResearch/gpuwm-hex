"""Remaining native inline boundary stages, including IEEE clamp branches."""
import os
import numpy as np
import pytest
from test_regional_native_v842 import (
    NC,NE,NL,NT,INPUT,oracle_root,read_arrays,record,mask_fixture,regional,record_cuda_compile)


@pytest.mark.parametrize('case',range(1,25))
def test_native_misc_cpu(case):
    path=oracle_root()/f'case-{case:02}'
    a=read_arrays(path/'input.bin',INPUT)
    velocity=read_arrays(path/'velocity-input.bin',[(n,(NL,NE)) for n in ('u','ru','drive_u','drive_ru')])
    regional.overwrite_speczone_u_ru(masks=mask_fixture(a),normal_velocity=velocity['u'],rho_u=velocity['ru'],
                                    u_driving_values=velocity['drive_u'],ru_driving_values=velocity['drive_ru'])
    expected=read_arrays(path/'velocity-reset.bin',[(n,(NL,NE)) for n in ('u','ru')])
    for n in expected:
        record(f'case-{case:02}-cpu-velocity',n,velocity[n],expected[n])
    scalars=read_arrays(path/'clamp-input.bin',[('q',(NT,NL,NC))])['q']
    with np.errstate(invalid='ignore'):
        regional.clamp_negative_scalars(scalars)
    expected=read_arrays(path/'clamp-output.bin',[('q',(NT,NL,NC))])['q']
    record(f'case-{case:02}-cpu-clamp','scalars',scalars,expected)


@pytest.mark.parametrize('case',range(1,25))
def test_native_misc_cuda(case):
    if os.environ.get('MPAS_REGIONAL_GPU')!='1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels
    path=oracle_root()/f'case-{case:02}'
    a=read_arrays(path/'input.bin',INPUT)
    velocity=read_arrays(path/'velocity-input.bin',[(n,(NL,NE)) for n in ('u','ru','drive_u','drive_ru')])
    d={n:cp.asarray(np.ascontiguousarray(v)) for n,v in velocity.items()}
    spec=cp.asarray(mask_fixture(a).spec_edges,dtype='i4');kernels=CudaRegionalKernels(KernelCache())
    kernels.launch('regional_speczone_u_ru_v841',NL*len(spec),
                   (np.int32(NL),np.int32(NE),np.int32(len(spec)),spec,d['drive_u'],d['drive_ru'],d['u'],d['ru']))
    expected=read_arrays(path/'velocity-reset.bin',[(n,(NL,NE)) for n in ('u','ru')])
    for n in expected:
        record(f'case-{case:02}-cuda-velocity',n,cp.asnumpy(d[n]),expected[n])
    scalars=cp.asarray(np.ascontiguousarray(read_arrays(path/'clamp-input.bin',[('q',(NT,NL,NC))])['q']))
    kernels.launch('regional_clamp_negative_scalars_v841',scalars.size,(np.int32(scalars.size),scalars))
    expected=read_arrays(path/'clamp-output.bin',[('q',(NT,NL,NC))])['q']
    record(f'case-{case:02}-cuda-clamp','scalars',cp.asnumpy(scalars),expected)
    if case==1:
        record_cuda_compile(kernels,"velocity-clamp")


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_subnormal_interpolation(device):
    """Native boundaries.F:631-642 with subnormal operands and results."""
    path = oracle_root()
    a = read_arrays(path/'subnormal-input.bin', [
        ('state', (NT, NL, NC)), ('tend', (NT, NL, NC)), ('remaining', ())])
    expected = read_arrays(path/'subnormal-output.bin', [('out', (NT, NL, NC))])['out']
    assert np.any((expected.view('u4') & 0x7fffffff) > 0)
    assert np.all((expected.view('u4') & 0x7f800000) == 0)
    if device == 'cpu':
        actual = a['state'] - a['remaining'] * a['tend']
    else:
        if os.environ.get('MPAS_REGIONAL_GPU') != '1':
            pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
        import cupy as cp
        from hexcore.cuda_backend import KernelCache
        from hexcore.cuda_regional_v841 import CudaRegionalKernels, MODULE_KEY
        kernels = CudaRegionalKernels(KernelCache())
        state, tend = (cp.asarray(np.ascontiguousarray(a[n])) for n in ('state', 'tend'))
        out = cp.empty_like(state)
        kernels.launch('regional_lbc_state_at_v841', state.size,
                       (np.int32(state.size), a['remaining'], state, tend, out))
        actual = cp.asnumpy(out)
        manifest = kernels.cache.compile_manifest()['modules'][MODULE_KEY]
        observations = manifest['effective_compile']['observations']
        assert observations, 'effective NVRTC flags must be observed'
        for observation in observations:
            flags = observation['effective_flags']
            assert '--ftz=false' in flags
            assert not any(f in flags for f in ('--ftz=true', '-ftz=true', '--use_fast_math'))
            assert all(f in flags for f in ('--fmad=false', '--prec-div=true', '--prec-sqrt=true'))
        record_cuda_compile(kernels, 'subnormal')
    record(f'subnormal-{device}', 'interpolation', actual, expected)
