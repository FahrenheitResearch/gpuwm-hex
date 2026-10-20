"""Measure WSM6 with existing shared engine maths, never an accepted fallback.

The glibc scalar headers are reused unchanged. Binary64 sin/atan and the
CuPy complex-library operations below remain uncertified against glibc 2.43.
This composition is test-only: it is never installed in rkind_libm, selected
by cuda_entry, or described as the campaign's accepted device math provider.
"""
import argparse
import hashlib
import json
from pathlib import Path


def assemble(engine, output):
    root = engine / 'gpuwm/core/kernels'
    names = ('glibc_flt32.cuh', 'glibc_flt64.cuh',
             'thompson_aerosol_libm.cuh', 'glibc_trig_flt32.cuh')
    sources = {name: (root/name).read_text() for name in names}
    urban = (root/'urban_bem.cuh').read_text()
    start = urban.index('// log10f: CORE-MATH ')
    end = urban.index('\n__device__ void bem_bem(', start)
    # The same unchanged shared window used by the Thompson lane. Preserve
    # the licence, constants and operation order; no new libm is transcribed.
    log10 = urban[start:end]
    text = '\n'.join(sources.values()) + '\n' + log10 + '\n'
    for operation in ('exp', 'log', 'log10'):
        single = 'ubm_log10f' if operation == 'log10' else 'thompson_aa_'+operation+'f'
        text += (f'__device__ float mp_{operation}(float x) {{return {single}(x);}}\n'
                 f'__device__ double mp_{operation}(double x) {{return thompson_aa_{operation}(x);}}\n')
    text += '''
__device__ float mp_pow(float x,float y) {return thompson_aa_powf(x,y);}
__device__ double mp_pow(double x,double y) {return thompson_aa_pow(x,y);}
__device__ float mp_sqrt(float x) {return __fsqrt_rn(x);}
__device__ double mp_sqrt(double x) {return __dsqrt_rn(x);}
__device__ float mp_sin(float x) {return glibc_sinf(x);}
__device__ float mp_cos(float x) {return glibc_cosf(x);}
__device__ float mp_atan(float x) {return glibc_atanf(x);}
__device__ double mp_cos(double x) {return uw_cos(x);}
// Existing library calls, explicitly uncertified diagnostic operations.
// No new approximation, native repair, or production fallback is supplied.
__device__ double mp_sin(double x) {return sin(x);}
__device__ double mp_atan(double x) {return atan(x);}
__device__ cx mp_log(cx x) {return thrust::log(x);}
__device__ cx mp_sqrt(cx x) {return thrust::sqrt(x);}
__device__ cx4 mp_sqrt(cx4 x) {return thrust::sqrt(x);}
__device__ double mp_abs(cx x) {return thrust::abs(x);}
__device__ cx mp_complex_div(cx x,cx y) {return x/y;}
__device__ cx4 mp_complex_div(cx4 x,cx4 y) {return x/y;}
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    manifest = dict(scope=__doc__, acceptance=False,
                    header_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    engine=str(engine.resolve()),
                    unchanged_shared_headers={name: hashlib.sha256(source.encode()).hexdigest()
                                              for name, source in sources.items()},
                    log10_source_sha256=hashlib.sha256(urban.encode()).hexdigest(),
                    log10_window_sha256=hashlib.sha256(log10.encode()).hexdigest(),
                    uncertified_operations=['sin(double)', 'atan(double)',
                                            'CuPy complex log/sqrt/abs/division'])
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assemble(args.engine, args.output)
