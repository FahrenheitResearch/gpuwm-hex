"""GF's output must not depend on what a reused workspace slot held before.

Defect (fixed by 9f3c7739, hex 0.4.6): the shallow dellaq loop read
qco(ktop+1), a word native never defines. The production kernel reuses
workspace slots across frames, columns and steps, so that word was whatever
the slot last held; when it was a NaN, rthcuten/rqvcuten went NaN at ktop and
the CUDA validation flag refused the step with whole columns non-finite. That
is how the 7.5 km VR global runs stopped at +6 h 36 min and +8 h 41 min
(2026-10-26, cold-teacher hexswap and ng-data-global-km train-main, hex
0.4.5). The failure depended on leftover memory, so the same case ran clean
on a rerun. Filling the freshly allocated qco with NaN on 0.4.5 makes the
first step refuse (53,449 columns); on 0.4.6+ it changes nothing.

A stale word is usually what an earlier frame (deep GF, an earlier column)
left in the slot, so NaN scratch at kernel entry does not reach it. This runs
the production lane-interleaved column entry on the host (as
test_gf_grid_stride does) twice over the same real columns: once as shipped,
and once with every binary32 workspace array filled with NaN the moment it is
allocated. Every output word must be finite and identical between the two:
any read of a workspace word the scheme did not write first fails it.
"""
import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from hexcore.mpasphys import gf_mpas_v842 as gf
from hexcore.mpasphys import gf_lanes

_ALLOCATION = re.compile(
    r"(D_\d+) = \(void \*\) workspace\.allocate \((.*?)\);\n(\s*)(\w+) = \(GFReal \*\) \1;\n")


def _poison_allocations(scheme):
    """Fill each binary32 workspace array with NaN right after it is allocated."""
    def fill(match):
        indent, nbytes, name = match.group(3), match.group(2), match.group(4)
        return (match.group(0) + f"{indent}{{ unsigned long long nb_ = ({nbytes}); "
                f"for (unsigned long long e_ = 0; e_ < nb_ / 4; ++e_) {name}[e_] = gf_nan(); }}\n")
    poisoned, count = _ALLOCATION.subn(fill, scheme)
    # Every allocation but the integer (22) and character (2) arrays.
    assert count == scheme.count("workspace.allocate (") - 24, count
    return poisoned


def _compile(compiler, source, path):
    path.write_text(source)
    library = path.with_suffix('.so')
    subprocess.run([compiler, '-std=c++17', '-O1', '-shared', '-fPIC',
                    '-ffp-contract=off', str(path), '-o', str(library)],
                   check=True, capture_output=True, text=True, timeout=300)
    return ctypes.CDLL(str(library))


_HOST_GRID = '''
extern "C" void gf_grid_host(int n,int nz,int slots,unsigned long long bytes,float* lev,float* surface,int* kpbl,
 float* outlev,float* outsurface,int* outint,const int* order,int* control,float* constants,
 unsigned int* diagnostics,unsigned long long* highwater){
 unsigned long long size=(unsigned long long)slots*(bytes+GF_CHARACTER_BYTES);
 auto scratch=(unsigned char*)calloc(size,1);
 blockDim.x=GF_LANES;gridDim.x=slots/GF_LANES;
 for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)
   for(threadIdx.x=0;threadIdx.x<GF_LANES;threadIdx.x++)
     gf_mpas_columns(n,nz,lev,surface,kpbl,outlev,outsurface,outint,order,
                     control[0],control[1],control[2],control[3],
                     constants[0],constants[1],constants[2],constants[3],
                     scratch,bytes,diagnostics,highwater);
 free(scratch);
}
'''


@pytest.mark.parametrize('numerics', ['strict', 'fast'])
def test_gf_output_does_not_read_unwritten_workspace_words(tmp_path, numerics):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host column entry check requires g++')
    prefix = '#define GF_HOST_WITNESS\n'
    if numerics == 'fast':
        prefix += '#define HEX_NUMERICS_FAST 1\n'

    def build(lowered, name):
        split = lowered.index(gf_lanes.ENTRY_MARKER)
        return _compile(compiler, prefix + f'#define GF_LANES {gf_lanes.LANES}\n' + lowered[:split] + '''
#undef GF_HOST_WITNESS
struct GFIndex { int x; };
GFIndex blockIdx, blockDim, threadIdx, gridDim;
''' + lowered[split:] + _HOST_GRID, tmp_path / f'{name}.cpp')

    shipped = build(gf._lowered_source(), 'shipped')
    poisoned = build(gf_lanes.lower(gf._SUPPORT, _poison_allocations(gf._SCHEME_SOURCE), gf._ENTRY),
                     'poisoned')
    # The real captured columns (deep, shallow-only and quiet; GFS 2025-05-19
    # 12Z zoom, step 19), cycled to 263 columns through 128 slots so slots
    # are reused across columns, as in a forecast.
    fixture = np.load(Path(__file__).parent / 'mpas_gf_v842' / 'zoom_columns.npz')
    n, nz, L = 263, int(fixture['levels'].shape[2]), gf_lanes.LANES
    pick = np.arange(n) % len(fixture['kpbl'])
    levels = np.ascontiguousarray(fixture['levels'][pick])
    levels[:, 4] *= np.linspace(.97, 1.03, n, dtype='f4')[:, None]
    surfaces = np.ascontiguousarray(fixture['surface'][pick])
    pbl = np.ascontiguousarray(fixture['kpbl'][pick])
    control = np.ascontiguousarray(fixture['control'])
    constants = np.ascontiguousarray(fixture['constants'])
    slabs = (n + L - 1) // L
    padded = np.zeros((slabs * L, 15, nz), dtype='f4')
    padded[:n] = levels
    lanes_in = np.ascontiguousarray(padded.reshape(slabs, L, 15, nz).transpose(0, 2, 3, 1))

    def run(port):
        out = [np.zeros((slabs, 6, nz, L), dtype='f4'), np.zeros((n, 6), dtype='f4'),
               np.zeros((n, 3), dtype='i4'), np.zeros(n, dtype='u4'), np.zeros(n, dtype='u8')]
        args = [lanes_in, surfaces, pbl, out[0], out[1], out[2], None, control, constants, out[3], out[4]]
        port.gf_grid_host.argtypes = [ctypes.c_int] * 3 + [ctypes.c_ulonglong] + [ctypes.c_void_p] * 11
        port.gf_grid_host(n, nz, L, gf._scratch_bytes(nz),
                          *[None if a is None else ctypes.c_void_p(a.ctypes.data) for a in args])
        return out[0].transpose(0, 3, 1, 2).reshape(slabs * L, 6, nz)[:n], out[1], out[2]

    want, got = run(shipped), run(poisoned)
    assert (want[0] != 0).any(), 'no column produced a tendency'
    for name, a, b in zip(('level tendencies', 'surface outputs'), want[:2], got[:2]):
        bad = ~np.isfinite(b)
        assert not bad.any(), (f'{name}: {int(bad.any(axis=tuple(range(1, b.ndim))).sum())} of {n} columns '
                               'read a workspace word before writing it (non-finite with NaN-filled allocations)')
        np.testing.assert_array_equal(b.view('u4'), a.view('u4'))
    np.testing.assert_array_equal(got[2], want[2])
