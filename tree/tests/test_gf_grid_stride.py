"""Execute the production lane-interleaved column entry on the host, without a device.

The lowered source (gf_lanes) is compiled with its device lane width (128)
and host math, its kernel body driven by host CUDA index stubs over a grid
smaller than the column count (scratch reuse, slab tails), and every output
word is compared with the untransformed scheme's single-column host entry
(gf_port_host, contiguous scratch). It is not a CUDA numerical or
performance proof; the device word replay is.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from hexcore.mpasphys import gf_mpas_v842 as gf
from hexcore.mpasphys import gf_lanes


def _compile(compiler, source, path):
    path.write_text(source)
    library = path.with_suffix('.so')
    subprocess.run([compiler, '-std=c++17', '-O1', '-shared', '-fPIC',
                    '-ffp-contract=off', str(path), '-o', str(library)],
                   check=True, capture_output=True, text=True, timeout=300)
    return ctypes.CDLL(str(library))


@pytest.mark.parametrize('numerics', ['strict', 'fast'])
def test_lane_interleaved_grid_matches_contiguous_single_columns(tmp_path, numerics):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host column entry check requires g++')
    prefix = '#define GF_HOST_WITNESS\n'
    if numerics == 'fast':
        prefix += '#define HEX_NUMERICS_FAST 1\n'
    lowered = gf._lowered_source()
    split = lowered.index(gf_lanes.ENTRY_MARKER)
    lanes = (prefix + f'#define GF_LANES {gf_lanes.LANES}\n' + lowered[:split] + '''
#undef GF_HOST_WITNESS
struct GFIndex { int x; };
GFIndex blockIdx, blockDim, threadIdx, gridDim;
''' + lowered[split:] + '''
extern "C" void gf_grid_host(int n,int nz,int slots,unsigned long long bytes,float* lev,float* surface,int* kpbl,
 float* outlev,float* outsurface,int* outint,const int* order,int* control,float* constants,
 unsigned int* diagnostics,unsigned long long* highwater){
 auto scratch=(unsigned char*)malloc((unsigned long long)slots*(bytes+GF_CHARACTER_BYTES));
 blockDim.x=GF_LANES;gridDim.x=slots/GF_LANES;
 for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)
   for(threadIdx.x=0;threadIdx.x<GF_LANES;threadIdx.x++)
     gf_mpas_columns(n,nz,lev,surface,kpbl,outlev,outsurface,outint,order,
                     control[0],control[1],control[2],control[3],
                     constants[0],constants[1],constants[2],constants[3],
                     scratch,bytes,diagnostics,highwater);
 free(scratch);
}
''')
    port = _compile(compiler, lanes, tmp_path / 'lanes.cpp')
    # Reference: the untransformed transcription, contiguous scratch.
    witness = _compile(compiler, prefix + gf._SUPPORT + gf._SCHEME_SOURCE + gf._ENTRY,
                       tmp_path / 'witness.cpp')
    # Eight real columns captured from a hex MPAS-physics zoom forecast
    # (GFS 2025-05-19 12Z, step 19: deep, shallow-only and quiet), cycled with
    # perturbations to 263 columns: three slabs, two full and one partial,
    # through 128 slots, so scratch is reused across columns and tails.
    fixture = np.load(Path(__file__).parent / 'mpas_gf_v842' / 'zoom_columns.npz')
    n, nz, slots, L = 263, int(fixture['levels'].shape[2]), gf_lanes.LANES, gf_lanes.LANES
    pick = np.arange(n) % len(fixture['kpbl'])
    levels = np.ascontiguousarray(fixture['levels'][pick])
    surfaces = np.ascontiguousarray(fixture['surface'][pick])
    pbl = np.ascontiguousarray(fixture['kpbl'][pick])
    # Distinct column inputs expose indexing/reuse errors across iterations.
    levels[:, 4] *= np.linspace(.97, 1.03, n, dtype='f4')[:, None]
    control = np.ascontiguousarray(fixture['control'])
    constants = np.ascontiguousarray(fixture['constants'])
    slabs = (n + L - 1) // L
    bytes_ = gf._scratch_bytes(nz)

    def pointers(arrays):
        return [None if x is None else ctypes.c_void_p(x.ctypes.data) for x in arrays]

    want = [np.zeros((n, 6, nz), dtype='f4'), np.zeros((n, 6), dtype='f4'), np.zeros((n, 3), dtype='i4')]
    witness.gf_port_host.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 8
    for i in range(n):
        witness.gf_port_host(nz, *pointers([levels[i], surfaces[i], pbl[i:i+1], want[0][i], want[1][i],
                                            want[2][i], control, constants]))
    assert (want[0] != 0).any() and (want[1][:, 4] > 0).any()
    port.gf_grid_host.argtypes = [ctypes.c_int] * 3 + [ctypes.c_ulonglong] + [ctypes.c_void_p] * 11
    # Natural order, then a column order (position -> column) that groups
    # columns by kind as the forecast path does: same words per column.
    for order in (None, np.ascontiguousarray(np.argsort(np.arange(n) % 8, kind='stable').astype('i4'))):
        by_position = np.arange(n) if order is None else order
        padded = np.zeros((slabs * L, 15, nz), dtype='f4')
        padded[:n] = levels[by_position]
        lanes_in = np.ascontiguousarray(padded.reshape(slabs, L, 15, nz).transpose(0, 2, 3, 1))
        lanes_out = np.zeros((slabs, 6, nz, L), dtype='f4')
        actual = [np.zeros((n, 6), dtype='f4'), np.zeros((n, 3), dtype='i4'),
                  np.zeros(n, dtype='u4'), np.zeros(n, dtype='u8')]
        port.gf_grid_host(n, nz, slots, bytes_, *pointers(
            [lanes_in, surfaces, pbl, lanes_out, actual[0], actual[1], order, control, constants,
             actual[2], actual[3]]))
        got_levels = np.empty((n, 6, nz), dtype='f4')
        got_levels[by_position] = lanes_out.transpose(0, 3, 1, 2).reshape(slabs * L, 6, nz)[:n]
        np.testing.assert_array_equal(got_levels.view('u4'), want[0].view('u4'))
        np.testing.assert_array_equal(actual[0].view('u4'), want[1].view('u4'))
        np.testing.assert_array_equal(actual[1], want[2])
        # The bounded scratch holds every column; the allocator traps otherwise.
        assert 0 < actual[3].max() <= bytes_
