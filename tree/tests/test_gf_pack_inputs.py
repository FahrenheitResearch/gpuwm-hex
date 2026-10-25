"""Execute the production fused gather on host CUDA indices, with strided views.

This checks copied words, arithmetic, bounds and Python launch arguments. It
is not a CUDA execution, speed or native bitwise proof.
"""
import ctypes
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys import gf_mpas_v842 as gf


@pytest.fixture(params=['strict', 'fast'])
def pack_library(tmp_path, request):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host gather witness requires g++')
    source = '#define GF_HOST_WITNESS\n'
    if request.param == 'fast':
        source += '#define HEX_NUMERICS_FAST 1\n'
    source += gf._SUPPORT + gf._PACK_SURFACE
    source += """
#undef GF_HOST_WITNESS
struct GFIndex { int x; };
GFIndex blockIdx, blockDim, threadIdx, gridDim;
"""
    source += gf._PACK_INPUTS + gf._SCATTER_OUTPUTS
    arguments = ['n', 'nz']
    for index in range(16):
        arguments += [f'p[{index}]', f's[{index*2}]', f's[{index*2+1}]']
    for index in range(7):
        arguments += [f'p[{index+16}]', f's[{index+32}]']
    arguments += ['len_disp', 'dt', 'levels', 'surface', 'lanes', 'position']
    source += """
extern "C" void gf_pack_host(int n,int nz,float** p,long long* s,
 float len_disp,float dt,float* levels,float* surface,int lanes,const int* position){
 blockDim.x=128;gridDim.x=(n*nz+127)/128;
 for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)
   for(threadIdx.x=0;threadIdx.x<128;threadIdx.x++)
     gf_mpas_pack_inputs(""" + ','.join(arguments) + """);
}
"""
    arguments = ['n', 'nz']
    pointer_index = stride_index = 0
    for index in range(15):
        rank = 2 if index < 6 else 1
        for _ in range(2):
            dtype = 'float' if index < 12 else 'int'
            arguments.append(f'({dtype}*)p[{pointer_index}]')
            pointer_index += 1
            for _ in range(rank):
                arguments.append(f's[{stride_index}]')
                stride_index += 1
    source += """
extern "C" void gf_scatter_host(int n,int nz,void** p,long long* s){
 blockDim.x=128;gridDim.x=(n*nz+127)/128;
 for(blockIdx.x=0;blockIdx.x<gridDim.x;blockIdx.x++)
   for(threadIdx.x=0;threadIdx.x<128;threadIdx.x++)
     gf_mpas_scatter_outputs(""" + ','.join(arguments) + """);
}
"""
    path = tmp_path / 'pack.cpp'
    path.write_text(source)
    library = tmp_path / 'pack.so'
    subprocess.run([compiler, '-std=c++17', '-O1', '-shared', '-fPIC',
                    '-ffp-contract=off', str(path), '-o', str(library)],
                   check=True, capture_output=True, text=True, timeout=30)
    port = ctypes.CDLL(str(library))
    port.gf_pack_host.argtypes = ([ctypes.c_int]*2 + [ctypes.c_void_p]*2 +
                                 [ctypes.c_float]*2 + [ctypes.c_void_p]*2 + [ctypes.c_int, ctypes.c_void_p])
    port.gf_pack_surface_host.argtypes = ([ctypes.c_int] + [ctypes.c_void_p]*7 +
                                         [ctypes.c_float]*2 + [ctypes.c_void_p])
    port.gf_scatter_host.argtypes = [ctypes.c_int]*2 + [ctypes.c_void_p]*2
    return port


def test_fused_pack_preserves_strided_words_and_surface(pack_library, monkeypatch):
    n, nz = 37, 5
    words = np.array([0x80000000, 0, 1, 0x7fc12345, 0x7f812345,
                      0xff800000, 0x3f800000], dtype='u4')
    fields = {}
    before = {}
    # Different layouts include halo columns, interface levels, reversed axes
    # and broadcast (zero-stride) views. No input can be silently materialized.
    for index, name in enumerate(gf._PACK_LEVEL_SOURCES):
        raw = np.resize(words, (nz+1, 2*n+3)).view('f4')
        if name in ('rthratenlw', 'rthratensw'):
            raw = np.arange(raw.size, dtype='f4').reshape(raw.shape) * .001
        view = raw[:nz, 1:1+2*n:2]
        if index % 3 == 0:
            view = view[::-1, ::-1]
        elif index % 3 == 1:
            view = np.array(view, order='F')
        fields[name] = view
        before[name] = view.copy().view('u4')
    fields['q'] = np.broadcast_to(fields['q'][:1, :], (nz, n))
    before['q'] = fields['q'].copy().view('u4')
    for index, name in enumerate(gf._PACK_VECTOR_SOURCES[1:]):
        raw = np.arange(2*n+1, dtype='f4') + index + 1
        fields[name] = raw[1::2][::-1]
    fields['zgrid'] = np.arange((nz+1)*(n+3), dtype='f4').reshape(nz+1, n+3)[:, :n]
    fields['kpbl'] = np.arange(2*n, dtype='i4')[::2]
    launched = []

    def get_function(name):
        assert name == 'gf_mpas_pack_inputs'
        def launch(grid, block, arguments):
            launched.append((grid, block))
            count, depth = arguments[:2]
            pointers, strides = [], []
            pos = 2
            for _ in range(16):
                pointers.append(arguments[pos].ctypes.data)
                strides.extend(arguments[pos+1:pos+3])
                pos += 3
            for _ in range(7):
                pointers.append(arguments[pos].ctypes.data)
                strides.append(arguments[pos+1])
                pos += 2
            ptrs = (ctypes.c_void_p*23)(*pointers)
            steps = np.asarray(strides, dtype='i8')
            length, dt, levels, surface, lanes, position = arguments[pos:]
            where = position.ctypes.data if isinstance(position, np.ndarray) else None
            pack_library.gf_pack_host(count, depth, ptrs, steps.ctypes.data,
                                      length, dt, levels.ctypes.data, surface.ctypes.data, int(lanes), where)
        return launch

    fake_cp = SimpleNamespace(ndarray=np.ndarray, float32=np.float32, int32=np.int32, empty=np.empty,
                              zeros=np.zeros, empty_like=np.empty_like, arange=np.arange,
                              ascontiguousarray=np.ascontiguousarray,
                              cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0)))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cp)
    monkeypatch.setattr(gf, '_module', lambda device: SimpleNamespace(get_function=get_function))
    levels, surface, kpbl = gf.X_from_MPAS(fields, len_disp=60000., dt=360.)
    expected = dict(fields)
    expected['rthften'] = fields['rthdynten']
    expected['rqvften'] = fields['rqvdynten']
    expected['rthraten'] = np.add(fields['rthratenlw'], fields['rthratensw'])
    want = np.stack([expected[name].T for name in gf.LEVEL_INPUTS], axis=1)
    np.testing.assert_array_equal(levels.view('u4'), want.view('u4'))
    vectors = [np.ascontiguousarray(fields['zgrid'][0])]
    vectors += [np.ascontiguousarray(fields[name]) for name in gf._PACK_VECTOR_SOURCES[1:]]
    want_surface = np.empty_like(surface)
    pack_library.gf_pack_surface_host(n, *(x.ctypes.data for x in vectors),
                                      60000., 360., want_surface.ctypes.data)
    np.testing.assert_array_equal(surface.view('u4'), want_surface.view('u4'))
    np.testing.assert_array_equal(kpbl, fields['kpbl'])
    for name, original in before.items():
        np.testing.assert_array_equal(fields[name].view('u4'), original)
        assert not np.shares_memory(levels, fields[name])
    assert launched == [(((n*nz+127)//128,), (128,))]

    # The forecast path packs the column kernel's lane-interleaved layout
    # [slab,15,nz,128] directly: same words, zero past the last column.
    lanes, lane_surface, _ = gf.X_from_MPAS(fields, len_disp=60000., dt=360., lanes=True)
    padded = np.zeros(((n + 127) // 128 * 128, 15, nz), dtype='f4')
    padded[:n] = want
    want_lanes = padded.reshape(-1, 128, 15, nz).transpose(0, 2, 3, 1)
    assert lanes.shape == want_lanes.shape
    np.testing.assert_array_equal(lanes.view('u4'), want_lanes.view('u4'))
    np.testing.assert_array_equal(lane_surface.view('u4'), want_surface.view('u4'))
    launched.pop()

    # With a column order (position -> column), position p holds column order[p].
    order = np.ascontiguousarray(np.argsort(np.arange(n) % 3, kind='stable').astype(np.int32))
    ordered, _, _ = gf.X_from_MPAS(fields, len_disp=60000., dt=360., lanes=True, order=order)
    padded[:n] = want[order]
    np.testing.assert_array_equal(ordered.view('u4'), padded.reshape(-1, 128, 15, nz).transpose(0, 2, 3, 1).view('u4'))
    launched.pop()

    # An empty solve-cell slice launches nothing and returns empty storage.
    empty = {name: value[..., :0] for name, value in fields.items()}
    a, b, c = gf.X_from_MPAS(empty, len_disp=60000., dt=360.)
    assert a.shape == (0, 15, nz) and b.shape == (0, 8) and c.shape == (0,)
    assert len(launched) == 1


def test_fused_scatter_preserves_words_and_halos(pack_library, monkeypatch):
    n, nz = 37, 5
    words = np.array([0x80000000, 0, 1, 0x7fc12345, 0x7f812345,
                      0xff800000, 0x3f800000], dtype='u4')
    result, tend, diag, expected = {}, {}, {}, {}
    names = gf._SCATTER_LEVEL_NAMES + gf._SCATTER_SURFACE_NAMES + gf.INTEGER_OUTPUTS
    for index, name in enumerate(names):
        if index < 6:
            raw = np.resize(words, (n, 6, nz)).view('f4')
            result[name] = raw[:, index, :].T
            destination = np.full((nz+1, n+3), -19., dtype='f4')
            if index % 2:
                destination = np.array(destination, order='F')[::-1]
            target = tend if index < 4 else diag
        else:
            dtype = 'f4' if index < 12 else 'i4'
            raw = np.resize(words, (n, 7)).view(dtype)
            result[name] = raw[:, index % 7]
            destination = np.full(2*(n+3), -19, dtype=dtype)[::2][::-1]
            target = diag
        target[name] = destination
        want = destination.copy()
        if index < 6:
            want[:nz, :n] = result[name]
        else:
            want[:n] = result[name]
        expected[name] = want.view('u4')
    launches = []

    def get_function(name):
        assert name == 'gf_mpas_scatter_outputs'
        def launch(grid, block, arguments):
            launches.append((grid, block))
            count, depth = arguments[:2]
            pointers, strides = [], []
            pos = 2
            for index in range(15):
                rank = 2 if index < 6 else 1
                for _ in range(2):
                    pointers.append(arguments[pos].ctypes.data)
                    strides.extend(arguments[pos+1:pos+rank+1])
                    pos += rank+1
            assert pos == len(arguments)
            ptrs = (ctypes.c_void_p*30)(*pointers)
            steps = np.asarray(strides, dtype='i8')
            pack_library.gf_scatter_host(count, depth, ptrs, steps.ctypes.data)
        return launch

    fake_cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0)))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cp)
    monkeypatch.setattr(gf, '_module', lambda device: SimpleNamespace(get_function=get_function))
    gf._couple_outputs(result, tend, diag, nz, n)
    for name in names:
        target = tend if name in gf.LEVEL_OUTPUTS[:4] else diag
        np.testing.assert_array_equal(target[name].view('u4'), expected[name])
    assert launches == [(((n*nz+127)//128,), (128,))]
    gf._couple_outputs({}, {}, {}, nz, 0)
    assert len(launches) == 1


def test_activity_order_groups_deep_then_shallow_then_quiet(monkeypatch):
    # Stand-in cupy over numpy: the order is a stable sort of last call's GF
    # activity read from the MPAS pools (raincv, k22_shallow).
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        where=np.where, int32=np.int32, cumsum=np.cumsum, empty=np.empty, arange=np.arange))
    n = 2048
    rain = np.zeros(n + 5, dtype='f4')
    k22 = np.zeros(n + 5, dtype='i4')
    rain[[7, 3000 % n, 9]] = [1e-3, 2., 0.5]
    k22[[1, 5, 9, 40]] = [3, 4, 5, 6]
    order = gf.activity_order({'raincv': rain, 'k22_shallow': k22}, n)
    assert order.dtype == np.int32 and sorted(order) == list(range(n))
    deep = sorted([7, 3000 % n, 9])
    assert list(order[:3]) == deep and list(order[3:6]) == [1, 5, 40]
    assert list(order[6:10]) == [0, 2, 3, 4]
    # Small solves, missing pools: natural order (None).
    assert gf.activity_order({'raincv': rain, 'k22_shallow': k22}, 100) is None
    assert gf.activity_order({'raincv': rain}, n) is None
