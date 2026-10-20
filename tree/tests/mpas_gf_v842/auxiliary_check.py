"""Native/host/device witnesses for GF auxiliary and driver-fixed modes."""
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def invoke(lib, name, scalars, arrays):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_int] * len(scalars) + [ctypes.c_void_p] * len(arrays)
    fn(*scalars, *[ctypes.c_void_p(a.ctypes.data) for a in arrays])


def check(root, receipt, fixture, gpu):
    differences = dict(g3init=0, gdinit=0, cup_ktop=0, cup_env=0, cup_dd_edt=0, cup_kbcon=0,
                       cup_up_moisture=0, cup_dd_moisture_new=0)
    cases = dict.fromkeys(differences, 0)
    saved = {}
    if gpu:
        import cupy as cp
        from hexcore.mpasphys import gf_mpas_v842 as gf
        source = gf.cuda_source() + '\n' + Path(__file__).with_name('auxiliary_entry.cuh').read_text()
        module = gf._compile_module(source, cp.cuda.runtime.getDevice())
        arena = cp.empty(65536, dtype=cp.uint8)
        fixture_data = np.load(fixture)
    else:
        native = ctypes.CDLL(str(root / 'gf_native.so'))
        host = ctypes.CDLL(str(root / 'gf_host.so'))
    nz = 8
    # Both restart states and all species-presence combinations.
    for index, values in enumerate(itertools.product((0, 1), (0, 1), (1, 2), (1, 2))):
        scheme, restart, qc, qi = values
        name = ('g3init', 'gdinit')[scheme]
        a = np.arange(11*4*nz, dtype='f4') + np.float32(.5)
        s = np.arange(9*4, dtype='f4') + np.float32(.5)
        if gpu:
            expected = [fixture_data[f'init{index}_{i}'] for i in range(2)]
            device = [cp.asarray(x) for x in (a, s)]
            module.get_function('gf_aux_init')((1,), (1,),
                tuple(np.int32(x) for x in (nz, *values)) + (*device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [a.copy(), s.copy()]
            got = [a.copy(), s.copy()]
            invoke(native, 'gf_native_init', (nz, *values), expected)
            invoke(host, 'gf_host_init', (nz, *values), got)
            for i, x in enumerate(expected):
                saved[f'init{index}_{i}'] = x
        cases[name] += 1
        differences[name] += sum(int(np.count_nonzero(x.view('u4') != y.view('u4')))
                                 for x, y in zip(expected, got))
    # Per column: existing error, immediate cloud top, later cloud top,
    # and missing cloud top. Both ilo branches and both error strings.
    for ilo in (1, 2):
        dby = np.ones((nz, 4), dtype='f4')
        dby[3:, 1] = -1.
        dby[5:, 2] = -1.
        kbcon = np.full(4, 3, dtype='i4')
        ktop = np.full(4, -9, dtype='i4')
        ierr = np.array([7, 0, 0, 0], dtype='i4')
        text = np.full((4, 50), ord(' '), dtype='u1')
        arrays = [dby, kbcon, ktop, ierr, text]
        if gpu:
            expected = [fixture_data[f'ktop{ilo}_{i}'] for i in range(5)]
            device = [cp.asarray(x) for x in arrays]
            module.get_function('gf_aux_ktop')((1,), (1,),
                (np.int32(nz), np.int32(ilo), *device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [x.copy() for x in arrays]
            got = [x.copy() for x in arrays]
            invoke(native, 'gf_native_ktop', (nz, ilo), expected)
            invoke(host, 'gf_host_ktop', (nz, ilo), got)
            for i, x in enumerate(expected):
                saved[f'ktop{ilo}_{i}'] = x
        cases['cup_ktop'] += 4
        differences['cup_ktop'] += sum(int(np.count_nonzero(
            x.view('u4') != y.view('u4') if x.dtype.itemsize == 4 else x != y))
            for x, y in zip(expected, got))
    # Height reconstruction modes unreachable through cu_grell_freitas,
    # plus very cold saturation floors and an existing-error column.
    for itest in (-1, 0, 1, 2, 3):
        a = np.full((7, nz, 4), np.float32(.5), dtype='f4')
        a[0] = np.arange(nz, dtype='f4')[:, None] * np.float32(1000.)
        a[2] = np.float32(350000.)
        a[4] = np.array([130., 250., 300., 275.], dtype='f4')
        a[5] = np.array([1.e-10, .03, .01, .001], dtype='f4')
        a[6] = np.float32(90000.) * np.exp(-np.arange(nz, dtype='f4')[:, None] / np.float32(8.))
        s = np.array([[95000.] * 4, [-10., 100., 1500., 0.]], dtype='f4')
        ierr = np.array([0, 0, 0, 7], dtype='i4')
        arrays = [a, s, ierr]
        if gpu:
            expected = [fixture_data[f'env{itest}_{i}'] for i in range(3)]
            device = [cp.asarray(x) for x in arrays]
            module.get_function('gf_aux_env')((1,), (1,),
                (np.int32(nz), np.int32(itest), *device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [x.copy() for x in arrays]
            got = [x.copy() for x in arrays]
            invoke(native, 'gf_native_env', (nz, itest), expected)
            invoke(host, 'gf_host_env', (nz, itest), got)
            for i, x in enumerate(expected):
                saved[f'env{itest}_{i}'] = x
        cases['cup_env'] += 4
        differences['cup_env'] += sum(int(np.count_nonzero(x.view('u4') != y.view('u4')))
                                     for x, y in zip(expected, got))
    # Direct module modes fixed off by the driver: aerosol evaporation,
    # extra buoyancy and the fourth cloud-base search.
    def direct(name, routine, controls, arrays, key):
        if gpu:
            expected = [fixture_data[f'{key}_{i}'] for i in range(len(arrays))]
            device = [cp.asarray(x) for x in arrays]
            module.get_function(f'gf_aux_{name}')((1,), (1,),
                tuple(np.int32(x) for x in (nz, *controls)) + (*device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [x.copy() for x in arrays]
            got = [x.copy() for x in arrays]
            invoke(native, f'gf_native_{name}', (nz, *controls), expected)
            invoke(host, f'gf_host_{name}', (nz, *controls), got)
            for i, x in enumerate(expected):
                saved[f'{key}_{i}'] = x
        cases[routine] += 4
        differences[routine] += sum(int(np.count_nonzero(
            x.view('u4') != y.view('u4') if x.dtype.itemsize == 4 else x != y))
            for x, y in zip(expected, got))
    for aero in (1, 2, 3):
        z = np.arange(nz, dtype='f4')[:, None] * 1000.
        a = np.zeros((6, nz, 4), dtype='f4')
        a[0] = z * np.array([0., .002, .02, .001], dtype='f4')
        a[1] = z * .001
        a[2] = z
        a[3] = 1000. - np.arange(nz, dtype='f4')[:, None] * 100.
        a[4] = .01
        a[5] = 1.
        s = np.array([[0.] * 4, [1.] * 4, [-1.] * 4,
                      [1., 100., 100000., 100.], [1., .01, 100., 1.], [1.] * 4], dtype='f4')
        e = np.full((3, 4), -9., dtype='f4')
        ints = np.array([[0, 0, 0, 7], [6] * 4, [2] * 4], dtype='i4')
        direct('edt', 'cup_dd_edt', (aero,), [a, s, e, ints], f'edt{aero}')
    for iloop, excess in itertools.product(range(1, 6), range(3)):
        a = np.full((3, nz, 4), 350000., dtype='f4')
        a[1] = 400000.
        a[1, :4, 2] = 500000.
        a[1, 4:, 2] = 300000.
        a[2] = 1000. - np.arange(nz, dtype='f4')[:, None] * 80.
        s = np.array([[450000., 300000., 350000., 450000.], [50.] * 4,
                      [10.] * 4, [.001] * 4, [1.] * 4], dtype='f4')
        ints = np.array([[2] * 4, [-9] * 4, [5] * 4, [0, 0, 0, 7]], dtype='i4')
        text = np.full((4, 50), ord(' '), dtype='u1')
        direct('kbcon', 'cup_kbcon', (iloop, excess), [a, s, ints, text], f'kbcon{iloop}_{excess}')
    for shallow, excess, autoconv in itertools.product(range(2), range(4), (1, 2)):
        a = np.zeros((17, nz, 4), dtype='f4')
        a[4] = np.arange(nz, dtype='f4')[:, None] * 500.
        a[5] = 1000. - np.arange(nz, dtype='f4')[:, None] * 80.
        a[6] = .001
        a[7] = 100.
        a[8] = 280.
        a[9] = np.array([.01, .03, .02, .01], dtype='f4')
        a[10] = 1.
        a[11] = 1.
        a[12] = .002
        a[13] = .02
        a[14] = 1.
        a[15] = .1
        a[16] = .02
        s = np.array([[0.] * 4, [0.] * 4, [0.] * 4,
                      [.005] * 4, [1., 100., 10000., 100.]], dtype='f4')
        ints = np.array([[0, 0, 0, 7], [3] * 4, [6] * 4, [2] * 4], dtype='i4')
        direct('up', 'cup_up_moisture', (shallow, excess, autoconv), [a, s, ints],
               f'up{shallow}_{excess}_{autoconv}')
    for iloop in (1, 2):
        a = np.zeros((15, nz, 4), dtype='f4')
        a[0] = 1.
        a[1] = np.array([300000., 450000., 300000., 350000.], dtype='f4')
        a[2] = 350000.
        a[4] = .01
        a[6] = np.array([.002, .02, .02, .01], dtype='f4')
        a[7] = np.arange(nz, dtype='f4')[:, None] * 1000.
        a[8] = .1
        a[9] = .02
        a[10] = 1.
        a[12] = a[6]
        a[13] = 350000.
        a[14] = 280.
        s = np.zeros((2, 4), dtype='f4')
        ints = np.array([[5] * 4, [0, 0, 0, 7]], dtype='i4')
        text = np.full((4, 50), ord(' '), dtype='u1')
        direct('down', 'cup_dd_moisture_new', (iloop,), [a, s, ints, text], f'down{iloop}')
    if not gpu:
        np.savez_compressed(fixture, **saved)
    report = {'backend': 'hex-cuda' if gpu else 'host-glibc-witness',
              'cases': cases, 'different_words_or_diagnostic_bytes': differences,
              'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'scope': 'Direct tests of auxiliary procedures and driver-fixed alternate modes; no forecast qualification'}
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(receipt.read_text())
    return bool(sum(differences.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args()
    raise SystemExit(check(a.root.resolve(), a.receipt, a.fixture, a.gpu))
