"""Standalone strict-Fortran oracle comparisons; GPU work requires the GPU queue.

Build with oml_native_oracle.py, then set MPAS_OML_ORACLE to the shared library
and MPAS_OML_GPU=1. MPAS_OML_RECEIPT names the receipt path. These synthetic
tests are development evidence, never whole-model acceptance evidence.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys import oml_v842 as oml

FIELDS = (*oml.STATE_FIELDS, "u", "v", "phase", "xland", "sst")
CONSTANTS = (5.67051e-8, 9.80616, 600., .14, 864000., 7.29212e-5, 30.)


def columns(n=4096):
    rng = np.random.default_rng(190842)
    a = np.empty((19, n), dtype=np.float32)
    a[0] = rng.uniform(260., 310., n)
    a[1] = rng.uniform(270., 310., n)
    a[2] = rng.uniform(.001, 200., n)
    a[3] = rng.uniform(.001, 200., n)
    a[4:6] = rng.uniform(-30., 30., (2,n))
    a[6] = rng.uniform(260., 310., n)
    a[7:9] = rng.uniform(-200., 1000., (2,n))
    a[9:11] = rng.uniform(0., 1000., (2,n))
    a[11] = a[1] - np.float32(5.)
    a[12] = rng.uniform(0., 3., n)
    a[13] = rng.uniform(.8, 1., n)
    a[14:16] = rng.uniform(-40., 40., (2,n))
    a[16] = rng.uniform(-1.5707964, 1.5707964, n)
    a[17] = rng.choice([1., 1.5, 2.], n)
    a[18] = rng.uniform(260., 310., n)
    # Branch witnesses: zero depth, calm wind, tml below/equal tmoml,
    # alpha floor, cooling/warming, depth floor, hemisphere/zero Coriolis.
    a[2, :8] = 0.
    a[4:6, :8] = 0.
    a[14:16, :12] = 0.
    a[14, 12:16] = [0., -0., 1.e-11, 1.e-10]
    a[15, 12:16] = 0.
    a[0, 16:20] = a[11, 16:20]
    a[0, 20:24] = 273.15
    a[11,24:28] = a[0,24:28] + np.float32(1.)
    a[16,28:32] = [0., -0., 1.5707964, -1.5707964]
    a[17,:32] = 2.
    return a


@pytest.fixture(scope="module")
def oracle():
    path = os.environ.get("MPAS_OML_ORACLE")
    if not path:
        pytest.skip("standalone pristine OML oracle not built; set MPAS_OML_ORACLE")
    lib = ctypes.CDLL(path)
    function = lib.replay
    function.argtypes = ([ctypes.c_int] * 5 +
                         [np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")] +
                         [ctypes.c_float] * 7)
    function.restype = None
    return function


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("MPAS_OML_GPU") != "1":
        pytest.skip("OML device tests require explicit MPAS_OML_GPU=1 inside the GPU queue")
    import cupy as cp
    return cp


@pytest.fixture(scope="module")
def receipt():
    rows = []
    yield rows
    path = os.environ.get("MPAS_OML_RECEIPT")
    if path:
        Path(path).write_text(json.dumps({
            "schema": "mpas-oml-standalone-oracle/v1", "acceptance": False,
            "processes": 1, "mpi": False, "host": socket.gethostname(),
            "authority": "pristine MPAS-A v8.4.2 module + literal driver/init slices",
            "tests": rows, "different_words": sum(row["different_words"] for row in rows),
            "arrays_per_comparison": list(FIELDS),
            "oracle_sha256": hashlib.sha256(Path(os.environ["MPAS_OML_ORACLE"]).read_bytes()).hexdigest(),
            "implementation_sha256": hashlib.sha256(Path(oml.__file__).read_bytes()).hexdigest(),
            "compiler": "direct NVRTC; explicit module load; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true",
            "device_math": "shared glibc_trig_flt32.cuh; correctly rounded sqrt/div; FTZ disabled",
        }, indent=2) + "\n")


def identical(expected, actual, receipt, name, **metadata):
    e = expected.view(np.uint32)
    a = actual.view(np.uint32)
    counts = {key: int(np.count_nonzero(e[i] != a[i])) for i,key in enumerate(FIELDS)}
    count = sum(counts.values())
    receipt.append(dict(name=name, different_words=count, per_array=counts,
                        words=expected.size, sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        **metadata))
    if count:
        first = np.argwhere(e != a)[:12]
        pytest.fail(f"{name}: {count} differing words; first " +
                    repr([(FIELDS[i],int(j),hex(int(e[i,j])),hex(int(a[i,j]))) for i,j in first]))


def invoke_device(cp, a, mode, ns, constants, enabled=True, restart=False, layout="C"):
    fields = {key: a[i] for i,key in enumerate(FIELDS)}
    stbolt, gravity, dt, gamma, relaxation, omega, hml0 = constants
    configs = dict(config_oml1d=enabled, config_do_restart=restart,
                   config_oml_gamma=gamma, config_oml_relaxation_time=relaxation,
                   config_oml_hml0=hml0)
    mesh = dict(nCellsSolve=ns, latCell=fields["phase"])
    if mode == 0:
        oml.oml1d(fields, u=fields["u"], v=fields["v"], coriolis=fields["phase"],
                  n_cells_solve=ns, stbolt=stbolt, gravity=gravity, dt_pbl=dt,
                  gamma=gamma, relaxation=relaxation)
    elif mode == 1:
        winds = {}
        for name,key in (("uReconstructZonal","u"),("uReconstructMeridional","v")):
            # Non-lowest levels are sentinels and cannot affect the port.
            wind = cp.full((3,a.shape[1]), np.float32(10000.), order=layout)
            wind[0] = fields[key]
            winds[name] = wind
        oml.driver_oml1d(configs, mesh, winds, fields, fields,
                         dt_pbl=dt, stbolt=stbolt, gravity=gravity, omega=omega)
    else:
        oml.initialize_oml1d(configs, mesh, fields, fields)


@pytest.mark.parametrize("gamma", [.001, .14, 2.])
@pytest.mark.parametrize("relaxation", [-1., 0., 600., 864000.])
def test_whole_module(oracle, gpu, receipt, gamma, relaxation):
    a = columns()
    a[16] *= np.float32(1.e-4)  # direct module takes f, driver takes latitude
    expected = a.copy()
    constants = (*CONSTANTS[:3],gamma,relaxation,*CONSTANTS[5:])
    ns = a.shape[1] - 7
    oracle(a.shape[1],ns,0,1,0,expected,*constants)
    device = gpu.asarray(a)
    invoke_device(gpu,device,0,ns,constants)
    identical(expected,device.get(),receipt,"whole-module",gamma=gamma,relaxation=relaxation)


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("enabled", [True, False])
def test_driver(oracle, gpu, receipt, layout, enabled):
    a = columns()
    expected = a.copy()
    ns = a.shape[1]-7
    oracle(a.shape[1],ns,1,int(enabled),0,expected,*CONSTANTS)
    device = gpu.asarray(a)
    invoke_device(gpu,device,1,ns,CONSTANTS,enabled=enabled,layout=layout)
    identical(expected,device.get(),receipt,"driver",layout=layout,enabled=enabled)


@pytest.mark.parametrize("hml0", [30., 0., -1.])
@pytest.mark.parametrize("enabled,restart", [(True,False),(True,True),(False,False)])
def test_initialization(oracle, gpu, receipt, hml0, enabled, restart):
    a = columns()
    expected = a.copy()
    constants = (*CONSTANTS[:6],hml0)
    ns = a.shape[1]-7
    oracle(a.shape[1],ns,2,int(enabled),int(restart),expected,*constants)
    device = gpu.asarray(a)
    invoke_device(gpu,device,2,ns,constants,enabled=enabled,restart=restart)
    identical(expected,device.get(),receipt,"initialization",hml0=hml0,enabled=enabled,restart=restart)


def test_repeated_driver(oracle, gpu, receipt):
    a = columns(256)
    a[17] = 2.
    expected = a.copy()
    device = gpu.asarray(a)
    invoke_device(gpu,device,2,256,CONSTANTS)
    oracle(256,256,2,1,0,expected,*CONSTANTS)
    # Five days of synthetic columns. This is explicitly NOT forecast proof.
    for _ in range(720):
        oracle(256,256,1,1,0,expected,*CONSTANTS)
        invoke_device(gpu,device,1,256,CONSTANTS)
    identical(expected,device.get(),receipt,"synthetic-five-day-recurrence",steps=720)


def test_native_subnormal_momentum(oracle, gpu, receipt):
    """MPAS module_sf_oml.F:99-110 must retain subnormal hu2/hv2 words."""
    a = columns(32)
    a[0:2] = 290.
    a[2:4] = 10.
    bits = np.asarray([1, 2, 0x7fffff, 0x80000001], dtype=np.uint32)
    a[4] = np.resize(bits.view(np.float32), 32)
    a[5] = -a[4]
    a[7:11] = 0.
    a[11] = 285.
    a[12] = 0.
    a[14:17] = 0.
    constants = (*CONSTANTS[:4], 0., *CONSTANTS[5:])
    expected = a.copy()
    oracle(32,32,0,1,0,expected,*constants)
    assert np.array_equal(expected[4:6].view(np.uint32), a[4:6].view(np.uint32))
    device = gpu.asarray(a)
    invoke_device(gpu,device,0,32,constants)
    identical(expected,device.get(),receipt,"native-subnormal-momentum",
              input_momentum_words=bits.tolist(), compiler="direct NVRTC, --ftz=false")


def test_empty_owned_region(oracle, gpu, receipt):
    a = columns(64)
    device = gpu.asarray(a)
    for mode in (0,1,2):
        invoke_device(gpu,device,mode,0,CONSTANTS)
    identical(a,device.get(),receipt,"empty-owned-region")


def test_off_and_restart_are_noops():
    oml.driver_oml1d({"config_oml1d":False},{},{},{},{},
                     dt_pbl=0.,stbolt=0.,gravity=0.,omega=0.)
    oml.initialize_oml1d({"config_oml1d":False},{},{},{})
    oml.initialize_oml1d({"config_oml1d":True,"config_do_restart":True},{},{},{})


def test_v842_citation_windows():
    import re
    from oml_native_oracle import SOURCE_PINS
    source = os.environ.get("MPAS_REF")
    if not source:
        pytest.skip("pinned MPAS tree unavailable; set MPAS_REF")
    physics = Path(source) / "src/core_atmosphere/physics"
    texts = {}
    for relative,digest in SOURCE_PINS.items():
        path = physics / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        texts[path.name] = path.read_text().splitlines()
    citations = re.findall(r"(\w+\.F):(\d+)-(\d+)",Path(oml.__file__).read_text())
    assert citations
    for name,first,last in citations:
        assert 1 <= int(first) <= int(last) <= len(texts[name])

@pytest.mark.parametrize("library", ["liboml_witness.so", "liboml_shared_math_witness.so"])
def test_host_transcription_witness(oracle, library):
    path = Path(os.environ["MPAS_OML_ORACLE"]).with_name(library)
    if not path.exists():
        pytest.skip("host arithmetic witness not built")
    lib = ctypes.CDLL(str(path))
    witness = lib.replay_cpp
    witness.argtypes = oracle.argtypes
    witness.restype = None
    rows = []
    for dt in (0., 30., 600.):
        for gamma in (.001,.14,2.):
            for relax in (-1.,0.,600.,864000.):
                expected = columns()
                expected[16] *= np.float32(1.e-4)
                actual = expected.copy()
                constants = (*CONSTANTS[:2],dt,gamma,relax,*CONSTANTS[5:])
                args = (expected.shape[1],expected.shape[1]-7,0,1,0)
                oracle(*args,expected,*constants)
                witness(*args,actual,*constants)
                differences = int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
                rows.append(dict(dt=dt,gamma=gamma,relaxation=relax,different_words=differences))
    # Exercise the exact kernel mask, Coriolis, SST writeback, and initialization
    # source too, bridging CUDA's shared glibc sine to the host's pinned glibc.
    for mode,enabled,restart,hml0 in (
            (1,1,0,30.), (1,0,0,30.),
            *((2,e,r,h) for h in (30.,0.,-1.) for e,r in ((1,0),(1,1),(0,0)))):
        expected = columns()
        actual = expected.copy()
        constants = (*CONSTANTS[:6],hml0)
        args = (expected.shape[1],expected.shape[1]-7,mode,enabled,restart)
        oracle(*args,expected,*constants)
        witness(*args,actual,*constants)
        differences = int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
        rows.append(dict(mode=mode,enabled=enabled,restart=restart,hml0=hml0,
                         different_words=differences))
    expected = columns(256)
    expected[17] = 2.
    actual = expected.copy()
    oracle(256,256,2,1,0,expected,*CONSTANTS)
    witness(256,256,2,1,0,actual,*CONSTANTS)
    for _ in range(720):
        oracle(256,256,1,1,0,expected,*CONSTANTS)
        witness(256,256,1,1,0,actual,*CONSTANTS)
    assert np.isfinite(expected).all(), "synthetic recurrence must stay finite"
    differences = int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32)))
    rows.append(dict(mode=1,steps=720,finite=True,different_words=differences))
    receipt_path = os.environ.get("MPAS_OML_HOST_RECEIPT")
    if receipt_path:
        if library == "liboml_shared_math_witness.so":
            p = Path(receipt_path)
            receipt_path = p.with_name(p.stem + "-shared-header" + p.suffix)
        Path(receipt_path).write_text(json.dumps(dict(
            acceptance=False, kind="same CUDA module/driver/init compiled on CPU",
            math_path=("shared device header on host" if "shared_math" in library
                       else "host glibc sinf"),
            witness_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            different_words=sum(r["different_words"] for r in rows), tests=rows),indent=2)+"\n")
    assert all(row["different_words"] == 0 for row in rows), rows


def test_bad_precision_and_layout(gpu):
    fields = {name: gpu.zeros(3, dtype=np.float32) for name in oml.STATE_FIELDS}
    kwargs = dict(u=fields["ust"],v=fields["ust"],coriolis=fields["ust"],n_cells_solve=3,
                  stbolt=CONSTANTS[0],gravity=CONSTANTS[1],dt_pbl=600.,gamma=.14,relaxation=0.)
    fields["h_oml"] = gpu.zeros(3,dtype=np.float64)
    with pytest.raises(TypeError,match="float32"):
        oml.oml1d(fields,**kwargs)
    fields["h_oml"] = gpu.zeros(6,dtype=np.float32)[::2]
    with pytest.raises(ValueError,match="contiguous"):
        oml.oml1d(fields,**kwargs)
