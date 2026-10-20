"""Real f1 OML call captures versus strict standalone MPAS and CUDA.

MPAS_OML_CAPTURES is a JSON list of capture directories, supplied by the
campaign, never a case default. Rust alone decodes and compares field words.
GPU tests require explicit opt-in and the GPU queue, as in the synthetic suite.
"""
import ctypes as ct
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys import oml_v842 as oml
from test_mpas_oml_v842 import oracle, gpu

NAMES = (*oml.STATE_FIELDS, "uair", "vair", "f")
CAPTURES = json.loads(os.environ.get("MPAS_OML_CAPTURES", "[]"))


@pytest.fixture(scope="module")
def transport():
    path = os.environ.get("MPAS_OML_CAPTURE_TRANSPORT")
    if not path:
        pytest.skip("Rust real-column transport not built")
    lib = ct.CDLL(path)
    lib.oml_batch_open.argtypes, lib.oml_batch_open.restype = [ct.c_char_p], ct.c_void_p
    lib.oml_batch_inputs.argtypes = [ct.c_void_p, ct.POINTER(ct.c_size_t), ct.POINTER(ct.POINTER(ct.c_float))]
    lib.oml_batch_inputs.restype = ct.POINTER(ct.c_float)
    lib.oml_batch_compare.argtypes = [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_size_t)]
    lib.oml_batch_close.argtypes = [ct.c_void_p]
    return lib


@pytest.fixture(scope="module")
def real_receipt():
    rows = []
    yield rows
    path = os.environ.get("MPAS_OML_REAL_RECEIPT")
    if path:
        Path(path).write_text(json.dumps(dict(
            schema="mpas-oml-real-column/v1", forecast_acceptance=False,
            native_ranks=1, replay_processes=1, host=socket.gethostname(),
            comparisons=rows, different_words=sum(r["different_words"] for r in rows),
            capture_transport_sha256=hashlib.sha256(Path(os.environ["MPAS_OML_CAPTURE_TRANSPORT"]).read_bytes()).hexdigest(),
            implementation_sha256=hashlib.sha256(Path(oml.__file__).read_bytes()).hexdigest(),
            compiler="direct NVRTC; explicit module load; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true",
            authority="actual f1 instrumented native OML arguments/outputs on corrected f0 cases",
            scalar_context="f1 scalar call events have no mapped cellID; no global-ID claim",
        ), indent=2) + "\n")


def run_capture(root, lib, native, cp, rows):
    handle = lib.oml_batch_open(str(Path(root).resolve()).encode())
    assert handle, "Rust rejected capture completeness/type/constant validation"
    try:
        n = ct.c_size_t()
        constants = ct.POINTER(ct.c_float)()
        pointer = lib.oml_batch_inputs(handle, ct.byref(n), ct.byref(constants))
        inputs = np.ctypeslib.as_array(pointer, shape=(17*n.value,)).reshape(17,n.value)
        resolved = np.ctypeslib.as_array(constants, shape=(5,))
        # ABI packing after Rust decoding, with two unused driver-only fields.
        host = np.zeros((19,n.value), dtype=np.float32)
        host[:17] = inputs
        stbolt, gravity, dt, gamma, relaxation = resolved
        native(n.value,n.value,0,1,0,host,stbolt,gravity,dt,gamma,relaxation,0.,0.)
        variants = [("strict-standalone",host[:17].copy())]
        if cp is not None:
            device = cp.asarray(inputs)
            fields = {name:device[i] for i,name in enumerate(oml.STATE_FIELDS)}
            oml.oml1d(fields, u=device[14], v=device[15], coriolis=device[16],
                      n_cells_solve=n.value, stbolt=stbolt,gravity=gravity,
                      dt_pbl=dt,gamma=gamma,relaxation=relaxation)
            variants.append(("hex-cuda",device.get()))
        for name,actual in variants:
            counts = (ct.c_size_t * 17)()
            assert lib.oml_batch_compare(handle,actual.ctypes.data,counts)==0
            per_array = dict(zip(NAMES,map(int,counts)))
            row = dict(capture=str(root), backend=name, columns=n.value,
                       words=actual.size, per_array=per_array,
                       different_words=sum(per_array.values()),
                       constants=dict(zip(("stbolt","gravity","dt_pbl","gamma","relaxation"),map(float,resolved))),
                       output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
            rows.append(row)
            assert row["different_words"]==0, row
            # A one-bit fault in a real replay output must be counted exactly.
            bad = actual.copy()
            bad.view(np.uint32)[0,0] ^= np.uint32(1)
            assert lib.oml_batch_compare(handle,bad.ctypes.data,counts)==0
            assert counts[0]==1 and sum(counts)==1
    finally:
        lib.oml_batch_close(handle)


@pytest.mark.parametrize("root", CAPTURES)
def test_real_native(root, transport, oracle, real_receipt):
    run_capture(root,transport,oracle,None,real_receipt)


@pytest.mark.parametrize("root", CAPTURES)
def test_real_cuda(root, transport, oracle, gpu, real_receipt):
    run_capture(root,transport,oracle,gpu,real_receipt)


def test_bulk_record_writer(transport, oracle):
    """Rust encoding must preserve native records and expose an output fault."""
    root=os.environ.get("MPAS_OML_WRITER_CAPTURE")
    output=os.environ.get("MPAS_OML_WRITER_OUTPUT")
    helper=os.environ.get("MPAS_OML_WORD_HELPER")
    if not all((root,output,helper)):
        pytest.skip("explicit small capture, owned output and Rust word helper required")
    import subprocess
    transport.oml_batch_write.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_char_p]
    handle=transport.oml_batch_open(str(Path(root).resolve()).encode())
    assert handle
    try:
        n=ct.c_size_t(); constants=ct.POINTER(ct.c_float)()
        pointer=transport.oml_batch_inputs(handle,ct.byref(n),ct.byref(constants))
        host=np.zeros((19,n.value),dtype=np.float32)
        host[:17]=np.ctypeslib.as_array(pointer,shape=(17*n.value,)).reshape(17,n.value)
        stbolt,gravity,dt,gamma,relaxation=np.ctypeslib.as_array(constants,shape=(5,))
        oracle(n.value,n.value,0,1,0,host,stbolt,gravity,dt,gamma,relaxation,0.,0.)
        output=Path(output).resolve()
        output.mkdir()
        counts={}
        for variant in ("native","fault"):
            actual=host[:17].copy()
            if variant=="fault":actual.view(np.uint32)[0,-1]^=np.uint32(1)
            target=output/variant
            assert transport.oml_batch_write(handle,actual.ctypes.data,str(target).encode())==0
            actual_set={p.name for p in target.glob("*.bin")}
            expected_set={p.name for p in Path(root).glob("*.post.*.bin")}
            assert actual_set==expected_set
            counts[variant]=0
            for name in expected_set:
                reference=Path(root)/name; candidate=target/name
                row=json.loads(subprocess.check_output([helper,"compare",str(reference),str(candidate)],text=True))
                counts[variant]+=row["different_words"]
                if variant=="native":assert hashlib.sha256(reference.read_bytes()).digest()==hashlib.sha256(candidate.read_bytes()).digest()
        assert counts=={"native":0,"fault":1}
        (output/"writer-receipt.json").write_text(json.dumps(dict(schema="p19-oml-rust-record-writer/v1",columns=n.value,records=len(expected_set),different_words=counts,forecast_acceptance=False),indent=2)+"\n")
    finally:
        transport.oml_batch_close(handle)
