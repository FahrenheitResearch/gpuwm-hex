"""Replay f1 ORCL0001 fields through the unchanged MPAS and resident CUDA IAU.

Manifest is metadata only: option/window/dt/itimestep, nCellsSolve,
nEdgesSolve, moist_start/end/index_qv, and fields mapping the 13 wrapper
arguments to f1 filenames (relative to the manifest). Rust decodes/permutates
all field bytes. No synthetic or missing field is substituted into a real deck.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.iau_v842 import IAUConfig, atm_add_tend_anal_incr, atm_iau_coef
from test_iau_v842 import _digest, _load_native, _pools, replay

FIELDS = ("zz", "rho_zz", "theta_m", "scalars", "rho_edge", "u_amb",
          "rho_amb", "theta_amb", "scalars_amb", "tend_ru", "tend_rtheta",
          "tend_rho", "tend_scalars")
REPLAY_SOURCES = json.loads(os.environ["HEXCORE_IAU_REAL_COLUMNS_LIST"]) if (
    "HEXCORE_IAU_REAL_COLUMNS_LIST" in os.environ) else [os.environ.get("HEXCORE_IAU_REAL_COLUMNS")]


def load_oracle_field(reader, path, *, post=False):
    """Borrow the native/Rust-packed buffers; metadata shape only in Python."""
    read = reader.iau_oracle_read_post if post else reader.iau_oracle_read
    read.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    read.restype = ctypes.c_ssize_t
    header = (ctypes.c_int32 * 21)()
    count = read(os.fsencode(path), header, None, None, 0)
    if count < 0:
        raise ValueError(f"Rust rejected oracle field {path}: {count}")
    shape = tuple(header[3:3+header[2]])
    original = np.empty(shape, np.float32, order="F")
    row_major = np.empty(shape, np.float32, order="C")
    if read(os.fsencode(path), header, original.ctypes.data,
            row_major.ctypes.data, count) != count:
        raise ValueError(f"Oracle field changed during reading: {path}")
    return original, row_major, list(header)


def test_compiled_rust_reader_returns_identical_words_in_both_layouts():
    decoder = os.environ.get("HEXCORE_IAU_ORACLE_READER")
    fixture = os.environ.get("HEXCORE_IAU_ORACLE_FIXTURE")
    if not decoder or not fixture:
        pytest.skip("compiled Rust reader and Rust-generated transport fixture required")
    original, row_major, header = load_oracle_field(ctypes.CDLL(decoder), Path(fixture))
    expected = np.array([[0,1,0x3f800000], [0x80000000,0x7fc00123,0xff800000]], np.uint32)
    assert header[3:5] == [2,3]
    assert original.flags.f_contiguous and row_major.flags.c_contiguous
    np.testing.assert_array_equal(original.view(np.uint32), expected)
    np.testing.assert_array_equal(row_major.view(np.uint32), expected)


@pytest.mark.parametrize("source",REPLAY_SOURCES)
def test_native_real_state_replay_matches_full_native_post_call(source):
    decoder = os.environ.get("HEXCORE_IAU_ORACLE_READER")
    native_path = os.environ.get("HEXCORE_IAU_NATIVE")
    if not source or not decoder or not native_path:
        pytest.skip("complete native capture and compiled native/Rust libraries required")
    path = Path(source)
    metadata = json.loads(path.read_text())
    assert set(metadata["fields"]) == set(FIELDS)
    assert set(metadata["native_post_fields"]) == {"tend_ru","tend_rtheta","tend_rho","tend_scalars"}
    reader = ctypes.CDLL(decoder)
    fields = {}
    contexts = set()
    for name in FIELDS:
        p = path.parent/metadata["fields"][name]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == metadata["record_sha256"][p.name]
        fields[name], _, header = load_oracle_field(reader,p)
        contexts.add(tuple(header[13:16]))
    assert len(contexts) == 1 and next(iter(contexts))[0] == metadata["itimestep"]
    nl,nc = fields["zz"].shape
    ne = fields["rho_edge"].shape[1]
    ns = fields["scalars"].shape[0]
    nsi = fields["scalars_amb"].shape[0]
    for name in FIELDS:
        shape = ((nsi,nl,nc) if name == "scalars_amb" else (ns,nl,nc)
                 if name in ("scalars","tend_scalars") else (nl,ne)
                 if name in ("rho_edge","u_amb","tend_ru") else (nl,nc))
        assert fields[name].shape == shape
    ncs,nes,ms,me,qv = (metadata[k] for k in (
        "nCellsSolve","nEdgesSolve","moist_start","moist_end","index_qv"))
    assert 0 <= ncs <= nc and 0 <= nes <= ne
    assert 1 <= qv <= ns and 1 <= ms <= ns+1 and 0 <= me <= ns and (ms > me or me <= nsi)
    weight = ctypes.c_float()
    _load_native(native_path)(nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,metadata["itimestep"],
        metadata["dt"],metadata["window"],1,*(fields[name] for name in FIELDS),ctypes.byref(weight))
    for name,filename in metadata["native_post_fields"].items():
        p = path.parent/filename
        assert hashlib.sha256(p.read_bytes()).hexdigest() == metadata["record_sha256"][filename]
        expected,_,header = load_oracle_field(reader,p,post=True)
        assert tuple(header[13:16]) in contexts and expected.shape == fields[name].shape
        np.testing.assert_array_equal(fields[name].view(np.uint32),expected.view(np.uint32),err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("source",REPLAY_SOURCES)
def test_real_iau_capture_matches_pristine_native_word_for_word(replay,source):
    decoder = os.environ.get("HEXCORE_IAU_ORACLE_READER")
    if not source or not decoder:
        pytest.skip("real IAU capture manifest and compiled Rust decoder required")
    cp, native, cache, records = replay
    path = Path(source)
    metadata = json.loads(path.read_text())
    if set(metadata["fields"]) != set(FIELDS):
        raise ValueError("real replay requires every native input/output field")
    reader = ctypes.CDLL(decoder)
    originals, host, provenance = {}, {}, {}
    for name in FIELDS:
        field_path = path.parent / metadata["fields"][name]
        digest = hashlib.sha256(field_path.read_bytes()).hexdigest()
        if "record_sha256" in metadata and digest != metadata["record_sha256"][field_path.name]:
            raise ValueError(f"capture hash changed for {name}")
        originals[name], host[name], header = load_oracle_field(reader, field_path)
        provenance[name] = {"file": str(field_path), "header": header,
                            "sha256": digest}
    # f1 snapshots must belong to one call, MPI rank and pre-call phase.
    contexts = {tuple(v["header"][13:16]) for v in provenance.values()}
    if len(contexts) != 1:
        raise ValueError("real IAU fields belong to different step/event/rank contexts")
    nl, nc = host["zz"].shape
    ne = host["rho_edge"].shape[1]
    ns = host["scalars"].shape[0]
    nsi = host["scalars_amb"].shape[0]
    ncs, nes = metadata["nCellsSolve"], metadata["nEdgesSolve"]
    ms, me, qv = (metadata[k] for k in ("moist_start", "moist_end", "index_qv"))
    config = IAUConfig(metadata["option"], metadata["window"])
    dt, step = metadata["dt"], metadata["itimestep"]
    if next(iter(contexts))[0] != step:
        raise ValueError("real IAU capture timestep differs from replay metadata")
    # Validate shapes and dimensions before native code can dereference arrays.
    for name in FIELDS:
        shape = ((nsi,nl,nc) if name == "scalars_amb" else
                 (ns,nl,nc) if name in ("scalars","tend_scalars")
                 else (nl,ne) if name in ("rho_edge","u_amb","tend_ru") else (nl,nc))
        if host[name].shape != shape:
            raise ValueError(f"real replay shape differs for {name}")
    if not (0 <= ncs <= nc and 0 <= nes <= ne and 1 <= qv <= ns
            and 1 <= ms <= ns+1 and 0 <= me <= ns and (ms > me or me <= nsi)):
        raise ValueError("real replay pool dimensions/indices invalid")
    device, pools = _pools(cp,host,ncs,nes,ms,me,qv)
    atm_add_tend_anal_incr(config,pools,step,dt,device["tend_ru"],
                         device["tend_rtheta"],device["tend_rho"],cache=cache)
    weight = ctypes.c_float()
    native(nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,step,dt,config.config_IAU_window_length_s,
           {"off":0,"on":1,"ON":2,"other":3}[config.config_IAU_option],
           *(originals[name] for name in FIELDS),ctypes.byref(weight))
    result = {name: cp.asnumpy(value) for name,value in device.items()}
    native_post = {}
    for name, filename in metadata.get("native_post_fields", {}).items():
        post_path = path.parent / filename
        digest = hashlib.sha256(post_path.read_bytes()).hexdigest()
        if digest != metadata["record_sha256"][filename]:
            raise ValueError(f"native post capture hash changed for {name}")
        expected, _, header = load_oracle_field(reader,post_path,post=True)
        if tuple(header[13:16]) not in contexts or expected.shape != originals[name].shape:
            raise ValueError(f"native post call/shape differs for {name}")
        np.testing.assert_array_equal(expected.view(np.uint32),originals[name].view(np.uint32),err_msg=f"full-native versus standalone {name}")
        native_post[name] = {"sha256": digest, "words": expected.size,
                             "different_words": 0}
    comparison = {}
    for name in FIELDS:
        comparison[name] = {
            "words": result[name].size,
            "different_words": int(np.count_nonzero(result[name].view(np.uint32)
                                                     != originals[name].view(np.uint32))),
            "input_sha256": _digest(host[name]), "native_sha256": _digest(originals[name]),
            "cuda_sha256": _digest(result[name]),
        }
    records.append({"real_capture": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "decoder_sha256": hashlib.sha256(Path(decoder).read_bytes()).hexdigest(),
                    "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "capture_origin": metadata.get("capture"), "native_post": native_post,
                    "provenance": provenance, "fields": comparison})
    assert np.float32(weight.value).view(np.uint32) == atm_iau_coef(config,step,dt).view(np.uint32)
    for name in FIELDS:
        np.testing.assert_array_equal(result[name].view(np.uint32),originals[name].view(np.uint32),err_msg=name)
