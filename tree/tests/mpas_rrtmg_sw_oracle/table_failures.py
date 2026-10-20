"""Pristine MPAS SW initialization error paths and Rust atomic refusal.

MPAS mpas_atmphys_rrtmg_swinit.F:70-1206 reads fourteen Fortran records.
No numerical table value is decoded here; only record framing locates EOF.
"""
import hashlib
from pathlib import Path
import struct
import subprocess


def replay_failures(oracle, bridge, scratch):
    oracle, bridge, scratch = Path(oracle).resolve(), Path(bridge).resolve(), Path(scratch)
    data = (oracle / "RRTMG_SW_DATA").read_bytes()
    starts, offset = [], 0
    while offset < len(data):
        starts.append(offset)
        length = struct.unpack_from(">I", data, offset)[0]
        assert struct.unpack_from(">I", data, offset + 4 + length)[0] == length
        offset += length + 8
    assert len(starts) == 14 and offset == len(data)
    scratch.mkdir()
    rows = []
    for band, start in enumerate(starts, 16):
        directory = scratch / f"band-{band}"
        directory.mkdir()
        source = directory / "RRTMG_SW_DATA"
        source.write_bytes(data[:start])
        native_output, decoded = directory / "unexpected.swd", directory / "decoded"
        native = subprocess.run([str(oracle / "sw_dump_tables"), str(native_output)], cwd=directory,
                                capture_output=True, text=True)
        port = subprocess.run([str(bridge), str(source), str(decoded)], capture_output=True, text=True)
        assert native.returncode != 0 and "error reading RRTMG_SW_DATA" in native.stdout + native.stderr
        assert not native_output.exists()
        assert port.returncode != 0 and not decoded.exists()
        rows.append(dict(band=band, bytes=start, native_exit=native.returncode,
                         native_message=native.stdout + native.stderr,
                         rust_exit=port.returncode, rust_message=port.stdout + port.stderr,
                         unpublished=True))
        source.unlink()
        directory.rmdir()
    # A self-link gives both readers a deterministic open failure without
    # changing filesystem permissions or depending on a particular user ID.
    directory = scratch / "open-failure"
    directory.mkdir()
    source = directory / "RRTMG_SW_DATA"
    source.symlink_to(source.name)
    native_output, decoded = directory / "unexpected.swd", directory / "decoded"
    native = subprocess.run([str(oracle / "sw_dump_tables"), str(native_output)], cwd=directory,
                            capture_output=True, text=True)
    port = subprocess.run([str(bridge), str(source), str(decoded)], capture_output=True, text=True)
    assert native.returncode != 0 and "error reading RRTMG_SW_DATA" in native.stdout + native.stderr
    assert not native_output.exists() and port.returncode != 0 and not decoded.exists()
    open_failure = dict(native_exit=native.returncode, native_message=native.stdout + native.stderr,
                        rust_exit=port.returncode, rust_message=port.stdout + port.stderr, unpublished=True)
    source.unlink()
    directory.rmdir()
    scratch.rmdir()
    return dict(authority="Pristine MPAS initialization EOF refusal; production Rust table decoder",
                input_sha256=hashlib.sha256(data).hexdigest(),
                native_executable_sha256=hashlib.sha256((oracle / "sw_dump_tables").read_bytes()).hexdigest(),
                rust_executable_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                rows=rows, open_failure=open_failure, whole_model_proof=False)
