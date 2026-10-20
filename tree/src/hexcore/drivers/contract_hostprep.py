"""Drive the exact float32 contract references in the host-prep crate."""
from functools import lru_cache
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

from hexcore.errors import MpasPortError
from hexcore.engines import EngineSpec, gpuwm_candidates
from hexcore.acoustic import AcousticStepState
from hexcore.transport import ScalarTransportResult

_ENGINE = EngineSpec(name="hex_contract_hostprep", flag="--engine",
    env_names=("GPUWM_HEX_CONTRACT_HOSTPREP",), gpuwm_env="GPUWM_HEX_CONTRACT_HOSTPREP",
    subject="the regional contract host references",
    what_breaks="the float32 contract host references cannot be computed",
    cargo_package="hex-contract-hostprep")


@lru_cache(maxsize=1)
def engine():
    explicit = os.environ.get("GPUWM_HEX_CONTRACT_HOSTPREP")
    root = Path(__file__).resolve().parents[3]
    candidates = ([Path(explicit)] if explicit else [
        Path(__file__).resolve().parents[1] / "libexec/hex_contract_hostprep",
        Path(__file__).resolve().parents[1] / "libexec/hex_contract_hostprep.exe",
        root / "rust/host-prep/target/release/hex_contract_hostprep",
        root / "rust/host-prep/target/release/hex_contract_hostprep.exe",
        *gpuwm_candidates(_ENGINE),
        Path(shutil.which("hex_contract_hostprep") or "__missing_hostprep__"),
    ])
    for path in candidates:
        if path.is_file():
            return path
    raise MpasPortError(
        "the regional contract needs hex_contract_hostprep to compute its exact "
        "float32 host references; build with cargo build --release --locked "
        "--manifest-path rust/host-prep/Cargo.toml and set "
        "GPUWM_HEX_CONTRACT_HOSTPREP to that executable")


def call(mode, dims, arrays, shapes):
    """Stream input, then read exactly the declared output, never a tolerance."""
    process = subprocess.Popen([str(engine())], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(struct.pack("<8s8Q", b"HEXREF01", mode, *dims, len(arrays)))
        for key, array in arrays.items():
            array = np.asarray(array)
            wire = "<i8" if array.dtype.kind in "iu" else "<f4"
            if array.dtype.kind not in "iu" and array.dtype != np.dtype("float32"):
                raise TypeError(f"contract reference {key} must be float32")
            value = np.ascontiguousarray(array, dtype=wire)
            name = key.encode("ascii")
            process.stdin.write(struct.pack("<Q", len(name))+name
                               +struct.pack("<BQ", value.dtype.itemsize, value.size))
            data = memoryview(value).cast("B")
            for offset in range(0, len(data), 1024*1024):
                process.stdin.write(data[offset:offset+1024*1024])
        process.stdin.close()
        outputs = []
        for shape in shapes:
            value = np.empty(shape, dtype="<f4")
            data = memoryview(value).cast("B")
            offset = 0
            while offset < len(data):
                read = process.stdout.readinto(data[offset:offset+1024*1024])
                if not read:
                    message = process.stderr.read().decode(errors="replace")
                    raise MpasPortError("contract host-prep returned incomplete arrays: "+message)
                offset += read
            outputs.append(value)
        if process.stdout.read(1):
            raise MpasPortError("contract host-prep returned excess bytes: output ABI differs")
        message = process.stderr.read().decode(errors="replace")
        if process.wait():
            raise MpasPortError("contract host-prep failed: "+message)
        return outputs
    except BrokenPipeError as error:
        raise MpasPortError("contract host-prep rejected input: "
            +process.stderr.read().decode(errors="replace")) from error
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            process.terminate()
        process.wait()


def dimensions(b, width=0, tracers=1):
    return (b.ncells, b.nedges, b.nlev, b.max_edges, width, tracers)


def horizontal(b, coefficients):
    stage = b.scalars_stage
    width = coefficients.adv_coefs.shape[1]
    return call(0, dimensions(b, width, stage.shape[0]), {
        "stage": stage, "velocity": np.asarray(b.state.rho_u, np.float32),
        "dv": b.dv_edge, "cells_on_edge": b.cells_on_edge,
        "mask_edge": b.masks.bdy_mask_edge,
        "adv": np.asarray(coefficients.adv_coefs, np.float32),
        "adv3": np.asarray(coefficients.adv_coefs_3rd, np.float32),
        "n_adv": coefficients.n_adv_cells_for_edge,
        "adv_cells": coefficients.adv_cells_for_edge,
    }, [(stage.shape[0], b.nlev, b.nedges)])[0]


def vertical(b):
    return call(1, dimensions(b), {
        "stage": b.scalars_stage[0],
        "ww": np.asarray(b.saved.vertical_velocity, np.float32),
        "fzm": np.asarray(b.vertical.fzm, np.float32),
        "fzp": np.asarray(b.vertical.fzp, np.float32),
    }, [(b.nlev+1, b.ncells)])[0]


def finish(b, edges, device_flux):
    old = np.asarray(b.state.scalars, np.float32)
    rho_old = np.asarray(b.state.rho, np.float32)
    rho_new = np.ascontiguousarray(rho_old*np.float32(1.000001))
    scalars, density = call(2, dimensions(b, tracers=old.shape[0]), {
        "old": old, "stage": b.scalars_stage, "rho_old": rho_old,
        "rho_new": rho_new, "velocity": np.asarray(b.state.rho_u, np.float32),
        "flux": device_flux, "edge_values": edges,
        "mask_cell": b.masks.bdy_mask_cell, "counts": b.n_edges_on_cell,
        "edges": b.edges_on_cell, "signs": -b.edge_sign_on_cell,
        "inv_area": b.inv_area_cell, "rdzw": np.asarray(b.vertical.rdzw, np.float32),
    }, [old.shape, rho_old.shape])
    return ScalarTransportResult(scalars, density)


def acoustic(b, inputs):
    off, coef, forcing, state = inputs
    arrays = {
        "cells_on_edge": b.cells_on_edge, "counts": b.n_edges_on_cell,
        "edges": b.edges_on_cell, "signs": b.edge_sign_on_cell,
        "dv": b.dv_edge, "inv_dc": b.inv_dc_edge, "inv_area": b.inv_area_cell,
        "spec_cell": b.masks.spec_zone_mask_cell, "spec_edge": b.masks.spec_zone_mask_edge,
        "rdzw": np.asarray(b.vertical.rdzw, np.float32),
        "fzm": np.asarray(b.vertical.fzm, np.float32),
        "fzp": np.asarray(b.vertical.fzp, np.float32), "theta": forcing.theta_m,
    }
    arrays.update({name: getattr(state, name) for name in
                   ("ru_p", "ru_avg", "rw_p", "ww_avg", "rho_pp", "rtheta_pp")})
    arrays.update({name: getattr(forcing, name) for name in
                   ("zz", "exner", "cqu", "zxu", "tend_ru", "tend_rho", "tend_rt",
                    "tend_rw", "rho_zz", "rw_save", "rw", "w", "dss")})
    arrays.update({name: getattr(coef, name) for name in
                   ("cofrz", "coftz", "cofwz", "cofwr", "cofwt", "a_tri", "alpha_tri", "gamma_tri")})
    arrays.update({name: getattr(off, name) for name in ("etp", "etm", "ewp", "ewm")})
    names = ("ru_p", "ru_avg", "rw_p", "ww_avg", "rho_pp", "rtheta_pp")
    values = call(3, dimensions(b), arrays, [getattr(state, name).shape for name in names])
    return AcousticStepState(**dict(zip(names, values)), rtheta_pp_old=state.rtheta_pp.copy())
