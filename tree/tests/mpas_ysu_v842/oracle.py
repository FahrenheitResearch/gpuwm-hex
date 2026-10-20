"""Test-only fixtures and native/witness invocation; no production data path."""
from __future__ import annotations
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from hexcore.mpasphys import ysu_v842 as ysu

PTR = ct.POINTER(ct.c_float)
IPTR = ct.POINTER(ct.c_int)
OUTPUTS = ysu._OUTPUT_LEVEL + ysu._OUTPUT_SURFACE + (
    "qmixtnp", "u10", "v10", "dusfc", "dvsfc", "dtsfc", "dqsfc")


def provenance(work: Path, folder: Path):
    """Identify the exact standalone binaries and synthetic fixture files."""
    return dict(native_binary_sha256=hashlib.sha256((work/"native.so").read_bytes()).hexdigest(),
                witness_binary_sha256=hashlib.sha256((work/"witness.so").read_bytes()).hexdigest(),
                synthetic_fixture_sha256={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                           for name in ("ysu-levels.csv", "ysu-surface.csv")})


def cuda_provenance():
    path = os.environ.get('HEX_MPAS_YSU_CUDA_ARTIFACT')
    return dict(cuda_compiler_route='verified prebuilt cubin' if path else 'direct NVRTC',
                compiled_artifact=json.loads(Path(path).with_suffix('.json').read_text()) if path else None,
                launcher_sha256=hashlib.sha256(Path(ysu.__file__).read_bytes()).hexdigest())


def constants():
    f = np.float32
    rd, rv, g = f(287), f(461.6), f(9.81)
    cp = f(f(7)*rd/f(2))
    return dict(dt=f(45), cp=cp, g=g, rd=rd, rv=rv, xlv=f(2.5e6),
                rovcp=f(rd/cp), rovg=f(rd/g), ep1=f(rv/rd-f(1)),
                ep2=f(rd/rv), karman=f(.4))


def fixtures(folder: Path):
    levels = np.genfromtxt(folder / "ysu-levels.csv", delimiter=",", names=True)
    surfaces = np.genfromtxt(folder / "ysu-surface.csv", delimiter=",", names=True)
    for surf in surfaces:
        rows = levels[levels["case"] == surf["case"]]
        nz = len(rows)
        arrays = {a: np.zeros(nz if a in ysu._OUTPUT_LEVEL + ("qmixtnp",) else 1,
                              np.int32 if a == "kpbl1d" else np.float32)
                  for a in ysu.ARRAY_ARGUMENTS}
        for a in ("ux", "vx", "tx", "qvx", "qcx", "qix", "p2d", "pi2d", "rthraten", "qmix"):
            arrays[a] = np.array(rows[a], np.float32)
        arrays["dz8w2d"] = np.array(rows["dz8w"], np.float32)
        arrays["p2di"] = np.array([*rows["p2di_k"], rows["p2di_kp1"][-1]], np.float32)
        for a in ysu._REQUIRED_SURFACE:
            arrays[a] = np.array([surf[a]], np.float32)
        for a in ysu._OPTIONAL_LEVEL + ysu._OPTIONAL_SURFACE:
            arrays[a] = None
        flags = dict(its=1, ite=1, kte=nz, kme=nz+1, nmix=1, f_qc=1, f_qi=1,
                     ysu_topdown_pblmix=int(surf["topdown"]), flag_bep=0)
        yield int(surf["case"]), arrays, flags


class Oracle:
    def __init__(self, work: Path):
        self.host_library = ct.CDLL(str((work / "witness.so").resolve()))
        self.native_library = ct.CDLL(str((work / "native.so").resolve()))
        self.host = self.host_library.run_host
        self.native = self.native_library.run_native
        self.host.argtypes = [ct.POINTER(PTR), PTR, IPTR, PTR, IPTR]
        self.native.argtypes = [ct.POINTER(PTR), PTR, IPTR]
        self.wrapper = self.native_library.run_wrapper
        self.wrapper.argtypes = self.native.argtypes
        self.driver = self.native_library.run_driver
        self.driver.argtypes = self.native.argtypes + [PTR, PTR, PTR]
        lifecycle = self.native_library.native_lifecycle
        lifecycle.argtypes = [IPTR]
        status = np.full(2, -1, np.int32)
        lifecycle(status.ctypes.data_as(IPTR))
        if np.any(status):
            raise RuntimeError("Native YSU lifecycle failed")

    def run(self, arrays, flags, *, native=False, wrapper=False, driver_radiation=None, values=None):
        wrapper = wrapper or driver_radiation is not None
        arrays = {a: None if arr is None else arr.copy() for a, arr in arrays.items()}
        if wrapper:
            # WRF dimensions include an unused top level. Keep its storage
            # allocated independently of the MMM scheme's nz-sized carriers.
            for a in ysu._REQUIRED_LEVEL + ysu._OUTPUT_LEVEL + ysu._OPTIONAL_LEVEL:
                if arrays[a] is not None:
                    arrays[a] = np.concatenate((arrays[a], np.zeros(1, np.float32)))
        ptrs = (PTR*len(ysu.ARRAY_ARGUMENTS))(*[
            PTR() if arrays[a] is None else arrays[a].ctypes.data_as(PTR)
            for a in ysu.ARRAY_ARGUMENTS])
        values = constants() if values is None else values
        scalars = np.array([values[a] for a in ysu.FLOAT_ARGUMENTS], np.float32)
        integers = np.array([flags[a] for a in ysu.INTEGER_ARGUMENTS], np.int32)
        status = np.zeros(1, np.int32)
        if driver_radiation is not None:
            lw, sw = driver_radiation
            diagnostic = np.full(flags["kte"], -999, np.float32)
            self.driver(ptrs, scalars.ctypes.data_as(PTR), integers.ctypes.data_as(IPTR),
                        lw.ctypes.data_as(PTR), sw.ctypes.data_as(PTR), diagnostic.ctypes.data_as(PTR))
            arrays["diagnostic_exch_h"] = diagnostic
        elif wrapper:
            self.wrapper(ptrs, scalars.ctypes.data_as(PTR), integers.ctypes.data_as(IPTR))
        elif native:
            self.native(ptrs, scalars.ctypes.data_as(PTR), integers.ctypes.data_as(IPTR))
        else:
            workspace = np.full(ysu.WORKSPACE_ARRAYS*(flags["kte"]+2), np.nan, np.float32)
            self.host(ptrs, scalars.ctypes.data_as(PTR), integers.ctypes.data_as(IPTR),
                      workspace.ctypes.data_as(PTR), status.ctypes.data_as(IPTR))
        if wrapper:
            for a in ysu._REQUIRED_LEVEL + ysu._OUTPUT_LEVEL + ysu._OPTIONAL_LEVEL:
                if arrays[a] is not None:
                    arrays[a] = arrays[a][:-1]
        return arrays, int(status[0])


def extended_fixtures(folder: Path):
    for case, base, initial in fixtures(folder):
        nz = initial["kte"]
        for topdown in (0, 1):
            for species in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for mode in ("absent", "driver", "urban", "currents"):
                    arrays = {a: None if v is None else v.copy() for a, v in base.items()}
                    flags = {**initial, "ysu_topdown_pblmix": topdown,
                             "f_qc": species[0], "f_qi": species[1]}
                    if mode != "absent":
                        arrays["ctopo"] = np.ones(1, np.float32)
                        arrays["ctopo2"] = np.ones(1, np.float32)
                        arrays["uox"] = np.zeros(1, np.float32)
                        arrays["vox"] = np.zeros(1, np.float32)
                    if mode in ("urban", "currents"):
                        for a in ("dusfc", "dvsfc", "dtsfc", "dqsfc"):
                            arrays[a] = np.zeros(1, np.float32)
                    if mode == "urban":
                        flags["flag_bep"] = 1
                        for a in ysu._OPTIONAL_LEVEL:
                            arrays[a] = np.zeros(nz, np.float32)
                        arrays["sfk"].fill(.8)
                        arrays["vlk"].fill(.9)
                        arrays["frcurb"] = np.array([.3], np.float32)
                        for a in ("a_u", "a_v", "a_t", "a_q"):
                            arrays[a].fill(-.0001)
                        arrays["b_u"].fill(.0003)
                        arrays["b_v"].fill(-.0007)
                        arrays["b_t"].fill(.0005)
                        arrays["b_q"].fill(.00000005)
                    if mode == "currents":
                        arrays["uox"].fill(.6)
                        arrays["vox"].fill(-.25)
                        arrays["ctopo"].fill(1.2)
                        arrays["ctopo2"].fill(.6)
                    # Preserve sentinel inactive-species carriers, matching
                    # the raw scheme's lack of writes when f_qc/f_qi is false.
                    arrays["qctnp"].fill(.123)
                    arrays["qitnp"].fill(-.321)
                    tag = f"column={case},topdown={topdown},species={species},mode={mode}"
                    yield tag, arrays, flags


def tracer_fixtures(folder: Path):
    """Exercise no tracers and the repeated tracer Thomas-solver branch."""
    for case, arrays, flags in fixtures(folder):
        for nmix in (0, 3):
            for topdown in (0, 1):
                state = {a: None if v is None else v.copy() for a, v in arrays.items()}
                state["qmix"] = np.concatenate([arrays["qmix"] * np.float32(n+1)
                                                for n in range(nmix)]) if nmix else np.empty(0, np.float32)
                state["qmixtnp"] = np.zeros_like(state["qmix"])
                yield f"column={case},tracers={nmix},topdown={topdown}", state, {
                    **flags, "nmix": nmix, "ysu_topdown_pblmix": topdown}


def branch_fixtures(folder: Path):
    for case, base, flags in fixtures(folder):
        nz = flags["kte"]
        for mode in ("negative-shear", "zero-shear", "deep-cloud", "neutral-top",
                     "stable-small-wind", "positive-radiation", "thin-inversion", "surface-br",
                     "shallow-topdown", "rounded-thermal", "zero-entrainment"):
            arrays = {a: None if v is None else v.copy() for a, v in base.items()}
            if mode == "negative-shear":
                for a in ("ux", "vx", "u10", "v10"):
                    arrays[a] *= np.float32(-1)
            elif mode == "zero-shear":
                arrays["ux"].fill(6)
                arrays["vx"].fill(-1.5)
            elif mode == "deep-cloud":
                arrays["qcx"][:nz//2] = np.float32(.002)
            elif mode == "neutral-top":
                theta = arrays["tx"][0]/arrays["pi2d"][0]
                arrays["tx"] = (theta + np.linspace(0, .2, nz, dtype=np.float32))*arrays["pi2d"]
                arrays["qvx"].fill(.001)
                arrays["qcx"].fill(0)
                arrays["qix"].fill(0)
            elif mode == "stable-small-wind":
                arrays["hfx"].fill(-100)
                arrays["qfx"].fill(0)
                arrays["br"].fill(1)
                arrays["ux"].fill(.1)
                arrays["vx"].fill(.1)
                arrays["wspd"].fill(.15)
                arrays["u10"].fill(.1)
                arrays["v10"].fill(.1)
            elif mode == "thin-inversion":
                arrays["dz8w2d"].fill(20)
                theta = np.float32(300) + np.arange(nz, dtype=np.float32)*np.float32(.1)
                theta[8:] += np.float32(10)
                arrays["tx"] = theta*arrays["pi2d"]
                arrays["qvx"].fill(.001)
                arrays["qcx"].fill(0)
                arrays["qix"].fill(0)
                arrays["ust"].fill(5)
                arrays["hfx"].fill(250)
                arrays["br"].fill(-.35)
            elif mode == "rounded-thermal":
                # Positive flux rounds away in the thermal enhancement.
                # brdn=brcr=0 reaches native's equality branch at line 758.
                arrays["pi2d"].fill(1)
                arrays["tx"].fill(310)
                arrays["tx"][:2] = np.float32(300)
                for a in ("qvx", "qcx", "qix", "qfx", "vx", "v10", "br"):
                    arrays[a].fill(0)
                arrays["hfx"].fill(1.e-10)
                arrays["dz8w2d"].fill(100)
                arrays["ust"].fill(.3)
                for a in ("ux", "u10", "wspd"):
                    arrays[a].fill(6)
            elif mode == "zero-entrainment":
                # Underflow of bfxpbl reaches cloud we>=0 while all output
                # words stay finite. This probes gradual underflow as well.
                arrays["pi2d"].fill(1)
                arrays["tx"] = np.float32(300) + np.arange(nz, dtype=np.float32)*np.float32(.1)
                arrays["tx"][4:] += np.float32(10)
                arrays["qvx"].fill(.001)
                arrays["qcx"].fill(0)
                arrays["qcx"][1:4] = np.float32(.002)
                for a in ("qix", "qfx", "rthraten"):
                    arrays[a].fill(0)
                arrays["hfx"].fill(-100)
                arrays["br"].fill(.5)
                arrays["ust"].fill(1.e-9)
                arrays["dz8w2d"].fill(1.e20)
            elif mode == "shallow-topdown":
                # First-guess PBL is retired within the first layer. The
                # theta-li loop reactivates it at k=1, then retires it again
                # at native lines 766-767 after the strong inversion at k=2.
                arrays["dz8w2d"].fill(100)
                theta = np.float32(330) + np.arange(nz, dtype=np.float32)*np.float32(.1)
                theta[0] = np.float32(300)
                arrays["tx"] = theta*arrays["pi2d"]
                for a in ("qvx", "qcx", "qix", "hfx", "qfx", "vx", "v10"):
                    arrays[a].fill(0)
                for a in ("ux", "u10", "wspd"):
                    arrays[a].fill(1)
                arrays["hfx"].fill(250)
                arrays["ust"].fill(.1)
                arrays["br"].fill(-.35)
            elif mode == "surface-br":
                # Positive bulk Richardson number despite positive heat flux
                # exercises brdn >= brcr before pblflg is retired.
                arrays["hfx"].fill(250)
                arrays["br"].fill(.5)
            elif mode == "positive-radiation":
                arrays["rthraten"] = abs(arrays["rthraten"])
            yield f"column={case},branch={mode}", arrays, {**flags, "ysu_topdown_pblmix": 1}
        # Every presence term in native's BEP conjunction can be false with
        # the remaining terms present. The scheme defines this as BEP off.
        for missing in ysu._OPTIONAL_LEVEL + ("frcurb",):
            arrays = {a: None if v is None else v.copy() for a, v in base.items()}
            for a in ysu._OPTIONAL_LEVEL:
                arrays[a] = np.zeros(nz, np.float32)
            arrays["sfk"].fill(1)
            arrays["vlk"].fill(1)
            arrays["frcurb"] = np.zeros(1, np.float32)
            arrays[missing] = None
            yield f"column={case},missing={missing}", arrays, {**flags, "flag_bep": 1}


def wrapper_fixtures(folder: Path):
    """Only defined wrapper inputs, including each partial-BEP presence term."""
    for tag, arrays, flags in extended_fixtures(folder):
        if flags["f_qc"] and flags["f_qi"] and 'mode=absent' not in tag:
            yield tag, arrays, flags
    for tag, arrays, flags in branch_fixtures(folder):
        if "missing=" not in tag:
            continue
        for name in ("ctopo", "ctopo2"):
            arrays[name] = np.ones(1, np.float32)
        for name in ("uox", "vox"):
            arrays[name] = np.zeros(1, np.float32)
        # Native always forwards every local BEP argument. With partial
        # caller inputs, those locals are undefined, so keep flag_bep false.
        yield tag, arrays, {**flags, "flag_bep": 0}
