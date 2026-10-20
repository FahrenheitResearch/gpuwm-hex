"""Full OML SlotCall replay using constants copied from the strict native ABI.

The literal standalone driver is the numerical oracle. The real table and
NativeDevicePoolViews constant mapping are exercised, so a mistaken binding
cannot be hidden by supplying the same mistaken value to both implementations.
Run inside the GPU queue with MPAS_OML_GPU=1, MPAS_OML_ORACLE and
MPAS_OML_NATIVE_LIBRARY; this column evidence is never forecast acceptance.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys import oml_v842 as oml
from hexcore.mpasphys.device_pools_v842 import _ConstantViews
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.physics_suites import load_table
from test_mpas_oml_v842 import FIELDS, columns


def test_oml_descriptor_keeps_framework_omega():
    """mpas_atmphys_driver_oml.F:131 uses mpas_constants.F:45, not eomeg."""
    descriptor = load_table()["mpasphys_v842"]["slots"]["oml"]["cuda_entries"]["oml1d"]
    assert descriptor.get("constant_aliases", {}).get("omega", "omega") == "omega"


@pytest.mark.parametrize("binding", ("published", "framework-diagnostic"))
def test_whole_slot_against_native_compiled_constants(binding):
    paths = {key: os.environ.get(key) for key in ("MPAS_OML_ORACLE", "MPAS_OML_NATIVE_LIBRARY")}
    if os.environ.get("MPAS_OML_GPU") != "1" or not all(paths.values()):
        pytest.skip("needs strict native ABI, standalone oracle and a queued GPU")
    import cupy as cp
    lib = ctypes.CDLL(paths["MPAS_OML_ORACLE"])
    oracle = lib.replay
    oracle.argtypes = ([ctypes.c_int] * 5 +
                       [np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")] +
                       [ctypes.c_float] * 7)
    oracle.restype = None
    descriptor = load_table()["mpasphys_v842"]["slots"]["oml"]["cuda_entries"]["oml1d"]
    rows = []
    with NativePhysics(paths["MPAS_OML_NATIVE_LIBRARY"], test_only=True) as native:
        # Read independently: the referee uses original native names, while
        # SlotCall receives precisely the published constant adapter/aliases.
        compiled = {name: native.get_constant(name) for name in ("stbolt", "gravity", "omega", "eomeg")}
        assert compiled["omega"].view(np.uint32) != compiled["eomeg"].view(np.uint32)
        aliases = descriptor.get("constant_aliases", {}) if binding == "published" else {}
        constants = _ConstantViews(native, aliases)
        for relaxation in (0., 864000.):
            expected = columns()
            actual = cp.asarray(expected)
            fields = {key: actual[i] for i, key in enumerate(FIELDS)}
            winds = {}
            for name, key in (("uReconstructZonal", "u"), ("uReconstructMeridional", "v")):
                winds[name] = cp.full((3, actual.shape[1]), np.float32(10000.))
                winds[name][0] = fields[key]
            call = SimpleNamespace(options=dict(config_oml1d=True, config_oml_gamma=.14,
                                                config_oml_relaxation_time=relaxation),
                                   dt=1200., pools=dict(constants=constants,
                                   mesh=dict(nCellsSolve=actual.shape[1]-1, latCell=fields["phase"]),
                                   diag=winds, diag_physics=fields, sfc_input=fields))
            for step in range(12):
                oracle(expected.shape[1], expected.shape[1]-1, 1, 1, 0, expected,
                       compiled["stbolt"], compiled["gravity"], call.dt, .14,
                       relaxation, compiled["omega"], 30.)
                oml.launch_slot(call)
                computed = actual.get()
                counts = {key: int(np.count_nonzero(expected[i].view(np.uint32) != computed[i].view(np.uint32)))
                          for i, key in enumerate(FIELDS)}
                rows.append(dict(step=step, relaxation=relaxation, words=int(expected.size),
                                 differing_words=sum(counts.values()), per_array=counts))
        report = dict(schema="p19-oml-native-constant-slot-replay/v1", forecast_acceptance=False,
                      process_count=1, comparisons=rows,
                      differing_words=sum(row["differing_words"] for row in rows),
                      native_constant_words={name: hex(int(value.view(np.uint32))) for name,value in compiled.items()},
                      binding=binding, instrumented_diagnostic=binding != "published",
                      descriptor=dict(module=descriptor["module"], call=descriptor["call"],
                                      constant_aliases=dict(descriptor.get("constant_aliases", {}))),
                      implementation_sha256=hashlib.sha256(Path(oml.__file__).read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(Path(paths["MPAS_OML_NATIVE_LIBRARY"]).read_bytes()).hexdigest())
        if os.environ.get("MPAS_OML_BINDING_RECEIPT"):
            destination = Path(os.environ["MPAS_OML_BINDING_RECEIPT"])
            destination.with_name(destination.stem + "-" + binding + destination.suffix).write_text(
                json.dumps(report, indent=2) + "\n")
        assert report["differing_words"] == 0, rows
