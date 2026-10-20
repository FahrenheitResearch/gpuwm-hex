"""Observe full OML driver transactions during a real forecast, without maths.

The strict native driver runs on the same incoming pools; its changed storage
is restored before the normal CUDA callback. libc compares bytes. This witness
isolates the OML transaction from upstream drift, and never establishes whole
forecast acceptance. Enable only in an explicitly instrumented diagnostic run.
"""
from __future__ import annotations

import ctypes
from collections import ChainMap
import hashlib
import json
import os
from pathlib import Path
import socket


def install(receipt: Path):
    from hexcore.mpasphys import oml_v842 as oml

    original = oml.launch_slot
    implementation_sha = hashlib.sha256(Path(oml.__file__).read_bytes()).hexdigest()
    witness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    correct_omega = os.environ.get("MPAS_OML_DIAGNOSTIC_NATIVE_OMEGA") == "1"
    libc = ctypes.CDLL(None)
    compare = libc.memcmp
    compare.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    compare.restype = ctypes.c_int
    fields = ([("diag_physics", name) for name in oml.STATE_FIELDS if name != "skintemp"]
              + [("sfc_input", name) for name in ("skintemp", "sst", "xland")]
              + [("diag", name) for name in ("uReconstructZonal", "uReconstructMeridional")]
              + [("mesh", "latCell")])
    calls = []

    def witnessed(call):
        provider = call.pools
        native = provider.native
        if provider.buffers:
            raise ValueError("OML diagnostic must begin before any device field is packed")
        before = {key: native.get_field(*key).copy(order="F") for key in fields}
        supplied_omega = provider["constants"]["omega"]
        native_omega = native.get_constant("omega")
        if correct_omega:
            # Explicit diagnostic of the proposed table fix. The published
            # table is never edited here, and this run cannot be acceptance.
            provider.values["constants"] = ChainMap({"omega": native_omega}, provider["constants"])
        # The literal pinned MPAS whole driver owns its native dt_pbl/configs.
        provider.session.hooks.invoke("driver_oml1d", call)
        expected = {key: native.get_field(*key).copy(order="F") for key in fields}
        for key, value in before.items():
            native.put_field(*key, value, nlevels=native._fields[key][2])
        original(call)
        # Same transaction implementation as the real backend's outer wrapper.
        # Its following commit is empty; no state is recomputed or replaced.
        provider.commit()
        rows = []
        for key, reference in expected.items():
            actual = native.get_field(*key).copy(order="F")
            if actual.dtype != reference.dtype or actual.shape != reference.shape:
                raise ValueError("Native/CUDA live field layout changed")
            equal = compare(reference.ctypes.data, actual.ctypes.data, actual.nbytes) == 0
            rows.append(dict(pool=key[0], field=key[1], words=int(actual.size),
                             byte_identical=equal, differing_words=0 if equal else None,
                             expected_sha256=hashlib.sha256(reference.tobytes(order="F")).hexdigest(),
                             actual_sha256=hashlib.sha256(actual.tobytes(order="F")).hexdigest()))
        calls.append(dict(itimestep=call.itimestep, time_seconds=str(call.time_seconds),
                          dt_pbl=str(call.dt), fields=rows,
                          supplied_omega_bytes=supplied_omega.tobytes().hex(),
                          native_omega_bytes=native_omega.tobytes().hex()))
        result = dict(schema="p19-oml-live-native-driver-witness/v1",
                      forecast_acceptance=False, instrumented_diagnostic=True,
                      host=socket.gethostname(), process_count=1,
                      implementation_sha256=implementation_sha,
                      witness_sha256=witness_sha,
                      diagnostic_binding_override="compiled framework omega" if correct_omega else None,
                      comparison="libc memcmp, all bytes including IEEE payloads",
                      native_authority="MPAS mpas_atmphys_driver_oml.F:41-148, full pinned native entry",
                      calls=calls, all_byte_identical=all(f['byte_identical'] for c in calls for f in c['fields']))
        receipt.write_text(json.dumps(result, indent=2) + "\n")

    oml.launch_slot = witnessed
