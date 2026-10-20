"""Exercise the real native six-field GFL group with empty MPI1 halo rings.

Transport qualification only: this does not invoke GFL or prove a forecast.
MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_halos.F:180-186;
MPAS-Model-rrfs/src/framework/mpas_dmpar.F:7010-7192.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.rrfs.driver_gfl_bf6a4b0 import HALO_FIELDS


@pytest.mark.parametrize("layers", [None, 2, 3])
def test_actual_native_empty_rings_retain_layer_two(layers):
    artifact = os.environ.get("MPAS_GFL_HALO_LIBRARY")
    if not artifact:
        pytest.skip("requires the published RRFS native backend artifact")
    assert Path(artifact).is_file()
    # Fixture words include signed zero, a subnormal, and a finite mantissa tail.
    words = np.array([0x80000000, 0x00000001, 0x3F800001, 0xBF800001], np.uint32)
    with NativePhysics(artifact, test_only=True, pin_set="rrfs") as native:
        native.block()
        native.dimension("mesh", "nCells", 4)
        native.dimension("mesh", "nCellsSolve", 4)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        options = {} if layers is None else {"halo_ends": [4] * layers}
        native.decomposition("nCells", global_ids=ids, owned=4, **options)
        for field in HALO_FIELDS:
            profile = field.startswith("sub3d_")
            value = np.tile(words, (59, 1)).view(np.float32) if profile else words.view(np.float32)
            native.put_field("diag_physics", field, value)
            native.register_field("diag_physics", field,
                                  dimensions=("nVertLevels", "nCells") if profile else ("nCells",),
                                  decomposed=True)
        if layers is None:
            with pytest.raises(RuntimeError, match="field registration failed: status=5"):
                native.exchange_group("physics:sub3d_cuten", HALO_FIELDS, halo_layers=[2])
            return
        native.exchange_group("physics:sub3d_cuten", HALO_FIELDS, halo_layers=[2])
        native.exchange_halo_group("physics:sub3d_cuten")
        for field in HALO_FIELDS:
            expected = np.tile(words, (59, 1)) if field.startswith("sub3d_") else words
            assert native.get_field("diag_physics", field).tobytes(order="C") == expected.tobytes(order="C")
