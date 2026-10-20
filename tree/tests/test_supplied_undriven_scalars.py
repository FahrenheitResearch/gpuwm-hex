"""The receipt's supplied-but-undriven species list (rrfs-h-h4a finding).

``select_lbc_scalar_names`` returns the stream's ``lbc_``-prefixed names, and
the undriven list compared the bare species name against them, so every
driven mass (lbc_qv ... lbc_qg) was recorded as undriven beside the aerosol
pair.  The list must hold only the species the stream carries and the row
does not drive.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gpuwm.netcdf_bridge")


def test_driven_masses_are_not_listed_as_undriven(tmp_path):
    from netCDF4 import Dataset  # test oracle writer only

    from hexcore.drivers.run_cuda_v841_forecast import supplied_undriven_scalars

    path = tmp_path / "lbc.2026-10-04_12.00.00.nc"
    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("nCells", 3)
        for name in ("lbc_qv", "lbc_qc", "lbc_qr", "lbc_nwfa", "lbc_nifa", "lbc_theta"):
            dataset.createVariable(name, "f4", ("Time", "nCells"))[:] = np.zeros((1, 3))
    scalar_names = ("qv", "qc", "qr", "qi", "nwfa", "nifa")
    driven = ("lbc_qv", "lbc_qc", "lbc_qr")
    assert supplied_undriven_scalars(path, scalar_names, driven) == ["lbc_nifa", "lbc_nwfa"]
    # A species the stream does not carry (qi) is neither driven nor supplied.
    assert "lbc_qi" not in supplied_undriven_scalars(path, scalar_names, driven)
