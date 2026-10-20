"""hexcore.init_gpu on a card: both kernel files compile from the installed package and the batch runs.

THE BREAKAGE THIS PREVENTS: the init kernels are .cu files beside the module, not Python strings, so a wheel that
left them out, or an engine whose float maths headers moved, would fail only on the first member init of a cycle.
This compiles horizontal.cu and physics.cu (with the engine's binary32 libm headers) the way the module does, and
runs the horizontal batch on a tiny synthetic member.  The all-field identity against ``rw_mpas_init`` is the
per-case gate named in ``hexcore.init_gpu``; this file does not claim it.
"""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from hexcore import init_gpu  # noqa: E402
from hexcore.init_gpu import physics  # noqa: E402


def _record(stream, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)) + payload + struct.pack(">I", len(payload)))


def _met(path: Path, slabs) -> Path:
    with path.open("wb") as stream:
        for field, level, values in slabs:
            ny, nx = values.shape
            _record(stream, struct.pack(">i", 5))
            header = bytearray(156)
            header[60:69] = field.ljust(9).encode()
            header[140:] = struct.pack(">fiii", level, nx, ny, 0)
            _record(stream, bytes(header))
            # a global 1-degree latitude/longitude grid starting at the south pole
            _record(stream, b"\0" * 8 + struct.pack(">5f", -90.0, 0.0, 1.0, 1.0, 6371.0))
            _record(stream, struct.pack(">i", 0))
            _record(stream, values.astype(">f4").tobytes())
    return path


def test_both_kernel_files_compile_with_strict_arithmetic():
    kernels = init_gpu.Kernels()
    assert kernels.module() is not None
    assert "--fmad=false" in kernels.options and "--ftz=false" in kernels.options
    kernel_dir = physics.engine_kernel_root() / "gpuwm" / "core" / "kernels"
    headers = [(kernel_dir / name).read_text() for name in ("glibc_flt32.cuh", "glibc_trig_flt32.cuh")]
    source = Path(physics.__file__).with_name("physics.cu").read_text()
    module = kernels.compile("\n".join(headers) + "\n" + source)
    for name in ("vertical_cells", "vertical_edges", "dynamics_cells", "w_cells", "soil_cells"):
        assert module.get_function(name) is not None


def test_a_uniform_member_interpolates_to_its_own_values(tmp_path):
    ny, nx = 181, 360
    land = np.ones((ny, nx), "f4")
    slabs = [("LANDSEA", 200100.0, land)]
    for level, t in ((200100.0, 290.0), (85000.0, 280.0)):
        for field, value in (("TT", t), ("RH", 50.0), ("GHT", 1500.0 if level != 200100.0 else 0.0),
                             ("UU", 5.0), ("VV", -3.0)):
            slabs.append((field, level, np.full((ny, nx), value, "f4")))
    for field, value in (("PSFC", 100000.0), ("PMSL", 101000.0), ("SOILHGT", 0.0), ("SKINTEMP", 291.0),
                         ("SNOW", 0.0), ("SEAICE", 0.0)):
        slabs.append((field, 200100.0, np.full((ny, nx), value, "f4")))
    for name in ("ST000010", "ST010040", "ST040100", "ST100200"):
        slabs.append((name, 200100.0, np.full((ny, nx), 285.0, "f4")))
    for name in ("SM000010", "SM010040", "SM040100", "SM100200"):
        slabs.append((name, 200100.0, np.full((ny, nx), 0.25, "f4")))
    member = init_gpu.read_met(_met(tmp_path / "MET", slabs))
    rng = np.random.default_rng(7)
    nc, ne = 64, 96
    mesh = {"latCell": rng.uniform(-1.4, 1.4, nc).astype("f4"), "lonCell": rng.uniform(0, 6.28, nc).astype("f4"),
            "latEdge": rng.uniform(-1.4, 1.4, ne).astype("f4"), "lonEdge": rng.uniform(0, 6.28, ne).astype("f4"),
            "landmask": np.ones(nc, "i4")}
    kernels = init_gpu.Kernels()
    result, receipt = init_gpu.horizontal_batch(kernels, [member, member], mesh, nfg=2, nsoil=4)
    t = cp.asnumpy(result["t"])
    assert t.shape == (2, nc, 2) and np.array_equal(t[0], t[1]), "two identical members give identical fields"
    assert np.allclose(t[0, :, 0], 290.0, rtol=0, atol=1e-3) and np.allclose(t[0, :, 1], 280.0, rtol=0, atol=1e-3)
    assert np.allclose(cp.asnumpy(result["u"])[0, :, 1], 5.0, rtol=0, atol=1e-4)
    assert np.allclose(cp.asnumpy(result["st"])[0], 285.0, rtol=0, atol=1e-3)
    assert np.allclose(cp.asnumpy(result["sm"])[0], 0.25, rtol=0, atol=1e-5)
    assert receipt["members"] == 2
