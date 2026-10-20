"""hexcore.init_gpu on the host: the WPS v5 reader, the field rules, the month weights and the member batching.

The kernels' arithmetic is proven per case against ``rw_mpas_init`` on a card (see ``hexcore.init_gpu``); these
tests hold the host half, which decides WHICH slab lands in WHICH member, field and level before any kernel runs.
A slab read at the wrong level, or a member's slabs filed under another member, would be a wrong init with no
error, so each of those is pinned here.  No device: the kernel stand-in returns identifiable values.
"""
from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import init_gpu
from hexcore.init_gpu import physics


def _record(stream, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)) + payload + struct.pack(">I", len(payload)))


def _slab(stream, field: str, level: float, values: np.ndarray, *, version: int = 5, proj: int = 0) -> None:
    ny, nx = values.shape
    _record(stream, struct.pack(">i", version))
    header = bytearray(156)
    header[60:69] = field.ljust(9).encode()
    header[140:] = struct.pack(">fiii", level, nx, ny, proj)
    _record(stream, bytes(header))
    _record(stream, b"\0" * 8 + struct.pack(">5f", -90.0, 0.0, 1.0, 1.0, 6371.0))
    _record(stream, struct.pack(">i", 0))
    _record(stream, values.astype(">f4").tobytes())


def _met(path: Path, slabs) -> Path:
    with path.open("wb") as stream:
        for field, level, values in slabs:
            _slab(stream, field, level, values)
    return path


def test_read_met_keeps_record_order_and_renames_hgt(tmp_path):
    a = np.arange(12, dtype="f4").reshape(3, 4)
    path = _met(tmp_path / "MET", [("TT", 200100.0, a), ("HGT", 85000.0, a + 1), ("LANDSEA", 200100.0, a * 0)])
    slabs = init_gpu.read_met(path)
    assert [s.field for s in slabs] == ["TT", "GHT", "LANDSEA"]
    assert slabs[0].level == np.float32(200100.0) and (slabs[0].nx, slabs[0].ny) == (4, 3)
    assert slabs[0].projection == (-90.0, 0.0, 1.0, 1.0)
    assert np.array_equal(slabs[1].values, (a + 1).ravel()) and slabs[1].values.dtype == np.float32


@pytest.mark.parametrize("damage, match", [
    ("version", "only WPS v5"),
    ("projection", "unsupported source geometry"),
    ("truncated", "truncated WPS record marker|invalid WPS record framing"),
    ("empty", "empty intermediate"),
])
def test_read_met_refuses_what_it_cannot_read_exactly(tmp_path, damage, match):
    path = tmp_path / "MET"
    values = np.zeros((3, 4), dtype="f4")
    with path.open("wb") as stream:
        if damage == "version":
            _slab(stream, "TT", 200100.0, values, version=4)
        elif damage == "projection":
            _slab(stream, "TT", 200100.0, values, proj=1)
        elif damage in ("truncated",):
            _slab(stream, "TT", 200100.0, values)
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-6])
    with pytest.raises(ValueError, match=match):
        init_gpu.read_met(path)


def test_field_rules_map_every_wps_name_the_cpu_init_reads():
    assert init_gpu.rule("TT")[0] == "t" and init_gpu.rule("PRESSURE")[0] == "p"
    assert init_gpu.rule("SEAICE")[:4] == ("xice", 0x432, 1.0, 1)
    assert init_gpu.rule("SNOW")[1] == 0x32
    target, method, mask_value, flag, fill, soil = init_gpu.rule("ST010200")
    assert (target, method, fill, soil) == ("st", 0x4321, 285.0, (1, 190.0))
    assert init_gpu.rule("SM100200")[5] == (3, 100.0) and init_gpu.rule("SOILM054")[5] == (4, 54.0)
    assert init_gpu.rule("LANDSEA") is None and init_gpu.rule("NOTAFIELD") is None


def test_month_weights_follow_the_mid_month_rule():
    # 13 June lies 29 days past mid-May and 2 days before mid-June (31 days apart).
    m1, m2, w1, w2, span = physics.monthly_weights("2022-06-13_21:00:00")
    assert (int(m1), int(m2)) == (4, 5) and (float(w1), float(w2), float(span)) == (2.0, 29.0, 31.0)
    # early January wraps to December
    m1, m2, *_ = physics.monthly_weights("2023-01-03_00:00:00")
    assert (int(m1), int(m2)) == (11, 0)


def test_the_kernel_sources_ship_beside_the_module():
    here = Path(init_gpu.__file__).parent
    assert (here / "horizontal.cu").is_file() and (here / "physics.cu").is_file()


def test_physics_headers_default_to_the_installed_engine():
    root = physics.engine_kernel_root()
    assert (root / "gpuwm" / "core" / "kernels" / "glibc_flt32.cuh").is_file()
    assert (root / "gpuwm" / "core" / "kernels" / "glibc_trig_flt32.cuh").is_file()


# ---------------------------------------------------------------- the member batch, with a kernel stand-in
class _FakeKernels:
    """``Kernels`` with numpy in place of cupy: ``sample`` returns, for slab i, the value 1000*i + its first value."""

    def __init__(self):
        self.cp = SimpleNamespace(zeros=np.zeros, asarray=np.asarray, cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
        self.calls = []

    def sample(self, values, offsets, meta, proj, lat, lon, land):
        self.calls.append(len(offsets))
        out = np.empty((len(offsets), lat.size), dtype="f4")
        for i, (offset, _mask) in enumerate(offsets):
            out[i] = values[offset]
        return out, {"search_points": 0, "search_batches": []}


def _member(tmp_path, tag: str, base: float, levels=(200100.0, 85000.0)):
    slabs = [("LANDSEA", 200100.0, np.ones((2, 3), "f4"))]
    for k, level in enumerate(levels):
        for field in ("TT", "RH", "GHT", "UU", "VV"):
            slabs.append((field, level, np.full((2, 3), base + 10 * k, "f4")))
    for field in ("PSFC", "PMSL", "SOILHGT", "SKINTEMP", "SNOW", "SEAICE"):
        slabs.append((field, 200100.0, np.full((2, 3), base, "f4")))
    for name in ("ST000010", "ST010040", "ST040100", "ST100200", "SM000010", "SM010040", "SM040100", "SM100200"):
        slabs.append((name, 200100.0, np.full((2, 3), base, "f4")))
    return init_gpu.read_met(_met(tmp_path / f"MET-{tag}", slabs))


def _mesh(nc=5, ne=7):
    return {"latCell": np.zeros(nc, "f4"), "lonCell": np.zeros(nc, "f4"), "latEdge": np.zeros(ne, "f4"),
            "lonEdge": np.zeros(ne, "f4"), "landmask": np.ones(nc, "i4")}


def test_each_slab_lands_in_its_own_member_field_and_level(tmp_path):
    members = [_member(tmp_path, "a", 1.0), _member(tmp_path, "b", 2.0)]
    kernels = _FakeKernels()
    result, receipt = init_gpu.horizontal_batch(kernels, members, _mesh(), nfg=2, nsoil=4)
    assert result["t"].shape == (2, 5, 2) and result["u"].shape == (2, 7, 2) and result["st"].shape == (2, 5, 4)
    for b, base in enumerate((1.0, 2.0)):
        assert np.all(result["t"][b, :, 0] == base) and np.all(result["t"][b, :, 1] == base + 10)
        assert np.all(result["u"][b, :, 1] == base + 10), "upper winds go on edges"
        assert np.all(result["u10"][b] == base), "surface winds are the 10 m wind on cells"
        assert np.all(result["st"][b] == base) and np.all(result["psfc"][b] == base)
    assert result["vert_level"].tolist() == [200100.0, 85000.0]
    assert result["dzs_cm"].tolist() == [[10.0, 30.0, 60.0, 100.0]] * 2
    assert receipt["members"] == 2 and sorted(g["edge"] for g in receipt["groups"]) == [False, True]


def test_members_with_a_different_level_order_are_refused(tmp_path):
    members = [_member(tmp_path, "a", 1.0), _member(tmp_path, "b", 2.0, levels=(200100.0, 70000.0))]
    with pytest.raises(ValueError, match="member level order differs"):
        init_gpu.horizontal_batch(_FakeKernels(), members, _mesh(), nfg=2, nsoil=4)


def test_more_source_levels_than_declared_are_refused(tmp_path):
    with pytest.raises(ValueError, match="more source levels than declared"):
        init_gpu.horizontal_batch(_FakeKernels(), [_member(tmp_path, "a", 1.0)], _mesh(), nfg=1, nsoil=4)
