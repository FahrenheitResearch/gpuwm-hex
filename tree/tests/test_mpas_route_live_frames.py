"""The MPAS-physics route publishes each history frame as it closes (parity run group 4, #15).

Before: the route marked every frame durable in one batch after the forecast, so the live first look rendered nothing
until the end (f001 55.6 s late on the 120 km globe, 194 s on the 80 -> 10 km zoom globe).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.first_look import DURABLE
from hexcore.mpasphys.streams_output_v842 import PhysicsStreams, _one_frame_per_file


def _stream(name, output="0_01:00:00", bucket="output_interval"):
    options = {"type": "output", "output_interval": output}
    if bucket is not None:
        options["filename_interval"] = bucket
    return SimpleNamespace(name=name, options=options)


def test_a_history_file_per_output_time_is_one_frame_per_file():
    assert _one_frame_per_file(_stream("output"))
    assert _one_frame_per_file(_stream("output", bucket=None))
    assert _one_frame_per_file(_stream("output", bucket="0_01:00:00"))
    assert not _one_frame_per_file(_stream("output", bucket="none"))
    assert not _one_frame_per_file(_stream("output", bucket="1_00:00:00"))


@pytest.mark.skipif(os.name != "posix", reason="the durable marker fsyncs a read-only handle (POSIX)")
def test_each_closed_frame_is_durable_and_listed_at_once(tmp_path):
    frame = tmp_path / "history.2025-05-19_13.00.00.nc"
    diag = tmp_path / "diag.2025-05-19_13.00.00.nc"
    frame.write_bytes(b"frame")
    diag.write_bytes(b"diag")
    writer = SimpleNamespace(output=tmp_path)
    jobs = [(_stream("output"), frame, 0), (_stream("diagnostics"), diag, 0)]
    PhysicsStreams._publish_closed(writer, jobs, 6)
    sidecar = json.loads((tmp_path / (frame.name + ".durable.json")).read_text())
    assert sidecar["bytes"] == frame.stat().st_size
    assert not (tmp_path / (diag.name + ".durable.json")).exists()
    rows = [json.loads(line) for line in (tmp_path / "history-ready.jsonl").read_text().splitlines()]
    assert rows == [{"step": 6, "path": str(frame)}]
    assert [p.name for p in tmp_path.glob(DURABLE)] == [frame.name + ".durable.json"]


def test_an_appended_file_waits_for_the_end(tmp_path):
    frame = tmp_path / "history.2025-05-19_12.00.00.nc"
    frame.write_bytes(b"two records")
    writer = SimpleNamespace(output=tmp_path)
    PhysicsStreams._publish_closed(writer, [(_stream("output", bucket="none"), frame, 0)], 6)
    PhysicsStreams._publish_closed(writer, [(_stream("output"), frame, 1)], 12)
    assert not (tmp_path / "history-ready.jsonl").exists()


def test_a_single_precision_static_on_the_earth_radius_identifies_its_unit_sphere_grid():
    # Parity run group 4: a region cut from the x1.40962 global row (grid on the unit sphere, native static in f32 on
    # the Earth radius) was refused at init: the static's areas are f32(area) * f32(r**2), not f32(area * r**2).
    from hexcore.moving_checkpoint import identify_source_geometry
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(500, 3))
    centers /= np.linalg.norm(centers, axis=-1, keepdims=True)
    area = rng.uniform(2e-4, 4e-4, 500)
    r = 6371229.0
    native = area.astype(np.float32) * np.float32(r * r)
    assert (native != (area * r * r).astype(np.float32)).any()
    mesh = SimpleNamespace(attrs={"sphere_radius": r}, arrays={"xCell": centers[:, 0] * r, "yCell": centers[:, 1] * r,
                                                               "zCell": centers[:, 2] * r, "areaCell": native})
    source = SimpleNamespace(radius_m=1.0, area=area, centers=centers)
    assert identify_source_geometry(mesh, source, r) == (r, r * r)
    mesh.arrays["areaCell"] = native * np.float32(1.001)
    with pytest.raises(ValueError, match="does not identify"):
        identify_source_geometry(mesh, source, r)
