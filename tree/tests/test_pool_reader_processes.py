"""rw_mpas_physics_pool split over reader processes: the single reader's files, without a GPU or netCDF."""
import json

import pytest

_FAKE_READER = r'''#!/usr/bin/env python3
"""A stand-in rw_mpas_physics_pool: field-<request index>.bin per numeric field, the manifest in request order."""
import hashlib, json, os, sys
request, out = json.load(open(sys.argv[1])), sys.argv[2]
os.makedirs(out, exist_ok=True)
log = os.environ.get("FAKE_READER_LOG")
if log:
    open(log, "a").write(f"{len(request['fields'])}\n")
records = []
for index, field in enumerate(request["fields"]):
    if field["name"] == "refused":
        sys.stderr.write("required input refused absent\n"); sys.exit(3)
    words = (field["name"] * 7).encode()
    record = {"pool": field["pool"], "name": field["name"], "kind": field["kind"], "dimensions": field["dimensions"],
              "shape": field["shape"], "payload": None, "sha256": None, "text": None, "origins": ["input:" + field["name"]]}
    if field["kind"] == "character":
        record["text"] = field["name"]
    else:
        name = f"field-{index:04}.bin"
        open(os.path.join(out, name), "wb").write(words)
        record.update(payload=name, sha256=hashlib.sha256(words).hexdigest())
    records.append(record)
open(os.path.join(out, "manifest.json"), "w").write(json.dumps({"schema": "mpas-native-pool-payload-v1",
                                                                  "fields": records}, indent=2, ensure_ascii=False))
'''


def _fake_request(tmp_path, names):
    reader = tmp_path / "reader.py"
    reader.write_text(_FAKE_READER)
    reader.chmod(0o755)
    fields = [{"pool": "state", "name": name, "kind": "character" if name.startswith("text") else "real",
               "dimensions": ["nCells"], "shape": [4 + 3 * (index % 5)]} for index, name in enumerate(names)]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema": "mpas-native-pool-request-v1", "sources": [], "fields": fields}))
    return reader, request


def test_pool_reader_split_over_processes_writes_the_single_readers_files(tmp_path, monkeypatch):
    """THE COST THIS REMOVES: rw_mpas_physics_pool serialises every netCDF-C read in one process, so the decode ran
    on one core (60 km: 7.4 s at 16 threads, 8.6 s at 1).  Split over reader processes, every payload file and the
    manifest are the single reader's bytes, by the single reader's names."""
    import sys
    from hexcore.mpasphys.pool_v842 import run_pool_reader
    names = [f"f{index}" for index in range(23)] + ["text0", "text1"]
    reader, request = _fake_request(tmp_path, names)
    command = tmp_path / "reader.sh"
    command.write_text(f"#!/bin/sh\nexec {sys.executable} {reader} \"$@\"\n")
    command.chmod(0o755)
    monkeypatch.setenv("GPUWM_HEX_POOL_READER_PROCESSES", "1")
    run_pool_reader(command, request, tmp_path / "single", 4)
    log = tmp_path / "parts.log"
    monkeypatch.setenv("FAKE_READER_LOG", str(log))
    monkeypatch.setenv("GPUWM_HEX_POOL_READER_PROCESSES", "4")
    run_pool_reader(command, request, tmp_path / "split", 4)
    assert len(log.read_text().split()) == 4
    single = sorted(p.name for p in (tmp_path / "single").iterdir())
    assert single == sorted(p.name for p in (tmp_path / "split").iterdir())
    for name in single:
        assert (tmp_path / "single" / name).read_bytes() == (tmp_path / "split" / name).read_bytes(), name


def test_a_refusing_part_reports_the_single_readers_refusal(tmp_path, monkeypatch):
    import subprocess
    import sys
    from hexcore.mpasphys.pool_v842 import run_pool_reader
    reader, request = _fake_request(tmp_path, ["a", "b", "refused", "c", "d", "e"])
    command = tmp_path / "reader.sh"
    command.write_text(f"#!/bin/sh\nexec {sys.executable} {reader} \"$@\"\n")
    command.chmod(0o755)
    monkeypatch.setenv("GPUWM_HEX_POOL_READER_PROCESSES", "3")
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_pool_reader(command, request, tmp_path / "out", 3)
    assert raised.value.cmd == [str(command), str(request), str(tmp_path / "out")]
    assert not list((tmp_path / "out").glob(".reader-part-*"))
