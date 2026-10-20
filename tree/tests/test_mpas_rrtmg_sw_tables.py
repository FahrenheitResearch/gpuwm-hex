"""Rust production table input against pristine MPAS's own table reader."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from mpas_rrtmg_sw_oracle.replay import read_swd
from mpas_rrtmg_sw_oracle.table_failures import replay_failures


def test_every_raw_coefficient_word_against_native_reader(tmp_path):
    bridge = os.environ.get("MPAS_RRTMG_SW_TABLES_BRIDGE")
    oracle = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not bridge or not oracle:
        pytest.skip("Rust table bridge and strict native SW oracle are required")
    source = Path(oracle)/"RRTMG_SW_DATA"
    destination = tmp_path/"decoded"
    subprocess.run([bridge, str(source), str(destination)], check=True)
    native = read_swd(Path(oracle)/"tables.swd")
    manifest = json.loads((destination/"manifest.json").read_text())
    assert manifest["schema"] == "mpas-rrtmg-sw-raw-v842/1"
    assert manifest["order"] == "F"
    rows = []
    for field in manifest["fields"]:
        got = np.frombuffer((destination/field["file"]).read_bytes(), dtype=field["dtype"])
        ref = native[field["name"]].ravel(order="F")
        assert list(native[field["name"]].shape) == field["shape"]
        differing = int(np.count_nonzero(got.view(np.uint32) != ref.view(np.uint32)))
        rows.append(dict(name=field["name"], words=int(ref.size), differing_words=differing))
    receipt = dict(authority="Pristine MPAS mpas_atmphys_rrtmg_swinit.F table reader",
                   input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   executable_sha256=hashlib.sha256(Path(bridge).read_bytes()).hexdigest(),
                   arrays=rows, differing_words=sum(row["differing_words"] for row in rows),
                   words=sum(row["words"] for row in rows), whole_model_proof=False)
    if path := os.environ.get("MPAS_RRTMG_SW_TABLE_RECEIPT"):
        Path(path).write_text(json.dumps(receipt, indent=2)+"\n")
    assert receipt["differing_words"] == 0


def test_every_native_record_eof_refuses_before_publication(tmp_path):
    bridge = os.environ.get("MPAS_RRTMG_SW_TABLES_BRIDGE")
    oracle = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not bridge or not oracle:
        pytest.skip("Rust table bridge and strict native SW oracle are required")
    receipt = replay_failures(oracle, bridge, tmp_path / "native-eof")
    (tmp_path / "native-table-failures.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert len(receipt["rows"]) == 14
