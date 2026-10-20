"""Word-exact replay of strict pristine MPAS SW column oracles.

Build fixtures with mpas_rrtmg_sw_oracle/build.py and fixtures.py, then run
through the GPU and build queues with MPAS_RRTMG_SW_ORACLE pointing at them.
Whole forecast acceptance remains separate from these diagnostic tests.
"""
from pathlib import Path
import os
import pytest
from mpas_rrtmg_sw_oracle.replay import read_swd, replay, replay_aerosols

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def oracle():
    path = os.environ.get("MPAS_RRTMG_SW_ORACLE")
    if not path:
        pytest.skip("MPAS_RRTMG_SW_ORACLE must name the strict native fixture directory")
    directory = Path(path)
    required = ("tables.swd", "columns.swd", "solar.bin", "driver_times.bin", "cloud.bin", "mcica.bin", "composition.bin", "composition-extreme.bin", "composition-dense.bin", "earth_sun.bin", "reftra.bin", "driver.swd", "aerosol_cases.json")
    missing = [name for name in required if not (directory / name).is_file()]
    assert not missing, f"Native fixture files missing: {missing}"
    return directory


def test_native_sw_tables_solar_cloud_wrapper_batches_and_streams(oracle, tmp_path):
    assert replay(oracle / "tables.swd", oracle / "columns.swd", tmp_path / "sw.json") == 0


def test_native_aerosol_options_and_thompson_driver(oracle, tmp_path):
    import cupy as cp
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    dump = read_swd(oracle / "tables.swd")
    engine = MPASShortwave(raw_coefficients={k: cp.asarray(v) for k,v in dump.items() if k.startswith("kg") and v.ndim})
    assert replay_aerosols(engine, oracle, tmp_path / "aerosol.json") == 0


def test_native_binary_tables_through_rust_and_cuda(oracle, tmp_path):
    bridge = os.environ.get("MPAS_RRTMG_SW_TABLES_BRIDGE")
    if not bridge:
        pytest.skip("Production Rust table bridge is required")
    import hashlib
    import json
    import numpy as np
    from gpuwm.core.rrtmg_sw import tables_from_dump, _pack_cuda_tables
    from hexcore.mpasphys.rrtmg_sw_v842 import MPASShortwave
    engine = MPASShortwave.from_mpas_data(oracle/"RRTMG_SW_DATA", decoder=bridge,
                                        decoded_directory=tmp_path/"decoded-tables")
    expected, _ = _pack_cuda_tables(tables_from_dump(read_swd(oracle/"tables.swd")))
    actual = engine.tab_gpu.get()
    differing = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
    receipt = dict(words=int(expected.size), differing_words=differing,
                   input_sha256=hashlib.sha256((oracle/"RRTMG_SW_DATA").read_bytes()).hexdigest(),
                   decoder_sha256=hashlib.sha256(Path(bridge).read_bytes()).hexdigest(),
                   whole_model_proof=False)
    (tmp_path/"production-initialization.json").write_text(json.dumps(receipt, indent=2)+"\n")
    assert differing == 0
