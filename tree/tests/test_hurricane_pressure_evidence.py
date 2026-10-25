"""Intensity evidence must not silently score a smoothed tracking diagnostic."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/hurricane'
spec = importlib.util.spec_from_file_location('marine_pressure', TOOLS / 'marine_pressure.py')
marine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(marine)


def fixture(tmp_path):
    folder = tmp_path / 's000-t0000000'
    folder.mkdir()
    arrays = dict(latCell=np.radians([22., 22.1, 25.]),
                  lonCell=np.radians([-92., -92.1, -90.]),
                  ter=np.array([[0., 0., 1000.]]),
                  surface_pressure=np.array([[95000., 94500., 80000.]]))
    np.savez(folder / 'surface.npz', **arrays)
    fields = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                     sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()}
    record = dict(history_sha256='fixture', frame=dict(model_seconds=0,
        member='start', member_center=[22., -92.], fix=dict(ok=True, lat=22., lon=-92.,
        vmax10_ms=50., mslp_min_hpa=970.)), fields=fields)
    (folder / 'receipt.json').write_text(json.dumps(record))
    return folder, arrays, record


def test_marine_core_minimum_is_native_not_smoothed_or_distant_terrain(tmp_path):
    folder, _, _ = fixture(tmp_path)
    row = marine.extract(folder, 50)
    assert row['mslp_hpa'] == 945
    assert row['cells'] == 2
    assert marine.extract(folder, 500)['mslp_hpa'] is None


def test_changed_surface_is_refused(tmp_path):
    folder, arrays, _ = fixture(tmp_path)
    arrays['surface_pressure'][0, 1] = 94000.
    np.savez(folder / 'surface.npz', **arrays)
    with pytest.raises(ValueError, match='changed retained field'):
        marine.extract(folder, 50)


def test_export_requires_hash_bound_native_pressure(tmp_path):
    folder, _, record = fixture(tmp_path)
    (tmp_path / 'collection.json').write_text(json.dumps(dict(error=None, producer=dict(returncode=0))))
    case = tmp_path / 'case.json'
    case.write_text(json.dumps(dict(start_time='2024-10-07_00:00:00',
                                   mesh=dict(fine_radius_km=450, fine_transition_km=54))))
    out = tmp_path / 'out'
    cmd = [sys.executable, str(TOOLS / 'export_track.py'), '--evidence', str(tmp_path),
           '--case', str(case), '--out', str(out)]
    subprocess.run(cmd, check=True)
    assert json.loads((out / 'fixes.json').read_text())['fixes'][0]['mslp_hpa'] is None
    sidecar = tmp_path / 'pressure.json'
    row = marine.extract(folder, 50)
    sidecar.write_text(json.dumps(dict(records=[row])))
    subprocess.run([*cmd, '--pressure-evidence', str(sidecar)], check=True)
    assert json.loads((out / 'fixes.json').read_text())['fixes'][0]['mslp_hpa'] == 945
    row['history_sha256'] = 'different'
    sidecar.write_text(json.dumps(dict(records=[row])))
    failed = subprocess.run([*cmd, '--pressure-evidence', str(sidecar)], capture_output=True)
    assert failed.returncode != 0 and b'does not match history' in failed.stderr
