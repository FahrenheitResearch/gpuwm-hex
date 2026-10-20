"""CPU product closure checks; no forecasts, fields or rendering are run."""
import datetime as dt
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import pytest

V = Path(__file__).parents[1]/'verification/wofs-ensemble'
sys.path.insert(0,str(V))
spec = importlib.util.spec_from_file_location('wofs_products',V/'products.py')
products = importlib.util.module_from_spec(spec)
spec.loader.exec_module(products)


def test_native_reusable_plane_protocol_owns_its_output(tmp_path):
    request=products.verification_request('control',tmp_path/'wrfout',dt.datetime(2022,6,13,18),tmp_path/'planes')
    assert request['schema']=='gpuwm.verify-visuals.request.v1'
    assert request['out_root']==str(tmp_path/'planes')
    assert request['arms'][0]['kind']=='netcdf'
    assert request['arms'][0]['fields']=={'composite_reflectivity':'REFL_10CM'}


def test_handoff_excludes_raw_output_and_checks_picture_identity(tmp_path,monkeypatch):
    import handoff
    monkeypatch.setattr(handoff,'W',tmp_path)
    source=tmp_path/'products/pilot'; source.mkdir(parents=True)
    root=tmp_path/'runs/pilot'; root.mkdir(parents=True)
    (root/'ensemble-receipt.json').write_text('{"jobs":[]}')
    (source/'complete.json').write_text('{}')
    (source/'CAPTIONS.md').write_text('caption')
    picture=source/'map.png'; picture.write_bytes(b'picture fixture')
    row=dict(path=str(picture),bytes=picture.stat().st_size,
             sha256=hashlib.sha256(picture.read_bytes()).hexdigest())
    (source/'map-artifacts.json').write_text(json.dumps([row]))
    (source/'raw.nc').write_bytes(b'raw must stay on the box')
    large=source/'member-scores.json'
    large.write_bytes(b'{"receipt":"'+b'x'*(129*1024)+b'"}')
    native=source/'aggregate/d01'; native.mkdir(parents=True)
    (native/'ensemble-manifest.json').write_text('{"native_receipt":true}')
    copied=handoff.stage('pilot')
    assert any(r['path']=='maps/map.png' for r in copied)
    assert not list((tmp_path/'handoff').rglob('*.nc'))
    assert gzip.decompress((tmp_path/'handoff/pilot/scores/member-scores.json.gz').read_bytes())==large.read_bytes()
    assert json.loads((tmp_path/'handoff/pilot/evidence/ensemble-products.json').read_text())['native_receipt']
    assert json.loads((tmp_path/'handoff/pilot/evidence/reducer-roster.json').read_text())==[]
    picture.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='picture size changed'):
        handoff.stage('pilot')

def test_every_hour_is_required_before_products_can_close():
    start = dt.datetime(2022,6,13,18)
    frames = [dict(nominal=(start+dt.timedelta(hours=h)).isoformat()+'Z') for h in range(7)]
    assert products.window_frames(dict(frames=frames),start,2)==frames[:3]
    with pytest.raises(ValueError,match='every required lead'):
        products.window_frames(dict(frames=frames[:3]+frames[4:]),start,6)
    with pytest.raises(ValueError,match='exactly once'):
        products.window_frames(dict(frames=frames+frames[-1:]),start,6)

def test_cleanup_logs_owned_size_and_refuses_parent_symlink_escape(tmp_path,monkeypatch):
    own=tmp_path/'own'; own.mkdir()
    shared=tmp_path/'shared'; shared.mkdir()
    outside=tmp_path/'outside'; outside.mkdir()
    monkeypatch.setattr(products,'W',own)
    monkeypatch.setattr(products,'R',shared)
    target=shared/'history.nc'; target.write_bytes(b'abc')
    products.retire(target)
    assert not target.exists()
    assert json.loads((own/'deleted-files.log').read_text())['bytes']==3
    target=outside/'important.nc'; target.write_bytes(b'preserve')
    (own/'escape').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='owned regular'):
        products.retire(own/'escape/important.nc')
    assert target.read_bytes()==b'preserve'

def test_cleanup_tree_removes_links_without_following_them(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('cleanup',V/'cleanup.py')
    cleanup=importlib.util.module_from_spec(spec); spec.loader.exec_module(cleanup)
    own=tmp_path/'own'; own.mkdir()
    outside=tmp_path/'outside'; outside.mkdir()
    target=outside/'preserved'; target.write_bytes(b'outside')
    raw=own/'raw'; raw.mkdir()
    (raw/'link').symlink_to(outside,target_is_directory=True)
    (raw/'frame').write_bytes(b'raw')
    monkeypatch.setattr(cleanup,'W',own); monkeypatch.setattr(cleanup,'R',own/'shared')
    cleanup.remove_tree(raw)
    assert not raw.exists() and target.read_bytes()==b'outside'
    rows=[json.loads(line) for line in (own/'deleted-files.log').read_text().splitlines()]
    assert len(rows)==2 and next(row for row in rows if row['kind']=='file')['bytes']==3

def test_native_cleanup_receipt_is_owned_absent_and_idempotent(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('cleanup',V/'cleanup.py')
    cleanup=importlib.util.module_from_spec(spec); spec.loader.exec_module(cleanup)
    monkeypatch.setattr(cleanup,'W',tmp_path)
    (tmp_path/'deleted-files.log').write_text('')
    root=tmp_path/'aggregate/d01'; root.mkdir(parents=True)
    receipt=root/'ensemble-manifest.json'
    receipt.write_text(json.dumps(dict(deleted_scratch=[dict(path='d01/retired.raw',bytes=17)])))
    cleanup.import_native_retirements(receipt)
    cleanup.import_native_retirements(receipt)
    rows=(tmp_path/'deleted-files.log').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['bytes']==17
    receipt.write_text(json.dumps(dict(deleted_scratch=[dict(path='../../escape',bytes=17)])))
    with pytest.raises(ValueError,match='escapes owned'):
        cleanup.import_native_retirements(receipt)
    (root/'survivor').write_bytes(b'still here')
    receipt.write_text(json.dumps(dict(deleted_scratch=[dict(path='d01/survivor',bytes=10)])))
    with pytest.raises(ValueError,match='surviving file'):
        cleanup.import_native_retirements(receipt)
