"""Landing real-input replay must reject an incomplete or changed native bank."""
import hashlib
import json

import pytest

from rrfs_native.replay_acoustic_capture import validate_capture_bank


def bank_contract(tmp_path):
    bank = tmp_path / 'bank'
    bank.mkdir()
    rows = []
    calls = {'dycore__first_work': 1, 'dycore__second_work': 1}
    for key in calls:
        for phase in ('pre', 'post'):
            name = f'{key}.s0000000001.e0000000001.r000000.{phase}.argument.value.bin'
            payload = (name + 'authenticated opaque fixture').encode()
            (bank / name).write_bytes(payload)
            rows.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(source_commit='fixture-source', records=rows)))
    contract = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    source_commit='fixture-source', calls=calls)
    return bank, manifest, contract


def test_authenticates_the_complete_stated_native_case_set(tmp_path):
    bank, manifest, contract = bank_contract(tmp_path)
    assert {k: len(v) for k, v in validate_capture_bank(bank, manifest, contract).items()} == contract['calls']


def test_authenticates_every_call_across_native_timesteps(tmp_path):
    bank, manifest, contract = bank_contract(tmp_path)
    data = json.loads(manifest.read_text())
    original_rows = list(data['records'])
    for row in original_rows:
        name = row['path'].replace('.s0000000001.', '.s0000000002.')
        payload = (bank / row['path']).read_bytes()
        (bank / name).write_bytes(payload)
        data['records'].append(dict(path=name, bytes=len(payload),
                                    sha256=hashlib.sha256(payload).hexdigest()))
    manifest.write_text(json.dumps(data))
    contract['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    contract['calls'] = {key: 2 for key in contract['calls']}
    assert {k: len(v) for k, v in validate_capture_bank(bank, manifest, contract).items()} == contract['calls']


@pytest.mark.parametrize('damage', [None, 'missing', 'label', 'binary', 'patch', 'ruling'])
def test_source_authority_cannot_be_dropped_or_relabelled(tmp_path, damage):
    bank, manifest, contract = bank_contract(tmp_path)
    authority = dict(label='separately labelled initializer derivative',
                     binary='approved initializer', patch='approved patch', ruling='specific ruling')
    data = json.loads(manifest.read_text())
    data['source_authority'] = authority
    manifest.write_text(json.dumps(data))
    contract['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    contract['source_authority'] = dict(authority)
    if damage == 'missing':
        del contract['source_authority']
    elif damage:
        contract['source_authority'][damage] = 'different authority'
    if damage is None:
        validate_capture_bank(bank, manifest, contract)
    else:
        with pytest.raises(ValueError, match='source authority'):
            validate_capture_bank(bank, manifest, contract)


@pytest.mark.parametrize('damage', ['missing_boundary', 'changed_record', 'extra_record', 'changed_manifest', 'wrong_source', 'wrong_calls'])
def test_rejects_incomplete_or_unauthenticated_capture_before_cuda(tmp_path, damage):
    bank, manifest, contract = bank_contract(tmp_path)
    if damage == 'missing_boundary':
        for path in bank.glob('dycore__second_work*'):
            path.unlink()
    elif damage == 'changed_record':
        next(bank.iterdir()).write_bytes(b'changed')
    elif damage == 'extra_record':
        (bank / 'extra.bin').write_bytes(b'extra')
    elif damage == 'changed_manifest':
        manifest.write_bytes(manifest.read_bytes() + b' ')
    elif damage == 'wrong_source':
        contract['source_commit'] = 'other-source'
    else:
        contract['calls']['dycore__missing_work'] = 1
    with pytest.raises(ValueError):
        validate_capture_bank(bank, manifest, contract)


@pytest.mark.parametrize('damage', [None, 'missing', 'extra', 'changed', 'bridge', 'source'])
def test_complete_cold_core_bank_authentication(tmp_path, damage):
    from rrfs_native.replay_cold_core import authenticate
    bank = tmp_path / 'bank'
    stage = tmp_path / 'stage'
    bank.mkdir()
    stage.mkdir()
    native = bank / 'native.bin'
    native.write_bytes(b'opaque native record')
    bridge = stage / 'bridge.rs'
    bridge.write_bytes(b'opaque pinned source')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    contract = dict(source_commit='1d6e370e7b43a332074809f170532a216c2212fb',
                    rust_sources=[dict(staged='bridge.rs', sha256=sha(bridge))],
                    cases=[dict(bank='bank', records=[dict(name=native.name,
                                bytes=native.stat().st_size, sha256=sha(native))])])
    if damage == 'missing': native.unlink()
    elif damage == 'extra': (bank / 'extra.bin').write_bytes(b'extra')
    elif damage == 'changed': native.write_bytes(b'changed')
    elif damage == 'bridge': bridge.write_bytes(b'changed')
    elif damage == 'source': contract['source_commit'] = 'other'
    if damage is None:
        authenticate(contract, tmp_path, stage)
    else:
        with pytest.raises(ValueError):
            authenticate(contract, tmp_path, stage)
