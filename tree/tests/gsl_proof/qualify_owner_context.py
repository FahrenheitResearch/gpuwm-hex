"""Qualify an unapplied missing-inventory proposal against original M0.

This reads sealed JSON receipts only. It never decodes weather, installs a
ratchet or supplies a proposed baseline to the executing verifier.
"""
import argparse
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned(ref):
    path = Path(ref['path'])
    assert digest(path) == ref['sha256'], 'changed proof input'
    return json.loads(path.read_text())


def qualify(owner, candidate):
    for key in ('mesh', 'date', 'process_count'):
        assert candidate[key] == owner[key], 'different owner scope: ' + key
    assert owner['process_count'] == 1
    assert owner['completed'] and owner['returncode'] == 0
    assert owner['forecast_progress']['completed']
    assert owner['forecast_progress']['steps_completed'] == owner['forecast_progress']['steps_requested']
    old_rows = owner['comparison']['comparison']['fields']
    rows = candidate['comparison']['fields']
    def index(items):
        result = {(r['stream'], r['field'], r['step']): r for r in items}
        assert len(result) == len(items), 'duplicate native sample'
        return result
    old, current = index(old_rows), index(rows)
    assert old.keys() == current.keys(), 'native sample inventory changed'
    missing, retired = [], []
    for key, row in current.items():
        previous = old[key]
        assert row['time'] == previous['time'], 'native clock changed'
        assert row['compared_words'] == previous['compared_words'], 'native word extent changed'
        assert previous['status'] in ('compared', 'field missing in hex'), 'invalid owner inventory'
        if previous['status'] == 'compared':
            assert previous['differing_words'] == 0, 'owner written words differ'
        if row['status'] == 'compared':
            assert row['differing_words'] == 0, 'candidate written words differ'
            if previous['status'] != 'compared':
                retired.append(row)
        else:
            assert row['status'] == previous['status'] == 'field missing in hex', 'new structural gap'
            assert row['differing_words'] == previous['differing_words'], 'missing word counter changed'
            missing.append(row)
    assert missing and retired, 'expected existing and retired owner omissions'
    return missing, retired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    matrix = pinned(cfg['owner_matrix'])
    assert matrix['source_clean'] and matrix['all_common_stored_words_identical']
    root = Path(cfg['owner_root'])
    custody = []
    for row in matrix['decks']:
        assert row['process_count'] == 1 and row['stored_differing_words'] == 0
        for kind in ('receipt', 'stored_word_receipt'):
            ref = dict(row[kind], path=str(root / row[kind]['path']))
            pinned(ref)
            custody.append(ref)
    owner_ref = cfg['owner_receipt']
    assert owner_ref in custody, 'selected owner receipt is not in original M0 matrix'
    owner = pinned(owner_ref)
    native, generated = pinned(cfg['native_receipt']), pinned(cfg['generated_receipt'])
    assert native['comparison'] == generated['comparison'], 'GSL substitution changed word receipt'
    assert owner['source_status'] == 'unmodified lander-supplied checkout'
    assert owner['forecast_progress']['native_library_sha256'] == cfg['provider_sha256']
    for ref in cfg['attempts']:
        attempt = pinned(ref)
        assert attempt['completed'] and attempt['returncode'] == 0 and attempt['process_count'] == 1
        assert attempt['run_host'] == owner['run_host']
        progress = attempt['progress']
        assert progress['completed'] and progress['steps_completed'] == progress['steps_requested']
        assert progress['steps_completed'] == owner['forecast_progress']['steps_completed']
        assert progress['native_library_sha256'] == cfg['provider_sha256']
    missing, retired = qualify(owner, native)
    assert qualify(owner, generated) == (missing, retired)
    # These are qualification controls, never passing forecasts or policies.
    controls = []
    def refuses(name, mutate):
        changed = deepcopy(native)
        mutate(changed)
        try:
            qualify(owner, changed)
        except (AssertionError, KeyError):
            controls.append(name)
        else:
            raise AssertionError('fault was accepted: ' + name)
    old_index = {(r['stream'], r['field'], r['step']): r for r in owner['comparison']['comparison']['fields']}
    compared_index = next(i for i, r in enumerate(native['comparison']['fields'])
                          if r['status'] == old_index[(r['stream'], r['field'], r['step'])]['status'] == 'compared')
    missing_index = next(i for i, r in enumerate(native['comparison']['fields']) if r['status'] != 'compared')
    for key in ('mesh', 'date', 'process_count'):
        refuses('changed-' + key, lambda d, k=key: d.__setitem__(k, -1))
    refuses('computed-word-difference', lambda d: d['comparison']['fields'][compared_index].__setitem__('differing_words', 1))
    refuses('growing-missing-words', lambda d: d['comparison']['fields'][missing_index].__setitem__('differing_words', missing[0]['differing_words'] + 1))
    refuses('changed-native-word-extent', lambda d: d['comparison']['fields'][missing_index].__setitem__('compared_words', missing[0]['compared_words'] + 1))
    refuses('incompatible-field-status', lambda d: d['comparison']['fields'][missing_index].__setitem__('status', 'incompatible'))
    refuses('changed-native-clock', lambda d: d['comparison']['fields'][missing_index].__setitem__('time', 'changed'))
    refuses('previously-written-field-missing', lambda d: d['comparison']['fields'][compared_index].__setitem__('status', 'field missing in hex'))
    refuses('dropped-native-sample', lambda d: d['comparison']['fields'].pop())
    refuses('duplicate-native-sample', lambda d: d['comparison']['fields'].append(deepcopy(d['comparison']['fields'][0])))
    refuses('invented-native-field', lambda d: d['comparison']['fields'][0].__setitem__('field', 'invented'))
    baseline = pinned(cfg['ratchet'])
    context = {k: native[k] for k in ('pin_set', 'mesh', 'date', 'deck', 'process_count')}
    from hexcore.verify_mpas import aggregate
    assert aggregate(native['comparison']['fields'], context) == native['candidate_ratchet'], 'candidate aggregate changed'
    assert not any(all(entry[k] == context[k] for k in ('pin_set', 'mesh', 'date', 'deck')) for entry in baseline['entries'])
    # Retain original entries exactly. This JSON/diff is an OWNER PROPOSAL only.
    proposal = deepcopy(baseline)
    proposal['entries'].extend(native['candidate_ratchet']['entries'])
    assert proposal['entries'][:len(baseline['entries'])] == baseline['entries']
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        path = args.out / name
        data = json.dumps(value, indent=2) + '\n'
        assert len(data.encode()) < 10 * 1024 * 1024
        path.write_text(data)
    write('ratchet-owner-proposal.json', proposal)
    original = Path(cfg['ratchet']['path']).read_text()
    proposed = (args.out / 'ratchet-owner-proposal.json').read_text()
    (args.out / 'owner-context-proposed.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), proposed.splitlines(True), fromfile='a/receipts/ratchet.json', tofile='b/receipts/ratchet.json')))
    write('missing-samples.json', missing)
    write('retired-owner-samples.json', retired)
    write('qualification.json', {
        'schema': 'i4-original-owner-context-review/v1', 'context': context,
        'process_count': 1, 'original_owner_receipt': owner_ref,
        'all_original_owner_receipts_authenticated': custody,
        'native_and_generated_comparison_identical': True,
        'all_native_samples_accounted': len(native['comparison']['fields']),
        'missing_samples': len(missing), 'missing_words': sum(r['compared_words'] for r in missing),
        'retired_owner_missing_samples': len(retired), 'retired_owner_missing_words': sum(r['compared_words'] for r in retired),
        'differing_computed_words': 0, 'new_or_growing_missing_samples': 0,
        'negative_controls_refused': controls, 'original_ratchet_entries_preserved': len(baseline['entries']),
        'proposed_context_entries': len(native['candidate_ratchet']['entries']),
        'proposal_applied': False, 'proposal_used_for_verification': False,
        'official_forecast_acceptance_claimed': False, 'config_sha256': digest(args.config),
        'inputs': cfg, 'files': {p.name: digest(p) for p in args.out.iterdir() if p.is_file()},
    })
    assert digest(cfg['ratchet']['path']) == cfg['ratchet']['sha256']
    print('Original owner inventory qualified; proposal remains unapplied.')


if __name__ == '__main__':
    main()
