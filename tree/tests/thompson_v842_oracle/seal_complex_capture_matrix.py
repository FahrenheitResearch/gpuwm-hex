"""Authenticate the full fresh thirteen-call witness without widening its scope."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('transport', type=Path)
    parser.add_argument('resources', type=Path)
    parser.add_argument('cpu', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--cuda', type=Path)
    args = parser.parse_args()
    manifest = load(args.inputs / 'manifest.json')
    source = sha(args.inputs / 'thompson_v842.py')
    assert source == manifest['candidate_sha256']
    for row in manifest['inputs']:
        assert sha(args.inputs / row['file']) == row['sha256']
    prepared = load(args.transport / 'preparation.json')
    for name, expected in prepared['metadata_sha256'].items():
        assert sha(args.transport / name) == expected
    assert sha(args.transport / 'oracle_words.rs') == prepared['oracle_words_source_sha256']
    assert sha(args.transport / 'dump_transport.rs') == prepared['ffi_source_sha256']
    native = load(args.transport / 'capture-manifest.json')
    audit = load(args.transport / 'module-state-source-audit.json')
    assert audit['declaration_inventory_matches'] and audit['capture_membership_complete']
    assert not audit['source_parser_issues_present'] and not audit['hidden_state_complete']
    assert audit['dump_root_sha256'] == native['dump_root_sha256'] == prepared['dump_root_sha256']
    rows = []
    reference = None
    for backend, path in [('cpu', args.cpu)] + ([('cuda', args.cuda)] if args.cuda else []):
        replay = load(path)
        assert replay['dump_root_sha256'] == native['dump_root_sha256']
        assert replay['capture_process_count'] == 1 and not replay['hidden_state_complete']
        assert replay['comparison_complete'] and replay['all_outputs_bitwise_equal']
        assert [c['capture'] for c in replay['captures']] == list(native['captures'])
        assert len(replay['captures']) == 13
        assert all(len(call['arrays']) == 218 for call in replay['captures'])
        words = 0
        for call in replay['captures']:
            for name, measured in call['arrays'].items():
                assert measured['different_words'] == 0
                assert measured['reference_sha256'] == measured['replay_sha256']
                assert measured.get('comparison_complete', True)
                words += measured['words']
            assert all('state.mp_radar.' + name in call['arrays'] for name in ('m_w_0', 'm_i_0'))
        assert words == 590115305
        if reference:
            for a, b in zip(replay['captures'], reference['captures']):
                assert a['arrays'].keys() == b['arrays'].keys()
                assert all(a['arrays'][k]['reference_sha256'] == b['arrays'][k]['reference_sha256']
                           for k in a['arrays'])
        reference = replay
        rows.append(dict(backend=backend, path=str(path), sha256=sha(path), calls=13,
                         arrays=2834, words=words, differing_words=0,
                         replay_recipe=replay['replay_recipe']))
    images = []
    if args.cuda:
        for path in sorted((args.resources / 'images').glob('*.json')):
            certificate = load(path)
            assert certificate['production_sha256'] == source
            assert '--fmad=false' in certificate['options'] and '--ftz=false' in certificate['options']
            assert certificate['cubin_sha256'] == sha(path.with_suffix('.cubin'))
            images.append(dict(path=str(path), sha256=sha(path), cubin_sha256=certificate['cubin_sha256']))
        assert len(images) == 5
    result = dict(acceptance=False, forecast_proven=False, production_changed=False,
        candidate_sha256=source, math_sha256=manifest['math_sha256'],
        inputs_manifest_sha256=sha(args.inputs / 'manifest.json'),
        host_library_sha256=sha(args.resources / 'host_shared_native_log10.so'),
        host_source_sha256=sha(args.resources / 'host_shared_native_log10.cpp'),
        rust_words_sha256=sha(args.transport / 'oracle_words'),
        rust_transport_sha256=sha(args.transport / 'dump_transport.so'),
        preparation_sha256=sha(args.transport / 'preparation.json'),
        source_module_audit_sha256=sha(args.transport / 'module-state-source-audit.json'),
        dump_root_sha256=native['dump_root_sha256'], capture_process_count=1,
        native_execution_host=native['input_provenance']['native_execution_host'],
        capture_duration=native['input_provenance']['duration'], hidden_state_complete=False,
        known_module_declarations_captured=True, global_state_closure=False,
        raw_complex_words_compared=True, normalized_words=False, exclusions=[],
        receipts=rows, images=images,
        scope='all thirteen fresh shortened-capture calls, including complex SAVE records; no forecast tier')
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print('full complex-state witness:', len(rows), 'backends, 590115305 words0 each')


if __name__ == '__main__':
    main()
