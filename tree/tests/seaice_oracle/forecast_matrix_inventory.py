"""Inventory every original ice option without assigning an identity grade.

MPAS mpas_atmphys_driver_seaice.F:461-558 is the lane boundary. Every attempt
stays linked to its full receipt; missing model samples never count as exact.
This reads JSON metadata only, not weather fields.
"""
import argparse
import hashlib
import json
from pathlib import Path


def seal(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summary(path, record):
    fields = record.get('comparison', {}).get('fields', [])
    computed = [row for row in fields if row['status'] == 'compared']
    different = [row for row in computed if row['differing_words']]
    first = min(different, key=lambda row: (row['step'], row['stream'], row['field'])) if different else None
    return dict(receipt=seal(path), evidence_kind=record.get('evidence_kind'),
                model_execution=record.get('model_execution'), passed=record.get('passed'),
                acceptance_tier=record.get('acceptance_tier'),
                numerical_compared_words=sum(row['compared_words'] for row in computed),
                numerical_differing_words=sum(row['differing_words'] for row in computed) if computed else None,
                missing_or_incompatible_samples=len(fields) - len(computed),
                first_computed_difference=None if first is None else
                    {key: first[key] for key in ('stream', 'field', 'step', 'differing_words')})


def run(args):
    prepared = json.loads(args.prepared_false.read_text())['decks']
    templates = [json.loads(path.read_text()) for path in args.templates.glob('*.deck.json')]
    false_native = json.loads(args.native_false.read_text())['runs']
    false_paths = [args.cuda_false, *(args.cuda_false_attempt or [])]
    false_cuda = []
    false_sources = []
    for index, path in enumerate(false_paths):
        if not path.is_file():
            continue
        # Controllers append to their matrix. Retain the exact full bytes read
        # by this audit so its receipt links survive later completed attempts.
        payload = path.read_bytes()
        snapshot = args.out.with_name(args.out.stem + f'-false-source-{index}.json')
        snapshot.write_bytes(payload)
        source = dict(original_path=str(path), snapshot=seal(snapshot))
        false_sources.append(source)
        false_cuda.extend((snapshot, record) for record in json.loads(payload)['runs'])
    publication_copies = (json.loads(args.publication_copies.read_text())
                          if args.publication_copies else {})
    attempts = []
    if args.attempt_directory:
        paths = sorted({path for directory in args.attempt_directory
                        for path in directory.glob('*.json')})
    else:
        paths = sorted(args.receipts.glob('whole*.json'))
    for path in paths:
        record = json.loads(path.read_text())
        if record.get('schema') == 'mpas-verification-v1' and record.get('process_count') == 1:
            command = record.get('provenance', {}).get('hex_command', [])
            if "config_seaice_backend='native-cpu'" not in command:
                attempts.append((path, record))
    rows = []
    for item in prepared:
        source = Path(item['native_source'])
        mesh, date, choice = source.parent.parent.name, source.parent.name, source.name
        template = next(deck for deck in templates if deck['mesh'] == mesh and
                        deck['date'] == date.replace('.', ':') and
                        (deck['suite'].removeprefix('suite-') == choice.removeprefix('suite-')
                         if choice.startswith('suite-') else '-' + choice + '-' in deck['id']))
        assert template['process_count'] == 1
        for fractional in (True, False):
            name = '-'.join((mesh, date, choice, 'frac-' + str(fractional).lower(), 'rank1'))
            row = dict(deck=name, mesh=mesh, date=date, native_option=choice,
                       fractional=fractional, process_count=1, acceptance_tier=None)
            if fractional:
                publication = Path(template['native']['path'])
                local = Path(publication_copies.get(str(publication), str(publication)))
                authenticated = seal(local)
                assert authenticated['sha256'] == template['native']['sha256']
                row['native'] = dict(completed=True, publication=template['native'],
                                     authenticated_publication_copy=authenticated)
                inputs = {key: value['sha256'] for key, value in template['inputs'].items()}
                selected = [(path, record) for path, record in attempts if
                            record['deck'] == template['id'] or
                            (not args.exact_deck_id and
                             {key: value['sha256'] for key, value in
                              record.get('provenance', {}).get('inputs', {}).items()} == inputs)]
                row['cuda_attempts'] = [summary(path, record) for path, record in selected]
            else:
                witness = next(record for record in false_native if record['deck'] == name)
                row['native'] = dict(completed=witness['completed'], sigsegv=witness['sigsegv'],
                                     witness_receipt=seal(args.native_false))
                row['cuda_attempts'] = [dict(receipt=seal(path), model_exit=record['model_exit'],
                                            model_execution=record['model_execution'],
                                            numerical_compared_words=0, numerical_differing_words=None,
                                            acceptance_tier=None)
                                        for path, record in false_cuda if record['deck'] == name]
            rows.append(row)
    assert len(rows) == 32 and len({row['deck'] for row in rows}) == 32
    result = dict(kind='32-option whole-model attempt inventory; no replacement acceptance grade',
                  process_count=1, expected_decks=len(rows),
                  decks_with_cuda_attempts=sum(bool(row['cuda_attempts']) for row in rows),
                  native_complete=sum(row['native']['completed'] for row in rows),
                  native_sigsegv=sum(row['native'].get('sigsegv', False) for row in rows),
                  attempted_directories=[str(path) for path in args.attempt_directory or []],
                  fractional_false_attempt_receipts=false_sources,
                  exact_deck_id=args.exact_deck_id,
                  decks=rows, exclusions=[], acceptance_tier=None)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'decks'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-false', 'templates', 'native-false', 'cuda-false', 'receipts', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--attempt-directory', type=Path, action='append',
                        help='inventory these explicit forecast receipt directories only')
    parser.add_argument('--exact-deck-id', action='store_true',
                        help='require the original deck identity, without input-only association')
    parser.add_argument('--publication-copies', type=Path,
                        help='map original publication paths to byte copies; original SHA256 still required')
    parser.add_argument('--cuda-false-attempt', type=Path, action='append',
                        help='retain additional actual false-option attempts alongside the original matrix')
    run(parser.parse_args())
