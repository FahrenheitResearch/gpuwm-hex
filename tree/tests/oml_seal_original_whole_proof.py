"""Seal completed real forecasts and independent original all-type replays.

Only proof metadata is copied. This does not grant final catalogue acceptance,
rewrite an official result, or invent an allowance for missing native fields.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seal(plan, row, output):
    folder = Path(plan['results']) / row['id']
    complete = folder / 'complete.json'
    stored = folder / 'all-written-stored-words.json'
    if not complete.is_file() or not stored.is_file():
        return None
    actual = json.loads(complete.read_text())
    proof = json.loads(stored.read_text())
    inventory = folder / 'original-written-output/ORIGINALS.json'
    originals = json.loads(inventory.read_text())
    numeric = Path(originals['numeric_comparison']['path'])
    native_pin = json.loads((folder / 'contract/PUBLICATION.json').read_text())['original_manifest']
    native = Path(native_pin['path'])
    if (actual.get('evidence_kind') != 'FORECAST'
            or actual.get('source_commit') != plan['source_commit']
            or actual.get('source_clean') is not True
            or actual.get('process_count') != proof.get('process_count')
            or proof.get('process_count') != 1
            or not proof.get('all_written_stored_words_equal')
            or proof.get('differing_words') != 0 or proof.get('failures')
            or actual['totals']['numeric_differing_words'] != 0
            or proof['original_inventory_sha256'] != sha(inventory)
            or proof['numeric_comparison_sha256'] != sha(numeric)
            or originals['numeric_comparison']['sha256'] != sha(numeric)
            or actual['receipt_sha256'] != sha(folder / 'verify.json')
            or native_pin['sha256'] != sha(native)
            or actual['native_manifest_sha256'] != sha(native)
            or [entry['original_files'] for entry in proof['files']] != originals['files']):
        raise ValueError('Original complete forecast/replay identity differs: ' + row['id'])
    output.mkdir(parents=True, exist_ok=True)
    target = output / row['id']
    target.mkdir(exist_ok=True)
    items = [(complete, 'complete.json'), (stored, 'all-written-stored-words.json'),
             (inventory, 'ORIGINALS.json'), (native, 'native-MANIFEST.json'),
             (numeric, 'numeric.json'), (folder / 'deck.json', 'deck.json'),
             (folder / 'verify.json', 'verify.json'),
             (folder / 'summary.json', 'summary.json'),
             (folder / 'hex/hex.log', 'hex.log'),
             (folder / 'hex/native-physics-proof.json', 'native-physics-proof.json'),
             (folder / 'stored-word-replay-worker.log', 'stored-word-replay-worker.log')]
    retirement = inventory.parent / 'COMPARED-COPY-RETIREMENT.json'
    if retirement.exists():
        journal = json.loads(retirement.read_text())
        if (not journal['complete'] or not journal['native_reference_files_untouched']
                or journal['stored_word_proof_sha256'] != sha(stored)
                or journal['original_inventory_sha256'] != sha(inventory)
                or len(journal['files']) != len(originals['files'])):
            raise ValueError('Preserve incomplete retirement evidence: ' + row['id'])
        items.append((retirement, retirement.name))
    metadata = {}
    for source, name in items:
        destination = target / name
        digest = sha(source)
        if source.stat().st_size >= 10_000_000:
            raise ValueError('Receipt exceeds delivery limit: ' + str(source))
        if destination.exists():
            if sha(destination) != digest:
                raise ValueError('Preserve conflicting sealed evidence: ' + str(destination))
        else:
            shutil.copy2(source, destination)
        if sha(destination) != digest:
            raise ValueError('Proof copy differs: ' + str(destination))
        metadata[name] = dict(path=str(destination.resolve()), sha256=digest)
    value = dict(schema='p19-original-all-type-whole-forecast-proof/v2',
                 id=row['id'], process_count=1, source_commit=plan['source_commit'],
                 native_execution_host=actual['native_execution_host'],
                 cuda_execution_host=actual['cuda_execution_host'],
                 cpu_comparison_host=proof['cpu_host'],
                 duration_hours=row['hours'], candidate_files=len(originals['files']),
                 all_written_stored_words_equal=True, words=proof['words'],
                 differing_words=0, numeric_compared_words=actual['totals']['compared_words'],
                 missing_words=actual['totals']['missing_words'],
                 missing_rows=actual['totals']['missing_rows'],
                 official_verifier_passed=actual['official_verifier_passed'],
                 full_native_field_identity=False, final_catalogue_acceptance=False,
                 metadata=metadata)
    final = target / 'seal.json'
    serialized = json.dumps(value, indent=2) + '\n'
    if final.exists() and final.read_text() != serialized:
        raise ValueError('Preserve prior whole-proof seal: ' + str(final))
    final.write_text(serialized)
    return dict(path=str(final.resolve()), sha256=sha(final),
                id=row['id'], words=proof['words'], differing_words=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Immutable whole forecast plan differs')
    plan = json.loads(args.plan.read_text())
    results = [value for row in plan['decks']
               if (value := seal(plan, row, args.out_root)) is not None]
    print(json.dumps(dict(sealed=results, complete=len(results) == len(plan['decks']),
                          final_catalogue_acceptance=False)))


if __name__ == '__main__':
    main()
