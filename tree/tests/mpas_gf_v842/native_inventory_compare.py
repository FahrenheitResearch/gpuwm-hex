"""Compare complete native word inventories using the pinned Rust decoder.

Only receipt metadata is processed in Python. Changed digests do not count
numeric word differences; matching complete digests establish word identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def compare(reference, candidate, tool, receipt):
    manifests = [json.loads((root / 'MANIFEST.json').read_text())
                 for root in (reference, candidate)]
    for key in ('ranks', 'binary_sha256', 'namelist_sha256', 'streams_sha256', 'inputs'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'Native runtime input differs: {key}')
    names = sorted(manifests[0]['outputs'])
    if not names or set(names) != set(manifests[1]['outputs']):
        raise ValueError('Native output inventories differ or are empty')
    rows = []
    for name in names:
        inventories = []
        for root, manifest in zip((reference, candidate), manifests):
            if sha(root / name) != manifest['outputs'][name]:
                raise ValueError(f'Native file does not match its manifest: {root / name}')
            run = subprocess.run([str(tool), '--field-digests', str(root / name)],
                                 capture_output=True, text=True, check=True)
            inventories.append(json.loads(run.stdout))
        if inventories[0]['xtime'] != inventories[1]['xtime']:
            raise ValueError(f'Native output clocks differ: {name}')
        fields = [{v['field']: v for v in x['variables']} for x in inventories]
        if set(fields[0]) != set(fields[1]):
            raise ValueError(f'Native fields differ: {name}')
        for field, want in fields[0].items():
            got = fields[1][field]
            if any(want[k] != got[k] for k in ('netcdf_type', 'shape', 'dimensions', 'words')):
                raise ValueError(f'Native storage differs: {name}/{field}')
            rows.append(dict(file=name, field=field, words=want['words'],
                native_word_sha256=want['word_sha256'], candidate_word_sha256=got['word_sha256'],
                word_hash_identical=want['word_sha256'] == got['word_sha256']))
    changed = sum(not row['word_hash_identical'] for row in rows)
    result = dict(schema='gf-complete-native-word-inventory-comparison-v1',
        reference=str(reference), candidate=str(candidate),
        reference_manifest_sha256=sha(reference / 'MANIFEST.json'),
        candidate_manifest_sha256=sha(candidate / 'MANIFEST.json'),
        tool_sha256=sha(tool), process_count=manifests[0]['ranks'], files=len(names),
        words=sum(row['words'] for row in rows), fields_with_changed_word_hash=changed,
        all_stored_word_hashes_identical=changed == 0, fields=rows,
        scope='Complete native run comparison; no hex forecast or excluded-word claim')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return changed != 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--tool', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(compare(a.reference.resolve(), a.candidate.resolve(), a.tool.resolve(), a.receipt))
