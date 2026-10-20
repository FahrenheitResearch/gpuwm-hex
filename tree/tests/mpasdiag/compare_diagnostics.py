"""Compare every native diagnostics record through f2's stored-word Rust tool.

This comparison-only receipt never substitutes for actual model execution,
f1 boundary agreement or trial-merge gates. Missing candidate files stay in
its inventory rather than disappearing with an early forecast failure.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deck',type=Path)
    parser.add_argument('native',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('comparator',type=Path)
    parser.add_argument('receipt',type=Path)
    args = parser.parse_args()
    deck = json.loads(args.deck.read_text())
    if deck['process_count'] != 1:
        raise ValueError('this comparison has one process')
    publication = args.native/'PUBLICATION.json'
    if sha(publication) != deck['native']['sha256']:
        raise ValueError('native publication bytes changed')
    published = json.loads(publication.read_text())
    manifest_path = args.native/'MANIFEST.json'
    if sha(manifest_path) != published['files']['MANIFEST.json']:
        raise ValueError('native manifest bytes changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['ranks'] != deck['process_count']:
        raise ValueError('native process count differs from the comparison')
    runtime = manifest['runtime']
    start = datetime.strptime(runtime['start'],'%Y-%m-%d_%H:%M:%S')
    streams = Path(deck['inputs']['streams']['path'])
    if sha(streams) != deck['inputs']['streams']['sha256']:
        raise ValueError('native stream specification changed')
    spec = next(s for s in ET.parse(streams).getroot() if s.get('name')=='diagnostics')
    template = spec.get('filename_template')
    samples, file_hashes = [], {}
    for name, expected in sorted(manifest['outputs'].items()):
        if not name.startswith('diagnostics.'):
            continue
        native = args.native/name
        if published['files'].get(name) != expected or sha(native) != expected:
            raise ValueError('native output bytes changed: '+name)
        metadata = subprocess.run([str(args.comparator),'--describe',str(native)],
                                  capture_output=True,text=True,check=True)
        info = json.loads(metadata.stdout)
        stamp = name[len('diagnostics.'):-len('.nc')]
        fallback = datetime.strptime(stamp,'%Y-%m-%d_%H.%M.%S').strftime('%Y-%m-%d_%H:%M:%S')
        times = info['times'] if info['times'] is not None else [fallback]
        if len(times) != info['records']:
            raise ValueError('native record clock cannot be inferred completely')
        steps = []
        for time in times:
            offset = int((datetime.strptime(time,'%Y-%m-%d_%H:%M:%S')-start).total_seconds())
            if offset % runtime['dt']:
                raise ValueError('native record is between model steps')
            steps.append(offset//runtime['dt'])
        sample = dict(stream='diagnostics',strict=str(native.resolve()),
                      hex=str((args.candidate/name).resolve()),steps=steps,times=times)
        if info['times'] is None:
            sample['clock_witness'] = dict(filename_template=template)
        samples.append(sample)
        file_hashes[name] = expected
    if sorted(step for sample in samples for step in sample['steps']) != deck['expected_samples']['diagnostics']:
        raise ValueError('native diagnostics inventory differs from the sealed deck')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    request = args.receipt.with_suffix('.request.json')
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1',samples=samples),indent=2)+'\n')
    raw = args.receipt.with_suffix('.raw.json')
    result = subprocess.run([str(args.comparator),'--request',str(request),'--out',str(raw)],
                            capture_output=True,text=True)
    comparison = json.loads(raw.read_text()) if raw.exists() else None
    fields = comparison.get('fields',[]) if comparison else []
    different = sum(field.get('differing_words',0) for field in fields)
    missing = sum(field.get('status') != 'compared' for field in fields)
    missing_words = sum(field.get('differing_words',0) for field in fields
                        if field.get('status') != 'compared')
    bitwise = bool(fields) and result.returncode == 0 and not different and not missing
    receipt = dict(schema='mpas-o2.diagnostics-compare/v1',process_count=1,
                   model_execution=False,identity_checked=False,passed=False,
                   native_publication_sha256=sha(publication),native_files=file_hashes,
                   comparator_sha256=sha(args.comparator),returncode=result.returncode,
                   stdout=result.stdout,stderr=result.stderr,comparison=comparison,
                   stream_bitwise=bitwise,different_words=different,
                   missing_words=missing_words,compared_different_words=different-missing_words,
                   noncompared_field_records=missing,
                   scope='all diagnostics fields/records; comparison-only, full proof unaccepted')
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(samples=len(samples),comparator_returncode=result.returncode,
                          different_words=different,noncompared_field_records=missing)),flush=True)
    return 0 if bitwise else 1


if __name__=='__main__':
    raise SystemExit(main())
