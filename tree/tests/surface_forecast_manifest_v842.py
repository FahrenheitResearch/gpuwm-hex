"""Bind completed private strict native runs to the real MPAS verifier.

Python handles provenance, receipt hashes and clock metadata. The Rust
comparator reads all stored netCDF metadata. No native output is filtered or
rewritten, and no MPAS/hex forecast identity is inferred by this preparation.
"""
import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def auxiliary_inputs(directory, init):
    """Pin tables and explicit input streams, including files outside the run."""
    auxiliary, publications = {}, {}

    def add(name, path):
        path = path.resolve(strict=True)
        sha = digest(path)
        auxiliary[name] = {'path': str(path), 'sha256': sha}
        for filename in ('PUBLICATION.json', 'MANIFEST.json'):
            manifest = path.parent / filename
            if not manifest.is_file():
                continue
            files = json.loads(manifest.read_text()).get('files', {})
            if isinstance(files, dict):
                declared = files.get(path.name)
            elif isinstance(files, list):
                declared = next((row.get('sha256') for row in files
                                 if row.get('name', row.get('path')) == path.name), None)
            else:
                raise SystemExit('Unsupported native publication file inventory: ' + str(manifest))
            if declared is not None:
                if declared != sha:
                    raise SystemExit('Native auxiliary differs from its publication: ' + str(path))
                publications[str(manifest)] = digest(manifest)

    for path in sorted(directory.iterdir()):
        if path.is_symlink() and path.name != 'atmosphere_model' and path.resolve() != init.resolve():
            add(path.name, path)
    for stream in ET.parse(directory / 'streams.atmosphere').getroot():
        if 'input' not in stream.get('type', '').split(';'):
            continue
        template = stream.get('filename_template', '')
        # These explicit decks have no active time-templated auxiliary input.
        # Inactive restart/IAU/LBC/DA templates are not invented as input files.
        if not template or '$' in template:
            continue
        path = directory / template
        if path.resolve() != init.resolve() and path.is_file():
            add('stream:' + stream.get('name', ''), path)
    return auxiliary, publications


def prepare(root, comparator, summary_path=None, submitted_decks=None,
            output_directory=None, output_receipt=None):
    from hexcore.config import parse_fortran_namelist
    from hexcore.verify_mpas import native_manifest
    summary_path = summary_path or root / 'receipts/native-surface-five-day-summary.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('all_native_references_complete') is not True or summary.get('process_count') != 1:
        raise SystemExit('Strict native five-day completion and schedules are not yet witnessed.')
    submitted_decks = submitted_decks or root / 'receipts/native-surface-five-day-decks.json'
    source_decks = json.loads(submitted_decks.read_text())
    pin_set = json.loads((root / 'receipts/whole-door-deck.json').read_text())['pin_set']
    out = output_directory or root / 'receipts/native-surface-five-day/manifests'
    out.mkdir(exist_ok=True)
    result = []
    for run in summary['decks']:
        source = next(d for d in source_decks if d['date'] == run['date'] and d['suite'] == run['suite'])
        directory = Path(run['directory'])
        log = directory / 'log.atmosphere.0000.out'
        if run['actual_exit_code'] != 0 or 'Finished running the atmosphere core' not in log.read_text():
            raise SystemExit('Native completion marker or exit absent: ' + str(directory))
        # These pins were captured before execution; a changed input cannot be certified.
        for filename, sha in source['source_sha256'].items():
            path = directory / Path(filename).name if filename.endswith(('namelist.atmosphere', 'streams.atmosphere')) else Path(filename)
            if digest(path) != sha:
                raise SystemExit('Changed submitted native input: ' + str(path))
        groups = parse_fortran_namelist((directory / 'namelist.atmosphere').read_text())
        model = groups['nhyd_model']
        start = datetime.strptime(model['config_start_time'], '%Y-%m-%d_%H:%M:%S')
        dt = Fraction(str(model['config_dt']))
        inputs = {key: {'path': str(directory / name), 'sha256': digest(directory / name)}
                  for key, name in (('namelist', 'namelist.atmosphere'), ('streams', 'streams.atmosphere'))}
        init = next(Path(name) for name in source['source_sha256'] if name.endswith('.init.nc'))
        inputs['init'] = {'path': str(init), 'sha256': digest(init)}
        auxiliary, publications = auxiliary_inputs(directory, init)
        outputs = []
        for output in run['outputs']:
            path = Path(output['path'])
            if digest(path) != output['sha256']:
                raise SystemExit('Changed completed native output: ' + str(path))
            completed = subprocess.run([str(comparator), '--describe', str(path)], capture_output=True, text=True, check=True)
            metadata = json.loads(completed.stdout)
            times = metadata['times']
            if metadata['records'] != 1 or times != [output['timestamp']]:
                raise SystemExit('Native record clock or record count differs: ' + str(path))
            elapsed = datetime.strptime(times[0], '%Y-%m-%d_%H:%M:%S') - start
            step = Fraction(elapsed.days * 86400 + elapsed.seconds) / dt
            if step.denominator != 1:
                raise SystemExit('Native output is off its dynamical clock: ' + str(path))
            outputs.append(dict(stream=output['stream'], path=path.name, sha256=output['sha256'],
                                steps=[int(step)], times=times, time_source=metadata['time_source'], records=1))
        manifest = dict(schema='mpas-native-run-v1', pin_set=pin_set, process_count=1,
                        inputs={k:v['sha256'] for k,v in inputs.items()},
                        auxiliary_inputs={k:v['sha256'] for k,v in auxiliary.items()},
                        complete=True, output_root=str(directory), outputs=outputs,
                        witness=dict(summary=str(summary_path), summary_sha256=digest(summary_path),
                                     completion_log_sha256=digest(log), submitted_inputs=source['source_sha256'],
                                     auxiliary_publications=publications))
        path = out / (run['date'] + '-' + run['suite'] + '.json')
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        deck = dict(pin_set=pin_set, process_count=1, inputs=inputs, auxiliary_inputs=auxiliary)
        # This is the same actual completeness/inventory validator used by verify-mpas.
        native_manifest({'path': str(path), 'sha256': digest(path)}, out, deck, comparator)
        result.append(dict(date=run['date'], suite=run['suite'], manifest=str(path),
                           sha256=digest(path), output_count=len(outputs), inputs=inputs,
                           auxiliary_inputs=auxiliary, auxiliary_publications=publications,
                           process_count=1, verifier_manifest_checked=True))
    receipt = dict(schema='p25-native-forecast-manifests/v1', process_count=1,
                   tier='native reference preparation only', hex_forecast_identity_claimed=False,
                   comparator_sha256=digest(comparator), manifests=result)
    output_receipt = output_receipt or root / 'receipts/native-surface-forecast-manifests.json'
    output_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(prepared=len(result), native_only=True, hex_forecast_identity_claimed=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--submitted-decks', type=Path)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--output-receipt', type=Path)
    arguments = parser.parse_args()
    prepare(arguments.directory.resolve(), arguments.comparator.resolve(), arguments.summary,
            arguments.submitted_decks, arguments.output_directory, arguments.output_receipt)
