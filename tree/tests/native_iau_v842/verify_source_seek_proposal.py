"""Replay the complete copied Rust transport using authentic o1 NEAREST.

All test NetCDF encoding and packing is Rust. Python compares test byte words
and orchestrates the real executables. No production or acceptance claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
for name in ('reader', 'legacy-reader', 'fixture', 'o1', 'out'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
subprocess.run([str(args.fixture), str(args.out / 'fixture')], check=True)
os.environ['GPUWM_HEX_RW_MPAS_STREAM'] = str((args.o1 / 'rw_mpas_stream').resolve())
for name in ('streams_v842', 'restart_v842'):
    path = args.o1 / 'hexcore' / (name + '.py')
    spec = importlib.util.spec_from_file_location('hexcore.' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from hexcore.restart_v842 import seek_stream

source = args.out / 'fixture' / 'increments.nc'
path = str(source.resolve())
seeks = []
for mode, target, expected in (
        ('nearest', '2001-01-01_05:00:00', 1),
        ('nearest', '2001-01-01_03:00:00', 0),
        ('exact_time', '2001-01-01_06:00:00', 1),
        ('latest_before', '2001-01-01_03:00:00', 0),
        ('earliest_after', '2001-01-01_03:00:00', 1),
        ('latest_strictly_before', '2001-01-01_06:00:00', 0),
        ('earliest_strictly_after', '2001-01-01_00:00:00', 1)):
    record, stamp = seek_stream(source, valid_time=target, whence=mode, scratch=args.out / 'seek')
    assert record == expected, (mode, target, record, expected)
    seeks.append(dict(mode=mode, target=target, record=record, selected_time=stamp))
fields = []
for name, kind, dims, shape, components in (
        ('u_amb', 'real', ['nVertLevels', 'nEdges'], [2, 4], []),
        ('rho_amb', 'real', ['nVertLevels', 'nCells'], [2, 3], []),
        ('theta_amb', 'real', ['nVertLevels', 'nCells'], [2, 3], []),
        ('scalars_amb', 'real', ['num_scalars', 'nVertLevels', 'nCells'], [6, 2, 3],
         ['qv_amb', 'qc_amb', 'qr_amb', 'qi_amb', 'qs_amb', 'qg_amb']),
        ('indexToCellID', 'integer', ['nCells'], [3], []),
        ('xtime', 'character', ['StrLen'], [64], [])):
    fields.append(dict(pool='test', name=name, kind=kind, dimensions=dims, shape=shape,
                       components=components, required=True, read_input=True, initial=''))
request = dict(schema='mpas-native-pool-request-v1', sources=[path], fields=fields,
               source_records={path: seeks[0]['record']})


def decode(label, reader, document, success=True):
    request_path = args.out / (label + '.request.json')
    request_path.write_text(json.dumps(document, indent=2) + '\n')
    output = args.out / label
    result = subprocess.run([str(reader), str(request_path), str(output)],
                            capture_output=True, text=True)
    (args.out / (label + '.log')).write_text(result.stdout + result.stderr)
    assert (result.returncode == 0) == success, (label, result.returncode, result.stderr)
    return json.loads((output / 'manifest.json').read_text()) if success else result.returncode


selected = decode('selected', args.reader, request)
assert selected['source_records'][path] == 1
rows = []
for field in selected['fields']:
    if field['payload']:
        expected = (args.out / 'fixture' / (field['name'] + '.expected.bin')).read_bytes()
        actual = (args.out / 'selected' / field['payload']).read_bytes()
        assert actual == expected, field['name']
        rows.append(dict(field=field['name'], words=len(actual)//4, differing_words=0,
                         sha256=hashlib.sha256(actual).hexdigest()))
    else:
        assert field['text'] == '2001-01-01_06:00:00', field
        rows.append(dict(field=field['name'], text=field['text']))
legacy = decode('legacy-ignores-selection', args.legacy_reader, request)
negative = []
for a, b in zip(selected['fields'], legacy['fields'], strict=True):
    if a['payload']:
        same = (args.out / 'selected' / a['payload']).read_bytes() == (
                args.out / 'legacy-ignores-selection' / b['payload']).read_bytes()
    else:
        same = a['text'] == b['text']
    assert not same, a['name']
    negative.append(a['name'])
default_request = {k: v for k, v in request.items() if k != 'source_records'}
default = decode('default', args.reader, default_request)
for a, b in zip(default['fields'], legacy['fields'], strict=True):
    if a['payload']:
        assert (args.out / 'default' / a['payload']).read_bytes() == (
                args.out / 'legacy-ignores-selection' / b['payload']).read_bytes()
    else:
        assert a['text'] == b['text']
invalid = []
for label, mapping in (('outside-time', {path: 2}), ('negative-time', {path: -1}),
                       ('unknown-source', {path + '.absent': 1})):
    invalid.append(dict(case=label, exit=decode(label, args.reader,
                       {**request, 'source_records': mapping}, success=False)))
receipt = dict(schema='mpas.iau-source-seek-replay/v1', acceptance=False,
               scope='full copied transport and actual o1 seek; diagnostic only',
               selected_records=seeks, fields=rows, legacy_negative_fields=negative,
               default_record_zero_preserved=True, invalid_requests=invalid,
               files_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.reader, args.legacy_reader, args.fixture,
                             args.o1 / 'rw_mpas_stream', args.o1 / 'hexcore/restart_v842.py',
                             args.o1 / 'hexcore/streams_v842.py', source)})
(args.out / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
