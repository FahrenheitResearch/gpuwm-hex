"""Replay a sealed first-differing driver call against strict native and host glibc.

Rust performs all input byte layout transport. The native oracle executes the
original MPAS from-MPAS loops and whole Noah driver; the host bridge executes
the shipped expression source. This is a column diagnostic, not a forecast.
"""
import argparse
import ctypes as ct
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from hexcore.mpasphys import noah_v842 as noah


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packer', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--host-library', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--metadata', type=Path,
                        help='explicit supplementary reconstructed metadata; diagnostic only')
    parser.add_argument('--candidate-source', type=Path)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text())
    metadata_witness = json.loads(args.metadata.read_text()) if args.metadata else capture
    if not capture.get('first_difference') or not metadata_witness.get('arguments'):
        print('No sealed first-differing call with actual scalar/table metadata')
        return 75
    assert capture['production_sha256'] == sha(noah.__file__)
    for name, digest in capture['array_files'].items():
        assert sha(args.arrays / name) == digest, 'captured driver input changed: ' + name
    arguments = metadata_witness['arguments']
    assert arguments['level_first'] and arguments['start'] == 0
    count = arguments['count']
    assert count == capture['inputs']['skintemp']['shape'][0]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for name, kind in noah._MPAS_INPUT_FIELDS.items():
        shape = capture['inputs'].get(name, {}).get('shape', [count])
        levels = shape[0] if len(shape) == 2 else 1
        width = {'soil': 4, 'atmosphere': 2, 'pressure': 2}.get(kind, 1)
        metadata.append('\t'.join(map(str, (name, 'int' if kind == 'integer_surface' else 'float', levels, width))))
    metadata_path = args.output / 'input-layout.tsv'
    metadata_path.write_text('\n'.join(metadata) + '\n')
    process = subprocess.run([str(args.packer), str(args.arrays), str(metadata_path),
                              str(count), str(args.output)], capture_output=True, text=True, check=True)
    transport = json.loads(process.stdout)
    fields = np.fromfile(args.output / 'fields.input.bin', dtype=np.float32)
    categories = np.fromfile(args.output / 'categories.input.bin', dtype=np.int32)
    table_path = Path(metadata_witness.get('tables_path', args.arrays / 'tables.input.bin'))
    table_bytes = table_path.read_bytes()
    assert hashlib.sha256(table_bytes).hexdigest() == metadata_witness['tables_sha256']
    tables = np.frombuffer(table_bytes, dtype=noah.table_dtype()).copy()
    native, host = ct.CDLL(str(args.native_library)), ct.CDLL(str(args.host_library))
    # Transport the actual native table words, including retained categories.
    # Constants remain those of the compiled strict standalone MPAS source.
    for name, spec in noah.TABLE_SCHEMA.items():
        if spec['parameter']:
            continue
        dtype, offset = tables.dtype.fields[name]
        try:
            target = (ct.c_ubyte * dtype.itemsize).in_dll(native, '__module_sf_noahlsm_MOD_' + name.lower())
        except ValueError:
            continue
        ct.memmove(ct.addressof(target), tables.ctypes.data + offset, dtype.itemsize)
    options = arguments['options']
    native_options = np.array([arguments['isurban'], arguments['isice'], arguments['itimestep'],
                              options['config_sfc_albedo'], options['config_microp_scheme'] != 'off',
                              options['config_convection_scheme'] != 'off'], dtype=np.int32)
    native_x, native_ix = np.empty((count, 79), np.float32), np.empty((count, 7), np.int32)
    host_x, host_ix = np.empty_like(native_x), np.empty_like(native_ix)
    function = native.io_from
    function.argtypes = [ct.c_void_p] * 4 + [ct.c_int] * 3 + [ct.c_float] * 2 + [ct.c_void_p]
    host_post_fields = fields.copy()
    function(fields.ctypes.data, categories.ctypes.data, native_x.ctypes.data, native_ix.ctypes.data,
             count, 0, count, arguments['dt_pbl'], arguments['xice_threshold'], native_options.ctypes.data)
    host_fields = {name: np.fromfile(args.output / (name + '.cellmajor.bin'),
                                   dtype=np.int32 if kind == 'integer_surface' else np.float32)
                   for name, kind in noah._MPAS_INPUT_FIELDS.items()}
    function = host.noah_from_mpas
    function.argtypes = [ct.c_void_p] * (2 + len(host_fields)) + [ct.c_int] * 4 + [ct.c_float] * 2 + [ct.c_int] * 6
    function(host_x.ctypes.data, host_ix.ctypes.data, *(value.ctypes.data for value in host_fields.values()),
             capture['inputs']['t']['shape'][0], capture['inputs']['pres2_hyd']['shape'][0], 0, count,
             arguments['dt_pbl'], arguments['xice_threshold'], native_options[4], native_options[5],
             arguments['isurban'], arguments['isice'], arguments['itimestep'], options['config_sfc_albedo'])
    def compare(stage):
        directory = args.output / stage
        directory.mkdir(exist_ok=True)
        (directory / 'fields.txt').write_text('driver_float\ndriver_integer\n')
        for name, reference, candidate in [('driver_float', native_x, host_x), ('driver_integer', native_ix, host_ix)]:
            (directory / (name + '.native.bin')).write_bytes(reference.tobytes())
            (directory / (name + '.cuda.bin')).write_bytes(candidate.tobytes())
        result = json.loads(subprocess.check_output([str(args.comparator), str(directory)], text=True))
        result['stage'] = stage
        return result
    results = [compare('from_mpas')]
    function = native.replay_driver
    function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
    function(native_x.ctypes.data, native_ix.ctypes.data, count)
    function = host.replay_driver
    function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
    function(host_x.ctypes.data, host_ix.ctypes.data, tables.ctypes.data, count)
    results.append(compare('whole_driver'))
    # Verify the reconstructed native inputs against every originally captured
    # surface/soil post-word. Rust alone transposes the opaque io_to ABI.
    function = native.io_to
    function.argtypes = [ct.c_void_p] * 4 + [ct.c_int] * 3 + [ct.c_float] * 2 + [ct.c_void_p]
    function(fields.ctypes.data, categories.ctypes.data, native_x.ctypes.data, native_ix.ctypes.data,
             count, 0, count, arguments['dt_pbl'], arguments['xice_threshold'], native_options.ctypes.data)
    packed_post = args.output / 'fields.post.bin'
    packed_post.write_bytes(fields.tobytes())
    post = args.output / 'native_captured_post'
    post.mkdir(exist_ok=True)
    subprocess.run([str(args.packer), str(packed_post), str(metadata_path), str(count), str(post)], check=True)
    names = []
    for name in noah._MPAS_OUTPUT_FIELDS:
        pool, field = noah._NATIVE_POOL_FIELDS[name]
        bound = '__'.join((pool, field, '1'))
        reference = args.arrays / (bound + '.native.bin')
        if not reference.is_file():
            raise RuntimeError('Captured native output absent: ' + bound)
        (post / (name + '.native.bin')).write_bytes(reference.read_bytes())
        (post / (name + '.cuda.bin')).write_bytes((post / (name + '.post.bin')).read_bytes())
        names.append(name)
    (post / 'fields.txt').write_text('\n'.join(names) + '\n')
    result = json.loads(subprocess.check_output([str(args.comparator), str(post)], text=True))
    result['stage'] = 'native_reconstruction_vs_captured_native_post'
    results.append(result)
    function(host_post_fields.ctypes.data, categories.ctypes.data, host_x.ctypes.data, host_ix.ctypes.data,
             count, 0, count, arguments['dt_pbl'], arguments['xice_threshold'], native_options.ctypes.data)
    packed_host_post = args.output / 'host.fields.post.bin'
    packed_host_post.write_bytes(host_post_fields.tobytes())
    host_post = args.output / 'host_captured_cuda_post'
    host_post.mkdir(exist_ok=True)
    subprocess.run([str(args.packer), str(packed_host_post), str(metadata_path), str(count), str(host_post)], check=True)
    for name in names:
        pool, field = noah._NATIVE_POOL_FIELDS[name]
        bound = '__'.join((pool, field, '1'))
        (host_post / (name + '.native.bin')).write_bytes((args.arrays / (bound + '.cuda.bin')).read_bytes())
        (host_post / (name + '.cuda.bin')).write_bytes((host_post / (name + '.post.bin')).read_bytes())
    (host_post / 'fields.txt').write_text('\n'.join(names) + '\n')
    result = json.loads(subprocess.check_output([str(args.comparator), str(host_post)], text=True))
    result['stage'] = 'host_vs_captured_original_cuda_post'
    results.append(result)
    packet = dict(schema='p16-noah-first-difference-host-glibc-replay-v1', acceptance=False,
                  process_count=1, columns=count, itimestep=arguments['itimestep'],
                  production_sha256=sha(noah.__file__), capture_sha256=sha(args.capture),
                  native_library_sha256=sha(args.native_library), host_library_sha256=sha(args.host_library),
                  packer_sha256=sha(args.packer), comparator_sha256=sha(args.comparator),
                  tables_sha256=metadata_witness['tables_sha256'], transport=transport, results=results,
                  metadata_sha256=sha(args.metadata) if args.metadata else None,
                  metadata_scope=metadata_witness.get('scope', 'actual captured scalar/table words'),
                  candidate_source_sha256=sha(args.candidate_source) if args.candidate_source else None,
                  scope='same sealed complete columns, strict standalone native and full host-glibc bridge; diagnostic only')
    args.receipt.write_text(json.dumps(packet, indent=2) + '\n')
    print(json.dumps(packet, indent=2))
    # The original CUDA is the preserved failing witness: a corrected candidate
    # must differ from it. Only the actual native comparisons gate this replay.
    return int(any(result['differing_words'] for result in results
                   if result['stage'] != 'host_vs_captured_original_cuda_post'))


if __name__ == '__main__':
    raise SystemExit(main())
