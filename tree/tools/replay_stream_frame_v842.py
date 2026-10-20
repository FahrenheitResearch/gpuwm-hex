"""Replay captured native stream storage through the Rust codec.

This is an I/O module proof using actual native field words and metadata.
It does not run physics or demonstrate a forecast through the new adapter.
MPAS v8.4.2 src/framework/mpas_io_streams.F:2430-4363.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from hexcore.restart_v842 import _call, inspect_stream
from hexcore.stream_headers_v842 import compare_headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--format', required=True, choices=('netcdf', 'pnetcdf', 'pnetcdf,cdf5', 'netcdf4'))
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--header-alignment', type=int, default=131072)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.candidate.exists():
        parser.error('candidate must not exist')
    header = inspect_stream(args.native, scratch=args.scratch)
    decoded = _call(dict(operation='read', path=str(args.native.resolve()),
                        output_directory=str((args.scratch/'raw').resolve())), args.scratch)
    request = dict(operation='write', path=str(args.candidate.resolve()), format=args.format,
                   dimensions=decoded['dimensions'], dimension_order=header['dimension_order'],
                   variables=decoded['variables'], global_attrs=decoded['global_attrs'],
                   global_attribute_order=header['global_attribute_order'],
                   header_alignment=args.header_alignment)
    started = perf_counter()
    response = _call(request, args.scratch)
    seconds = perf_counter()-started
    grade = compare_headers(args.native, args.candidate, scratch=args.scratch)
    result = dict(schema='mpas-native-frame-codec-replay-v1',
                  evidence_kind='CAPTURED-NATIVE-STREAM-REPLAY', forecast_proof=False,
                  writer_seconds=seconds, writer=response, **grade)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=grade['passed'], byte_identical=grade['byte_identical'],
                         writer_seconds=seconds, differences=len(grade['differences']))))
    return int(not grade['passed'] or not grade['byte_identical'])


if __name__ == '__main__':
    raise SystemExit(main())
