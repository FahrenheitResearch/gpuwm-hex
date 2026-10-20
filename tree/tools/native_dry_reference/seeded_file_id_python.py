"""Run a proof script/module with a declared MPAS stream RNG input.

This changes only the stream metadata clock lookup, before writer creation.
It neither reads native output nor changes any generated file. Production
code has no seed option. Use file_id_clock.c with the same seed for native.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--script')
    target.add_argument('--module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.seed <= 0xffffffff:
        parser.error('seed must fit unsigned int32')
    from hexcore.mpasphys import stream_metadata_v842 as metadata
    metadata.time = lambda: args.seed
    print(json.dumps(dict(proof_only=True, file_id_seed=args.seed,
        metadata_source=str(Path(metadata.__file__).resolve()),
        metadata_sha256=hashlib.sha256(Path(metadata.__file__).read_bytes()).hexdigest())),
        file=sys.stderr, flush=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [args.script or args.module, *arguments]
    if args.script:
        runpy.run_path(args.script, run_name='__main__')
    else:
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
