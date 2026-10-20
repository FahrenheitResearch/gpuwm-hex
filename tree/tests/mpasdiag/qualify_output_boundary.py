"""Qualify the Rust boundary word comparator on sealed native payloads."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                           'tools/native_physics_reference/backend'))
    from oracle_replay import Capture, _Metadata
    library = ct.CDLL(str(args.library.resolve()))
    library.oracle_open.argtypes, library.oracle_open.restype = [ct.c_char_p], ct.c_void_p
    library.oracle_metadata.argtypes = [ct.c_void_p, ct.POINTER(_Metadata)]
    library.oracle_close.argtypes = [ct.c_void_p]
    library.oracle_compare_live_owned.argtypes = [ct.c_void_p, ct.c_void_p,
        ct.c_size_t, *[ct.POINTER(ct.c_size_t)] * 3]
    rows = []
    for name, item in json.loads(args.selection.read_text())['files'].items():
        path = args.captures/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with Capture(library, path) as capture:
            payload = (ct.c_ubyte*capture.meta.payload_bytes)()
            ct.memmove(payload, capture.meta.payload, capture.meta.payload_bytes)
            def compare():
                cells, words, different = ct.c_size_t(), ct.c_size_t(), ct.c_size_t()
                code = library.oracle_compare_live_owned(capture.handle, payload,
                    len(payload), ct.byref(cells), ct.byref(words), ct.byref(different))
                assert code == 0
                return dict(cells=cells.value, words=words.value, different_words=different.value)
            exact = compare()
            assert exact['different_words'] == 0 and exact['cells'] > 0
            # Fixture corruption acts on raw bytes, without atmospheric maths.
            payload[0] ^= 1
            corrupted = compare()
            assert corrupted['different_words'] == 1
            payload[0] ^= 1
            payload[-1] ^= 1
            dummy = compare()
            assert dummy['different_words'] == 0
            rows.append(dict(capture=name, exact=exact, physical_corruption=corrupted,
                             unowned_dummy_corruption=dummy))
    args.receipt.write_text(json.dumps(dict(schema='mpas-o2.output-boundary-comparator-test/v1',
        passed=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        files=len(rows), compared_words=sum(row['exact']['words'] for row in rows),
        comparisons=rows), indent=2)+'\n')
    print('Rust boundary comparator:', len(rows), 'native payloads; exact, physical corruption and dummy checks pass')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
