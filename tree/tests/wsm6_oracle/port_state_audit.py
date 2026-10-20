"""Compare actual port initialization against every sealed native state record.

Rust decodes/authenticates ORCL0001, Python binds its buffers to the test C ABI.
This is a host-glibc whole-module state witness, not official CUDA replay.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', type=Path, required=True)
    p.add_argument('--host-library', type=Path, required=True)
    p.add_argument('--capture', nargs=2, action='append', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    lib = ct.CDLL(str(args.host_library.resolve()))
    snapshot = lib.wsm6_state_host
    snapshot.argtypes = [ct.POINTER(ct.c_uint32), ct.c_int, ct.c_char_p,
                         ct.POINTER(ct.c_ubyte), ct.c_int]
    snapshot.restype = ct.c_int
    captures = []
    for root, prefix in args.capture:
        fixture = json.loads(subprocess.check_output(
            [str(args.transport.resolve()), 'state', root, prefix], text=True))
        constants = (ct.c_uint32 * 26)(*fixture['constants'])
        rows = {}
        for member, native in fixture['records'].items():
            expected = native['bytes']
            output = (ct.c_ubyte * len(expected))()
            module, name = member.split('.', 1)
            assert module in ('mp_wsm6', 'mp_radar')
            size = snapshot(constants, 0, name.encode(), output, len(expected))
            assert size == len(expected), (member, size, len(expected))
            width = native['header'][1]
            actual = bytes(output)
            expected = bytes(expected)
            differences = [i // width for i in range(0, len(expected), width)
                           if expected[i:i+width] != actual[i:i+width]]
            rows[member] = {'words': len(expected)//width, 'different_words': len(differences),
                            'first_difference': differences[0] if differences else None,
                            'native_record_sha256': native['sha256'],
                            'port_payload_sha256': hashlib.sha256(actual).hexdigest()}
        captures.append({'capture': prefix, 'dump_root_sha256': fixture['dump_root_sha256'],
                         'records': rows, 'different_words': sum(r['different_words'] for r in rows.values())})
    result = {'scope': __doc__.strip(), 'host_library_sha256': sha(args.host_library),
              'transport_sha256': sha(args.transport), 'captures': captures}
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    different = sum(c['different_words'] for c in captures)
    print('port hidden-state host witness:', different, 'different words')
    return int(different != 0)


if __name__ == '__main__':
    raise SystemExit(main())
