"""Orchestrate f1 capture replay through the built native library.

Rust owns typed field decoding and exact comparisons. A JSON manifest explicitly
maps each capture to a native pool, module variable, table or argument; this does
not infer missing private/SAVE/derived state or claim forecast acceptance.
"""
from contextlib import ExitStack, nullcontext
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from hexcore.mpasphys.native_backend import NativePhysics


class _Metadata(ct.Structure):
    _fields_ = [('header', ct.c_int32 * 21), ('strings', ct.c_ubyte * 768),
                ('payload', ct.c_void_p), ('payload_bytes', ct.c_size_t), ('ids', ct.c_void_p)]


class Capture:
    """A live Rust-owned capture; returned arrays borrow its storage."""
    def __init__(self, transport, path):
        self.lib, self.path = transport, Path(path)
        self.handle = self.lib.oracle_open(str(self.path.resolve()).encode())
        if not self.handle:
            raise ValueError(f'Rust rejected malformed or unreadable oracle capture: {self.path}')
        self.meta = _Metadata()
        assert self.lib.oracle_metadata(self.handle, ct.byref(self.meta)) == 0
        self.header = tuple(self.meta.header)
        self.shape = tuple(self.header[3:3 + self.header[2]])
        self.lower = tuple(self.header[8:8 + self.header[2]])

    def array(self):
        code, width = self.header[:2]
        dtype = {1: np.dtype('float32'), 2: np.dtype('float64'),
                 3: np.dtype('int32'), 4: np.dtype('int32')}.get(code)
        if dtype is None or dtype.itemsize != width:
            raise TypeError('numeric oracle array required')
        raw = np.ctypeslib.as_array((ct.c_ubyte * self.meta.payload_bytes).from_address(self.meta.payload))
        result = raw.view(dtype).reshape(self.shape, order='F')
        result.flags.writeable = False
        return result

    def scalar(self):
        if self.header[2] != 0:
            raise ValueError('scalar oracle input required')
        if self.header[0] == 5:
            value = self.lib.oracle_text(self.handle)
            if value is None:
                raise ValueError('Rust rejected scalar character capture')
            return value.decode('utf8')
        value = self.array().item()
        return bool(value) if self.header[0] == 4 else value

    def compare(self, value, lower=None):
        if self.header[0] == 5:
            if value.dtype != np.dtype('uint8') or value.ndim != 1 or value.nbytes != self.meta.payload_bytes:
                raise ValueError(f'oracle character storage width differs: {self.path}')
        else:
            if value.shape != self.shape or (lower is not None and tuple(lower) != self.lower):
                raise ValueError(f'oracle output shape/lower bounds differ: {self.path}')
            dtype = {1: np.dtype('float32'), 2: np.dtype('float64'),
                     3: np.dtype('int32'), 4: np.dtype('int32')}[self.header[0]]
            if value.dtype != dtype:
                raise TypeError(f'oracle output precision differs: {self.path}')
        words = np.array(value, order='F', copy=True)
        different = ct.c_size_t()
        status = self.lib.oracle_compare(self.handle, words.ctypes.data, words.nbytes, ct.byref(different))
        if status:
            raise ValueError(f'Rust rejected oracle comparison: status={status}')
        return {'capture': str(self.path), 'capture_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'words': words.nbytes // self.header[1], 'different_words': different.value,
                'word_bytes': self.header[1], 'kind': 'character' if self.header[0] == 5 else 'numeric'}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lib.oracle_close(self.handle)


def transport(library):
    path = Path(library).with_name('libmpas_oracle_transport.so')
    receipt = json.loads(Path(library).with_name('build-receipt.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['oracle_transport_sha256']:
        raise ValueError('Rust oracle transport does not match the native build receipt')
    lib = ct.CDLL(str(path))
    lib.oracle_open.argtypes, lib.oracle_open.restype = [ct.c_char_p], ct.c_void_p
    lib.oracle_metadata.argtypes = [ct.c_void_p, ct.POINTER(_Metadata)]
    lib.oracle_text.argtypes, lib.oracle_text.restype = [ct.c_void_p], ct.c_char_p
    lib.oracle_compare.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
    lib.oracle_compare_owned.argtypes = [ct.c_void_p, ct.c_void_p,
                                        *[ct.POINTER(ct.c_size_t)] * 3]
    lib.oracle_close.argtypes = [ct.c_void_p]
    return lib


def compare_owned(library, reference, candidate):
    """Rust compares raw owned column words using capture global IDs."""
    lib = transport(library)
    with Capture(lib, reference) as a, Capture(lib, candidate) as b:
        cells, words, different = ct.c_size_t(), ct.c_size_t(), ct.c_size_t()
        status = lib.oracle_compare_owned(a.handle, b.handle, ct.byref(cells),
                                          ct.byref(words), ct.byref(different))
        if status:
            raise ValueError(f'Rust rejected global-ID column comparison: status={status}')
        return {'cells': cells.value, 'words': words.value, 'different_words': different.value,
                'reference_sha256': hashlib.sha256(a.path.read_bytes()).hexdigest(),
                'candidate_sha256': hashlib.sha256(b.path.read_bytes()).hexdigest()}


def replay(library, manifest_path, *, native=None):
    """Replay into a fresh context, or an explicitly initialized native context."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'mpas-native-replay-v1':
        raise ValueError('unsupported replay manifest')
    lib = transport(library)
    comparisons = []
    input_captures = []
    context = NativePhysics(library, test_only=True) if native is None else nullcontext(native)
    with context as native:
        if native.build['library_sha256'] != hashlib.sha256(Path(library).read_bytes()).hexdigest():
            raise ValueError('prepared native context differs from the replay library')
        for call in manifest['calls']:
            arguments = dict(call['arguments'])
            with ExitStack() as inputs:
                for mapping in call.get('inputs', []):
                    record = inputs.enter_context(Capture(lib, manifest_path.parent / mapping['capture']))
                    input_captures.append({'capture': str(record.path),
                                           'sha256': hashlib.sha256(record.path.read_bytes()).hexdigest(),
                                           'header': record.header, 'role': mapping['role'], 'name': mapping['name']})
                    role, name = mapping['role'], mapping['name']
                    if role == 'field':
                        native.put_field(mapping['pool'], name, record.array(),
                                         nlevels=mapping.get('nlevels', 1), level=mapping.get('level', 1))
                    elif role == 'text_field':
                        native.put_text_field(mapping['pool'], name, record.scalar(),
                                              nlevels=mapping.get('nlevels', 1),
                                              level=mapping.get('level', 1))
                    elif role == 'variable':
                        native.put_variable(name, record.array(), lower=record.lower)
                    elif role == 'config':
                        native.config(name, record.scalar(), pool=mapping['pool'])
                    elif role == 'dimension':
                        native.dimension(mapping['pool'], name, record.scalar() if not record.shape else record.array())
                    elif role == 'argument':
                        arguments[name] = record.array().copy(order='F') if record.shape else record.scalar()
                    else:
                        raise ValueError(f'unsupported native input role: {role}')
                if call['entry'] == 'block':
                    native.block(**arguments)
                elif call['entry'] == 'clock':
                    native.clock(**arguments)
                elif call['entry'] == 'streams':
                    native.streams()
                else:
                    # Explicit framework references never decode scientific data.
                    arguments = {key: native.framework_object(value['framework'])
                                 if isinstance(value, dict) and 'framework' in value else value
                                 for key, value in arguments.items()}
                    native.call(call['entry'], **arguments)
                for mapping in call.get('outputs', []):
                    with Capture(lib, manifest_path.parent / mapping['capture']) as record:
                        role, name = mapping['role'], mapping['name']
                        lower = None
                        if role == 'field':
                            value = native.get_field(mapping['pool'], name, level=mapping.get('level', 1))
                            lower = (1,) * value.ndim
                        elif role == 'text_field':
                            value = native.get_text_field_storage(mapping['pool'], name, level=mapping.get('level', 1))
                        elif role in {'variable', 'table'}:
                            result = native.get_variable(name) if role == 'variable' else native.get_table(name)
                            if result is None:
                                raise ValueError(f'required captured output is unallocated: {name}')
                            value, lower = result
                        elif role == 'argument':
                            value = arguments[name]
                        else:
                            raise ValueError(f'unsupported native output role: {role}')
                        comparisons.append(record.compare(value, lower))
    return {'schema': 'mpas-native-replay-receipt-v1', 'forecast_proof': False,
            'process_count': manifest.get('process_count', 'unspecified'),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'library_sha256': native.build['library_sha256'], 'calls': len(manifest['calls']),
            'input_captures': input_captures,
            'comparisons': comparisons, 'different_words': sum(v['different_words'] for v in comparisons),
            'input_completeness': manifest.get('input_completeness', 'unassessed')}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    receipt = replay(args.library, args.manifest)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(1 if receipt['different_words'] else 0)


if __name__ == '__main__':
    main()
