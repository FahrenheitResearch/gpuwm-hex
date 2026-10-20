"""Ordered native stream header proof, using Rust NetCDF inspection.

MPAS v8.4.2 src/framework/mpas_io_streams.F:4086-4363 defines metadata.
This complements the field-word gate, which does not inspect attributes.
"""
import hashlib
from pathlib import Path

from .restart_v842 import inspect_stream


def compare_headers(reference, candidate, *, scratch, written_fields_only=False, ignored_global_attributes=()):
    """Compare stored types, order and attribute words, without array decode.

    MPAS mpas_io_streams.F:4086-4363. The written-field scope projects only
    native-only variables and unused dimensions out of the reference. It does
    not exempt any type, dimension extent or ordering difference. Only named
    global attributes may be excluded explicitly; variable attributes stay strict.
    Full file hashes are also recorded. Equal headers do not imply equal data.
    """
    reference, candidate = Path(reference), Path(candidate)
    headers = [inspect_stream(path, scratch=scratch) for path in (reference, candidate)]
    required = {'format', 'dimension_order', 'variable_order', 'global_attribute_order',
                'global_attribute_words'}
    if any(not required <= h.keys() for h in headers):
        raise ValueError('rw_mpas_stream must be rebuilt with ordered header inspection')
    expected, actual = headers
    differences = []

    def check(label, left, right):
        if left != right:
            differences.append(dict(item=label, native=left, hex=right))

    dims, variables = actual['dimensions'], actual['variables']
    check('format', expected['format'], actual['format'])
    expected_dims = expected['dimension_order']
    expected_vars = expected['variable_order']
    if written_fields_only:
        expected_dims = [d for d in expected_dims if d in dims]
        expected_vars = [v for v in expected_vars if v in variables]
    check('dimension_order', expected_dims, actual['dimension_order'])
    check('variable_order', expected_vars, actual['variable_order'])
    check('dimensions', {d: expected['dimensions'][d] for d in expected_dims}, dims)
    check('unlimited', [d for d in expected['unlimited'] if not written_fields_only or d in dims], actual['unlimited'])
    ignored = frozenset(ignored_global_attributes)
    for key in ('global_attribute_order', 'global_attribute_words'):
        if key.endswith('_order'):
            left, right = ([n for n in h[key] if n not in ignored] for h in headers)
        else:
            left, right = ({n: v for n, v in h[key].items() if n not in ignored} for h in headers)
        check(key, left, right)
    for name in actual['variable_order']:
        if name not in expected['variables']:
            continue  # The variable-order check reports this extra variable.
        left, right = expected['variables'][name], variables[name]
        for key in ('dtype', 'dimensions', 'shape', 'attribute_order', 'attribute_words'):
            check(f'{name}.{key}', left[key], right[key])
    hashes = []
    for path in (reference, candidate):
        with path.open('rb') as source:
            hashes.append(hashlib.file_digest(source, 'sha256').hexdigest())
    return dict(passed=not differences, scope='written fields' if written_fields_only else 'complete file',
                ignored_global_attributes=sorted(ignored),
                native=str(reference), hex=str(candidate), native_sha256=hashes[0],
                hex_sha256=hashes[1], byte_identical=hashes[0] == hashes[1], differences=differences)
