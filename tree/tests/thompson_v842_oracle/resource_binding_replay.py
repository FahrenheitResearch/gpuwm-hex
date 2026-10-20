"""Replay the actual f4 Rust table-resource loader on CPU.

This transport witness loads all33 strict native arrays through the compiled
Rust decoder and authenticates the bytes against the standalone preparation.
The mathematics packet is test-only and explicitly diagnostic. No CUDA or
forecast identity is inferred from resource transport.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from hexcore.mpasphys import thompson_v842 as port
from replay import tables, engine_math_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(provider_source, directory, native_tables, auxiliary, engine, packet, receipt):
    spec = importlib.util.spec_from_file_location('hexcore.mpasphys._thompson_resource_witness', provider_source)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    source = packet.with_suffix('.diagnostic.cuh')
    source.write_text(engine_math_source(engine.resolve(), promote_log10=True))
    resources = dict(table_directory=str(native_tables.resolve()), auxiliary=str(auxiliary.resolve()),
        rust_decoder=dict(path=str((directory/'lookup_decode.so').resolve()),sha256=digest(directory/'lookup_decode.so')),
        device_math_source=dict(path=str(source.resolve()),sha256=digest(source)))
    packet.write_text(json.dumps({'mp_thompson':resources},indent=2)+'\n')
    loaded = provider.load_thompson_resources(resources,np)
    native = tables(directory)
    comparisons = []
    for name, shape in port.TABLE_SHAPES.items():
        value = loaded['tables'][name]
        assert value.shape == shape and value.flags.f_contiguous
        expected = native[name]
        actual_sha = hashlib.sha256(value.tobytes(order='F')).hexdigest()
        expected_sha = hashlib.sha256(expected.tobytes(order='F')).hexdigest()
        assert value.dtype == expected.dtype and actual_sha == expected_sha, name
        comparisons.append(dict(name=name, shape=shape, precision=str(value.dtype),bytes=value.nbytes,
            reference_sha256=expected_sha,resource_sha256=actual_sha,identical=True))
    assert loaded['tables']['_file_sha256'] == port.TABLE_FILE_SHA256
    assert loaded['device_math_source'] == source.read_text()
    result = dict(schema='mpas-thompson-resource-binding-witness/v1',acceptance=False,process_count=1,
        scope='Actual f4/Rust CPU transport; diagnostic maths packet, no CUDA or forecast',
        provenance={str(p):digest(p) for p in (provider_source,Path(__file__),Path(port.__file__),
                                              auxiliary,directory/'lookup_decode.so',packet,source)},
        tables=comparisons,all_tables_identical=True)
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tables=len(comparisons),all_tables_identical=True,acceptance=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('provider_source','directory','native_tables','auxiliary','engine','packet','receipt'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    replay(**vars(parser.parse_args()))
