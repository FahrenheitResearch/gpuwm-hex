"""Fault controls for transport words that can invalidate native replay proof."""
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest


@pytest.fixture
def transport():
    path = os.environ.get('P21_NATIVE_TAGGED_TRANSPORT')
    if not path:
        pytest.skip('requires the built, authenticated Rust native tagged transport')
    return str(Path(path).resolve())


def record(order, name=b'column/in/words', words=(0, 0x80000000, 1, 0x7fc00001, 0x7fc00002)):
    return (struct.pack(order + 'i', len(name)) + name +
            struct.pack(order + '3i', 0, 1, len(words)) +
            struct.pack(order + str(len(words)) + 'I', *words))


@pytest.mark.parametrize('order,magic', [('<', b''), ('>', b'SWD1')])
def test_exact_subnormal_signed_zero_and_nan_payloads(transport, tmp_path, order, magic):
    source, out = tmp_path / 'native.bin', tmp_path / 'views'
    source.write_bytes(magic + record(order))
    run = subprocess.run([transport, str(source), str(out)], capture_output=True)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((out / 'manifest.json').read_text())
    assert len(manifest['records']) == 1
    meta = manifest['records'][0]
    assert meta['shape'] == [5] and meta['dtype'] == 0
    assert (out / meta['file']).read_bytes() == struct.pack('<5I', 0, 0x80000000, 1, 0x7fc00001, 0x7fc00002)


@pytest.mark.parametrize('fault', ['truncated', 'duplicate', 'unsafe_name', 'negative_rank'])
def test_corrupted_native_stream_cannot_produce_complete_manifest(transport, tmp_path, fault):
    data = record('<')
    if fault == 'truncated':
        data = data[:-1]
    elif fault == 'duplicate':
        data += data
    elif fault == 'unsafe_name':
        data = record('<', b'column/../words')
    else:
        name = b'column/in/words'
        data = struct.pack('<i', len(name)) + name + struct.pack('<2i', 0, -1)
    source, out = tmp_path / 'native.bin', tmp_path / 'views'
    source.write_bytes(data)
    run = subprocess.run([transport, str(source), str(out)], capture_output=True)
    assert run.returncode != 0
    assert not (out / 'manifest.json').exists()
