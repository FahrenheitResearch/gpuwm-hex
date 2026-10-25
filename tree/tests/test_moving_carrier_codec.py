import io
import pickle
import time

import numpy as np
import pytest

from hexcore import moving_carrier_codec as codec
from hexcore.moving_checkpoint import _seal, load_boundary


def test_exact_buffers_aliases_orders_and_blocks(monkeypatch):
    monkeypatch.setattr(codec, "BLOCK", 31)
    monkeypatch.setattr(codec, "threads", lambda: 3)
    bits = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000,
                     1, 0x3f800001, 0x00800000], dtype=np.uint32)
    array = np.tile(bits, 17).view(np.float32)
    readonly = array.copy()
    readonly.flags.writeable = False
    payload = dict(a=array, alias=array, f=np.asfortranarray(array.reshape(8, 17)),
                   stride=array[::3], empty=np.empty((3, 0), dtype='>f8'),
                   readonly=readonly, text='checkpoint', scalar=np.float64(-0.0))
    stream = io.BytesIO()
    record = codec.dump(payload, stream)
    assert record['workers'] == 3
    stream.seek(0)
    restored = codec.load(stream)
    assert restored['a'] is restored['alias']
    assert restored['f'].flags.f_contiguous
    assert not restored['readonly'].flags.writeable
    for name in ('a', 'f', 'stride', 'empty', 'readonly'):
        assert restored[name].dtype == payload[name].dtype
        assert restored[name].shape == payload[name].shape
        assert restored[name].tobytes(order='A') == payload[name].tobytes(order='A')
    assert restored['scalar'].tobytes() == payload['scalar'].tobytes()


def test_compressed_size_and_truncation(monkeypatch):
    monkeypatch.setattr(codec, 'threads', lambda: 2)
    source = dict(state=np.zeros((1024, 1024), dtype=np.float64))
    stream = io.BytesIO()
    codec.dump(source, stream)
    data = stream.getvalue()
    assert len(data) < source['state'].nbytes / 50
    for cut in (9, 24, len(data)-1):
        with pytest.raises(ValueError, match='truncated'):
            codec.load(io.BytesIO(data[:cut]))
    with pytest.raises(ValueError, match='trailing'):
        codec.load(io.BytesIO(data+b'bad'))


def test_legacy_read_and_receipt_validation(tmp_path):
    payload = dict(schema='gpuwm-hex.moving-boundary/v1', time_seconds=21600,
                   field=np.arange(100, dtype=np.float32))
    assert np.array_equal(codec.load(io.BytesIO(pickle.dumps(payload, protocol=5)))['field'],
                          payload['field'])
    path = tmp_path/'boundary.pkl'
    receipt = _seal(path, payload, [], {'field':payload['field']}, time.perf_counter())
    assert receipt['storage']['codec'] == 'protocol5-zlib-blocks-v1'
    assert np.array_equal(load_boundary(path)['field'], payload['field'])
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='sealed receipt'):
        load_boundary(path)
