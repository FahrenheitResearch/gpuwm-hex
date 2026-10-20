"""Exercise the Rust native-file bridge without requesting a GPU slot."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from surface_native_v842 import (load_real_land_use, load_real_surface_fields,
                                surface_reader, replay_fields)
from test_surface_update_v842 import initial, load_real_monthly


@pytest.fixture
def native_input():
    path = os.environ.get('P25_REAL_MONTHLY_INPUT')
    library = os.environ.get('P25_MONTHLY_READER_LIB')
    if not path or not library:
        pytest.skip('native reader checks require an input and prebuilt Rust library')
    return path, library


def test_generic_reader_matches_real_monthly_words(native_input):
    path, library = native_input
    first = int(os.environ.get('P25_FIRST_CELL', '0'))
    expected, actual = initial(), initial()
    load_real_monthly(expected, path, library, first=first)
    reader = surface_reader(library)
    want, got = replay_fields(expected), replay_fields(actual)
    words = 0
    for key in ('greenfrac', 'albedo12m', 'landmask'):
        output = got[key]
        assert reader(os.fsencode(path), key.encode(), 4 if key == 'landmask' else 5,
                      None if key == 'landmask' else b'nMonths',
                      1 if key == 'landmask' else 12, first, 256, 0,
                      output.ctypes.data, output.size) == 0
        np.testing.assert_array_equal(output.view('u4'), want[key].view('u4'))
        words += output.size
    receipt = os.environ.get('P25_READER_RECEIPT')
    if receipt:
        import hashlib
        with open(path, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        Path(receipt).write_text(json.dumps(dict(
            tier='Rust decoder consistency on native monthly input; no new CUDA/Fortran claim',
            source=path, source_sha256=digest, first_cell=first,
            carriers=['greenfrac', 'albedo12m', 'landmask'],
            words=words, differing_words=0), indent=2) + '\n')


def test_native_scalar_land_use_matches_oracle(native_input):
    path, library = native_input
    values, sources = load_real_land_use([path], library)
    assert values == dict(isice_lu=24, iswater_lu=16)
    assert sources == dict(isice_lu=path, iswater_lu=path)


def test_incomplete_native_file_cannot_claim_real_surface_replay(native_input):
    path, library = native_input
    # This dry native input has real static terrain/monthly fields but lacks
    # physics state. A complete SST replay must fail instead of filling it.
    with pytest.raises(ValueError, match='native surface input missing vegfra; no synthetic replacement'):
        load_real_surface_fields(initial(), [path], library, 2,
                                 first=int(os.environ.get('P25_FIRST_CELL', '0')))
