"""hexcore.nest_boundary: a regional nest's boundaries from its parent's forecast trajectory."""
from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset
from hexcore.lbc_stream import checked_marker, READY
from test_analysis_remap import fields

from hexcore import nest_boundary as module


def test_boundary_publication_preserves_native_staggering_and_all_water(tmp_path):
    state, saved, vertical = fields()
    moment = datetime(2022, 6, 14, 0, 15)
    document = module.publish_frame(tmp_path, moment, state, saved, vertical, module.transfer.SPECIES,
                                    dict(parent_sha256='a'*64))
    path = tmp_path/document['path']
    assert checked_marker(path.with_name(path.name+'.ready.json'), schema=READY) == path
    assert module.valid_time(path) == moment
    with Dataset(path) as d:
        np.testing.assert_array_equal(d['lbc_u'][0].T, saved.normal_velocity)
        np.testing.assert_array_equal(d['lbc_w'][0].T, saved.vertical_velocity)
        np.testing.assert_array_equal(d['lbc_rho'][0].T, state.rho * vertical.zz)
        for index, name in enumerate(module.transfer.SPECIES):
            np.testing.assert_array_equal(d['lbc_'+name][0].T, state.scalars[index])
        np.testing.assert_allclose(d['lbc_theta'][0], 299, rtol=1e-7)


def test_old_history_capsule_clock_is_parsed_without_rewriting_parent(tmp_path):
    path = tmp_path/'cuda-history.2022-06-14_00.30.00.nc'
    with Dataset(path, 'w') as d:
        d.createDimension('Time', 1)
    assert module.valid_time(path) == datetime(2022, 6, 14, 0, 30)


def test_streamed_parent_requires_complete_bytes_and_source_digest(tmp_path):
    import hashlib
    path=tmp_path/'parent.nc'
    payload=b'complete parent endpoint'
    record=dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    path.write_bytes(payload[:4])
    with pytest.raises(TimeoutError, match='incomplete'):
        module.wait_parent(record,0.)
    path.write_bytes(b'x'*len(payload))
    with pytest.raises(ValueError, match='SHA-256'):
        module.wait_parent(record,0.)
    path.write_bytes(payload)
    module.wait_parent(record,0.)
