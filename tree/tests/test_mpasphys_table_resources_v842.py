"""Certified Rust record transport; no CUDA mathematics or forecast claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys.table_resources_v842 import RustLookupRecords, load_thompson_tables, certified_path


def decoder():
    path=os.environ.get('MPAS_THOMPSON_RUST_DECODER')
    if not path:pytest.skip('explicit published Rust decoder is required')
    path=Path(path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_rust_records_preserve_special_words_and_validate_record_extents(tmp_path):
    reader=RustLookupRecords(decoder())
    words=np.array([0x8000000000000000,0x7ff8123456789abc,0x0000000000000001],np.uint64)
    payload=(24).to_bytes(4,'big')+words.astype('>u8').tobytes()+(24).to_bytes(4,'big')
    path=tmp_path/'records';path.write_bytes(payload)
    arrays=reader.arrays(path,['test'],{'test':(3,)},records=True,little=False)
    assert np.array_equal(arrays['test'].view(np.uint64),words)
    with pytest.raises(ValueError,match='extent'):reader.arrays(path,['test'],{'test':(2,)},records=True,little=False)
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError,match='truncated'):reader.arrays(path,['test'],{'test':(3,)},records=True,little=False)


def test_resource_hash_is_verified_before_library_or_packet_is_used(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256 differs'):certified_path({'path':path,'sha256':'0'*64})


def test_complete_certified_native_thompson_tables(tmp_path):
    resources=os.environ.get('MPAS_THOMPSON_TABLE_PACKET')
    if not resources:pytest.skip('explicit published strict table packet is required')
    resources=json.loads(Path(resources).read_text())
    spec=importlib.util.spec_from_file_location('strict_thompson_shapes',resources.pop('scheme_source'))
    scheme=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheme)
    metadata=json.loads(Path(resources.pop('auxiliary_metadata')).read_text())
    tables=load_thompson_tables(resources,np,scheme)
    assert set(tables)==set(scheme.TABLE_SHAPES)|{'_file_sha256'}
    assert tables['_file_sha256']==scheme.TABLE_FILE_SHA256
    for name,shape in scheme.TABLE_SHAPES.items():
        assert tables[name].shape==shape and tables[name].flags.f_contiguous
        assert tables[name].dtype==np.dtype('float32' if name=='tnccn_act' else 'float64')
    for row in metadata['layout']:
        array=tables[row['name']].ravel(order='F')
        assert hashlib.sha256(memoryview(array).cast('B')).hexdigest()==row['decoded_payload_sha256']
    receipt={'scope':'all 33 strict table arrays transported by Rust; six auxiliary payload hashes match; no CUDA or forecast identity',
             'table_count':len(scheme.TABLE_SHAPES),'decoder':resources['rust_decoder'],
             'table_certificate':tables['_file_sha256'],'auxiliary_sha256':metadata['sha256'],
             'auxiliary_words_match':True}
    destination=os.environ.get('MPAS_THOMPSON_TABLE_RECEIPT')
    if destination:Path(destination).write_text(json.dumps(receipt,indent=2)+'\n')
