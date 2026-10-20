"""CAM callback routing and word-comparison integrity, without GPU acquisition."""
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import struct
import json

import pytest

from hexcore.mpasphys import cam_rad_v842 as cam


def test_native_allocatable_borrows_data_without_overwriting_descriptor():
    import ctypes as ct
    import numpy as np
    # GNU 15 descriptor for a 2x3 real8 table. The words surrounding the
    # descriptor remain distinct from its separately allocated data buffer.
    class Dtype(ct.Structure):
        _fields_=[('elem_len',ct.c_size_t),('version',ct.c_int),
                  ('rank',ct.c_byte),('type',ct.c_byte),('attribute',ct.c_short)]
    class Dimension(ct.Structure):
        _fields_=[('stride',ct.c_ssize_t),('lower',ct.c_ssize_t),('upper',ct.c_ssize_t)]
    class Descriptor(ct.Structure):
        _fields_=[('base',ct.c_void_p),('offset',ct.c_size_t),('dtype',Dtype),
                  ('span',ct.c_ssize_t),('dim',Dimension*2)]
    data=(ct.c_double*6)(1.,2.,3.,4.,5.,6.)
    descriptor=Descriptor(ct.addressof(data),-3,Dtype(8,0,2,3,0),8,
                          (Dimension*2)(Dimension(1,1,2),Dimension(2,1,3)))
    original=bytes(descriptor)
    words=cam._native_allocatable_words(ct.addressof(descriptor),ct.c_double,
                                       ((1,1),(2,3),(1,2)))
    view=np.ctypeslib.as_array(words)
    assert view.tolist()==[1.,2.,3.,4.,5.,6.]
    view[5]=9.
    assert data[5]==9. and bytes(descriptor)==original
    descriptor.dim[1].upper=4
    with pytest.raises(ValueError,match='descriptor differs'):
        cam._native_allocatable_words(ct.addressof(descriptor),ct.c_double,
                                      ((1,1),(2,3),(1,2)))


def test_registration_routes_each_native_selector(monkeypatch):
    entries={}
    class Registry:
        def register(self, slot, scheme, backend, callback):
            key=(slot,scheme,backend)
            assert key not in entries
            entries[key]=callback
    frames=[]
    def provider(call):
        frame=object();frames.append((call,frame));return frame
    executed=[]
    def run(call,frame):
        executed.append((call,frame));return frame
    monkeypatch.setattr(cam,'run_driver',run)
    cam.register_backends(Registry(),provider)
    assert set(entries)=={('radiation_lw','cam_lw','hex-cuda'),
                          ('radiation_sw','cam_sw','hex-cuda')}
    for (slot,scheme,_),callback in entries.items():
        call=SimpleNamespace(slot=slot,scheme=scheme)
        result=callback(call)
        assert frames[-1]==executed[-1]==(call,result)


@pytest.mark.parametrize('slot,scheme', [('radiation_lw','rrtmg_lw'),
                                       ('radiation_sw','cam_lw'),('pbl','cam_sw')])
def test_driver_rejects_misrouted_selector_before_device_use(slot,scheme):
    with pytest.raises(ValueError,match='different MPAS radiation slot'):
        cam.run_driver(SimpleNamespace(slot=slot,scheme=scheme),None)


def test_exact_receipt_detects_one_word_and_preserves_signed_zero(tmp_path):
    path=Path(__file__).parent/'cam_rad_oracle'/'compare_slices.py'
    spec=importlib.util.spec_from_file_location('cam_exact_receipt',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Positive/negative zero and adjacent floats must stay distinct words.
    native=tmp_path/'native.bin';port=tmp_path/'port.bin';receipt=tmp_path/'receipt.json'
    header=struct.pack('>iii',1,1,3)
    native.write_bytes(header+struct.pack('>III',0,0x80000000,0x3f800000))
    port.write_bytes(header+struct.pack('>III',0,0x80000000,0x3f800001))
    groups=[('boundary',[{'name':'field','typ':'float','dims':[':']}],1)]
    assert module.compare(native,port,groups,receipt,'test witness')==1
    result=json.loads(receipt.read_text())
    assert result['words']==3 and result['differing_words']==1
    assert result['groups'][0]['first'][0]['element']==2
    port.write_bytes(header+struct.pack('>III',0,0,0x3f800000))
    assert module.compare(native,port,groups,receipt,'test witness')==1
    assert json.loads(receipt.read_text())['groups'][0]['first'][0]['element']==1


def test_unwritten_scratch_still_counts_in_total_word_result(tmp_path):
    path=Path(__file__).parent/'cam_rad_oracle'/'compare_boundary.py'
    spec=importlib.util.spec_from_file_location('cam_boundary_receipt',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fields=[{'name':'dosw','typ':'bool','dims':[]},
            {'name':'taucldc','typ':'float','dims':[':']}]
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'schema':{'arguments':fields}}))
    native=tmp_path/'native.bin';port=tmp_path/'port.bin';receipt=tmp_path/'receipt.json'
    header=struct.pack('>iiiii',0,0,1,1,1)
    native.write_bytes(header+struct.pack('>I',0x3f800000))
    port.write_bytes(header+struct.pack('>I',0))
    assert module.compare(native,port,inventory,receipt)==1
    result=json.loads(receipt.read_text())
    assert result['total_differing_words']==1
    assert result['defined_differing_words']==0
    assert result['comparison_result']=='FAIL' and result['excluded_words']==0


def test_replay_resume_rejects_changed_artifact_or_lost_device_words(tmp_path):
    import hashlib
    path=Path(__file__).parent/'cam_rad_oracle'/'replay_cuda.py'
    spec=importlib.util.spec_from_file_location('cam_replay_checkpoint',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=('cam-cuda-checkpoint-state.bin','cam-cuda-checkpoint-workspace.bin','cam-cuda-after.bin')
    for name in names:(tmp_path/name).write_bytes(bytes.fromhex('800000003f800001'))
    identity={'artifact':'strict-artifact','input':'complete-native-capture'}
    saved=dict(identity=identity,records=1,input_offset=8,
               files={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names})
    module.validate_checkpoint(saved,identity,tmp_path)
    with pytest.raises(ValueError,match='input/artifact changed'):
        module.validate_checkpoint(saved,dict(identity,artifact='different'),tmp_path)
    (tmp_path/names[1]).write_bytes(bytes.fromhex('000000003f800001'))
    with pytest.raises(ValueError,match='bytes changed'):
        module.validate_checkpoint(saved,identity,tmp_path)


def test_interrupted_replay_recovers_only_authenticated_prefix(tmp_path):
    import hashlib
    path=Path(__file__).parent/'cam_rad_oracle'/'replay_cuda.py'
    spec=importlib.util.spec_from_file_location('cam_replay_recovery',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=('cam-cuda-checkpoint-state.bin','cam-cuda-checkpoint-workspace.bin','cam-cuda-after.bin')
    prefix=bytes.fromhex('800000003f800001')
    for name in names:(tmp_path/name).write_bytes(prefix)
    identity={'artifact':'strict-artifact','input':'complete-native-capture'}
    saved=dict(identity=identity,records=1,input_offset=len(prefix),
               files={name:hashlib.sha256(prefix).hexdigest() for name in names})
    output=tmp_path/names[-1];output.write_bytes(prefix+b'interrupted suffix')
    module.recover_incomplete_tail(saved,identity,tmp_path)
    assert output.read_bytes()==prefix
    assert json.loads((tmp_path/'cam-cuda-interrupted-tail.json').read_text())[0]['discarded_bytes']==18
    output.write_bytes(b'changed!'+b'interrupted suffix')
    with pytest.raises(ValueError,match='bytes changed'):
        module.recover_incomplete_tail(saved,identity,tmp_path)
    assert output.read_bytes()==b'changed!'+b'interrupted suffix'


def test_optional_capture_retains_absence_and_rejects_required_missing_array():
    import io
    path=Path(__file__).parent/'cam_rad_oracle'/'replay_cuda.py'
    spec=importlib.util.spec_from_file_location('cam_optional_capture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    field={'name':'swupt','typ':'float','dims':['ims:ime','jms:jme'],'optional':True}
    raw=struct.pack('>iiiii',2,1,0,1,0)
    assert module.read_value(io.BytesIO(raw),field) is None
    out=io.BytesIO();module.write_value(out,field,None)
    assert out.getvalue()==raw
    with pytest.raises(ValueError,match='Absent nonoptional CAM descriptor'):
        module.read_value(io.BytesIO(raw),dict(field,name='taucldc',optional=False))
    with pytest.raises(ValueError,match='Absent nonoptional CAM descriptor'):
        module.write_value(io.BytesIO(),dict(field,name='taucldc',optional=False),None)
