"""Borrow complete standalone CAM capture words through the real Rust ABI.

Rust owns parsing, byte order, bounds, raw comparisons and encoding. Python
only passes descriptors/buffers between that ABI and the CUDA driver.
"""
import ctypes as ct
import json
import os
from pathlib import Path
import weakref


class View(ct.Structure):
    _fields_=[('rank',ct.c_uint32),('low',ct.c_int32*5),('extent',ct.c_int32*5),
              ('bytes',ct.c_uint64),('next',ct.c_uint64),('data',ct.c_void_p)]


class Codec:
    def __init__(self, path):
        self.path=Path(path).resolve()
        self.lib=ct.CDLL(str(self.path))
        if self.lib.rw_mpas_cam_compact_abi()!=1:
            raise ValueError('Unsupported Rust CAM compact ABI')
        self.lib.rw_mpas_cam_compact_open.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_size_t]
        self.lib.rw_mpas_cam_compact_open.restype=ct.c_void_p
        self.lib.rw_mpas_cam_compact_read.argtypes=[ct.c_void_p,ct.c_uint64,ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_size_t]
        self.lib.rw_mpas_cam_compact_read.restype=ct.c_void_p
        self.lib.rw_mpas_cam_compact_view.argtypes=[ct.c_void_p,ct.POINTER(View)]
        self.lib.rw_mpas_cam_compact_free.argtypes=[ct.c_void_p]
        self.lib.rw_mpas_cam_compact_close.argtypes=[ct.c_void_p]
        self.lib.rw_mpas_cam_compact_encode.argtypes=[ct.c_uint32,ct.c_uint32,ct.POINTER(ct.c_int32),ct.POINTER(ct.c_int32),ct.c_void_p,ct.c_size_t,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_size_t]
        self.lib.rw_mpas_cam_compact_encode.restype=ct.c_void_p
        self.lib.rw_mpas_cam_compact_compare.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_char_p,ct.c_void_p,ct.c_size_t]
        self.lib.rw_mpas_cam_compact_compare.restype=ct.c_void_p
        self.lib.rw_mpas_cam_compact_string_free.argtypes=[ct.c_void_p]
        self.readers=weakref.WeakKeyDictionary()

    def read(self, stream, field):
        import numpy as np
        if stream not in self.readers:
            self.readers[stream]=Reader(self,stream.name)
        reader=self.readers[stream]
        error=ct.create_string_buffer(4096)
        handle=self.lib.rw_mpas_cam_compact_read(reader.handle,stream.tell(),
            {'float':1,'double':2,'int':3,'bool':4}[field['typ']],len(field['dims']),
            bool(field.get('optional')),error,len(error))
        if not handle:raise ValueError(error.value.decode())
        try:
            view=View()
            if self.lib.rw_mpas_cam_compact_view(handle,ct.byref(view)):
                raise ValueError('Rust CAM compact view unavailable')
            stream.seek(view.next)
            if view.rank and not view.bytes:return None
            typ={'float':ct.c_float,'double':ct.c_double,'int':ct.c_int32,'bool':ct.c_int32}[field['typ']]
            # Copy the borrowed storage before releasing its Rust owner. No
            # Python byte decoding or floating-point calculation occurs here.
            words=np.ctypeslib.as_array(ct.cast(view.data,ct.POINTER(typ)),
                                       shape=(view.bytes//ct.sizeof(typ),)).copy()
            if not view.rank:
                return words[0]
            lower=tuple(view.low[:view.rank]);extent=tuple(view.extent[:view.rank])
            stride=[];count=1
            for n in extent:stride.append(count);count*=n
            return words,lower,extent,tuple(stride)
        finally:self.lib.rw_mpas_cam_compact_free(handle)

    def write(self, stream, field, value):
        import numpy as np
        from hexcore.mpasphys.cam_rad_v842 import DeviceArray
        kind={'float':1,'double':2,'int':3,'bool':4}[field['typ']]
        rank=len(field['dims'])
        if value is None:
            if not field.get('optional'):raise ValueError('Absent nonoptional CAM field')
            lower=[1]*5;extent=[0]*rank+[1]*(5-rank)
            words=np.empty(0,dtype={'float':'float32','double':'float64','int':'int32','bool':'int32'}[field['typ']])
        elif isinstance(value,DeviceArray):
            lower=list(value.lower)+[1]*(5-rank);extent=list(value.extent)+[1]*(5-rank)
            words=value.storage.get()
        else:
            lower=extent=[1]*5
            words=np.asarray([value],dtype={'float':'float32','double':'float64','int':'int32','bool':'int32'}[field['typ']])
        self.encode(stream,kind,rank,lower,extent,words,field.get('optional',False))

    def encode(self, stream, kind, rank, lower, extent, words, optional=False):
        if not words.flags.c_contiguous:raise ValueError('CAM compact writer needs contiguous raw words')
        error=ct.create_string_buffer(4096)
        handle=self.lib.rw_mpas_cam_compact_encode(kind,rank,(ct.c_int32*5)(*lower),
            (ct.c_int32*5)(*extent),words.ctypes.data,words.nbytes,words.dtype.itemsize,
            optional,error,len(error))
        if not handle:raise ValueError(error.value.decode())
        try:
            view=View()
            if self.lib.rw_mpas_cam_compact_view(handle,ct.byref(view)):
                raise ValueError('Rust CAM compact encoded view unavailable')
            stream.write(ct.string_at(view.data,view.bytes))
        finally:self.lib.rw_mpas_cam_compact_free(handle)

    def compare(self, reference, candidate, groups):
        error=ct.create_string_buffer(4096)
        handle=self.lib.rw_mpas_cam_compact_compare(os.fsencode(reference),
            os.fsencode(candidate),json.dumps(groups).encode(),error,len(error))
        if not handle:raise ValueError(error.value.decode())
        try:return json.loads(ct.string_at(handle))
        finally:self.lib.rw_mpas_cam_compact_string_free(handle)


class Reader:
    def __init__(self, codec, path):
        self.codec=codec
        error=ct.create_string_buffer(4096)
        self.handle=codec.lib.rw_mpas_cam_compact_open(os.fsencode(path),error,len(error))
        if not self.handle:raise ValueError(error.value.decode())
    def __del__(self):
        if getattr(self,'handle',None):self.codec.lib.rw_mpas_cam_compact_close(self.handle)


_codec=None
def codec():
    global _codec
    if _codec is None:
        path=os.environ.get('HEX_CAM_COMPACT_LIBRARY')
        if not path:raise ValueError('Standalone CAM capture transport requires HEX_CAM_COMPACT_LIBRARY')
        _codec=Codec(path)
    return _codec
