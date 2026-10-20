"""Adapters for explicit pristine CPU stand-ins used by the forecast referee."""
import ctypes as ct
import hashlib
import json
from pathlib import Path


class NativeCompanions:
    def __init__(self, native, library):
        self.native = native
        path = Path(library)
        self.receipt = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.receipt['library_sha256']:
            raise ValueError('companion provider bytes changed')
        if hashlib.sha256(Path(native.path).read_bytes()).hexdigest() != self.receipt['provider_sha256']:
            raise ValueError('companion framework provider differs')
        self.lib = ct.CDLL(str(path.resolve()))
        self.lib.o2_companions_init.argtypes = [ct.c_void_p]*6
        self.lib.o2_companions_init.restype = None
        self.lib.o2_companion_call.argtypes = [ct.c_int, ct.c_int]
        self.lib.o2_companion_call.restype = None
        self.lib.o2_companions_init(native._runtime, *(native.pool(name) for name in
            ('state', 'diag', 'tend', 'tend_physics', 'configs')))

    def adapters(self):
        return {name: NativeCompanion(self, index) for index, name in
                enumerate(('cloud', 'convective', 'pv', 'soundings'), 1)}


class NativeCompanion:
    def __init__(self, owner, index):
        self.owner, self.index = owner, index

    def setup(self, *args):
        self.owner.lib.o2_companion_call(self.index, 0)

    def update(self):
        self.owner.lib.o2_companion_call(self.index, 1)

    def compute(self):
        self.owner.lib.o2_companion_call(self.index, 2)

    def reset(self):
        self.owner.lib.o2_companion_call(self.index, 3)

    def cleanup(self):
        self.owner.lib.o2_companion_call(self.index, 4)
