"""Expose authenticated PRE records through the native backend's transfer ABI.

No scientific calculation or expected POST is available to this fixture.
Scalar aliases below are the actual keyword values in pinned fork driver_lsm
1010-1035, driver_seaice 787-811 and sfc_diagnostics 138-143. Numeric record I/O
remains Rust's responsibility. Full native shapes and bounds are preserved.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np


@dataclass
class CapturedCall:
    dt: float = 0
    itimestep: int = 0
    slot: str = 'land_surface'
    scheme: str = 'sf_ruc'
    options: dict = field(default_factory=dict)
    pools: dict = field(default_factory=dict)


class CapturedNative:
    def __init__(self, records, headers):
        self.records, self.headers = records, headers
        self._fields = {}
        self.reads, self.writes = [], []
        for name, value in records.items():
            pool, _, key = name.partition('.')
            if key.endswith('.t1'):
                self._fields[pool,key[:-3]] = (value.dtype,value.shape,1)

    def get_field(self, pool, name, *, level=1):
        assert level == 1
        key = pool+'.'+name+'.t1'
        self.reads.append(key)
        return self.records[key]

    def get_text_field(self, pool, name):
        return bytes(self.get_field(pool,name)).decode('ascii').rstrip()

    def put_field(self, pool, name, value, *, level=1, nlevels=1):
        assert level == nlevels == 1
        key = pool+'.'+name+'.t1'
        self._put(key,value)

    def _put(self, key, value):
        host = value.get() if hasattr(value,'get') else np.asarray(value)
        original = self.records[key]
        assert host.dtype == original.dtype and host.shape == original.shape, (key,host.dtype,host.shape,original.dtype,original.shape)
        self.records[key] = host.copy(order='F')
        self.writes.append(key)

    def get_variable(self, name):
        key = 'physics.'+name
        if key not in self.records:
            keyword = {'num_soils':'nsl','num_landcat':'nlcat','num_soilcat':'nscat',
                       'dt_pbl':'dt'}.get(name,name)
            key = 'scheme.'+keyword
        self.reads.append(key)
        header = self.headers[key]
        return self.records[key],tuple(header[8:8+header[2]])

    def put_variable(self, name, value, *, lower):
        key = 'physics.'+name
        assert tuple(lower) == tuple(self.headers[key][8:8+self.headers[key][2]])
        self._put(key,value)

    def get_constant(self, name):
        keyword = {'cp':'cp','R_d':'r_d','rcp':'rovcp','gravity':'g0',
                   'xlv':'lv','stbolt':'stbolt'}[name]
        key = 'scheme.'+keyword
        self.reads.append(key)
        return self.records[key].item()

    def get_table(self, name):
        key = 'state.'+name
        self.reads.append(key)
        header = self.headers[key]
        return self.records[key],tuple(header[8:8+header[2]])


def native_provider(pre, headers, cp, *, slot):
    options = {name.removeprefix('configs.'):
               bytes(v).decode('ascii').rstrip() if v.dtype == np.uint8 else v.item()
               for name,v in pre.items() if name.startswith('configs.')}
    # The observer captured the value passed to the scheme even where the
    # framework configuration object was not separately recorded.
    if slot == 'land_surface':
        options['config_lakemodel'] = pre['scheme.lakemodel'].item()
    native = CapturedNative(pre,headers)
    args = SimpleNamespace(its=int(pre['scheme.its'].item()),ite=int(pre['scheme.ite'].item()))
    provider = SimpleNamespace(native=native,cp=cp,
                session=SimpleNamespace(layout=SimpleNamespace(configs=options,dimensions={}),arguments=args))
    call = CapturedCall(dt=pre['scheme.dt'].item() if 'scheme.dt' in pre else 0,
                        itimestep=int(pre['argument.itimestep'].item()) if 'argument.itimestep' in pre else 0,
                        slot=slot,options=options)
    return provider,call
