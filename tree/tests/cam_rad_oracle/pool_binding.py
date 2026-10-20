"""Emit CAM native pool bindings from the pinned driver declarations.

Python copies ABI storage only. Native framework owns initialization and halo
preparation; CAM arithmetic and conversions execute through the device port.
"""
import re
import translate as tr


def metadata(source, fields):
    symbols = {}
    for filename in ('module_ra_cam_support.F', 'module_ra_cam.F'):
        module = None
        for line, statement in tr.statements(source / filename):
            if statement.startswith('module '):
                module = statement.split()[1]
            elif statement == 'contains':
                module = None
            if module:
                for variable in tr.declaration(statement) or ():
                    symbols[variable.name] = (module, filename, line)
    symbols = {v.name: symbols[v.name] for v in fields}
    pools = {}
    for suffix in ('lw', 'sw'):
        path = source / ('mpas_atmphys_driver_radiation_' + suffix + '.F')
        for line, statement in tr.statements(path):
            match = re.fullmatch(r"callmpas_pool_get_array\((\w+),'(\w+)',(\w+)(?:,time_lev)?\)",
                                 statement.replace(' ', ''))
            if match:
                pool, field, name = match.groups()
                binding = (pool, field)
                if name in pools and pools[name] != binding:
                    raise ValueError((name, pools[name], binding))
                pools[name] = binding
    path = source / 'mpas_atmphys_driver_radiation_lw.F'
    call = next(statement for _, statement in tr.statements(path)
                if statement.startswith('call camrad('))
    arguments = dict(value.strip().split('=', 1)
                     for value in tr.split(call[call.index('(')+1:-1]))
    arguments = {k.strip(): v.strip() for k,v in arguments.items()}
    return symbols, pools, arguments, tuple(v.name for v in fields if not v.const)


def emit(source, fields):
    symbols, pools, arguments, mutable = metadata(source, fields)
    import hashlib
    pins = {n:hashlib.sha256((source / n).read_bytes()).hexdigest()
            for n in ('module_ra_cam_support.F', 'module_ra_cam.F')}
    return ('\nNATIVE_SOURCE_PINS = ' + repr(pins) + '\nNATIVE_STATE_SYMBOLS = ' + repr(symbols) +
            '\nNATIVE_POOL_BINDINGS = ' + repr(pools) +
            '\nNATIVE_ARGUMENT_BINDINGS = ' + repr(arguments) +
            '\nNATIVE_STATE_MUTABLE = ' + repr(mutable) + '\n' + API)


API = r'''

class NativeCAMWords:
    """Raw strict GNU module storage, module_ra_cam_support.F:22-383.

    The native stand-in initializes constants/tables once. These copies retain
    its exact kind and every mutable module word across mixed CPU/CUDA slots.
    This GNU binding is restricted to f3's authenticated v8.4.2 library; it
    evaluates no physics and is not an independent forecast acceptance claim.
    """
    def __init__(self, native):
        """Pristine globals; module_ra_cam.F:1-176; module_ra_cam_support.F:22-383."""
        import ctypes as ct
        import hashlib
        import math
        import numpy as np
        self.native = native
        sources = native.build['sources']
        for filename, digest in NATIVE_SOURCE_PINS.items():
            key = 'src/core_atmosphere/physics/physics_wrf/' + filename
            if sources.get(key) != digest:
                raise ValueError('CAM native module source differs: ' + filename)
        if native.build['source_pin'] != 'f34984b2c89353cf31c235b08d3b4acd96b37bb3':
            raise ValueError('CAM native initialized tables require pinned MPAS v8.4.2')
        self.views = {}
        for name, kind, rank in STATE_FIELDS:
            module, _, _ = NATIVE_STATE_SYMBOLS[name]
            scalar = {'float':ct.c_float, 'double':ct.c_double,
                      'int':ct.c_int32, 'bool':ct.c_int32}[kind]
            count = math.prod(STATE_LAYOUT[name][1]) if rank else 1
            native_words = (scalar * count).in_dll(native.lib, '__' + module + '_MOD_' + name)
            self.views[name] = np.ctypeslib.as_array(native_words)

    def read(self):
        """Copy every initialized module word; module_ra_cam_support.F:22-383."""
        return {name: words.copy() for name, words in self.views.items()}

    def write(self, values):
        """Return every mutable module word; module_ra_cam.F:424-443,709-710."""
        if set(values) != set(self.views):
            raise ValueError('CAM native module transfer requires its whole state')
        for name, words in self.views.items():
            value = values[name]
            if value.shape != words.shape or value.dtype != words.dtype:
                raise ValueError('CAM native module transfer kind/extent differs: ' + name)
        for name, words in self.views.items():
            if name not in NATIVE_STATE_MUTABLE and values[name].tobytes() != words.tobytes():
                raise ValueError('CAM transfer attempted to change a native PARAMETER: ' + name)
        for name in NATIVE_STATE_MUTABLE:
            self.views[name][:] = values[name]


@dataclass(frozen=True)
class NativeCAMFrame:
    """Owned ABI copy transaction; mpas_atmphys_driver_radiation_lw.F:804-960."""
    frame: DriverFrame
    owner: object


class _NativeCAMTransfer:
    """Retain raw Fortran bounds and shared aliases; mpas_atmphys_vars.F:179-217,770-803."""
    def __init__(self, provider, call):
        """Prepared arrays follow native allocation; mpas_atmphys_driver.F:185-244."""
        self.provider, self.native, self.cp, self.call = provider, provider.native, provider.cp, call
        self.copies = {}
        self.words = NativeCAMWords(self.native)
        # MPAS src/core_atmosphere/physics/mpas_atmphys_utilities.F:18; mpas_kind_types.F:32 (StrKIND=512).
        import ctypes as ct
        import numpy as np
        words=(ct.c_ubyte*512).in_dll(self.native.lib,
            '__mpas_atmphys_utilities_MOD_mpas_err_message')
        self.message_words=np.ctypeslib.as_array(words)
        self.utility_state={'mpas_err_message':self.cp.asarray(self.message_words)}

    def variable(self, name):
        """Prepared i,k,j descriptors, including rank4; mpas_atmphys_vars.F:179-217,770-803."""
        key = ('prepared', name)
        if key in self.copies:
            return self.copies[key][1]
        original = self.native.get_variable(name)
        if original is None:
            raise RuntimeError('CAM native driver preparation omitted allocated variable ' + name)
        host, lower = original
        if host.ndim == 0:
            return host[()]
        return self._copy(key, host, lower)

    def field(self, pool, name):
        """Original pool order includes halos; mpas_atmphys_driver_radiation_lw.F:343-362,646-658."""
        name = next(field for p, field in self.native._fields
                    if p == pool and field.casefold() == name.casefold())
        level = self.provider.level if pool == 'state' else 1
        key = (pool, name, level)
        if key in self.copies:
            return self.copies[key][1]
        host = self.native.get_field(pool, name, level=level)
        return self._copy(key, host, (1,) * host.ndim)

    def _copy(self, key, host, lower):
        """No numerical conversion of pool words; mpas_atmphys_vars.F:179-217,770-803."""
        stride, count = [], 1
        for size in host.shape:
            stride.append(count)
            count *= size
        value = DeviceArray(self.cp.asarray(host.ravel(order='F')), tuple(lower),
                            tuple(host.shape), tuple(stride))
        self.copies[key] = (host, value, tuple(lower))
        return value

    def argument(self, name):
        """Native pool retrieval and scalar bounds; mpas_atmphys_driver_radiation_lw.F:343-362,905-951."""
        if name == 'its':
            return 1
        if name == 'ite':
            return self.provider.session.layout.dimensions['nCellsSolve']
        if name == 'xtime_s':
            return self.call.time_seconds
        if name in NATIVE_POOL_BINDINGS:
            return self.field(*NATIVE_POOL_BINDINGS[name])
        if name in ('curr_julday', 'gmt', 'julday', 'year'):
            value, lower = self.native.get_table('mpas_atmphys_manager.' + name)
            return value[()]
        if name in ('degrad', 'dpd', 'solcon_0'):
            return self.native.get_constant(name)
        return self.variable(name)

    def state(self):
        """Share native initialized tables with the test seam; mpas_atmphys_camrad_init.F:39-76,119-293."""
        values = {}
        for name, kind, rank in STATE_FIELDS:
            host = self.words.views[name]
            if rank:
                lower, extent, stride = STATE_LAYOUT[name]
                storage = self.cp.asarray(host)
                if kind == 'bool':
                    storage = storage.astype(self.cp.bool_)
                values[name] = DeviceArray(storage, lower, extent, stride)
            else:
                values[name] = host[0]
        return make_device_state(values, int(self.cp.cuda.Device().id))

    def commit(self, result):
        """Copy native driver outputs and shared state; mpas_atmphys_driver_radiation_lw.F:960; mpas_atmphys_driver_radiation_sw.F:1006."""
        import numpy as np
        boundary, stages, copied, frame = result
        error = int(copied.error.get()[0])
        if error:
            raise RuntimeError(f'MPAS CAM output conversion fatal source line {error}')
        for key, (host, value, lower) in self.copies.items():
            host[...] = value.storage.get().reshape(host.shape, order='F')
            if key[0] == 'prepared':
                self.native.put_variable(key[1], host, lower=lower)
            else:
                self.native.put_field(key[0], key[1], host, level=key[2],
                                      nlevels=self.native._fields[key[0], key[1]][2])
        metadata = frame.state.metadata.get()
        values = {}
        for name, kind, rank in STATE_FIELDS:
            if rank:
                value = frame.state.values[name].storage.get()
                values[name] = value.astype(self.words.views[name].dtype, copy=False)
            else:
                offset, _ = STATE_SCALARS[name]
                word = metadata[offset:offset+1]
                if kind == 'double':
                    value = word.view(np.float64)
                elif kind == 'float':
                    value = word.astype(np.uint32).view(np.float32)
                else:
                    value = word.astype(np.uint32).view(np.int32)
                values[name] = value
        self.words.write(values)
        for stage, scalar in ((stages[1], 'declin'), (stages[1], 'solcon')):
            value = stage_scalar(stage, 'radconst', scalar)
            word = value.storage[value.offset:value.offset+1].get().astype(np.uint32).view(np.float32)[0]
            self.native.put_variable(scalar, np.asarray(word))
        # MPAS module_ra_cam.F:477 writes the framework's complete CHARACTER word set.
        self.message_words[:]=self.utility_state['mpas_err_message'].get()
        self.native.put_variable('doabsems', np.asarray(int(boundary.metadata[boundary.doabsems_offset].get()), dtype=np.int32))


def bind_native_pools(provider, call):
    """Whole native CAM driver interface; mpas_atmphys_driver_radiation_lw.F:804-960; mpas_atmphys_driver_radiation_sw.F:834-1006."""
    owner = _NativeCAMTransfer(provider, call)
    suffix = 'lw' if call.slot == 'radiation_lw' else 'sw'
    def stage(name):
        return {field: owner.argument(field) for field, _, _ in STAGE_FIELDS[name]}
    dynamic = {'dolw', 'dosw', 'doabsems', 'n_cldadv', 'declin', 'solcon', 'xtime', 'radt'}
    arguments = {field: owner.argument(NATIVE_ARGUMENT_BINDINGS[field])
                 for field, _, _ in ARGUMENT_FIELDS if field not in dynamic}
    # run_driver replaces these words with exact native manager/device outputs.
    arguments.update(dolw=False, dosw=False, doabsems=False, n_cldadv=3,
                     declin=0., solcon=0., xtime=0., radt=0.)
    solar = {field: owner.argument('curr_julday' if field == 'julian' else field)
             for field, _, _ in STAGE_FIELDS['radconst']}
    clock = {'xtime_s':call.time_seconds, 'dt_radt':owner.variable('dt_radt' + suffix),
             'xtime_m':0., 'radt':0.}
    frame = DriverFrame(owner.state(), stage('radiation_' + suffix + '_from_mpas'),
                        arguments, stage('radiation_' + suffix + '_to_mpas'), solar, clock, owner.utility_state)
    return NativeCAMFrame(frame, owner)


def run_slot(call):
    """Commit complete CAM driver execution; mpas_atmphys_driver_radiation_lw.F:804-960; mpas_atmphys_driver_radiation_sw.F:834-1006."""
    binding = call.pools
    if not isinstance(binding, NativeCAMFrame):
        raise TypeError('CAM native-pool callback requires its bound whole driver frame')
    result = run_driver(call, binding.frame)
    binding.owner.commit(result)
    return result
'''
