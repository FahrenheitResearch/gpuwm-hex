"""Explicit test-only whole-driver witness using the port's actual host bodies.

This callback transports caller buffers to the strictly compiled C++ port and
back. It does no weather arithmetic in Python and calls no native WSM6 routine.
It is a host-glibc diagnostic, never a CUDA acceptance or a provider fallback.
MPAS driver argument list: mpas_atmphys_driver_microphysics.F:681-687.
"""
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@lru_cache(maxsize=1)
def _driver(manifest_path):
    from hexcore.mpasphys import wsm6_v842

    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['acceptance'] or manifest['backend'] != 'actual-port-host-glibc':
        raise ValueError('host diagnostic must explicitly disclaim CUDA acceptance')
    source = Path(wsm6_v842.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['production_source_sha256']:
        raise ValueError('host witness source differs from current port')
    library = Path(manifest['library']['path'])
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest['library']['sha256']:
        raise ValueError('strict actual-port host library changed')
    loaded = ct.CDLL(str(library.resolve()))
    function = loaded.wsm6_driver_host
    function.argtypes = ([ct.c_int] * 4 + [ct.c_float] * 2 +
        [ct.c_int, ct.POINTER(ct.c_float), ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int)] +
        [ct.c_int] * 2 + [ct.c_float, ct.c_int] + [ct.c_float] * 3)
    function.restype = None
    return loaded, function


def invoke(call):
    """MPAS interface: mpas_atmphys_interface.F:657-675,903-946,1076-1101."""
    import cupy as cp
    from hexcore.mpasphys.wsm6_v842 import FIELD_NAMES

    _, driver = _driver(os.environ['WSM6_HOST_GLIBC_MANIFEST'])
    binding = call.pools
    fields, options = binding['fields'], binding['options']
    nz, nc = fields['theta_m'].shape
    host = {name: cp.asnumpy(fields[name]) for name in FIELD_NAMES}
    counter = cp.asnumpy(fields['i_rainnc'])
    constants = cp.asnumpy(binding['constants'])
    pointers = (ct.c_void_p * len(FIELD_NAMES))(*[host[name].ctypes.data for name in FIELD_NAMES])
    driver(nz, nc, int(options['solve_cells']), int(options['mp_top_level']),
        options['dt_microp'], options['dt_dyn'], int(options['hail_opt']),
        constants.ctypes.data_as(ct.POINTER(ct.c_float)), pointers,
        counter.ctypes.data_as(ct.POINTER(ct.c_int)),
        int(options['has_reqc'] == options['has_reqi'] == options['has_reqs'] == 1),
        int(options['l_diags']), options['bucket_rainnc'], int(options['l_acrain']),
        options['rvord'], options['rcv'], options['p0'])
    for name in FIELD_NAMES:
        cp.copyto(fields[name], cp.asarray(host[name]))
    cp.copyto(fields['i_rainnc'], cp.asarray(counter))
