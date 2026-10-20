def cuda_source():
    """Maths for MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:5239-5395.

    Use shared glibc 2.43 elementary functions, preserving subnormals.
    """
    from gpuwm.core import kernels
    header = Path(kernels.__file__).with_name('glibc_flt32.cuh').read_text()
    trig = Path(kernels.__file__).with_name('glibc_trig_flt32.cuh').read_text()
    double = Path(kernels.__file__).with_name('glibc_flt64.cuh').read_text()
    extra = Path(kernels.__file__).with_name('thompson_aerosol_libm.cuh').read_text()
    # Reuse the engine's unchanged CORE-MATH/glibc 2.43 implementation.
    # The older Thompson log10f reduction targets glibc 2.39 instead.
    urban = Path(kernels.__file__).with_name('urban_bem.cuh').read_text()
    log10 = urban[urban.index('// log10f: CORE-MATH'):urban.index('\n'+'__device__ '+'void bem_bem(')]
    return header + '\n' + double + '\n' + extra + '\n' + trig + '\n' + log10 + '\n' + _SUPPORT + '\n' + _SCHEME_SOURCE + '\n' + _ENTRY + '\n' + _DRIVER


@lru_cache(maxsize=None)
def _module(device):
    """CUDA build for MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    with cp.cuda.Device(device):
        arch = cp.cuda.Device(device).compute_capability
        program = nvrtc.createProgram(cuda_source(), 'cu_gfl_bf6a4b0.cu', (), ())
        try:
            nvrtc.compileProgram(program, COMPILER_OPTIONS + (f'--gpu-architecture=sm_{arch}',))
            binary = nvrtc.getCUBIN(program)
        except Exception as error:
            raise RuntimeError(nvrtc.getProgramLog(program)) from error
        finally:
            nvrtc.destroyProgram(program)
        module = cp.cuda.function.Module()
        module.load(binary)
        return module


@lru_cache(maxsize=None)
def _saved_fct_error(device):
    """SAVE from MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:2641-2705."""
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.zeros(1,dtype=cp.uint32)


def launch_columns(fields, *, ncol, nz, pbl_scheme, row_dx_first, native_dimensions=None, saved_fct_error=None):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/MPAS/module_cu_gfl_mpas.F90:33-745 at 1d6e370e.

    Boundary arrays have column-major batches: surface [ncol], level [ncol,nz+1],
    chemistry [ncol,nchem,nz+1]. Scalar arguments are CUDA single-element arrays.
    All inout fields, aliasing and held level words belong to the caller.
    row_dx_first is the first owned cell's dx from the unsplit native row.
    Captures may supply the original six vertical bounds through
    native_dimensions; the driver uses its ordinary interface allocation.
    Values are never decoded or processed on the host.
    """
    import cupy as cp
    import numpy as np
    vertical_names = ('kms','kme','kds','kde','kts','kte')
    vertical = (1,nz+1,1,nz+1,1,nz) if native_dimensions is None else tuple(int(native_dimensions[name]) for name in vertical_names)
    storage_levels = vertical[1]-vertical[0]+1
    if saved_fct_error is None:
        saved_fct_error = _saved_fct_error(cp.cuda.runtime.getDevice())
    if not isinstance(saved_fct_error,cp.ndarray) or saved_fct_error.dtype!=cp.uint32 or saved_fct_error.size!=1:
        raise ValueError('GFL fct1d3 SAVE must be one CUDA logical word')
    if not isinstance(row_dx_first, cp.ndarray) or row_dx_first.dtype != cp.float32 or row_dx_first.size != 1:
        raise ValueError("GFL native row dx must be one CUDA float32 word")
    pointers, strides = [], []
    for name in FIELDS:
        array = fields.get(name)
        if array is None and name in ('gdc', 'gdc2', 'chem3d_in', 'rthften', 'rqvften'):
            pointers.append(0); strides.append(0); continue
        dtype = cp.int32 if name in INTEGER_FIELDS else cp.float32
        if not isinstance(array, cp.ndarray) or array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError(f'GFL {name} must be contiguous CUDA {dtype}')
        if name in SCALAR_FIELDS:
            if array.size != 1:
                raise ValueError(f'GFL scalar {name} must contain one word')
            stride = 0
        else:
            if array.ndim == 0 or array.shape[0] != ncol:
                raise ValueError(f'GFL {name} must have {ncol} columns')
            words = array.size // max(1, ncol)
            if name in SURFACE_FIELDS and words != 1:
                raise ValueError(f'GFL surface {name} must contain one word per column')
            if name not in SURFACE_FIELDS and name != 'chem3d_in' and words != storage_levels:
                raise ValueError(f'GFL level {name} must contain {storage_levels} native allocation words per column')
            stride = words * array.itemsize
        pointers.append(array.data.ptr); strides.append(stride)
    diagnostics = cp.zeros(ncol, dtype=cp.uint32)
    highwater = cp.zeros(ncol, dtype=cp.uint64)
    if ncol == 0:
        return {'diagnostics': diagnostics, 'scratch_highwater': highwater, 'saved_fct_error':saved_fct_error}
    tile = min(ncol, 64)
    capacity = 32768 * max(storage_levels,vertical[5]-vertical[4]+1) + 131072
    scratch = cp.empty((tile, capacity), dtype=cp.uint8)
    pbl = pbl_scheme.encode('ascii')
    pbl_device = cp.asarray(np.frombuffer(pbl, dtype=np.uint8))
    strides_host = np.asarray(strides, dtype=np.uint64)
    pointers_host = np.asarray(pointers, dtype=np.uint64)
    strides_device = cp.asarray(strides_host)
    vertical_device = cp.asarray(vertical,dtype=cp.int32)
    kernel = _module(cp.cuda.runtime.getDevice()).get_function('gfl_columns')
    # Pointer metadata, including null optionals, stays alive until the stream completes.
    pointer_tables = []
    for first in range(0, ncol, tile):
        count = min(tile, ncol-first)
        table = cp.asarray(np.where(pointers_host == 0, 0, pointers_host+np.uint64(first)*strides_host))
        pointer_tables.append(table)
        kernel(((count+31)//32,), (32,), (np.int32(count), np.int32(nz), table,
            strides_device, pbl_device, np.int64(len(pbl)), scratch, np.uint64(capacity),
            diagnostics[first:first+count], highwater[first:first+count], row_dx_first, vertical_device, saved_fct_error))
    cp.cuda.get_current_stream().synchronize()
    return {'diagnostics': diagnostics, 'scratch_highwater': highwater, 'saved_fct_error':saved_fct_error}


def cells_on_cells_on_cell(mesh, *, nCellsSolve):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584 at 1d6e370e.

    One-based MPAS indices; row-contiguous [cell,edge] arrays. Writes only owned
    second-ring rows and counts. Unused edge slots retain their input words.
    """
    import cupy as cp
    import numpy as np
    ne, coc = mesh['nEdgesOnCell'], mesh['cellsOnCell']
    n2, coc2 = mesh['nCellsOnCellsOnCell'], mesh['cellsOnCellsOnCell']
    _module(cp.cuda.runtime.getDevice()).get_function('gfl_cells_on_cells')(
        ((nCellsSolve+127)//128,), (128,), (np.int32(nCellsSolve), np.int32(coc.shape[1]),
         np.int32(coc2.shape[1]), ne, coc, n2, coc2))


def spread_subsidence(tendencies, subsidence, *, mesh, sigma_deep, sigma_deep_far,
                       config_gfl_sub3d, nCellsSolve, exchange_halo_group=None):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1336-1433 at 1d6e370e.

    Arrays [4,nCells,nVertLevels] ordered theta,vapor,u,v. The layer-two
    halo group completes before any neighboring subsidence is read.
    """
    import cupy as cp
    import numpy as np
    if config_gfl_sub3d > 0:
        if exchange_halo_group is None:
            raise ValueError('GFL spreading requires the physics:sub3d_cuten halo exchange')
        exchange_halo_group('physics:sub3d_cuten')
    return _spread_after_halo(tendencies, subsidence, mesh=mesh, sigma_deep=sigma_deep,
        sigma_deep_far=sigma_deep_far, config_gfl_sub3d=config_gfl_sub3d, nCellsSolve=nCellsSolve)


def _spread_after_halo(tendencies, subsidence, *, mesh, sigma_deep, sigma_deep_far,
                       config_gfl_sub3d, nCellsSolve):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1367-1433 at 1d6e370e."""
    import cupy as cp
    import numpy as np
    _, nc, nz = tendencies.shape
    _module(cp.cuda.runtime.getDevice()).get_function('gfl_spread')(
        ((nCellsSolve*nz+127)//128,), (128,), (np.int32(nc), np.int32(nCellsSolve),
         np.int32(nz), np.int32(mesh['cellsOnCell'].shape[1]),
         np.int32(mesh['cellsOnCellsOnCell'].shape[1]), np.int32(config_gfl_sub3d),
         mesh['bdyMaskCell'], mesh['nEdgesOnCell'], mesh['cellsOnCell'],
         mesh['nCellsOnCellsOnCell'], mesh['cellsOnCellsOnCell'], sigma_deep,
         sigma_deep_far, subsidence, tendencies))


def run_slot(call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710 at 1d6e370e.

    The RRFS adapter supplies the original routine argument arrays, native row
    context and inout storage in gfl_arguments; step2 calls spread_subsidence.
    """
    from .driver_gfl_bf6a4b0 import run_slot as run_driver
    return run_driver(call)


def bind_native_pools(provider, call):
    """RRFS fork MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710,1336-1367 at 1d6e370e."""
    from .driver_gfl_bf6a4b0 import bind_native_pools as bind
    return bind(provider, call)


def register(registry):
    """Select MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:668-710."""
    registry.register('convection', SCHEME, BACKEND, run_slot)


BACKEND_ENTRY = ('convection', SCHEME, BACKEND, run_slot)
