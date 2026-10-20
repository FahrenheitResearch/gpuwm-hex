"""Captured-initial-state diagnostic of the complete production column launch.

The actual launcher, tiling, strides and scientific transcription are reused.
Only a passively captured initial ierrm vector is supplied to the omitted native
initialization. This altered diagnostic is never a production initialization,
native-defect reproduction claim, or whole-Hex forecast witness.
Authority: fork GFL/MPAS/module_cu_gfl_mpas.F90:116,233-253,515 at 1d6e370e.
"""
from functools import lru_cache
import hashlib
import inspect
from hexcore.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl
from row_cuda import observe_initial_state


def source():
    return observe_initial_state(gfl.cuda_source(), columns=True)


@lru_cache(maxsize=None)
def module(device):
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    with cp.cuda.Device(device):
        arch = cp.cuda.Device(device).compute_capability
        program = nvrtc.createProgram(source(), 'diagnostic_gfl_captured_columns.cu', (), ())
        try:
            nvrtc.compileProgram(program, gfl.COMPILER_OPTIONS + (f'--gpu-architecture=sm_{arch}',))
            binary = nvrtc.getCUBIN(program)
        except Exception as error:
            raise RuntimeError(nvrtc.getProgramLog(program)) from error
        finally:
            nvrtc.destroyProgram(program)
        result = cp.cuda.function.Module()
        result.load(binary)
        return result


@lru_cache(maxsize=1)
def launcher():
    # Derive the diagnostic from the real launcher. Source guards ensure no
    # private duplicate of its validation, tiling or pointer tables drifts.
    text = inspect.getsource(gfl.launch_columns)
    changes = (
        ('def launch_columns(fields, *,', 'def launch_columns(fields, *, observed_ierrm,', 1),
        ('vertical_device, saved_fct_error))',
         'vertical_device, saved_fct_error, observed_ierrm[first:first+count]))', 1),
    )
    for old, new, count in changes:
        if text.count(old) != count:
            raise ValueError('production column launcher seam changed: ' + old)
        text = text.replace(old, new)
    namespace = vars(gfl).copy()
    namespace['_module'] = module
    exec(compile(text, '<captured-state GFL column diagnostic>', 'exec'), namespace)
    return namespace['launch_columns']


def launch(fields, *, observed_ierrm, **kwargs):
    import cupy as cp
    if (not isinstance(observed_ierrm, cp.ndarray) or observed_ierrm.dtype != cp.int32 or
            observed_ierrm.shape != (kwargs['ncol'],)):
        raise ValueError('captured ierrm must supply exactly one native integer per column')
    result = launcher()(fields, observed_ierrm=observed_ierrm, **kwargs)
    return result, dict(source_preserved=False, captured_ierrm_substituted=True,
        automatic_storage_initialized=True, hidden_state_complete=False,
        scientific_expressions_preserved=True, production_column_decomposition=True,
        diagnostics=int(cp.count_nonzero(result['diagnostics'])),
        scratch_highwater=int(cp.max(result['scratch_highwater'])),
        emitted_source_sha256=hashlib.sha256(source().encode()).hexdigest(),
        production_launcher_sha256=hashlib.sha256(inspect.getsource(gfl.launch_columns).encode()).hexdigest())
