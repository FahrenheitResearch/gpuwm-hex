"""Shared GNU elementary math for MPAS carriers, with no private transcription."""
from functools import lru_cache
from importlib.resources import files
import re


# Pure in its argument and the installed gpuwm headers. Breakage it prevents: U-MPASPHYS-1, the
# per-step carrier launches re-read three header files from disk and re-ran this regex every call.
@lru_cache(maxsize=64)
def bind_common_carrier_math(source):
    """Bind native math; mpas_vector_reconstruction.F:303-323; mpas_atmphys_interface.F:504-519."""
    root = files('gpuwm.core.kernels')
    headers = '\n'.join(root.joinpath(name).read_text() for name in (
        'glibc_flt32.cuh', 'flt32_expf_fma.cuh', 'glibc_trig_flt32.cuh'))
    names = dict(sinf='glibc_sinf', cosf='glibc_cosf',
                 logf='gfk_log', expf='gfk_exp_fma')
    source = re.sub(r'\b(sinf|cosf|logf|expf)\s*(?=\()',
                    lambda match: names[match.group(1)], source)
    return headers + '\n' + source
