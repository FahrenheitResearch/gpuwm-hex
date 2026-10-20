"""Compile the exact production kernels on a CPU build host, without a GPU.

NVRTC accepts an explicit target architecture. Only cubin generation runs here;
Loading and numerical verification remain inside the assigned GPU slot queue.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
from cupy.cuda import compiler, nvrtc
from hexcore.mpasphys import noah_v842 as noah

p = argparse.ArgumentParser()
p.add_argument('--arch', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert a.arch.isdecimal()
assert not compiler._use_ptx, 'production cache requires the cubin compiler path'
a.output.mkdir(parents=True, exist_ok=True)
base = Path(__file__).resolve().parent
math = noah._device_math_source()
sources = {
    'production': math + noah._CORE_CUDA + noah._ENTRY_CUDA + noah._LANDUSE_CUDA
                  + noah._MPAS_IO_CUDA + noah._STATE_CUDA,
    'helpers': math + noah._CORE_CUDA + (base / 'helper_entry.cu').read_text(),
}
version = nvrtc.getVersion()
options = noah.CUDA_OPTIONS + (f'-arch=sm_{a.arch}', '--device-as-default-execution-space')
if (version[0] >= 13 or version >= (12, 8)) and compiler._use_pch:
    options += ('--pch',)
rows = []
for name, source in sources.items():
    # Match noah_v842._compile_module's identity and CuPy's exact NVRTC options.
    identity = repr((noah.CUDA_OPTIONS, a.arch, version, cp.__version__)) + source
    key = hashlib.sha256(identity.encode()).hexdigest()
    program = compiler._NVRTCProgram(source, '', method='cubin')
    binary, _ = program.compile(options)
    path = a.output / (key + '.cubin')
    path.write_bytes(binary)
    rows.append({'entry': name, 'file': path.name, 'bytes': len(binary),
                 'sha256': hashlib.sha256(binary).hexdigest(),
                 'cuda_source_sha256': hashlib.sha256(source.encode()).hexdigest()})
    print(f'compiled {name}: {len(binary)} bytes', flush=True)
(a.output / 'manifest.json').write_text(json.dumps({
    'scope': 'CPU-only NVRTC build; no numerical or forecast proof',
    'arch': a.arch, 'nvrtc_version': version, 'cupy_version': cp.__version__,
    'compile_options': options, 'production_source_sha256':
        hashlib.sha256(Path(noah.__file__).read_bytes()).hexdigest(),
    'artifacts': rows,
}, indent=2) + '\n')
