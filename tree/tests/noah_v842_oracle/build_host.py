"""Build a host-glibc witness from the exact shipped CUDA expression source."""
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.mpasphys.noah_v842 import _CORE_CUDA, _ENTRY_CUDA, _LANDUSE_CUDA, _MPAS_IO_CUDA

head = """#include <cmath>
#define __device__
#define AD(a,b) ((float)((a)+(b)))
#define SU(a,b) ((float)((a)-(b)))
#define MU(a,b) ((float)((a)*(b)))
#define DV(a,b) ((float)((a)/(b)))
#define __fsqrt_rn sqrtf
#define r_pow powf
#define r_log logf
#define r_log10 log10f
#define r_exp expf
#define r_atan atanf
"""
out = []
depth = 0
in_kernel = False
for line in (_ENTRY_CUDA + _LANDUSE_CUDA + _MPAS_IO_CUDA + Path(__file__).with_name("helper_entry.cu").read_text()).splitlines():
    if 'extern "C" __global__' in line:
        in_kernel = True
    if in_kernel:
        depth += line.count('{')-line.count('}')
        line = line.replace('__global__','')
        line = line.replace('const NoahTables* ts,int* status,int n','NoahTables* ts,int n')
        line = line.replace('const LanduseTables* ts,int* status,int n','const LanduseTables* ts,int n')
        line = line.replace('int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;', 'for(int c=0;c<n;++c){')
        line = line.replace('int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=n)return;int c=start+local;', 'for(int local=0;local<n;++local){int c=start+local;')
        line = line.replace('int local=blockIdx.x*blockDim.x+threadIdx.x;if(local>=n || status[local])return;int c=start+local;', 'for(int local=0;local<n;++local){if(status[local])continue;int c=start+local;')
        line = re.sub(r'status\[c\]\s*=.*?;', '', line).replace('return;', 'continue;')
        if depth == 0:
            line = '}' + line
            in_kernel = False
    out.append(line)
Path(__file__).with_name('host.cpp').write_text(head + _CORE_CUDA + '\n'.join(out))
