"""Prepare a host-only candidate finder; all found inputs require native replay.

The trace records the fallback without changing finite arithmetic. Nonfinite
iteration states are discarded before table indexing; this debug filter is
never used by either native oracle or production CUDA.
"""
from pathlib import Path

root=Path('oracle')
source=(root/'witness.cpp').read_text()
source='#include <limits>\nstatic bool found_fallback=false, rejected_state=false;\nstatic int trace_last_n=0;\n'+source
source=source.replace('n = (n+1);','n = (n+1); trace_last_n=n;',1)
old='m_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));'
# This exact call occurs once in zolri and once in zolrib.
assert source.count(old)==2
source=source.replace(old,'found_fallback=true; '+old,1)
old='zolt = __fdiv_rn(__fmul_rn(zol2,zt),za);'
assert source.count(old)==1
source=source.replace(old,old+'''
if (!std::isfinite(zol20) || !std::isfinite(zol3) || !std::isfinite(zolt)) {
// Debug search only: sf_mynn.F90:2135-2232 table indices are undefined for NaN.
rejected_state=true; return std::numeric_limits<float>::quiet_NaN();
}
''')
source+='''
extern "C" void trace_secant(float* data,int psi,int n) {
m_psi_init(psi);
for (int i=0;i<n;++i) {
found_fallback=false; rejected_state=false; trace_last_n=0;
float v=m_zolri(data[i],data[n+i],data[2*n+i],data[3*n+i],data[4*n+i],psi);
data[5*n+i]=v; data[6*n+i]=rejected_state ? 0.f : (found_fallback ? 1.f : (trace_last_n==20 ? 2.f : 0.f));
}
}
'''
(root/'trace.cpp').write_text(source)
(root/'TraceMakefile').write_text('''all: trace.so
trace.so: trace.cpp
\tg++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -std=c++17 -shared $< -o $@
''')
