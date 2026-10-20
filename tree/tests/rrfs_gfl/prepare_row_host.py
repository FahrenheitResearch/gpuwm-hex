"""Full original-row host entry for debugging the complete GFL transcription.

No scientific expressions are edited. This preserves the original horizontal
extent rather than invoking the production column entry. Supporting evidence only.
"""
import argparse
from pathlib import Path
from hexcore.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl


def prepare(target, observed=False):
    entry = gfl._ENTRY.split('#ifndef GF_HOST_WITNESS')[0]
    entry = entry.replace('void gfl_column(int column,int nz,', 'void gfl_row(int ncol,int nz,')
    entry = entry.replace(' int ims=1,ime=1,', ' int column=0; int ims=1,ime=ncol,')
    entry = entry.replace('ids=1,ide=2,', 'ids=1,ide=ncol+1,')
    entry = entry.replace('its=1,ite=1,', 'its=1,ite=ncol,')
    entry += '''
extern "C" void gfl_row_host(int ncol,int nz,const unsigned long long* ptrs,
 const unsigned long long* strides,const char* pbl,long long pbl_len,char* scratch,
 unsigned long long capacity,unsigned int* diagnostics,unsigned long long* highwater,
 float row_dx_first,const int* vertical,unsigned int* saved_fct_error){
 gfl_row(ncol,nz,ptrs,strides,pbl,pbl_len,scratch,capacity,diagnostics,highwater,
         row_dx_first,vertical,saved_fct_error);
}
'''
    if observed:
        from row_cuda import observe_initial_state
        text = observe_initial_state(gfl._SUPPORT + gfl._SCHEME_SOURCE + entry)
        (target / 'host_row.cpp').write_text('#define GF_HOST_WITNESS\n' + text)
    else:
        (target / 'host_row.cpp').write_text('#define GF_HOST_WITNESS\n#include "gfl_support.cuh"\n'
            '#include "gfl_lowered.cuh"\n' + entry)
    (target / 'host_row.mk').write_text('''
.PHONY: host-row
host-row: gfl_host_row.so
gfl_host_row.so: host_row.cpp gfl_support.cuh gfl_lowered.cuh
\tg++ -std=c++17 -O3 -fPIC -shared -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations host_row.cpp -o $@
''')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--observed-ierrm', action='store_true', help='diagnostic substitution of captured native words')
    args = p.parse_args()
    prepare(args.target, args.observed_ierrm)
