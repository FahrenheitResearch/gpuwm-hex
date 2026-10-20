"""Record exact shared-device-libm call witnesses on the pinned CPU host.

No transcendental is implemented here. Both the shared header and native
glibc entrypoints are called, and the shared result drives the column.
"""
from pathlib import Path
import argparse
import re


def prepare(directory):
    source=(directory/'host_shared_port.cpp').read_text()
    wrappers='\n#include <cstdio>\n'
    rows=[]
    for i,name in enumerate(('expf','exp','logf','log','log10f','log10','powf','pow')):
        source=source.replace(f'#define mpas_{name} thompson_aa_{name}','')
        typ='float' if name.endswith('f') else 'double';power=name.startswith('pow')
        args='x,y' if power else 'x';decl=f'{typ} x,{typ} y' if power else f'{typ} x'
        wrappers+=f'static unsigned long calls_{name}=0,diffs_{name}=0;\n'
        wrappers+=f'inline {typ} mpas_{name}({decl}) {{ calls_{name}++;{typ} actual=thompson_aa_{name}({args});{typ} native=::{name}({args});\n'
        bit='__float_as_uint' if typ=='float' else '__double_as_longlong'
        wrappers+=f'if({bit}(actual)!={bit}(native)) {{ if(diffs_{name}++==0) '
        wrappers+=f'fprintf(stderr,"FIRST {name} x=%a y=%a native=%a shared=%a\\n",(double)x,(double)'+('y' if power else '0')+',(double)native,(double)actual); } return actual; }\n'
        rows.append(f'fprintf(stderr,"SUMMARY {name} calls=%lu differences=%lu\\n",calls_{name},diffs_{name});')
    wrappers+='extern "C" void report_math_trace() {'+''.join(rows)+'}\n'
    marker='// Scalar arithmetic for generated Thompson columns.'
    assert marker in source
    source=source.replace(marker,wrappers+marker)
    (directory/'host_shared_trace_port.cpp').write_text(source)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    prepare(p.parse_args().directory)
