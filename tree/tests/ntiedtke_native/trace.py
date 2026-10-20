"""Diagnostic trace of actual MPAS cumastrn calls; never acceptance authority.

The copied source gains output statements only. Final driver outputs must be
compared with the pristine build before using intermediate traces for debugging.
MPAS cu_ntiedtke.F90:342-1085.
"""
from pathlib import Path
import re
import sys


def generate(source, destination):
    text = Path(source).read_text()
    start = text.lower().index('      subroutine cumastrn')
    end = text.lower().index('      end subroutine cumastrn', start)
    body = text[start:end]
    clean = re.sub(r'!.*', '', body).replace('&', '').lower()
    ranks = {}
    for dims, names in re.findall(r'real\(kind=kind_phys\)[^\n]*?dimension\(([^)]*)\)\s*::([^\n]*)', clean):
        rank = len(dims.split(','))
        if dims.startswith('klon'):
            ranks.update({n.strip(): rank for n in names.split(',')})
    stages = ('cuinin','cutypen','cuascn','cudlfsn','cuddrafn','cuflxn','cudtdqn','cududvn')
    manifest = {}
    for stage in stages:
        pattern = rf'(?im)^\s*call {stage}[\s&]*\([\s\S]*?\)'
        match = re.search(pattern, body)
        args = re.sub(r'\s|&', '', match.group().split('(',1)[1][:-1]).split(',')
        traced = {n:ranks[n] for n in args if n in ranks}
        formal = re.search(rf'(?im)^\s*subroutine {stage}[\s&]*\(([\s\S]*?)\)', text)
        formal_names = re.sub(r'\s|&', '', formal.group(1)).split(',')
        manifest[stage] = {actual: {'rank': ranks[actual], 'formal': parameter}
                           for actual,parameter in zip(args,formal_names,strict=True) if actual in ranks}
        calls = [f'\n      if (klon == 18) then\n        open(97,file="trace-{stage}.csv",status="replace")']
        calls += [f'        call nt_trace{rank}("{name}",{name})' for name,rank in traced.items()]
        calls += ['        close(97)\n      endif\n']
        body = body[:match.end()]+'\n'.join(calls)+body[match.end():]
    helpers = '''
      subroutine nt_trace1(name,x)
      character(len=*),intent(in)::name
      real(kind=kind_phys),intent(in)::x(:)
      integer::i
      do i=1,size(x)
        write(97,'(a,",",i0,",",i0,",",z8.8)')name,1,i,transfer(x(i),0)
      enddo
      end subroutine
      subroutine nt_trace2(name,x)
      character(len=*),intent(in)::name
      real(kind=kind_phys),intent(in)::x(:,:)
      integer::i,k
      do k=1,size(x,2)
      do i=1,size(x,1)
        write(97,'(a,",",i0,",",i0,",",z8.8)')name,k,i,transfer(x(i,k),0)
      enddo
      enddo
      end subroutine
'''
    text = text[:start]+body+text[end:]
    marker = ' end module cu_ntiedtke\n'
    assert text.lower().count(marker)==1
    text = text.replace(marker,helpers+marker)
    Path(destination).write_text(text)
    import json
    Path(destination).with_suffix('.json').write_text(json.dumps(manifest,indent=2))


def compare(directory, library, receipt):
    import csv
    import json
    import numpy as np
    from host import HostPipeline
    from test_mpas_ntiedtke_v842 import load_native, _word
    from hexcore.mpasphys.ntiedtke_v842 import Constants, LEVEL_INPUTS, INTERFACE_INPUTS
    from gpuwm.core import ntiedtke as nt
    directory = Path(directory)
    columns,surface = load_native(directory)
    dx = max(k[1] for k in columns)
    keys = sorted(k for k in columns if k[1]==dx)
    nz = len(columns[keys[0]])-1
    settings = json.loads((directory/'settings.json').read_text()) if (directory/'settings.json').exists() else {'stepcu':1,'initflag':0}
    p = HostPipeline(Path(library),len(keys),nz,60.0,settings['stepcu'],settings['initflag'],Constants())
    for name in LEVEL_INPUTS+INTERFACE_INPUTS:
        rows = nz+1 if name in INTERFACE_INPUTS else nz
        source = 'pi3d' if name=='exner' else name
        p.w.bind(name,0)[:rows] = np.array([[_word(columns[k][r][source]) for k in keys] for r in range(rows)],dtype=np.float32)
    for name in ('xland','hfx','qfx'):
        p.w.bind(name,1)[:] = [_word(surface[k][name]) for k in keys]
    p.w.bind('dx',1)[:] = dx
    manifest = json.loads((directory/'manifest.json').read_text())
    p.zero_run_head()
    results = {}
    for stage in nt.NT_CALL_ORDER:
        if stage=='ntiedtke_cuascn': p.reduce_llo3()
        if stage=='ntiedtke_cududvn': p.snapshot_momentum()
        p.run_stage(stage)
        if stage=='ntiedtke_convert': p.snapshot_forcing()
        native_stage = stage.removeprefix('ntiedtke_')
        if native_stage not in manifest: continue
        arrays = {}
        with (directory/f'trace-{native_stage}.csv').open() as f:
            for name,row,col,word in csv.reader(f):
                arrays.setdefault(name,[]).append((int(row)-1,int(col)-1,int(word,16)))
        result = {}
        names = {name.split()[-1] for name in p._signatures[stage]}
        for actual,info in manifest[native_stage].items():
            formal = info['formal']
            if formal not in names: continue
            a = p.w.bind(nt.NT_STAGE_ALIASES.get((stage,formal),formal),p.base(stage)).view(np.uint32)
            count = 0; samples = []
            for row,col,word in arrays[actual]:
                offset = 1 if p.base(stage)==1 else 0
                got = int(a[row+offset,col] if a.ndim==2 else a[col])
                if got != word:
                    count+=1
                    if len(samples)<3: samples.append(dict(row=row,col=col,got=hex(got),native=hex(word)))
            result[formal] = dict(differing_words=count,samples=samples)
        results[stage]=result
    Path(receipt).write_text(json.dumps(results,indent=2)+'\n')
    print({stage:{n:a['differing_words'] for n,a in arrays.items() if a['differing_words']}
           for stage,arrays in results.items()})


if __name__=='__main__':
    if sys.argv[1]=='compare': compare(*sys.argv[2:])
    else: generate(sys.argv[1],sys.argv[2])
