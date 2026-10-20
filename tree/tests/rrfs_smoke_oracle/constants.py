from pathlib import Path
import runpy
ns=runpy.run_path('scratch/translate.py');mods=ns['modules'];rs=ns['routines']
procs=[]
for scope,uses,symbols in [(n,m['uses'],m['globals']) for n,m in mods.items()]+[(n,[{'module':r['module'],'only':None}]+r['uses'],{k:d for k,d in r['symbols'].items() if d['parameter']}) for n,r in rs.items()]:
 symbols={k:d for k,d in symbols.items() if d['init'] is not None and d['type'] not in ('FStr','Plume*')}
 if not symbols:continue
 lines=['subroutine const_'+scope+'()','use mpas_kind_types']
 for u in uses:
  if u['module']=='mpas_kind_types':continue
  lines.append('use '+u['module']+(', only: '+','.join(u['only']) if u['only'] else ''))
 lines+=['implicit none','integer :: const_index']
 for k,d in symbols.items():
  t='real(RKIND)' if d['type']=='float' else 'integer' if d['type']=='int' else 'logical'
  lines.append(t+(', parameter' if d['parameter'] else '')+' :: '+k+('('+','.join(d['dim'])+')' if d['dim'] else '')+' = '+d['init'])
 for k,d in symbols.items():
  val=k
  if d['dim']:lines.append('do const_index=1,size('+k+')');val+='(const_index)'
  prefix="'"+scope+'.'+k+"', "+('const_index' if d['dim'] else '1')
  if d['type']=='float':lines.append("write(*,'(A,1X,I0,1X,Z8.8)') "+prefix+',transfer('+val+',0)')
  else:lines.append("write(*,'(A,1X,I0,1X,I0)') "+prefix+','+('merge(1,0,'+val+')' if d['type']=='bool' else val))
  if d['dim']:lines.append('end do')
 lines.append('end subroutine');procs.append(('const_'+scope,lines))
lines=['program smoke_constants']+['call '+n+'()' for n,_ in procs]+['end program']+sum([l for _,l in procs],[])
Path('scratch/native/constants.f90').write_text('\n'.join(lines)+'\n')
p=Path('scratch/native/Makefile');s=p.read_text();s+='\nconstants: constants.f90 libsmoke_oracle.so\n\t$(FC) $(FFLAGS) -o $@ $< -L. -lsmoke_oracle -Wl,-rpath,\'$$ORIGIN\'\n';p.write_text(s)
