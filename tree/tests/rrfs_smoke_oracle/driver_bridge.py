from pathlib import Path
import json
rs=json.loads(Path('scratch/routines.json').read_text())
lines=['module smoke_driver_bridge','use iso_c_binding','use smoke_driver_slice','implicit none','type, bind(c) :: argument','type(c_ptr) :: data','integer(c_int) :: lo(4),hi(4),ivalue','real(c_float) :: fvalue','integer(c_int) :: present','end type','contains']
for n,r in rs.items():
 if r['module']!='smoke_driver_slice':continue
 lines+=['subroutine oracle_'+n+'(args) bind(c)','type(argument), intent(inout) :: args(*)','integer :: string_index']
 for a in r['args']:
  d=r['symbols'][a];t='real(c_float)' if d['type']=='float' else 'integer(c_int)' if d['type']=='int' else 'logical' if d['type']=='bool' else 'character(100)'
  if d['dim']:lines.append(t+', pointer :: '+a+'('+','.join(':' for _ in d['dim'])+')')
  elif d['type']=='FStr':lines+=[t+' :: '+a,'character(c_char), pointer :: bytes_'+a+'(:)']
  else:lines.append(t+' :: '+a)
 for i,a in enumerate(r['args'],1):
  d=r['symbols'][a];arg='args('+str(i)+')'
  if d['dim']:lines.append('call c_f_pointer('+arg+'%data,'+a+',['+','.join(arg+'%hi('+str(k)+')-'+arg+'%lo('+str(k)+')+1' for k in range(1,len(d['dim'])+1))+'])')
  elif d['type']=='FStr':lines+=['call c_f_pointer('+arg+'%data,bytes_'+a+',[100])','do string_index=1,100',a+'(string_index:string_index)=bytes_'+a+'(string_index)','end do']
  else:lines.append(a+' = '+arg+('%fvalue' if d['type']=='float' else '%ivalue /= 0' if d['type']=='bool' else '%ivalue'))
 lines.append('call '+n+'('+','.join(r['args'])+')');lines.append('end subroutine')
lines.append('end module')
Path('scratch/native/driver_bridge.f90').write_text('\n'.join(lines)+'\n')
