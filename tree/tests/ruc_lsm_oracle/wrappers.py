import json
from pathlib import Path
rs=json.loads(Path('scratch/routines.json').read_text())
rs=[r for r in rs if r['name'] in ('ruc_land','ruc_ice','sfcdiags_ruclsm','rucinit','seaice_prepare','seaice_finish','correct_tsk_over_seaice','driver_forcing')]
out=['module ruc_oracle_wrappers','use iso_c_binding','contains']
cu=[];layout={}
for r in rs:
 name=r['unit']+'_'+r['name'];layout[name]=[]
 out+=['subroutine oracle_'+name+'(ptrs) bind(c)','use '+{'land':'module_ruc_land','ice':'module_ruc_ice','diags':'module_sf_sfcdiags_ruclsm','init':'ruc_init_slice','drivers':'ruc_driver_slice'}[r['unit']]+' , only: '+r['name'],'type(c_ptr),intent(in)::ptrs(*)']
 for i,a in enumerate(r['args']):
  d=r['symbols'][a];dims=d['dim'];ty={'float':'real(c_float)','int':'integer(c_int)'}[d['type']]
  if d['logical']:ty='logical(c_int)'
  if d['character']:
   out+=['integer(c_int),pointer :: '+a+'_code','character(len=32) :: '+a]
  else:out+=[ty+',pointer :: '+a+('('+','.join(':' for _ in dims)+')' if dims else '')]
  layout[name].append({'name':a,**d})
 # scalar dimensions must be associated first
 for is_array in (False,True):
  for i,a in enumerate(r['args']):
   d=r['symbols'][a];dims=d['dim']
   if bool(dims)!=is_array:continue
   if d['character']:
    out+=['call c_f_pointer(ptrs('+str(i+1)+'),'+a+'_code)',"select case("+a+"_code)","case(1)",a+"='USGS'","case(2)",a+"='MODIS'","case default",a+"='MODIFIED_IGBP_MODIS_NOAH'","end select"]
   else:
    shape=[]
    for dim in dims:
     lo,hi=dim.split(':') if ':' in dim else ('1',dim);shape.append('('+hi+')-('+lo+')+1')
    out+=['call c_f_pointer(ptrs('+str(i+1)+'),'+a+(',['+','.join(shape)+']' if shape else '')+')']
 for j,a in enumerate(a for a in r['args'] if r['symbols'][a]['optional']):
  out+=['block','integer(c_int),pointer::flag','call c_f_pointer(ptrs('+str(len(r['args'])+j+1)+'),flag)','if(flag==0) nullify('+a+')','end block']
 out+=['call '+r['name']+'('+','.join(r['args'])+')','end subroutine']
 if name=='land_ruc_land':
  lake=r['args'].index('lakemask');model=r['args'].index('lakemodel')
  cu += [
   '// RRFS-MPAS 1d6e370e src/core_atmosphere/physics/physics_noaa/RUCLSM/MPAS/module_ruc_land.F90:498-582',
   'extern "C" __global__ void ruc_land_tile_preflight(unsigned long long *ptrs,int *stop,int nc) {',
   ' if(blockIdx.x!=0 || threadIdx.x!=0)return;',
   ' stop[0]=nc; stop[1]=0;',
   ' if(((int*)ptrs['+str(model)+'])[0]!=1)return;',
   ' for(int c=0;c<nc;++c)if(((float*)ptrs['+str(lake)+'])[c]==1.f){stop[0]=c;stop[1]=1;return;}',
   '}']
 # Device argument buffers are per-column, cell-fastest profiles; scalar
 # inputs repeat for each column too. This ABI keeps all array shapes exact.
 cu+=['extern "C" __global__ void oracle_'+name+'(unsigned long long *ptrs,float *tables,float *work,int *error,int nc) {',' int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;',' Tables tb(tables); Arena arena{work+c,0,nc,error};']
 if name=='land_ruc_land':
  stop_index=len(r['args'])+sum(r['symbols'][a]['optional'] for a in r['args'])
  cu+=[' int *tile_stop=(int*)ptrs['+str(stop_index)+']; if(c>=tile_stop[0])return; arena.preparation_only=tile_stop[1]!=0;']
 if name in ('land_ruc_land','ice_ruc_ice'):
  entry_index=len(r['args'])+sum(r['symbols'][a]['optional'] for a in r['args'])+int(name=='land_ruc_land')
  cu+=[' if(ptrs['+str(entry_index)+']) arena.entry_ilnb=((int*)ptrs['+str(entry_index)+'])+c;',
       ' if(ptrs['+str(entry_index+1)+']) arena.snowtemp_calls=((int*)ptrs['+str(entry_index+1)+'])+c;']
 for i,a in enumerate(r['args']):
  d=r['symbols'][a]
  if not d['dim']:cu+=[' '+d['type']+' &'+a+' = (('+d['type']+'*)ptrs['+str(i)+'])[c];']
 for i,a in enumerate(r['args']):
  d=r['symbols'][a];dims=d['dim']
  if dims:
   bounds=[];sizes=[]
   for dim in dims:
    lo,hi=dim.split(':') if ':' in dim else ('1',dim);bounds.append(lo);sizes.append('('+hi+'-'+lo+'+1)')
   bounds+=['1']*(3-len(bounds));sizes+=['1']*(3-len(sizes))
   cu+=[' FArray<'+d['type']+'> '+a+'{(('+d['type']+'*)ptrs['+str(i)+'])+c,'+','.join(bounds)+',nc,nc*'+sizes[0]+',nc*'+sizes[0]+'*'+sizes[1]+','+','.join(sizes)+'};']
 cu+=[' '+name+'(tb,arena,'+','.join(r['args']+['(((int*)ptrs['+str(len(r['args'])+j)+'])[c]!=0)' for j,a in enumerate(a for a in r['args'] if r['symbols'][a]['optional'])])+');','}']
out+=['subroutine oracle_tables() bind(c)','use ruc_init_slice,only:ruclsm_soilvegparm,dm_info','type(dm_info)::dminfo',"character(len=8)::veg='MODI-RUC',soil='STAS-RUC'",'call ruclsm_soilvegparm(dminfo,veg,soil)','end subroutine','end module']
Path('scratch/native/wrappers.f90').write_text('\n'.join(out))
Path('scratch/launch.cu').write_text('\n'.join(cu))
Path('scratch/argument-layout.json').write_text(json.dumps(layout,indent=2))
