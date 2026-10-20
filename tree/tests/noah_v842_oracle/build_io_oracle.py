"""Extract native MPAS field conversion loops with framework-free array storage."""
import os
import re
from pathlib import Path
from hexcore.mpasphys.noah_v842 import COLUMN_LAYOUTS, _MPAS_INPUT_FIELDS

source = (Path(os.environ['MPAS_REF']) / 'src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F').read_text().splitlines()
fields = _MPAS_INPUT_FIELDS
layout = COLUMN_LAYOUTS['driver']
widths = {'soil':4, 'atmosphere':2, 'pressure':2}
offsets = {}; width = 0
decls = []; load = []; save = []
for key, shape in fields.items():
    if shape == 'integer_surface':
        decls += [f'integer::{key}(ncells)']
        load += [f'{key}=ivals({1 if key=="ivgtyp" else 2},:)']
        continue
    size = widths.get(shape, 1)
    offsets[key] = width; width += size
    decls += [f'real:: {key}({str(size)+"," if size>1 else ""}ncells)']
    for k in range(size):
        ref = f'{key}({k+1},:)' if size>1 else key
        load += [f'{ref}=fields({offsets[key]+k+1},:)']
        save += [f'fields({offsets[key]+k+1},:)={ref}']
for key in ('cpm','swddif','swddir'):
    decls += [f'real::{key}(ncells)']; load += [f'{key}=0']

body = '\n'.join(source[356:474])
body = '\n'.join(line for line in body.splitlines() if 'call mpas_pool_get_array' not in line)
pnames = set(re.findall(r'\b[a-z_0-9]+_p\b', body.lower()))
soilnames = {'sh2o_p','smcrel_p','smois_p','tslb_p'}
for key in sorted(pnames):
    dim = '4' if key=='dzs_p' else 'ncells,4,1' if key in soilnames else 'ncells,1'
    typ = 'integer' if key in ('ivgtyp_p','isltyp_p','utype_urb_p') else 'real'
    decls += [f'{typ}::{key}({dim})']
decls += ['integer::i,j,n,c,its,ite,jts,jte','integer,parameter::num_soils=4,RKIND=c_float',
          'character(len=8)::config_microp_scheme,config_convection_scheme']
load += ['its=start+1;ite=start+count;jts=1;jte=1',
         "config_microp_scheme='off';config_convection_scheme='off'",
         "if(options(5)/=0)config_microp_scheme='on'",
         "if(options(6)/=0)config_convection_scheme='on'"]
rename = {'DZ8W':'dz8w','QV3D':'qv','P8W3D':'pres2_hyd','T3D':'t','TSK':'tsk','ALBEDO':'sfc_albedo',
          'ALBBCK':'sfc_albbck','EMISS':'sfc_emiss','EMBCK':'sfc_emibck','RIB':'br','RAINBL':'rainbl',
          'FLX4_2D':'flxsnow','FVB_2D':'fvbsnow','FBUR_2D':'fbursnow','FGSN_2D':'fgsnsnow'}
def reference(name):
    base = name.split('[')[0]; key=rename.get(base,base.lower())
    sub = int(name.split('[')[1][:-1])+1 if '[' in name else None
    if base=='DT':return 'dt'
    if base=='XICE_THRESHOLD':return 'threshold'
    if base in ('DZ8W','QV3D','P8W3D','T3D'):return f'{key}({sub},i)'
    if base=='DZS':return f'dzs_p({sub})'
    if sub:return f'{key}_p(i,{sub},1)'
    return f'{key}_p(i,1)'
header = f'''use iso_c_binding
implicit none
integer(c_int),value::ncells,start,count
integer(c_int)::ivals(2,ncells),options(6),ix(7,count)
real(c_float)::fields({width},ncells),x(79,count)
real(c_float),value::dt,threshold
'''+ '\n'.join(decls) + '\n' + '\n'.join(load) + '\n'
packing = ['do i=its,ite','c=i-start']
packing += [f'x({k+1},c)={reference(nm)}' for k,nm in enumerate(layout['float'])]
packing += ['ix(1,c)=ivgtyp_p(i,1)','ix(2,c)=isltyp_p(i,1)','ix(3,c)=options(1)',
            'ix(4,c)=options(2)','ix(5,c)=options(3)','ix(6,c)=options(4)','ix(7,c)=1','enddo']
out = '! MPAS v8.4.2 mpas_atmphys_driver_lsm.F:357-477; original numeric loops.\n'
out += 'subroutine io_from(fields,ivals,x,ix,ncells,start,count,dt,threshold,options) bind(C,name="io_from")\n'
out += header + body + '\n' + '\n'.join(packing) + '\nend subroutine\n'
to_body = '\n'.join(source[564:631])
to_body = '\n'.join(line for line in to_body.splitlines() if 'call mpas_pool_get_array' not in line)
out += '! MPAS v8.4.2 mpas_atmphys_driver_lsm.F:565-628; original numeric loops.\n'
out += 'subroutine io_to(fields,ivals,x,ix,ncells,start,count,dt,threshold,options) bind(C,name="io_to")\n'
out += header + body + '\ndo i=its,ite\nc=i-start\n'
for k,nm in enumerate(layout['float']):
    ref=reference(nm)
    if '_p(' in ref and not ref.startswith('dzs_'):out += f'{ref}=x({k+1},c)\n'
out += 'enddo\n' + to_body + '\n' + '\n'.join(save) + '\nend subroutine\n'
Path(__file__).with_name('io_replay.F90').write_text(out)
