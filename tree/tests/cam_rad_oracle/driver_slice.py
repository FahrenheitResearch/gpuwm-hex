"""Extract only the native CAM branches of MPAS radiation driver conversions."""
from pathlib import Path
import re
import translate as tr


NAMES=('radiation_lw_from_mpas','radiation_lw_to_mpas',
       'radiation_camlw_to_mpas','radiation_sw_from_mpas','radiation_sw_to_mpas','radconst')


def declarations(path):
    result={}
    for _,line in tr.statements(path):
        if line.startswith(('real','integer','logical')) and '::' in line:
            for v in tr.declaration(line):result[v.name]=v
    return result


def generate(reference):
    global LAST_ROUTINES
    LAST_ROUTINES={}
    physics=reference/'src/core_atmosphere/physics'
    globals_=declarations(physics/'mpas_atmphys_vars.F')
    globals_.update(declarations(physics/'mpas_atmphys_constants.F'))
    globals_['degrad']=tr.Variable('degrad','float',[])
    result=[];schemas={};inventory=[]
    for suffix in ('lw','sw'):
        path=physics/('mpas_atmphys_driver_radiation_'+suffix+'.F')
        current=None;selected=True;stack=[]
        for n,line in tr.statements(path):
            match=re.match(r'subroutine (\w+)\s*\(',line)
            if match:
                current=tr.Routine(match[1],tr.split(line[line.index('(')+1:-1]),None,path.name,n)
                continue
            if line.startswith('end subroutine'):
                if current and current.name in NAMES:
                    current.end=n
                    symbols=globals_|current.variables
                    text='\n'.join(s for _,s in current.body).replace('_rkind','')
                    used={s for s in tr.TOKEN.findall(text) if re.fullmatch(r'[a-z_]\w*',s)}
                    locals_={s for s in ('i','j','k','n') if s in current.variables and s in used}
                    if current.name=='radconst':
                        locals_={s for s in current.variables if s not in current.args and s in used}
                    fields=[v for name,v in sorted(symbols.items()) if name in used and name not in locals_ and v.typ!='char']
                    assigned={re.match(r'\s*([a-z_]\w*)\s*=',s)[1]
                              for _,s in current.body if re.match(r'\s*[a-z_]\w*\s*=(?!=)',s)}
                    for v in fields:
                        if not v.dims: v.intent='inout' if v.name in assigned else 'in'
                    unknown=used-set(symbols)-set(tr.PREC)-{'do','enddo','end','if','then','endif','else','exp','sin','cos','asin','sqrt','mod','real','int','abs','max','min','and','or','not','rkind','return'}
                    if unknown:raise ValueError((current.name,'unknown fields',sorted(unknown)))
                    args=['CamArray<'+v.typ+'> '+v.name if v.dims else v.typ+('& ' if v.intent=='inout' else ' ')+v.name for v in fields]
                    result+=['// MPAS '+path.name+':'+str(current.start)+'-'+str(n),
                             'CAM_HD void '+current.name+'('+','.join(args)+') {']
                    result+=[symbols[s].typ+' '+s+';' for s in sorted(locals_)]
                    for lineno,stmt in current.body:
                        result+=['// MPAS '+path.name+':'+str(lineno),tr.body(lineno,stmt.replace('_rkind',''),symbols,current)]
                    result+=['}']
                    schemas[current.name]=fields
                    LAST_ROUTINES[current.name]=(current,locals_,symbols)
                    inventory.append({'name':current.name,'source':path.name,'first_line':current.start,'last_line':n})
                current=None;continue
            if not current or current.name not in NAMES:continue
            if line.startswith(('type(', 'character', 'implicit','use ','return')):continue
            decl=tr.declaration(line)
            if decl:
                current.variables.update({v.name:v for v in decl});continue
            if 'select case' in line:
                stack.append(selected);selected=False;continue
            if line.startswith('case'):
                selected=stack[-1] and ('"cam_'+suffix+'"' in line or "'cam_"+suffix+"'" in line)
                continue
            if line.startswith('end select'):
                selected=stack.pop();continue
            if not selected or line.startswith(('call mpas_pool_', 'call mpas_log_')):continue
            current.body.append((n,line))
    assert set(schemas)==set(NAMES)
    clock=[tr.Variable(n,'float',[],intent='inout' if n in ('xtime_m','radt') else 'in')
           for n in ('xtime_s','dt_radt','xtime_m','radt')]
    schemas['driver_clock']=clock
    result+=['// MPAS mpas_atmphys_driver_radiation_lw.F:888-902; mpas_atmphys_driver_radiation_sw.F:874-899',
             'CAM_HD void driver_clock(float xtime_s,float dt_radt,float& xtime_m,float& radt) {',
             'xtime_m=cam_div(xtime_s,60.f);radt=cam_div(dt_radt,60.f);','}']
    return '\n'.join(result)+'\n',schemas,inventory
