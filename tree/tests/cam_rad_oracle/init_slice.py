"""MPAS initialization numeric slices with Rust-owned table/pool inputs.

Source line selection removes allocation, file I/O, MPI and pool retrieval.
Every retained numeric statement is translated by the closed CAM parser.
The aerosol local maximum must still pass through the framework MPI reduction.
"""
from pathlib import Path
import re
import translate as tr

RANGES={
 'camradinit':[(110,142)],
 'radini':[(206,247)],
 'radaeini':[(309,331),(381,388)],
 'aerosol_init':[(456,479),(486,489),(503,512),(525,550)],
 'aer_optics_initialize':[(684,732)],
 'oznini':[(790,794),(836,871)],
}

class NumericSource:
    name='mpas_atmphys_camrad_init.F'
    def __init__(self,text):self.text=text
    def read_text(self):return self.text
    def read_bytes(self):return self.text.encode()


def extract(path):
    # Drop only the top macro-definition block; keep source line numbering.
    raw=path.read_text().splitlines()
    filtered=['' if i<15 and line.startswith('#') else line for i,line in enumerate(raw,1)]
    for i,line in enumerate(filtered):
        if 'DM_BCAST_MACRO' in line:filtered[i]=''
    source=NumericSource('\n'.join(filtered))
    routines={};current=None
    for n,line in tr.statements(source):
        start=re.match(r'subroutine (\w+)\((.*?)\)',line)
        if start:
            current=tr.Routine(start[1],tr.split(start[2]),None,path.name,n)
            routines[current.name]=current
        elif line.startswith('end subroutine'):
            current.end=n;current=None
        elif current:
            if line.startswith(('type(','implicit','character')):continue
            decl=tr.declaration(line)
            if decl:
                current.variables.update({v.name:v for v in decl})
            elif line.startswith('data '):
                name,vals=line[5:].split('/',1);current.variables[name.strip()].initial='DATA:'+vals.rsplit('/',1)[0]
            elif any(lo<=n<=hi for lo,hi in RANGES[current.name]):
                if n in (471,791):continue # scratch allocation / Rust decoded table
                current.body.append((n,line.replace('_rkind','')))
    return routines


def generate(path,module,core_routines):
    routines=extract(path)
    const_fields=('cp','r_d','r_v','gravity','ep_2','stbolt')
    r=routines['camradinit'];r.args=list(const_fields)
    for a in const_fields:r.variables[a]=tr.Variable(a,'float',[],intent='in')
    r.body=[(n,s.replace('radini(dminfo,','radini(')) for n,s in r.body]
    for name in ('radini','radaeini'):
        r=routines[name];r.args.remove('dminfo')
        r.body=[(n,s.replace('radaeini(dminfo,','radaeini(')) for n,s in r.body]
    r=routines['aerosol_init']
    r.args=['naerlevels','ncells','ncellssolve','nvertlevels','m_hybi','m_psp','pressure_b','aerosolcn','aerosolcp']+[a for a in r.variables if a.startswith('idx')]+['max_mxaerl']
    r.variables['mxaerl_local'].dims=['ncellssolve']
    for a in r.args:r.variables[a].intent='inout' if a=='max_mxaerl' else 'in'
    r=routines['aer_optics_initialize']
    # The provider decodes these records without interpolation or conversion.
    r.args=['rh_opac','ksul_opac','wsul_opac','gsul_opac',
            'kssam_opac','wssam_opac','gssam_opac','ksscm_opac','wsscm_opac','gsscm_opac',
            'kcphil_opac','wcphil_opac','gcphil_opac']
    for a in r.args:r.variables[a].intent='in'
    r.variables['_opac_ratio']=tr.Variable('_opac_ratio','double',['irh'])
    body=[]
    for n,s in r.body:
        if 'interpol(' in s:
            pattern=r'(\w+_opac)\(:,kbnd\)\s*/\s*\1\(1,kbnd\)'
            match=re.search(pattern,s)
            assert match,(n,s)
            body.append((n,'_opac_ratio(:)='+match[0]))
            s=s[:match.start()]+'_opac_ratio'+s[match.end():]
        body.append((n,s))
    r.body=body
    r=routines['oznini']
    r.args=['ncells','num_months','levsiz','latcell','loncell','pin','ozmixm','lat_ozone','ozmixin','degrad']
    r.variables['degrad']=tr.Variable('degrad','float',[],intent='in')
    r.variables['latsiz']=tr.Variable('latsiz','int',[],initial='64',const=True)
    r.variables['lonsiz']=tr.Variable('lonsiz','int',[],initial='1',const=True)
    for a in r.args:r.variables[a].intent='in'
    tr.ROUTINES=core_routines|routines
    lines=[];schemas={};inventory=[]
    for r in routines.values():
        symbols=module|r.variables
        used=set(tr.TOKEN.findall('\n'.join(s for _,s in r.body)))
        params=tuple(a for a,v in r.variables.items() if v.const)
        args=[]
        for a in r.args:
            v=r.variables[a]
            args.append(('CamArray<'+v.typ+'> _' if v.dims else ('const ' if v.intent=='in' else '')+v.typ+'& ')+a)
        lines+=['// MPAS '+r.file+':'+str(r.start)+'-'+str(r.end),
                'CAM_HD void '+r.name+'('+','.join(args)+') {','CamFrame _frame(arena);']
        for a in params:
            v=r.variables[a];lines.append('const '+v.typ+' '+a+'='+tr.scalar_initial(v.initial,symbols)+';')
        for a in r.args:
            v=r.variables[a]
            if v.dims:
                lines.append('auto '+a+'=_'+a+(';' if ':' in v.dims else '.rebound('+tr.dimensions(v,symbols)+');'))
        for a,v in r.variables.items():
            if a in r.args or a in params or a not in used:continue
            if v.dims:
                lines+=['auto '+a+'=arena.array<'+v.typ+'>('+tr.dimensions(v,symbols)+');']
                if v.initial:
                    lines+=['{const '+v.typ+' init[]={'+tr.initializer(v,symbols)+'};'+a+'.init(init,sizeof(init)/sizeof(init[0]));}']
            elif v.typ!='char':lines+=[v.typ+' '+a+';']
        for n,s in r.body:
            if s.startswith('call radaeini') or s.startswith('call radini'):
                # Inputs produced by expressions require typed temporaries because
                # the native reference ABI takes scalar actuals by address.
                callee=s[5:s.index('(')];actuals=tr.split(s[s.index('(')+1:-1]);converted=[]
                lines+=['// MPAS '+r.file+':'+str(n),'{']
                for j,a in enumerate(actuals):
                    v=routines[callee].variables[routines[callee].args[j]]
                    lines+=[v.typ+' temp'+str(j)+'='+tr.expr(a,symbols)+';'];converted+=['temp'+str(j)]
                lines+=[callee+'('+','.join(converted)+');if(error) return;}']
            else:
                if n==479:
                    # Reduction is over one dimension here, unlike the core's
                    # columnwise maxval(dim=2) assignment.
                    emitted='max_mxaerl=cam_maxval(mxaerl_local.section(cam_section(1,ncellssolve)));'
                else:emitted=tr.body(n,s,symbols,r)
                lines+=['// MPAS '+r.file+':'+str(n),emitted]
        lines+=['}']
        schemas[r.name]=[r.variables[a] for a in r.args]
        inventory.append({'name':r.name,'source':r.file,'first_line':r.start,'last_line':r.end,
                          'numeric_ranges':RANGES[r.name]})
    tr.ROUTINES=core_routines
    return '\n'.join(lines)+'\n',schemas,inventory
