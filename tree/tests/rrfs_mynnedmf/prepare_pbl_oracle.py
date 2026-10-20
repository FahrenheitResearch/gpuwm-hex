"""Extract the complete pinned MYNN PBL packing/writeback loops, unchanged."""
import argparse,hashlib,json,re
from pathlib import Path

PIN='8fc75f442ccd390afcab603cdbf1f266266d377812bd9a8c069b066af9505389'

def prepare(source,out):
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:
        raise RuntimeError('Pinned original PBL driver SHA-256 mismatch')
    lines=raw.decode().splitlines()
    # Only framework pointer lookup calls are replaced by bank pointer aliases.
    pre='\n'.join(lines[786:846])
    post='\n'.join(line for line in lines[1134:1237]
                   if not line.lstrip().startswith(('call mpas_pool_get_array','case(')))
    names={}
    for name,indices in re.findall(r'\b(\w+)\((i,k,j|k,i|i,j|i)\)',pre+'\n'+post):
        names[name]=indices
    groups={kind:sorted(n for n,v in names.items() if v==kind)
            for kind in ('i,k,j','k,i','i,j','i')}
    assert [len(groups[k]) for k in groups]==[32,31,9,9],groups
    groups['i'].remove('meshDensity')
    aliases=[];declarations=[]
    banks={'i,k,j':('l3','ims:ime,kms:kme,jms:jme'),
           'k,i':('p3','1:nk,1:nc'),'i,j':('l2','ims:ime,jms:jme'),'i':('p2','1:nc')}
    for kind,rows in groups.items():
        bank,bounds=banks[kind]
        rank=len(kind.split(','))
        for index,name in enumerate(rows,1):
            declarations.append('real(c_float),pointer::'+name+'('+','.join(':' for _ in range(rank))+')')
            aliases.append(name+'('+bounds+') => '+bank+'('+','.join(':' for _ in range(rank))+','+str(index)+')')
    mappings=dict(re.findall(r'\b(\w+)\s*=\s*(\w+)', '\n'.join(lines[1592:1660])))
    reverse={local:arg for arg,local in mappings.items()}
    assert all(n in reverse for kind in ('i,k,j','i,j') for n in groups[kind])
    text='''module pbl_slice_oracle
use iso_c_binding
implicit none
contains
subroutine pbl_slice(mode,nc,nk,b,flags,len_disp,meshDensity,p3,l3,p2,l2) bind(C)
integer(c_int),value::mode,nc,nk
integer(c_int),intent(in)::b(12),flags(4)
real(c_float),value::len_disp
real(c_float),intent(in)::meshDensity(nc)
real(c_float),target,intent(inout)::p3(nk,nc,31),l3(b(2)-b(1)+1,b(4)-b(3)+1,b(6)-b(5)+1,32)
real(c_float),target,intent(inout)::p2(nc,8),l2(b(2)-b(1)+1,b(6)-b(5)+1,9)
integer,parameter::RKIND=c_float
integer::ims,ime,kms,kme,jms,jme,its,ite,kts,kte,jts,jte,i,j,k
logical::f_ni,f_nc,f_nifa,f_nwfa
'''+ '\n'.join(declarations)+'''
ims=b(1);ime=b(2);kms=b(3);kme=b(4);jms=b(5);jme=b(6)
its=b(7);ite=b(8);kts=b(9);kte=b(10);jts=b(11);jte=b(12)
f_ni=flags(1)/=0;f_nc=flags(2)/=0;f_nifa=flags(3)/=0;f_nwfa=flags(4)/=0
'''+ '\n'.join(aliases)+'\nif(mode==0) then\n'+pre+'\nelse\n'+post+'''
endif
end subroutine
end module
'''
    out.mkdir(parents=True,exist_ok=True)
    target=out/'pbl_slice.F90';target.write_text(text)
    manifest={'schema':'r6-original-pbl-slice-v1','source_sha256':PIN,
              'source_ranges':['787-846','1135-1237'],
              'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'groups':groups,'local_arguments':reverse,
              'scope':'Original complete packing/writeback loops; framework lookups replaced only. Synthetic boundary evidence, not real pool capture or forecast.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.out)))
