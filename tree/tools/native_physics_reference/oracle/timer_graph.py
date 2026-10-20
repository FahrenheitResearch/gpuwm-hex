"""Typed capture of the complete native timer object graph.

MPAS src/framework/mpas_timer.F:45 and mpas_timer_types.inc:1-23.
Timer values are observed native state; external MPI resources, domain aliases
and reproducibility of future wall-clock observations are separate obligations.
"""
from dataclasses import asdict
import hashlib
import re
from fortran_source import active_source, statements, declarations


def schema(root):
    paths=['src/framework/mpas_timer_types.inc','src/framework/mpas_dmpar_types.inc']
    types={}
    for path in paths:
        current=None
        for statement in statements(active_source((root/path).read_text())):
            match=re.fullmatch(r'type\s+(\w+)',statement.text,re.I)
            if match:
                current=match[1].lower()
                types[current]=dict(path=path,line=statement.start+1,fields=[])
            elif re.match(r'end\s*type\b',statement.text,re.I):current=None
            elif current:
                types[current]['fields'] += [dict(asdict(v),supported=v.supported,
                    line=statement.start+1)for v in declarations(statement.text)]
    needed=set();pointers={'mpas_timer_root':dict(type='type(mpas_timer_root)',rank=0,nested='mpas_timer_root')}
    def visit(typ):
        if typ in needed:return
        if typ not in types:raise ValueError('unknown native timer type '+typ)
        needed.add(typ)
        for field in types[typ]['fields']:
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            rank=len(field['dims'])
            if rank>1:raise ValueError('native timer storage rank changed')
            if nested:
                if rank or 'allocatable'in field['attrs']:
                    raise ValueError('native timer derived storage changed')
                target=nested[1].lower();visit(target)
                if 'pointer'in field['attrs']:
                    pointers[target]=dict(type='type('+target+')',rank=0,nested=target)
            elif not field['supported']:
                raise ValueError('unsupported native timer leaf '+typ+'.'+field['name'])
            elif 'pointer'in field['attrs']:
                if rank!=1:raise ValueError('native timer primitive pointer changed')
                tag='leaf_'+re.sub(r'[^a-z0-9]','',field['type'].lower())
                pointers[tag]=dict(type=field['type'],rank=rank,nested=None)
            elif 'allocatable'in field['attrs']and rank!=1:
                raise ValueError('native timer primitive allocation changed')
    visit('mpas_timer_root')
    # The pinned native builds enable MPAS_USE_MPI_F08. Inventory both source
    # branches; their handles are opaque native words, not recreated resources.
    f08=[];inside=False
    for statement in statements(active_source((root/paths[1]).read_text(),{'MPAS_USE_MPI_F08':1})):
        if re.fullmatch(r'type\s+dm_info',statement.text,re.I):inside=True
        elif inside and re.match(r'end\s*type\b',statement.text,re.I):break
        elif inside:
            f08.extend(dict(asdict(v),line=statement.start+1)for v in declarations(statement.text)if v.name in ('comm','info'))
    if {v['name']:re.sub(r'\s+','',v['type'].lower())for v in f08}!={'comm':'type(mpi_comm)','info':'type(mpi_info)'}:
        raise ValueError('native MPI F08 timer handle declarations changed')
    for field in types['dm_info']['fields']:
        if field['name']in ('comm','info'):
            field['opaque_mpi_handle']=True
    return dict(types={n:types[n]for n in sorted(needed)},pointers=pointers,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()for p in paths},
        mpi_f08_handle_declarations=f08,mpi_handle_encoding='native storage transferred to one default integer word; width checked at runtime',
        external_aliases_complete=False,mpi_resources_restored=False,
        wall_clock_reproducibility_proven=False)


def generate(root):
    info=schema(root);types=info['types'];pointers=info['pointers']
    text='''! GENERATED from native MPAS timer declarations.
module mpas_oracle_timer
 use mpas_kind_types
 use mpas_derived_types,only:mpas_timer_root,mpas_timer_node,dm_info
 use mpas_oracle_dump,only:oracle_write,oracle_read,oracle_header,oracle_enabled
 implicit none
 private
 public::oracle_timer_capture,oracle_timer_restore
'''
    for name,p in pointers.items():
        shape='(:)'if p['rank']else''
        text+=f' type ref_{name}\n {p["type"]},pointer::p{shape}=>null()\n end type\n type(ref_{name}),allocatable::refs_{name}(:)\n'
    text+=' contains\n subroutine clear_refs\n'
    for name in pointers:text+=f' if(allocated(refs_{name}))deallocate(refs_{name})\n'
    text+=''' end subroutine
 function number(i)result(s)
 integer,intent(in)::i
 character(len=32)::s
 write(s,'(i0)')i
 end function
 subroutine oracle_timer_capture(key,phase,name,value,event)
 character(len=*),intent(in)::key,phase,name
 integer,intent(in)::event
 type(mpas_timer_root),pointer,intent(in)::value
 if(.not.oracle_enabled(key))return
 call clear_refs
 call write_ptr_mpas_timer_root(value,key,phase,name,name,event)
 call clear_refs
 end subroutine
 subroutine oracle_timer_restore(prefix,name,value)
 character(len=*),intent(in)::prefix,name
 type(mpas_timer_root),pointer,intent(inout)::value
 call clear_refs
 call read_ptr_mpas_timer_root(value,prefix,name,name)
 call clear_refs
 end subroutine
'''
    def leaves(typ,mode):
        result=''
        for field in types[typ]['fields']:
            value='p%'+field['name'];label="trim(node)//'."+field['name']+"'"
            nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
            if 'pointer'in field['attrs']:
                tag=nested[1].lower()if nested else'leaf_'+re.sub(r'[^a-z0-9]','',field['type'].lower())
                args=f'{value},key,phase,base,{label},event'if mode=='write'else f'{value},prefix,base,{label}'
                result+=f' call {mode}_ptr_{tag}({args})\n'
            elif nested:
                raise ValueError('native timer inline derived value needs a new typed visitor')
            elif field.get('opaque_mpi_handle'):
                result+=f" if(storage_size({value})/=storage_size(0))error stop 'oracle timer MPI handle width changed'\n"
                if mode=='write':
                    result+=f' call oracle_write(key,phase,{label},transfer({value},0),event)\n'
                else:
                    result+=f" call oracle_read(prefix//'.pre.'//{label}//'.bin',mpi_handle)\n {value}=transfer(mpi_handle,{value})\n"
            else:
                alloc='allocatable'in field['attrs'];rank=len(field['dims'])
                if alloc:
                    if mode=='write':
                        result+=f" call oracle_write(key,phase,{label}//'.present',allocated({value}),event)\n if(allocated({value}))then\n"
                    else:
                        result+=f" call oracle_read(prefix//'.pre.'//{label}//'.present.bin',present)\n if(allocated({value}))deallocate({value})\n if(present)then\n call oracle_header(prefix//'.pre.'//{label}//'.bin',code,width,rank_n,dims,lb,unit)\n close(unit)\n if(rank_n/=1)error stop 'timer allocation rank changed'\n allocate({value}(lb(1):lb(1)+dims(1)-1))\n"
                if mode=='write':
                    lower=f',lower=lbound({value})'if rank else''
                    result+=f' call oracle_write(key,phase,{label},{value},event{lower})\n'
                else:result+=f" call oracle_read(prefix//'.pre.'//{label}//'.bin',{value})\n"
                if alloc:result+=' endif\n'
        return result
    for name,p in pointers.items():
        shape='(:)'if p['rank']else'';refs='refs_'+name
        for mode in ('write','read'):
            writer=mode=='write';args='p,key,phase,base,path,event'if writer else'p,prefix,base,path'
            text+=f' recursive subroutine {mode}_ptr_{name}({args})\n {p["type"]},pointer,intent({"in"if writer else"inout"})::p{shape}\n'
            text+=(' character(len=*),intent(in)::key,phase,base,path\n integer,intent(in)::event\n'if writer else' character(len=*),intent(in)::prefix,base,path\n')
            text+=f' type(ref_{name}),allocatable::grown(:)\n integer::id,n,j,extent(2),code,width,rank_n,dims(5),lb(5),unit,mpi_handle\n logical::present\n character(len=256)::node\n n=0\n if(allocated({refs}))n=size({refs})\n'
            if writer:
                text+=f" id=0\n if(associated(p))then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n id=j\n exit\n endif\n enddo\n if(id==0)id=n+1\n endif\n call oracle_write(key,phase,path//'.target',id,event)\n"
                if p['rank']:
                    text+=" if(id>0)then\n extent=[lbound(p,1),size(p)]\n call oracle_write(key,phase,path//'.bounds',extent,event)\n endif\n"
                text+=' if(id<=n)return\n'
            else:
                text+=" call oracle_read(prefix//'.pre.'//path//'.target.bin',id)\n if(id==0)then\n nullify(p)\n return\n endif\n"
                text+=" if(id<0.or.id>n+1)error stop 'timer invalid target ordinal or forward reference'\n"
                if p['rank']:text+=" call oracle_read(prefix//'.pre.'//path//'.bounds.bin',extent)\n if(extent(2)<0)error stop 'timer negative pointer extent'\n"
                text+=f' if(id<=n)then\n'
                if p['rank']:
                    text+=f" if(size({refs}(id)%p)/=extent(2))error stop 'timer alias extent differs'\n p(extent(1):)=>{refs}(id)%p\n"
                else:text+=f' p=>{refs}(id)%p\n'
                text+=' return\n endif\n'
                text+=f' if(associated(p))then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n nullify(p)\n exit\n endif\n enddo\n endif\n'
                if p['rank']:
                    text+=' if(associated(p))then\n if(size(p)/=extent(2).or.lbound(p,1)/=extent(1))nullify(p)\n endif\n if(.not.associated(p))allocate(p(extent(1):extent(1)+extent(2)-1))\n'
                else:
                    text+=' if(.not.associated(p))then\n allocate(p)\n'
                    for field in types[p['nested']]['fields']:
                        if 'pointer'in field['attrs']:text+=' nullify(p%'+field['name']+')\n'
                    text+=' endif\n'
            text+=f" node=base//'.node.{name}.'//trim(number(id))\n allocate(grown(n+1))\n if(n>0)grown(:n)={refs}\n grown(n+1)%p=>p\n call move_alloc(grown,{refs})\n"
            if p['nested']:text+=leaves(p['nested'],mode)
            elif writer:text+=" call oracle_write(key,phase,trim(node)//'.value',p,event,lower=lbound(p))\n"
            else:text+=" call oracle_read(prefix//'.pre.'//trim(node)//'.value.bin',p)\n"
            text+=' end subroutine\n'
    text+='''end module mpas_oracle_timer
! Framework is built before physics; these external native-ABI bridges avoid
! making mpas_timer USE a generated physics module during that earlier build.
subroutine oracle_framework_timer_capture(key,phase,name,value,event)
 use mpas_derived_types,only:mpas_timer_root
 use mpas_oracle_timer,only:oracle_timer_capture
 implicit none
 character(len=*),intent(in)::key,phase,name
 integer,intent(in)::event
 type(mpas_timer_root),pointer,intent(in)::value
 call oracle_timer_capture(key,phase,name,value,event)
end subroutine
subroutine oracle_framework_timer_restore(prefix,name,value)
 use mpas_derived_types,only:mpas_timer_root
 use mpas_oracle_timer,only:oracle_timer_restore
 implicit none
 character(len=*),intent(in)::prefix,name
 type(mpas_timer_root),pointer,intent(inout)::value
 call oracle_timer_restore(prefix,name,value)
end subroutine
'''
    return text,info


def interface_source():
    """Native pointer descriptors require an explicit external interface."""
    return '''
 private::oracle_framework_timer_capture,oracle_framework_timer_restore
 interface
 subroutine oracle_framework_timer_capture(key,phase,name,value,event)
  use mpas_derived_types,only:mpas_timer_root
  character(len=*),intent(in)::key,phase,name
  integer,intent(in)::event
  type(mpas_timer_root),pointer,intent(in)::value
 end subroutine
 subroutine oracle_framework_timer_restore(prefix,name,value)
  use mpas_derived_types,only:mpas_timer_root
  character(len=*),intent(in)::prefix,name
  type(mpas_timer_root),pointer,intent(inout)::value
 end subroutine
 end interface
'''


def audit_membership(info,prefix,phase,name,filenames,read):
    """Audit native graph membership and alias views using Rust controls.

    The registry is populated before descending, exactly as in the generated
    reader. This covers cycles without authorizing MPI handles, external
    aliases or future wall-clock observations as restored resources.
    """
    expected,missing,issues=set(),set(),[]
    counts={tag:0 for tag in info['pointers']};extents={}
    stem=f'{prefix}.{phase}.'
    def member(path,code=None,width=None,shape=None,lower=None):
        filename=stem+path+'.bin';expected.add(filename)
        if filename not in filenames:
            missing.add(filename);return None
        header=read('header',filename,'-')
        if (header['name']!=path or header['phase']!=phase or
            code is not None and header['code']!=code or
            width is not None and header['width']!=width or
            shape is not None and header['shape']!=shape or
            lower is not None and header['lower']!=lower):
            issues.append(dict(file=filename,reason='timer record layout or context disagrees with native declaration'))
            return None
        return header
    def control(path,code,indices='-'):
        if member(path,code,4,[]if indices=='-'else[2])is None:return None
        value=read('sample',stem+path+'.bin',indices)['value']
        if (code==4 and type(value)is not bool)or(code==3 and type(value)is not int):
            issues.append(dict(path=path,reason='timer control is not a native integer/logical'));return None
        return value
    def leaf(field,path,shape=None,lower=None):
        typ=field['type'].lower()
        code=5 if typ.startswith('character')else 4 if typ.startswith('logical')else 3 if typ.startswith('integer')else 2 if 'r8kind'in typ else 1
        header=member(path,code,1 if code==5 else 8 if code==2 else 4,shape,lower)
        if header is not None and len(header['shape'])!=len(field['dims']):
            issues.append(dict(path=path,reason='timer primitive storage rank changed'))
    def leaves(typ,path):
        for field in info['types'][typ]['fields']:
            label=path+'.'+field['name']
            if 'pointer'in field['attrs']:
                nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',field['type'],re.I)
                tag=nested[1].lower()if nested else'leaf_'+re.sub(r'[^a-z0-9]','',field['type'].lower())
                pointer(tag,label)
            elif 'allocatable'in field['attrs']:
                if control(label+'.present',4)is True:leaf(field,label)
            else:leaf(field,label,[]if not field['dims']else None)
    def pointer(tag,path):
        ordinal=control(path+'.target',3)
        if ordinal is None or ordinal==0:return
        current=counts[tag]
        if ordinal<0 or ordinal>current+1:
            issues.append(dict(path=path,ordinal=ordinal,reason='timer target ordinal missing or forward reference'));return
        declaration=info['pointers'][tag]
        lower=extent=None
        if declaration['rank']:
            lower=control(path+'.bounds',3,'1');extent=control(path+'.bounds',3,'2')
            if lower is None or extent is None:return
            if extent<0:
                issues.append(dict(path=path,reason='timer pointer extent is negative'));return
        if ordinal<=current:
            if declaration['rank'] and extent!=extents[(tag,ordinal)]:
                issues.append(dict(path=path,reason='timer alias extent differs'))
            return
        counts[tag]=ordinal
        node=name+f'.node.{tag}.{ordinal}'
        if declaration['nested']:leaves(declaration['nested'],node)
        else:
            extents[(tag,ordinal)]=extent
            leaf(dict(type=declaration['type'],dims=[':']),node+'.value',[extent],[lower])
    pointer('mpas_timer_root',name)
    observed={f for f in filenames if f.startswith(stem+name+'.')}
    unexpected=sorted(observed-expected)
    return dict(expected_records=len(expected),missing=sorted(missing),unexpected=unexpected,
        topology_issues=issues,typed_node_counts=counts,external_aliases_complete=False,
        mpi_resources_restored=False,wall_clock_reproducibility_proven=False,
        global_hidden_state_complete=False,complete=not missing and not unexpected and not issues,
        scope='Represented native timer graph membership and internal typed topology only')
