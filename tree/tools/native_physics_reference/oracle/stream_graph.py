"""Native stream-manager connectivity saves, with explicit external pointers.

MPAS mpas_stream_manager.F:96-103,4894-5397, mpas_field_types.inc:246-276.
The native temporary nodes initialize ARRAY/NEXT; their other primitive words
are observed storage, not invented defaults. Non-null external pointees remain
unrepresented. Their presence never aborts an instrumented forecast.
"""
from dataclasses import asdict
import hashlib
import re
from fortran_source import active_source, statements, declarations


def schema(root):
    field_path='src/framework/mpas_field_types.inc'
    module_path='src/framework/mpas_stream_manager.F'
    fields=[];inside=False
    for st in statements(active_source((root/field_path).read_text())):
        if re.fullmatch(r'type\s+field2dinteger',st.text,re.I):inside=True
        elif inside and re.match(r'end\s*type\b',st.text,re.I):break
        elif inside:
            fields.extend(dict(asdict(v),supported=v.supported,line=st.start+1)for v in declarations(st.text))
    roots=[]
    for st in statements(active_source((root/module_path).read_text())):
        if re.match(r'contains\b',st.text,re.I):break
        roots.extend(v.name for v in declarations(st.text)
                     if re.fullmatch(r'type\s*\(\s*field2dinteger\s*\)',v.type,re.I))
    if len(roots)!=8 or len(set(roots))!=8:raise ValueError('native stream-manager roots changed')
    external=[]
    for f in fields:
        nested=re.fullmatch(r'type\s*\(\s*(\w+)\s*\)',f['type'],re.I)
        if nested and nested[1].lower()!='field2dinteger':
            if 'pointer'not in f['attrs']:raise ValueError('stream external storage changed')
            f['external_pointer']=True;external.append(f['name'])
        elif nested:
            if f['dims']or'pointer'not in f['attrs']:raise ValueError('stream node storage changed')
        elif not f['supported']or len(f['dims'])>2:raise ValueError('unsupported native stream leaf')
    return dict(root_names=roots,fields=fields,external_pointer_fields=external,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                       for p in (field_path,module_path)},
        external_aliases_complete=False,external_pointees_complete=False,
        uninitialized_primitive_storage_possible=True)


def tag(field):
    if re.fullmatch(r'type\s*\(\s*field2dinteger\s*\)',field['type'],re.I):return 'node'
    return 'leaf_'+re.sub(r'[^a-z0-9]','',field['type'].lower())+str(len(field['dims']))


def generate(root):
    info=schema(root);roots=info['root_names'];fields=info['fields']
    pointers={'node':dict(type='type(field2DInteger)',rank=0)}
    for f in fields:
        if 'pointer'in f['attrs']and not f.get('external_pointer'):
            pointers[tag(f)]=dict(type=f['type'],rank=len(f['dims']))
    text='''! GENERATED native stream connectivity graph.
module mpas_oracle_stream
 use mpas_kind_types
 use mpas_derived_types,only:field2DInteger
 use mpas_oracle_dump,only:oracle_write,oracle_read,oracle_header,oracle_enabled
 implicit none
 private
 public::oracle_stream_capture,oracle_stream_restore
'''
    def shape(rank):return '('+','.join(':'for _ in range(rank))+')'if rank else''
    for name,p in pointers.items():
        text+=f' type ref_{name}\n {p["type"]},pointer::p{shape(p["rank"])}=>null()\n end type\n type(ref_{name}),allocatable::refs_{name}(:)\n'
    text+=' contains\n subroutine clear_refs\n'
    for name in pointers:text+=f' if(allocated(refs_{name}))deallocate(refs_{name})\n'
    text+=''' end subroutine
 function number(i)result(s)
 integer,intent(in)::i
 character(len=32)::s
 write(s,'(i0)')i
 end function
'''
    for mode in ('capture','restore'):
        args='key,phase,name,'+','.join(roots)+',event'if mode=='capture'else'prefix,name,'+','.join(roots)
        text+=f' subroutine oracle_stream_{mode}({args})\n'
        text+=(' character(len=*),intent(in)::key,phase,name\n integer,intent(in)::event\n'if mode=='capture'else' character(len=*),intent(in)::prefix,name\n')
        text+=' type(field2DInteger),pointer,intent('+('in'if mode=='capture'else'inout')+')::'+','.join(roots)+'\n'
        if mode=='capture':text+=' if(.not.oracle_enabled(key))return\n'
        text+=' call clear_refs\n'
        for r in roots:
            args=f"{r},key,phase,name,name//'.{r}',event"if mode=='capture'else f"{r},prefix,name,name//'.{r}'"
            text+=f' call {"write"if mode=="capture"else"read"}_ptr_node({args})\n'
        text+=' call clear_refs\n end subroutine\n'
    for name,p in pointers.items():
        rank=p['rank'];refs='refs_'+name
        for mode in ('write','read'):
            writer=mode=='write';args='p,key,phase,base,path,event'if writer else'p,prefix,base,path'
            text+=f' recursive subroutine {mode}_ptr_{name}({args})\n {p["type"]},pointer,intent({"in"if writer else"inout"})::p{shape(rank)}\n'
            text+=(' character(len=*),intent(in)::key,phase,base,path\n integer,intent(in)::event\n'if writer else' character(len=*),intent(in)::prefix,base,path\n')
            text+=f' type(ref_{name}),allocatable::grown(:)\n integer::id,n,j,bounds({max(1,2*rank)})\n logical::present\n character(len=256)::node\n n=0\n if(allocated({refs}))n=size({refs})\n'
            if writer:
                text+=f" id=0\n if(associated(p))then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n id=j\n exit\n endif\n enddo\n if(id==0)id=n+1\n endif\n call oracle_write(key,phase,path//'.target',id,event)\n"
                if rank:text+=" if(id>0)then\n bounds=[lbound(p),shape(p)]\n call oracle_write(key,phase,path//'.bounds',bounds,event)\n endif\n"
                text+=' if(id<=n)return\n'
            else:
                text+=" call oracle_read(prefix//'.pre.'//path//'.target.bin',id)\n if(id==0)then\n nullify(p)\n return\n endif\n if(id<0.or.id>n+1)error stop 'stream invalid ordinal or forward reference'\n"
                if rank:text+=f" call oracle_read(prefix//'.pre.'//path//'.bounds.bin',bounds)\n if(any(bounds({rank+1}:)<0))error stop 'stream negative pointer extent'\n"
                text+=' if(id<=n)then\n'
                if rank:
                    lbs=','.join(f'bounds({i}):'for i in range(1,rank+1))
                    text+=f" if(any(shape({refs}(id)%p)/=bounds({rank+1}:)))error stop 'stream alias extent differs'\n p({lbs})=>{refs}(id)%p\n"
                else:text+=f' p=>{refs}(id)%p\n'
                text+=' return\n endif\n'
                text+=f' if(associated(p))then\n do j=1,n\n if(associated(p,{refs}(j)%p))then\n nullify(p)\n exit\n endif\n enddo\n endif\n'
                if rank:
                    alloc=','.join(f'bounds({i}):bounds({i})+bounds({i+rank})-1'for i in range(1,rank+1))
                    text+=f' if(associated(p))then\n if(any(shape(p)/=bounds({rank+1}:)).or.any(lbound(p)/=bounds(:{rank})))nullify(p)\n endif\n if(.not.associated(p))allocate(p({alloc}))\n'
                else:
                    text+=' if(.not.associated(p))then\n allocate(p)\n'
                    for f in fields:
                        if 'pointer'in f['attrs']:text+=' nullify(p%'+f['name']+')\n'
                    text+=' endif\n'
            text+=f" node=base//'.node.{name}.'//trim(number(id))\n allocate(grown(n+1))\n if(n>0)grown(:n)={refs}\n grown(n+1)%p=>p\n call move_alloc(grown,{refs})\n"
            if name=='node':
                for f in fields:
                    value='p%'+f['name'];label="trim(node)//'."+f['name']+"'"
                    if f.get('external_pointer'):
                        if writer:text+=f" call oracle_write(key,phase,{label}//'.present',associated({value}),event)\n"
                        else:text+=f" call oracle_read(prefix//'.pre.'//{label}//'.present.bin',present)\n if(present)error stop 'stream replay external pointee not represented'\n nullify({value})\n"
                    elif 'pointer'in f['attrs']:
                        args=f'{value},key,phase,base,{label},event'if writer else f'{value},prefix,base,{label}'
                        text+=f' call {mode}_ptr_{tag(f)}({args})\n'
                    elif writer:
                        lower=f',lower=lbound({value})'if f['dims']else''
                        text+=f' call oracle_write(key,phase,{label},{value},event{lower})\n'
                    else:text+=f" call oracle_read(prefix//'.pre.'//{label}//'.bin',{value})\n"
            elif writer:text+=" call oracle_write(key,phase,trim(node)//'.value',p,event,lower=lbound(p))\n"
            else:text+=" call oracle_read(prefix//'.pre.'//trim(node)//'.value.bin',p)\n"
            text+=' end subroutine\n'
    text+='end module mpas_oracle_stream\n'
    text+='''subroutine oracle_framework_stream_begin(event)
 use mpas_oracle_dump,only:oracle_begin
 implicit none
 integer,intent(out)::event
 call oracle_begin('stream_reindex',event)
end subroutine
'''
    text+=interface_source(roots,definitions=True)
    return text,info


def interface_source(roots,definitions=False):
    text=''if definitions else''' private::oracle_framework_stream_capture,oracle_framework_stream_restore,oracle_framework_stream_begin
 interface
 subroutine oracle_framework_stream_begin(event)
 integer,intent(out)::event
 end subroutine
'''
    for mode in ('capture','restore'):
        args='key,phase,name,'+','.join(roots)+',event'if mode=='capture'else'prefix,name,'+','.join(roots)
        text+=f' subroutine oracle_framework_stream_{mode}({args})\n use mpas_derived_types,only:field2DInteger\n'
        if definitions:text+=f' use mpas_oracle_stream,only:oracle_stream_{mode}\n implicit none\n'
        text+=(' character(len=*),intent(in)::key,phase,name\n integer,intent(in)::event\n'if mode=='capture'else' character(len=*),intent(in)::prefix,name\n')
        text+=' type(field2DInteger),pointer,intent('+('in'if mode=='capture'else'inout')+')::'+','.join(roots)+'\n'
        if definitions:text+=f' call oracle_stream_{mode}({args})\n'
        text+=' end subroutine\n'
    if not definitions:text+=' end interface\n'
    return text


def instrument_stream_writes(text):
    """Observe actual saved nodes around native I/O, inside the native branch.

    MPAS mpas_stream_manager.F:3391-3406. Capture follows PREWRITE_REINDEX
    and precedes POSTWRITE_REINDEX. Neither conversion nor write order changes.
    """
    first='              call prewrite_reindex(manager % allFields, manager % allPackages, stream % field_pool, stream % field_pkg_pool)'
    last='              call postwrite_reindex(manager % allFields, stream % field_pool)'
    if text.count(first)!=1 or text.count(last)!=1:raise ValueError('native stream reindex call sites changed')
    begin=text.rfind('           if ( .not. stream % blockWrite ) then',0,text.index(first))
    if begin<0:raise ValueError('native stream prewrite branch changed')
    text=text[:begin]+'           block\n           integer :: oracle_stream_event\n'+text[begin:]
    text=text.replace(first,first+"\n              call oracle_framework_stream_begin(oracle_stream_event)\n              call oracle_state_mpas_stream_manager('stream_reindex','pre',oracle_stream_event)",1)
    text=text.replace(last,"              call oracle_state_mpas_stream_manager('stream_reindex','post',oracle_stream_event)\n"+last,1)
    at=text.index(last)+len(last);end=text.index('           end if',at)+len('           end if')
    return text[:end]+'\n           end block'+text[end:]


def audit_membership(info,prefix,phase,name,filenames,read):
    expected,missing,issues,external=set(),set(),[],[];counts={'node':0};layouts={};stem=f'{prefix}.{phase}.'
    def member(path,code=None,width=None,rank=None,layout=None):
        filename=stem+path+'.bin';expected.add(filename)
        if filename not in filenames:missing.add(filename);return None
        h=read('header',filename,'-')
        if (h['name']!=path or h['phase']!=phase or code is not None and h['code']!=code
            or width is not None and h['width']!=width or rank is not None and len(h['shape'])!=rank
            or layout is not None and (h['shape'],h['lower'])!=layout):
            issues.append(dict(path=path,reason='native stream layout or context differs'));return None
        return h
    def control(path,code,rank=0,index='-'):
        if member(path,code,4,rank)is None:return None
        v=read('sample',stem+path+'.bin',index)['value']
        if (code==3 and type(v)is not int)or(code==4 and type(v)is not bool):
            issues.append(dict(path=path,reason='native stream control type differs'));return None
        return v
    def leaf(f,path,layout=None):
        t=f['type'].lower();code=5 if t.startswith('character')else 4 if t.startswith('logical')else 3
        member(path,code,None if code==5 else 4,len(f['dims']),layout)
    def pointer(t,rank,path,f=None):
        ordinal=control(path+'.target',3)
        if ordinal is None or ordinal==0:return
        count=counts.setdefault(t,0)
        if ordinal<0 or ordinal>count+1:
            issues.append(dict(path=path,reason='native stream invalid ordinal or forward reference'));return
        layout=None
        if rank:
            h=member(path+'.bounds',3,4,1)
            if h is None:return
            if h['shape']!=[2*rank]:issues.append(dict(path=path,reason='stream pointer bounds size differs'));return
            bounds=[control(path+'.bounds',3,1,str(i))for i in range(1,2*rank+1)]
            if None in bounds:return
            lower=bounds[:rank];extent=bounds[rank:];layout=(extent,lower)
            if any(n<0 for n in extent):issues.append(dict(path=path,reason='negative stream pointer extent'));return
        if ordinal<=count:
            if rank and layouts[(t,ordinal)][0]!=layout[0]:issues.append(dict(path=path,reason='stream alias extent differs'))
            return
        counts[t]=ordinal;node=name+f'.node.{t}.{ordinal}'
        if t=='node':
            for fld in info['fields']:
                label=node+'.'+fld['name']
                if fld.get('external_pointer'):
                    if control(label+'.present',4)is True:external.append(label)
                elif 'pointer'in fld['attrs']:pointer(tag(fld),len(fld['dims']),label,fld)
                else:leaf(fld,label)
        else:layouts[(t,ordinal)]=layout;leaf(f,node+'.value',layout)
    for r in info['root_names']:pointer('node',0,name+'.'+r)
    unexpected=sorted({f for f in filenames if f.startswith(stem+name+'.')}-expected)
    return dict(expected_records=len(expected),missing=sorted(missing),unexpected=unexpected,
        topology_issues=issues,typed_node_counts=counts,unrepresented_external_pointers=external,
        external_aliases_complete=False,global_hidden_state_complete=False,
        complete=not missing and not unexpected and not issues and not external,
        scope='Observed stream connectivity nodes, primitive storage and internal aliases; non-null external pointees remain unrepresented')
