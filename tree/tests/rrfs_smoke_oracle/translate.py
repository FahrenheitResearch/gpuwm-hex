"""Translate the complete pinned SMOKE statement graph, rejecting unknown syntax."""
from pathlib import Path
import re,json
ROOT=Path('scratch/native')
def split(s,sep=','):
 out=[];depth=0;last=0;quote=None
 for i,c in enumerate(s):
  if quote:
   if c==quote:quote=None
  elif c in "'\"":quote=c
  elif c=='(':depth+=1
  elif c==')':depth-=1
  elif c==sep and depth==0:out.append(s[last:i].strip());last=i+1
 out.append(s[last:].strip());return out

def statements(p):
 out=[];pending='';start=0
 # Native CPP joins physical backslash-newline before Fortran comment removal.
 # Keep the original first line number for source citations.
 physical=[]; joined=''; first=0
 for n,line in enumerate(p.read_text().splitlines(),1):
  if not joined:first=n
  if line.endswith('\\'):
   joined+=line[:-1];continue
  physical.append((first,joined+line));joined=''
 if joined:physical.append((first,joined))
 for n,line in physical:
  line=re.sub(r'!.*$','',line).strip()
  if not line:continue
  if not pending:start=n
  pending+=' '+line.lstrip('&').rstrip('&').strip()
  if line.endswith('&'):continue
  for s in split(pending,';'):
   parts=re.split(r"('(?:[^']|'')*')",s)
   s=''.join(x if i%2 else x.lower() for i,x in enumerate(parts))
   if s:out.append((start,s))
  pending=''
 return out

def declaration(s,syms):
 m=re.match(r'(real(?:\s*\([^)]*\))?|integer|logical|character(?:\s*\([^)]*\))?|type\(plumegen_coms\)|class\(plumegen_coms\))\s*(.*)',s)
 if not m:return False
 typ,tail=m.groups();kind='float' if typ.startswith('real') else 'int' if typ=='integer' else 'bool' if typ=='logical' else 'FStr' if typ.startswith('character') else 'Plume*'
 attrs,vars=tail.split('::',1) if '::' in tail else ('',tail)
 dm=re.search(r'dimension\s*\(([^)]*)\)',attrs)
 intent=re.search(r'intent\s*\(\s*(\w+)\s*\)',attrs)
 for var in split(vars):
  init=None
  if '=' in var:var,init=map(str.strip,var.split('=',1))
  vm=re.fullmatch(r'(\w+)(?:\s*\((.*)\))?',var)
  if not vm:raise ValueError(('declaration',s,var))
  name,localdim=vm.groups()
  dim=split(localdim) if localdim is not None else split(dm[1]) if dm else []
  syms[name]={'type':kind,'dim':dim,'intent':intent[1] if intent else None,'optional':'optional' in attrs,'parameter':'parameter' in attrs,'init':init,'save':'save' in attrs or init is not None and 'parameter' not in attrs,'decl':s,'str_len':re.search(r'\d+',typ)[0] if kind=='FStr' and re.search(r'\d+',typ) else '100'}
 return True

modules={};routines={};fields={}
for p in ROOT.glob('*.F90'):
 mod=None;routine=None;in_type=False
 for n,s in statements(p):
  if s.startswith('module ') and not s.startswith('module procedure'):
   mod={'name':s.split()[1],'globals':{},'uses':[],'path':str(p.name),'start':n};modules[mod['name']]=mod;continue
  if s=='type plumegen_coms':in_type=True;continue
  if s.startswith('end type'):in_type=False;continue
  if in_type:
   if s not in ('contains',) and not s.startswith('procedure'):declaration(s,fields)
   continue
  m=re.match(r'(?:(real)\s+)?(subroutine|function)\s+(\w+)\s*(?:\((.*?)\))?(?:\s+result\((\w+)\))?$',s)
  if m:
   rt,kind,name,args,res=m.groups()
   routine={'name':name,'args':split(args) if args else [],'symbols':{},'uses':[],'body':[],'module':mod['name'],'start':n,'path':p.name,'kind':kind,'return':res or name,'return_type':'float' if rt else None}
   routines[name]=routine;continue
  if routine and re.match(r'^end(?:\s+(?:subroutine|function)(?:\s+\w+)?)?$',s):routine['end']=n;routine=None;continue
  if s.startswith('use '):
   target=routine if routine else mod
   um=re.match(r'use\s+(\w+)(?:\s*,\s*only\s*:\s*(.*))?',s)
   target['uses'].append({'module':um[1],'only':split(um[2]) if um[2] else None});continue
  syms=routine['symbols'] if routine else mod['globals']
  if declaration(s,syms):continue
  if s.startswith('parameter '):
   for expr in split(s[s.index('(')+1:-1]):
    name,init=map(str.strip,expr.split('=',1));syms[name].update(parameter=True,init=init)
   continue
  if routine and not s.startswith(('implicit','intrinsic','public','private','contains','save ')):
   routine['body'].append((n,s))

# Keep module declarations scoped; USE re-exports are resolved recursively.
def imported(mod,seen=None):
 seen=set() if seen is None else seen
 if mod['name'] in seen:return {}
 seen.add(mod['name']);out={}
 for u in mod['uses']:
  if u['module'] not in modules:continue
  ss=imported(modules[u['module']],set(seen))
  out.update({k:v for k,v in ss.items() if u['only'] is None or k in u['only']})
 out.update({k:{**v,'cpp':'state.'+mod['name']+'.'+k,'global_module':mod['name']} for k,v in mod['globals'].items()})
 return out

def symbols(r):
 out=imported(modules[r['module']])
 for u in r['uses']:
  if u['module'] in modules:
   out.update({k:v for k,v in imported(modules[u['module']]).items() if u['only'] is None or k in u['only']})
 out.update({k:{**v,'cpp':k} for k,v in r['symbols'].items()})
 for obj in ('coms','this'):
  if obj in out:
   out.update({obj+'%'+k:{**v,'cpp':obj+'->'+k} for k,v in fields.items()})
 return out

if Path('scratch/driver-origins.json').exists():
 for name,origin in json.loads(Path('scratch/driver-origins.json').read_text()).items():
  r=routines[name];r['start']=origin['first'];r['end']=origin['last'];r['body']=[(origin['map'][str(n)],s) for n,s in r['body']]

Path('scratch/routines.json').write_text(json.dumps(routines,indent=2))
Path('scratch/modules.json').write_text(json.dumps(modules,indent=2))
Path('scratch/plume-fields.json').write_text(json.dumps(fields,indent=2))
print('Parsed',len(modules),'modules',len(routines),'routines',len(fields),'plume members')
class Expr:
 def __init__(self,s,syms):
  s=re.sub(r',\s*kind\s*=\s*rkind','',s)
  self.syms=syms;self.s=s;self.i=0
  self.ts=re.findall(r"'(?:[^']|'')*'|\d+(?:\.(?![a-z]+\.)\d*)?(?:[ed][+-]?\d+)?(?:_rkind)?|\.\d+(?:[ed][+-]?\d+)?(?:_rkind)?|\.\w+\.|[a-z]\w*(?:%\w+)?|\*\*|>=|<=|==|/=|=>|[():+*/<>,=-]",s)
 def peek(self):return self.ts[self.i] if self.i<len(self.ts) else ''
 def pop(self):v=self.peek();self.i+=1;return v
 def read(self,prec=0):
  t=self.pop()
  if t in ('+','-','.not.'):
   v,kind=self.read(65 if t!='.not.' else 25);v=f'sd_neg({v})' if t=="-" and kind=="float" else f'({"!" if t==".not." else t}{v})'
  elif t=='(':
   v,kind=self.read();assert self.pop()==')',(self.s,self.ts[self.i:])
  elif t==':':v='Slice()';kind='slice'
  elif re.match(r'^(\d|\.\d)',t):
   literal=t.replace('_rkind','')
   kind='double' if 'd' in literal else 'float' if any(x in literal for x in '.e') or '_rkind' in t else 'int'
   v=literal.replace('d','e')+('f' if kind=='float' else '')
  elif t in ('.true.','.false.'):v='true' if t=='.true.' else 'false';kind='bool'
  elif t.startswith("'"):v='FStr('+json.dumps(t[1:-1].replace("''","'"))+')';kind='FStr'
  else:
   sym=self.syms.get(t)
   kind=sym['type'] if sym else 'float';v=sym.get('cpp',t) if sym else t
   if self.peek()=='(':
    self.pop();args=[];kinds=[]
    if self.peek()!=')':
     while True:
      a,k=self.read();args.append(a);kinds.append(k)
      if self.peek()!=',':break
      self.pop()
    assert self.pop()==')',(self.s,self.ts[self.i:])
    if sym and sym['dim']:
     if 'slice' in kinds:
      v+='.section('+','.join(args)+')';kind='array_'+sym['type']
     else:v+='('+','.join(args)+')'
    elif t in routines:
     rr=routines[t];v=t+'(state,arena'+(','+','.join(args) if args else '')+')';kind=rr['return_type'] or rr['symbols'].get(rr['return'],{}).get('type','float')
    elif t in ('real','float'):
     v='float('+args[0]+')';kind='float'
    elif t in ('int','ifix','nint','floor'):
     v=('sd_nint' if t=='nint' else 'sd_floor' if t=='floor' else 'int')+'('+args[0]+')';kind='int'
    elif t=='present':v='has_'+args[0];kind='bool'
    elif t=='associated':v='('+args[0]+' != nullptr)';kind='bool'
    elif t in ('max','min','amax1','amin1','max0','min0'):
     fn='sd_max' if 'max' in t else 'sd_min';kind='double' if 'double' in kinds else 'float' if 'float' in kinds else 'int';v=args[0]
     for a in args[1:]:v=fn+'('+v+','+a+')'
    elif t in ('exp','log','alog','log10','sqrt','erf','cos','sin','abs','mod','sum','minval','maxval','aint','dble','sign','isnan'):
     v='sd_'+('log' if t=='alog' else t)+'('+','.join(args)+')';kind=kinds[0]
     if t in ('sum','minval','maxval'):kind=kind.replace('array_','')
     if t=='dble':kind='double'
     if t=='isnan':kind='bool'
    else:raise ValueError(('function',t,self.s))
   elif sym and sym['dim']:kind='array_'+sym['type']
   elif not sym:raise ValueError(('unknown-symbol',t,self.s))
  ops={'.or.':10,'.and.':20,'.eq.':30,'.ne.':30,'.lt.':30,'.le.':30,'.gt.':30,'.ge.':30,'==':30,'/=':30,'>':30,'<':30,'>=':30,'<=':30,':':35,'+':40,'-':40,'*':50,'/':50,'**':70}
  mapping={'.or.':'||','.and.':'&&','.eq.':'==','.ne.':'!=','.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>=','/=':'!='}
  while self.peek() in ops and ops[self.peek()]>prec:
   op=self.pop()
   if op==':' and self.peek() in (',',')',''):
    v=f'Slice({v}, SD_END)';kind='slice';continue
   rhs,rkind=self.read(ops[op]-1 if op=='**' else ops[op]);nk='double' if 'double' in (kind,rkind) else 'float' if 'float' in (kind,rkind) else 'int'
   if op==':':v=f'Slice({v},{rhs})';kind='slice'
   elif ops[op]<=30:v=f'({v} {mapping.get(op,op)} {rhs})';kind='bool'
   elif op=='**':
    # Integer powers follow GNU's multiply chain; real exponent 2/3
    # is also lowered by GNU when it is exact (verified by oracle).
    if re.fullmatch(r'(?:2(?:\.0*)?f?|3(?:\.0*)?f?|4(?:\.0*)?f?)',rhs):v='sd_ipow('+v+','+str(int(float(rhs.rstrip('f'))))+')'
    else:v='sd_pow('+v+','+rhs+')'
    kind=nk
   elif nk=='int' and not kind.startswith('array') and not rkind.startswith('array'):v=f'({v} {op} {rhs})';kind='int'
   else:
    v={'+' :'sd_add','-':'sd_sub','*':'sd_mul','/':'sd_div'}[op]+'('+v+','+rhs+')';kind=kind if kind.startswith('array') else rkind if rkind.startswith('array') else nk
  return v,kind
 def full(self):
  v,k=self.read();assert self.i==len(self.ts),(self.s,self.ts[self.i:]);return v,k

def call_actuals(s):
 m=re.match(r'call\s+(\w+)\s*(?:\((.*)\))?',s)
 if not m or m[1] not in routines:return None
 rr=routines[m[1]]; mapping={};pos=0
 for a in split(m[2]) if m[2] else []:
  am=re.match(r'^(\w+)\s*=(?!=)(.*)',a)
  if am:mapping[am[1]]=am[2]
  else:mapping[rr['args'][pos]]=a;pos+=1
 return rr,mapping
for r in routines.values():
 r['mutated']=set(a for a in r['args'] if r['symbols'][a]['intent'] in ('out','inout'))
 for _,s in r['body']:
  mm=re.search(r'(?:^|\)\s*)(\w+)(?:%\w+|\([^=]*\))?\s*=(?!=|>)',s)
  if mm and mm[1] in r['args']:r['mutated'].add(mm[1])
for _ in range(len(routines)):
 changed=False
 for r in routines.values():
  for _,s in r['body']:
   result=call_actuals(s)
   if not result:continue
   rr,mapping=result
   for a in rr['mutated']:
    aa=mapping.get(a)
    if aa in r['args'] and aa not in r['mutated']:r['mutated'].add(aa);changed=True
 if not changed:break

def signature(r):
 args=['State &state','Arena &arena']
 for a in r['args']:
  d=r['symbols'][a]
  ty='FArray<'+d['type']+'>' if d['dim'] else d['type']+(' &' if d['type']!='Plume*' and a in r['mutated'] else '')
  args.append(ty+' '+a)
 for a in r['args']:
  if r['symbols'][a]['optional']:args.append('bool has_'+a)
 rt='void' if r['kind']=='subroutine' else r['return_type'] or r['symbols'][r['return']]['type']
 return 'SDDEV '+rt+' '+r['name']+'('+','.join(args)+')'

def bounds(d,ex):
 out=[]
 for dim in d['dim']:
  lo,hi=dim.split(':') if ':' in dim else ('1',dim)
  if hi=='*':hi='1'
  out.extend((ex(lo),ex(hi)))
 return ','.join(out)

native_constants={}
if Path('scratch/native-constants.txt').exists():
 for line in Path('scratch/native-constants.txt').read_text().splitlines():
  name,index,value=line.split();native_constants[name+'.'+index]=value
def constant(scope,k,i,d):
 value=native_constants.get(scope+'.'+k+'.'+str(i))
 if value is None:return None
 return '__uint_as_float(0x'+value+'U)' if d['type']=='float' else 'true' if d['type']=='bool' and value=='1' else 'false' if d['type']=='bool' else value

failures=[]
def generate(r):
 syms=symbols(r);ex=lambda s:Expr(s,syms).full()[0]
 out=[f'// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/{"mpas_atmphys_driver_smoke.F" if r["path"]=="driver_slice.F90" else "physics_noaa/SMOKE/"+("MPAS/" if r["path"]=="mpas_smoke_wrapper.F90" else "")+r["path"]}:{r["start"]}-{r["end"]} @1d6e370e; SMOKE @9652bbf',signature(r)+' {','ArenaScope scope(arena);']
 for a in r['args']:
  d=r['symbols'][a]
  if d['dim'] and ':' not in d['dim']:out.append(a+'.bind('+bounds(d,ex)+');')
 for name,d in r['symbols'].items():
  if name in r['args']:continue
  if d['dim']:
   out.append(f'auto {name} = arena.alloc<{d["type"]}>('+bounds(d,ex)+');')
   if d['init']:
    ini=d['init']
    if ini.startswith('(/'):
     for idx,v in enumerate(split(ini[2:-2]),1):out.append(f'{name}({idx}) = '+(constant(r['name'],name,idx,d) or ex(v))+';')
    else:out.append(f'sd_assign({name},'+ex(ini)+',arena);')
  elif d['save']:
   out.append(f'auto &{name} = state.saved_{r["name"]}_{name};')
  else:out.append(d['type']+' '+name+(' = '+(constant(r['name'],name,1,d) or ex(d['init'])) if d['init'] else '')+';')
 if r['kind']=='function' and r['return'] not in r['symbols']:
  out.append((r['return_type'] or 'float')+' '+r['return']+';');syms[r['return']]={'cpp':r['return'],'type':r['return_type'] or 'float','dim':[]}
 loops=[];loopid=0;selects=[];selid=0
 def statement(s,n):
  nonlocal loopid,selid
  s=re.sub(r'^\w+:\s*(?=if)', '', s)
  label=''
  mm=re.match(r'^(\d+)\s+(.*)',s)
  if mm:label,s=mm.groups()
  prefix='label_'+label+': ; ' if label else ''
  if s.startswith(('write','print','format')):return '; /* Framework diagnostic I/O; no field mutation. */'
  if s.startswith('call mpas_log_write'):
   assert len(split(s[s.index('(')+1:s.rindex(')')])) == 1, 'typed framework callback required for optional logging arguments'
   return 'sd_framework_log(state);'
  if s.startswith(('call mpas_timer_start','call mpas_timer_stop')):return '; /* Framework timing only. */'
  if s.startswith('allocate(private_thread_coms)'):return 'state.module_zero_plumegen_coms.private_thread_coms = state.plume_store;'
  if '=>' in s:
   lhs,rhs=map(str.strip,s.split('=>'));return ex(lhs)+' = '+ex(rhs)+';'
  if s=='continue':return prefix+';'
  m=re.match(r'(?:(\w+):\s*)?do\b\s*(.*)',s)
  if m:
   name,spec=m.groups();loopid+=1;name=name or f'loop{loopid}';loops.append(name)
   if spec.startswith('while'):return 'while ('+ex(spec[spec.index('(')+1:-1])+') {'
   if not spec:return 'for (;;) {'
   var,bs=spec.split('=',1);bs=split(bs);lo,hi=map(ex,bs[:2]);step=ex(bs[2]) if len(bs)>2 else '1';cmp='>=' if step.startswith('(-') else '<='
   return f'for ({var.strip()}={lo};{var.strip()}{cmp}{hi};{var.strip()}+={step}) {{'
  if s.startswith(('enddo','end do')):
   name=loops.pop();return name+'_continue: ; } '+name+'_exit: ;'
  if s.startswith(('exit','cycle')):
   cmd,*target=s.split();name=target[0] if target else loops[-1];return 'goto '+name+('_exit' if cmd=='exit' else '_continue')+';'
  if s.startswith(('else if','elseif')):return '} else '+statement(re.sub(r'^else\s*if','if',s),n)
  if s=='else':return '} else {'
  if s.startswith(('endif','end if')):return '}'
  if re.match(r'if\s*\(',s):
   idx=s.index('(');dep=1;k=idx+1;qt=None
   while dep:
    if qt:
     if s[k]==qt:qt=None
    elif s[k]=="'":qt="'"
    elif s[k]=='(':dep+=1
    elif s[k]==')':dep-=1
    k+=1
   cond=ex(s[idx+1:k-1]);tail=s[k:].strip();return f'if ({cond}) '+('{' if tail=='then' else '{ '+statement(tail,n)+' }')
  if s.startswith('select case'):
   selid+=1;name='sel'+str(selid);selects.append([name,False]);return '{ auto '+name+' = '+ex(s[s.index('(')+1:-1])+';'
  if s.startswith('case'):
   name,prev=selects[-1];selects[-1][1]=True
   if 'default' in s:return ('} else {' if prev else '{')
   vals=split(s[s.index('(')+1:-1]);cond=' || '.join(name+' == '+ex(v) for v in vals)
   return ('} else ' if prev else '')+'if ('+cond+') {'
  if s.startswith('end select'):selects.pop();return '} }'
  if s.startswith('go to') or s.startswith('goto'):return 'goto label_'+s.split()[-1]+';'
  if s=='return':return 'return'+(' '+r['return'] if r['kind']=='function' else '')+';'
  if s.startswith('call '):
   name,args=re.match(r'call\s+(\w+)\s*(?:\((.*)\))?',s).groups();rr=routines[name];actual=split(args) if args else []
   argmap={};pos=0
   for a in actual:
    am=re.match(r'^(\w+)\s*=(?!=)(.*)',a)
    if am:argmap[am[1]]=am[2]
    else:argmap[rr['args'][pos]]=a;pos+=1
   rendered=[];flags=[]
   for a in rr['args']:
    d=rr['symbols'][a]
    if a not in argmap:
     assert d['optional'],(name,a)
     rendered.append('FArray<'+d['type']+'>()' if d['dim'] else 'state.absent_'+d['type']);flags.append('false');continue
    aa=argmap[a];value,value_kind=Expr(aa,syms).full()
    if d['dim'] and not value_kind.startswith('array'):
     am=re.fullmatch(r'(\w+(?:%\w+)?)\s*\((.*)\)',aa)
     assert am and am[1] in syms and syms[am[1]]['dim'],('sequence association',aa)
     value='sd_sequence('+syms[am[1]]['cpp']+','+','.join(ex(v) for v in split(am[2]))+')'
    rendered.append(value)
    if d['optional']:flags.append('has_'+aa if aa in syms and syms[aa]['optional'] else 'true')
   return name+'(state,arena'+(','+','.join(rendered+flags) if rendered or flags else '')+');'
  if '=' in s:
   lhs,rhs=map(str.strip,s.split('=',1));l,lk=Expr(lhs,syms).full();v,vk=Expr(rhs,syms).full()
   # The pinned strict object uses MAXSS with +0 as its source operand
   # at module_smoke_plumerise.F90:1550-1553. Preserve its unordered/tie
   # selection at these sites, rather than guessing a universal MIN/MAX rule.
   if r['name']=='update_plumerise' and n in (1550,1551,1552,1553):
    assert rhs==f'max(0., {lhs})',(n,s)
    v=f'sd_max({l},0.f)'
   # Published strict MAKEPLUME 5fc1d6--5fc21a: WMAX is the product
   # destination; MINSS/MAXSS select their finite source on unordered
   # inputs. Newly computed DT precedes stored TIME. Both microphysics
   # and final saturation passes keep the computed sum/pressure first;
   # plume-top MAXSS keeps the previous maximum as its destination
   # (5fcb26--5fcb2e). The capture has the same field-level operations.
   if r['name']=='makeplume':
    lowering={
     521: ('coms%dt','sd_max(sd_min(sd_div(sd_sub(coms->zm(2),coms->zm(1)),sd_mul(wmax,coms->tstpf)),5.f),0.01f)'),
     526: ('coms%time','sd_add(coms->dt,coms->time)'),
     573: ('coms%wbar','sd_mul(sd_add(coms->w(coms->l),coms->w((coms->l - 1))),0.5f)'),
     581: ('coms%rho (coms%l)','sd_div(sd_mul(coms->pe(coms->l),3483.8f),coms->t(coms->l))'),
     632: ('coms%rho (k)','sd_div(sd_mul(coms->pe(k),3483.8f),coms->t(k))'),
     653: ('ztopmax','sd_max(ztopmax,coms->ztop)'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Original strict full-build object, module_smoke_plumerise.F90:
   # scalar SSE destinations retain computed terms before adding stored
   # tendencies. Preserve the measured commutations, not a global rule.
   if r['name']=='advtndc_plumerise' and n==1333:
    assert lhs=='sct(k)' and rhs=='sct(k) + (sca(k)-scp(k)) * dtli',(n,s)
    v='sd_add(sd_mul(sd_sub(sca(k),scp(k)),dtli),sct(k))'
   if r['name']=='fa_zc_plumerise' and n==1290:
    assert lhs=='vt3dg(k)',(n,s)
    v='sd_mul(sd_add(sd_add(sd_mul(vctr2(k),scr1((k + 1))),sd_mul(vctr1(k),scr1(k))),sd_mul(sd_sub(scr1(k),scr1((k + 1))),vt3df(k))),vt3dc(k))'
   if r['name']=='fa_zc_plumerise' and n==1317:
    assert lhs=='scr1(k)',(n,s)
    v='sd_add(sd_mul(sd_add(sd_mul(sd_sub(vt3dc(k),vt3dc((k - 1))),scp(k)),sd_sub(vt3dg((k - 1)),vt3dg(k))),vt3dk(k)),scr1(k))'
   # Original strict object scl_misc: temperature uses subtraction of the
   # positive adiabatic term (2175/217e), while the stored ADIABAT sign is
   # toggled after multiplication/division (2532). The entrainment products
   # keep scalar differences/contents as SSE destinations, and RAD_T keeps
   # the computed term before adding its previous value (2292).
   if r['name']=='scl_misc':
    lowering={
     1366: ('coms%adiabat','sd_neg(sd_div(sd_mul(coms->wbar,g),cp))'),
     1369: ('dmdtm','sd_div(sd_mul(sd_abs(coms->wbar),sd_add(alpha,alpha)),coms->radius(k))'),
     1372: ('coms%tt(k)','sd_sub(sd_sub(coms->tt(k),sd_div(sd_mul(coms->wbar,g),cp)),sd_mul(sd_sub(coms->t(k),coms->te(k)),dmdtm))'),
     1375: ('coms%qvt(k)','sd_sub(coms->qvt(k),sd_mul(sd_sub(coms->qv(k),coms->qvenv(k)),dmdtm))'),
     1377: ('coms%qct(k)','sd_sub(coms->qct(k),sd_mul(coms->qc(k),dmdtm))'),
     1378: ('coms%qht(k)','sd_sub(coms->qht(k),sd_mul(coms->qh(k),dmdtm))'),
     1379: ('coms%qit(k)','sd_sub(coms->qit(k),sd_mul(coms->qi(k),dmdtm))'),
     1382: ('coms%vel_t(k)','sd_sub(coms->vel_t(k),sd_mul(sd_sub(coms->vel_p(k),coms->vel_e(k)),dmdtm))'),
     1385: ('coms%rad_t(k)','sd_add(sd_mul(sd_mul(sd_mul(dmdtm,0.5f),sd_div(6.f,5.f)),coms->radius(k)),coms->rad_t(k))'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Original evaporate object e28/e2e/e3a/e46/e56/e5a: evaluate the
   # supersaturation factor first, then powers; negate DEVIDT only after
   # the complete product/quotient. Applying unary minus to CVI early has
   # different NaN-sign behavior even though finite values agree.
   if r['name']=='evaporate':
    lowering={
     1814: ('dividend','sd_mul(sd_mul(sd_mul(sd_sub(sd_div(sd,coms->qsat(coms->l)),1.f),sd_pow(sd_div(1.e6f,coms->rho(coms->l)),0.475f)),sd_pow(coms->qi(coms->l),0.525f)),1.13f)'),
     1816: ('divisor','sd_add(sd_div(4.1e6f,sd_mul(10.f,coms->est(coms->l))),7.e5f)'),
     1818: ('devidt','sd_neg(sd_div(sd_mul(dividend,coms->cvi(coms->l)),divisor))'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Original update_plumerise and its scalar .part.0 clone keep the
   # tendency*DT term as addition destination before the prior state.
   # Existing measured MAXSS zero clamps remain separate above.
   if r['name']=='update_plumerise':
    updates={1529:('w','wt'),1536:('t','tt'),1542:('qv','qvt'),
             1544:('qc','qct'),1545:('qh','qht'),1546:('qi','qit'),
             1555:('vel_p','vel_t'),1556:('rad_p','rad_t')}
    if n in updates:
     field,tendency=updates[n]
     assert lhs==f'coms%{field}(k)',(n,s)
     v=f'sd_add(sd_mul(coms->{tendency}(k),coms->dt),coms->{field}(k))'
   # Published r0 strict atmosphere_model, visc_w 5f9388--5f9524:
   # doubling lowers to ADDSS. W uses DZ2M as product destination; T
   # keeps its previous level as addition destination. Every final
   # tendency update keeps the computed diffusion term first. The r1
   # instrumented object commutes some NaN operands, so use the published
   # strict binary rather than assuming instrumentation is neutral.
   if r['name']=='visc_w':
    derivatives={1484:('d2wdz','w','dz2m'),1485:('d2tdz','t','dz2t'),
                 1486:('d2qvdz','qv','dz2t'),1487:('d2qhdz','qh','dz2t'),
                 1488:('d2qcdz','qc','dz2t'),1489:('d2qidz','qi','dz2t'),
                 1491:('d2vel_pdz','vel_p','dz2t'),1492:('d2rad_dz','rad_p','dz2t')}
    if n in derivatives:
     expected,field,spacing=derivatives[n]
     assert lhs==expected,(n,s)
     delta=f'sd_sub(coms->{field}((k + 1)),sd_add(coms->{field}(k),coms->{field}(k)))'
     previous=f'coms->{field}((k - 1))'
     summed=f'sd_add({previous},{delta})' if field=='t' else f'sd_add({delta},{previous})'
     v=f'sd_mul({spacing},{summed})' if field=='w' else f'sd_mul({summed},{spacing})'
    tendencies={1494:('wt','d2wdz'),1496:('tt','d2tdz'),1504:('qvt','d2qvdz'),
                1505:('qct','d2qcdz'),1506:('qht','d2qhdz'),1507:('qit','d2qidz'),
                1508:('vel_t','d2vel_pdz'),1509:('rad_t','d2rad_dz')}
    if n in tendencies:
     field,term=tendencies[n]
     assert lhs==f'coms%{field}(k)',(n,s)
     v=f'sd_add({term},coms->{field}(k))'
   # Published strict friction 5fb3b3/5fb420--5fb428/5fb49c--5fb4a4
   # and buoyancy 5faec4--5faef1/5faf10/5faf48--5faf50 retain
   # computed terms as measured SSE destinations before prior state.
   if r['name']=='friction':
    lowering={
     942: ('c1','sd_div(1.f,sd_mul(sd_sub(ztop,zmkf),distim))'),
     948: ('vart(k)','sd_add(sd_mul(sd_sub(var2(k),var1(k)),sd_mul(sd_sub(zt(k),zmkf),c1)),vart(k))'),
     953: ('var1(k)','sd_add(sd_mul(sd_mul(sd_sub(zt(k),zmkf),c2),sd_sub(var2(k),var1(k))),var1(k))'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   if r['name']=='buoyancy_plumerise':
    lowering={
     1085: ('tv','sd_div(sd_mul(sd_add(sd_div(qv(k),eps),1.f),t(k)),sd_add(qv(k),1.f))'),
     1086: ('tve','sd_div(sd_mul(sd_add(sd_div(qvenv(k),eps),1.f),te(k)),sd_add(qvenv(k),1.f))'),
     1091: ('scr1(k)','sd_mul(sd_sub(sd_div(sd_sub(tv,tve),tve),qwtotl),sd_mul(g,umgamai))'),
     1097: ('wt(k)','sd_add(sd_mul(sd_add(scr1(k),scr1((k + 1))),0.5f),wt(k))'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Published strict plume transport context: ENTRAINMENT 5fac45/5facae/
   # 5fad4b/5fadd9 retains previous radius before current radius in both
   # unrolled iterations and tails. Dynamic entrainment's absolute velocity
   # difference is the multiplication destination before its finite factor.
   if r['name']=='entrainment':
    lowering={
     1127: ('radius_bar','sd_mul(sd_add(radius((k - 1)),radius(k)),0.5f)'),
     1139: ('dyn_entr','sd_div(sd_mul(sd_abs(sd_sub(sd_add(sd_sub(coms->vel_p(k),coms->vel_e(k)),coms->vel_p((k - 1))),coms->vel_e((k - 1)))),sd_mul(sd_div(2.f,3.1416f),0.5f)),radius_bar)'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Published SCL_ADVECTC 5fa0b4/5fa0bb/5fa12f--5fa143. The summed
   # density is the destination of the interior flux multiplication.
   # Keep first-level grouping distinct, as the pinned source does.
   if r['name']=='scl_advectc_plumerise':
    lowering={
     1160: ('dtlto2','sd_mul(coms->dt,0.5f)'),
     1163: ('coms%vt3df(1)','sd_mul(sd_mul(sd_mul(sd_add(coms->w(1),coms->wc(1)),0.5f),dtlto2),coms->dzm(1))'),
     1167: ('coms%vt3dc(k)','sd_mul(sd_mul(sd_add(coms->rho(k),coms->rho((k + 1))),sd_mul(sd_mul(sd_add(coms->w(k),coms->wc(k)),dtlto2),0.5f)),1.e-3f)'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Published SCL_DYN_ENTRAIN 5f9618--5f96e5. Scalar differences/
   # cloud and hail contents are multiplication destinations before DMDTM;
   # the ice term intentionally keeps DMDTM first (5f96cd). Radius keeps
   # its freshly computed tendency as the addition destination (5f9630).
   if r['name']=='scl_dyn_entrain':
    lowering={
     1434: ('rad_t(k)','sd_add(sd_div(sd_abs(sd_sub(vel_e(k),vel_p(k))),pi),rad_t(k))'),
     1438: ('dmdtm','sd_div(sd_mul(sd_abs(sd_sub(vel_e(k),vel_p(k))),sd_div(2.f,3.1416f)),radius(k))'),
     1441: ('vel_t(k)','sd_sub(vel_t(k),sd_mul(sd_sub(vel_p(k),vel_e(k)),dmdtm))'),
     1446: ('tt(k)','sd_sub(tt(k),sd_mul(sd_sub(t(k),te(k)),dmdtm))'),
     1449: ('qvt(k)','sd_sub(qvt(k),sd_mul(sd_sub(qv(k),qvenv(k)),dmdtm))'),
     1451: ('qct(k)','sd_sub(qct(k),sd_mul(qc(k),dmdtm))'),
     1452: ('qht(k)','sd_sub(qht(k),sd_mul(qh(k),dmdtm))'),
    }
    if n in lowering:
     expected,v=lowering[n]
     assert lhs==expected,(n,s)
   # Published VEL_ADVECTC 5fb242,5fb2ea--5fb33b: previous/current
   # and next/current wind sums, wind-sum destination in the second
   # flux product, ADDSS doubling, bracket*coefficient then prior WT.
   # r1's instrumentation reallocates array-base registers; field-level
   # arithmetic agrees, so register-name inequality alone is not a defect.
   if r['name']=='vel_advectc_plumerise':
    if n==972:
     assert lhs=='flxw(1)',(n,s)
     v='sd_mul(dn0(1),wc(1))'
    if n==984:
     assert lhs=='wt(k)',(n,s)
     delta='sd_sub(flxw((k + 1)),flxw((k - 1)))'
     first='sd_mul(sd_add(flxw(k),flxw((k - 1))),sd_add(wc((k - 1)),wc(k)))'
     second='sd_mul(sd_add(wc((k + 1)),wc(k)),sd_add(flxw(k),flxw((k + 1))))'
     last=f'sd_mul(sd_add({delta},{delta}),wc(k))'
     coeff='sd_div(sd_mul(dzm(k),c1z),sd_add(dn0(k),dn0((k + 1))))'
     v=f'sd_add(sd_mul(sd_add(sd_sub({first},{second}),{last}),{coeff}),wt(k))'
   return prefix+('sd_assign('+l+','+v+',arena);' if lk.startswith('array') else l+' = '+v+';')
  raise ValueError(('statement',n,s))
 for n,s in r['body']:
  try:out.append(statement(s,n)+f' // L{n}')
  except Exception as e:failures.append({'routine':r['name'],'line':n,'source':s,'error':str(e)})
 assert not loops,(r['name'],loops)
 if r['kind']=='function':out.append('return '+r['return']+';')
 out.append('}');return '\n'.join(out)
parts=[]
for r in routines.values():
 try:parts.append(generate(r))
 except Exception as e:failures.append({'routine':r['name'],'error':str(e)})
Path('scratch/translation-errors.json').write_text(json.dumps(failures,indent=2))
Path('scratch/generated-body.cuh').write_text('\n'.join(signature(r)+';' for r in routines.values())+'\n\n'+'\n\n'.join(parts))
print('Generated',len(parts),'bodies;',len(failures),'unsupported statements')
for f in failures[:20]:print(f)

assert not failures, failures
