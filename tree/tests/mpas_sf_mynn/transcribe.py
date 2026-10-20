"""Development-only transcription of the pinned MPAS module, never a runtime path.

It fails on unknown syntax. The emitted scalar expressions retain Fortran's
association and float32 rounding. Regenerate with the pinned source path.
"""
from __future__ import annotations
import ctypes
import json
from pathlib import Path
import re
import sys
import numpy as np

LIB = ctypes.CDLL('libm.so.6')
for name, arity in [('powf', 2), ('logf', 1), ('expf', 1), ('atanf', 1), ('sqrtf', 1)]:
    f = getattr(LIB, name)
    f.argtypes = [ctypes.c_float] * arity
    f.restype = ctypes.c_float


def statements(text):
    pending = ''
    begin = 0
    for n, raw in enumerate(text.splitlines(), 1):
        raw = raw.split('!')[0].strip().lower()
        if not raw:
            continue
        if not pending:
            begin = n
        pending += ' ' + raw.lstrip('&').rstrip('&').strip()
        if raw.endswith('&'):
            continue
        for part in pending.strip().split(';'):
            yield begin, part.strip()
        pending = ''
    assert not pending


def split_args(s):
    depth = 0
    start = 0
    out = []
    for n, c in enumerate(s):
        depth += (c == '(') - (c == ')')
        if c == ',' and depth == 0:
            out.append(s[start:n].strip())
            start = n+1
    return out + [s[start:].strip()]


def lit(v, typ):
    if typ == 'bool':
        return 'true' if v else 'false'
    if typ == 'int':
        return str(int(v))
    return float(np.float32(v)).hex() + 'f'


class E:
    def __init__(self, code, typ='float', value=None):
        self.code, self.typ, self.value = code, typ, value


TOKEN = re.compile(r'\s*(\.[a-z]+\.|(?:\d+\.?\d*|\.\d+)(?:[ed][+-]?\d+)?(?:_kind_phys)?|[a-z_]\w*|\*\*|<=|>=|==|/=|[()+*/,<>-])')
PREC = {'.or.':1, '.and.':2, '.eq.':3, '.ne.':3, '.gt.':3, '.lt.':3, '.ge.':3, '.le.':3, '==':3, '/=':3, '>':3, '<':3, '>=':3, '<=':3, '+':4, '-':4, '*':5, '/':5, '**':7}
OP = {'.or.':'||', '.and.':'&&', '.eq.':'==', '.ne.':'!=', '.gt.':'>', '.lt.':'<', '.ge.':'>=', '.le.':'<=', '/=':'!='}


class Expr:
    def __init__(self, routine, text):
        self.r = routine
        self.tokens = TOKEN.findall(text)
        if ''.join(self.tokens) != re.sub(r'\s+', '', text):
            raise ValueError(('expression tokens', text, self.tokens))
        self.n = 0
    def take(self):
        v = self.tokens[self.n]; self.n += 1; return v
    def peek(self):
        return self.tokens[self.n] if self.n < len(self.tokens) else ''
    def parse(self, minimum=0):
        tok = self.take()
        if tok in ('+', '-', '.not.'):
            x = self.parse(6 if tok != '.not.' else 3)
            if x.value is not None:
                a = (not x.value) if tok == '.not.' else (-x.value if tok == '-' else x.value)
                x = E(lit(a, x.typ), x.typ, a)
            else:
                x = E('(' + ('!' if tok == '.not.' else tok) + x.code + ')', x.typ)
        elif tok == '(':
            x = self.parse(); assert self.take() == ')'
        elif tok in ('.true.', '.false.'):
            x = E(tok == '.true.' and 'true' or 'false', 'bool', tok == '.true.')
        elif tok[0].isdigit() or tok[0] == '.':
            typ = 'float' if any(c in tok for c in '.ed_') else 'int'
            val = np.float32(tok.replace('_kind_phys', '').replace('d', 'e')) if typ == 'float' else int(tok)
            x = E(lit(val, typ), typ, val)
        elif self.peek() == '(':
            self.take(); args=[]
            if self.peek() != ')':
                while True:
                    args.append(self.parse())
                    if self.peek() != ',': break
                    self.take()
            assert self.take() == ')'
            x = self.call(tok, args)
        else:
            typ, const = self.r.variables.get(tok, GLOBALS.get(tok, ('float', None)))[:2]
            x = E('result' if self.r.result and tok == self.r.name else tok, typ, const)
            if const is not None: x.code = lit(const, typ)
        while self.peek() in PREC and PREC[self.peek()] >= minimum:
            op = self.take(); p = PREC[op]
            rhs = self.parse(p if op == '**' else p+1)
            x = self.binary(op, x, rhs)
        return x
    def binary(self, op, a, b):
        typ = 'float' if 'float' in (a.typ, b.typ) else 'int'
        oper = OP.get(op, op)
        if PREC[op] <= 3: typ = 'bool'
        if a.value is not None and b.value is not None:
            av, bv = a.value, b.value
            if typ == 'float': av, bv = np.float32(av), np.float32(bv)
            if op == '**': v = np.float32(LIB.powf(float(av), float(bv))) if typ == 'float' else av**bv
            elif op == '+': v = av+bv
            elif op == '-': v = av-bv
            elif op == '*': v = av*bv
            elif op == '/': v = av/bv if typ == 'float' else int(av/bv)
            else:
                v = {'==':lambda:av==bv,'!=':lambda:av!=bv,'>':lambda:av>bv,'<':lambda:av<bv,'>=':lambda:av>=bv,'<=':lambda:av<=bv,'&&':lambda:av and bv,'||':lambda:av or bv}[oper]()
            if typ == 'float': v=np.float32(v)
            return E(lit(v,typ),typ,v)
        if op == '**':
            if b.value is not None and float(b.value) in (0.,1.,2.,3.,4.):
                p=int(b.value)
                if p==0: return E('1.0f','float')
                if p==1: return a
                sq='__fmul_rn('+a.code+','+a.code+')'
                if p==2: return E(sq)
                if p==3: return E('__fmul_rn('+a.code+','+sq+')')
                return E('__fmul_rn('+sq+','+sq+')')
            return E('SL_POW('+a.code+','+b.code+')')
        if typ == 'float':
            fun={'+':'__fadd_rn','-':'__fsub_rn','*':'__fmul_rn','/':'__fdiv_rn'}[op]
            return E(fun+'('+a.code+','+b.code+')')
        return E('('+a.code+oper+b.code+')',typ)
    def call(self, name, args):
        if name == 'present':
            return E('has_'+args[0].code, 'bool')
        if name in self.r.arrays:
            if name in TABLES:
                return E(name+'['+args[0].code+']')
            assert len(args)==1 and args[0].code == 'i', (name,args)
            return E(name)
        if name in ('float','real','int'):
            typ = 'int' if name == 'int' else 'float'
            if args[0].value is not None:
                v = int(args[0].value) if typ == 'int' else np.float32(args[0].value)
                return E(lit(v,typ),typ,v)
            return E('('+typ+')('+args[0].code+')',typ)
        maps={'log':'SL_LOG','alog':'SL_LOG','exp':'SL_EXP','atan':'SL_ATAN','sqrt':'SL_SQRT','abs':'fabsf','max':'fmaxf','amax1':'fmaxf','min':'fminf'}
        if name in maps:
            if all(a.value is not None for a in args):
                vals=[a.value for a in args]
                if name in ('log','alog','exp','atan','sqrt'):
                    v=getattr(LIB, {'alog':'log'}.get(name,name)+'f')(float(vals[0]))
                elif name=='abs': v=abs(vals[0])
                elif name in ('max','amax1'): v=max(vals)
                else: v=min(vals)
                v=np.float32(v)
                return E(lit(v,'float'),'float',v)
            out=maps[name]+'('+','.join(a.code for a in args)+')'
            return E(out)
        if name in ROUTINES:
            target=ROUTINES[name]
            extra=[E('has_'+n,'bool') for n in target.optional]
            return E('m_'+name+'('+','.join(a.code for a in args+extra)+')')
        raise ValueError(('unknown function',name))


def expr(r,s):
    p=Expr(r,s); e=p.parse(); assert p.n==len(p.tokens), (s,p.tokens[p.n:]); return e


TABLES=('psim_stab','psim_unstab','psih_stab','psih_unstab')
GLOBALS={'debug_code':('bool',False),'psi_opt':('int',0),'wmin':('float',np.float32(.1)),'vconvc':('float',np.float32(1.25)),'snowz0':('float',np.float32(.011)),'coare_opt':('float',np.float32(3.))}
ROUTINES={}


class Routine:
    def __init__(self,name,args,items,result):
        self.name,self.args,self.items,self.result=name,args,items,result
        self.variables={}; self.arrays=set(TABLES); self.optional=[]; self.intent={}; self.body=[]
        for line,s in items:
            if '::' in s:
                decl, fields=s.split('::',1)
                typ='float' if decl.startswith('real') else 'int' if decl.startswith('integer') else 'bool' if decl.startswith('logical') else 'char'
                for field in split_args(fields):
                    field=field.strip()
                    name=re.match(r'\w+',field)[0]
                    value=None
                    if '=' in field:
                        value=expr(self,field.split('=',1)[1]).value
                        assert value is not None
                        if typ=='float': value=np.float32(value)
                    self.variables[name]=(typ,value)
                    if 'dimension' in decl or '(' in field.split('=')[0]: self.arrays.add(name)
                    if 'optional' in decl: self.optional.append(name)
                    mt=re.search(r'intent\((\w+)\)',decl)
                    if mt: self.intent[name]=mt[1]
            elif s=='implicit none': pass
            else: self.body.append((line,s))
        self.args=[a for a in args if self.variables[a][0]!='char' and a!='errflg']
        if result: self.variables[self.name]=('float',None)
    def signature(self):
        parts=[]
        for a in self.args:
            typ=self.variables[a][0]
            ref=' &' if self.intent.get(a) in ('out','inout') else ' '
            parts.append(typ+ref+a)
        parts += ['bool has_'+n for n in self.optional]
        return ('float' if self.result else 'void')+' m_'+self.name+'('+', '.join(parts)+')'
    def emit(self):
        out=['// MPAS physics_mmm/sf_mynn.F90:'+str(self.start)+'-'+str(self.end)+' '+self.name,
             '__device__ __noinline__ '+self.signature()+' {']
        if self.result: out.append('float result;')
        for n,(typ,v) in self.variables.items():
            if n in self.args or typ=='char' or n=='errflg' or n==self.name: continue
            out.append(typ+' '+n+(' = '+lit(v,typ) if v is not None else '')+';')
        skip=0
        for line,s in self.body:
            if re.match(r'\d+\s+format',s) or s.startswith(('write(', 'print*')): continue
            if s.startswith('errmsg =') or s.startswith('errflg ='): continue
            if s in ('do i=its,ite','do i = its,ite'):
                out.append('{ // independent MPAS column loop'); continue
            if s.startswith('enddo') or s=='end do': out.append('}'); continue
            if s=='endif' or s=='end if': out.append('}'); continue
            if s=='else': out.append('} else {'); continue
            mt=re.match(r'(else\s*if|elseif|if)\s*\(',s)
            if mt:
                start=s.index('('); depth=0; end=None
                for n in range(start,len(s)):
                    depth += (s[n]=='(')-(s[n]==')')
                    if depth==0: end=n; break
                cond=expr(self,s[start+1:end]).code
                tail=s[end+1:].strip()
                prefix='} else if' if mt[1]!='if' else 'if'
                if tail=='then': out.append(prefix+' ('+cond+') {'); continue
                out.append(prefix+' ('+cond+') '+self.assign(tail)); continue
            if s.startswith('do while'):
                out.append('while ('+expr(self,s[s.index('(')+1:-1]).code+') {'); continue
            if s.startswith('do '):
                var,rr=s[3:].split('=',1); lo,hi=split_args(rr)
                var=var.strip(); out.append('for ('+var+'='+expr(self,lo).code+'; '+var+'<='+expr(self,hi).code+'; ++'+var+') {'); continue
            if s.startswith('call '):
                call=s[5:]; fun=call[:call.index('(')]; argtext=call[call.index('(')+1:-1]
                target=ROUTINES[fun]; args=split_args(argtext)
                args=[a for a,n in zip(args,target.args)]
                out.append('m_'+fun+'('+','.join(expr(self,a).code for a in args)+(',' if target.optional else '')+','.join('true' for a in target.optional)+');'); continue
            if '=' in s: out.append(self.assign(s)); continue
            raise ValueError(('statement',self.name,line,s))
        if self.result: out.append('return result;')
        return '\n'.join(out+['}'])
    def assign(self,s):
        a,b=s.split('=',1)
        left=expr(self,a.strip()).code
        if left==self.name: left='result'
        right=expr(self,b.strip()).code
        if self.result:
            right=re.sub(r'\b'+self.name+r'\b','result',right)
        return left+' = '+right+';'


def parse(text):
    active=None; items=[]
    for line,s in statements(text):
        mt=re.match(r'(?:real\(kind=kind_phys\) )?(subroutine|function) (\w+)\((.*)\)',s)
        if mt:
            active=(mt[2],split_args(mt[3]),mt[1]=='function',line); items=[]
        elif active and s.startswith(('end subroutine','end function')):
            name,args,result,start=active
            r=Routine(name,args,items,result);r.start=start;r.end=line
            ROUTINES[name]=r;active=None
        elif active: items.append((line,s))


def main():
    source=Path(sys.argv[1]).read_text(); parse(source)
    # init/finalize error strings have no arithmetic; table init remains whole.
    keep=[r for n,r in ROUTINES.items() if n not in ('sf_mynn_init','sf_mynn_finalize')]
    declarations='\n'.join('__device__ __noinline__ '+r.signature()+';' for r in keep)
    code='\n'.join(r.emit() for r in keep)
    run=ROUTINES['sf_mynn_run']
    arrays=[a for a in run.args if a in run.arrays]
    scalars=[a for a in run.args if a not in run.arrays and a not in ('its','ite')]
    metadata={"fields":arrays,"scalars":scalars,"outputs":[a for a in arrays if run.intent[a]!='in'],"optional":run.optional,"routines":{r.name:[r.start,r.end] for r in keep}}
    wrapper='extern "C" __global__ void sf_mynn_columns(float *fields, const float *constants, int itimestep, int isfflx, int spp_pbl, int isftcflx, int iz0tlnd, unsigned present, int n) {\nint i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;\n'
    fa=[a for a in scalars if run.variables[a][0]=='float']
    wrapper+='m_sf_mynn_run('+','.join('fields['+str(arrays.index(a))+'*n+i]' if a in arrays else 'constants['+str(fa.index(a))+']' if a in fa else '0' if a in ('its','ite') else a for a in run.args)
    wrapper+=''.join(', (present & (1u<<'+str(k)+'))!=0' for k,a in enumerate(run.optional))+');\n}\n'
    wrapper+='extern "C" __global__ void sf_mynn_tables() { if(blockIdx.x==0 && threadIdx.x==0) m_psi_init(0); }\n'
    head='\n'.join('__device__ float '+a+'[1001];' for a in TABLES)+'\n'
    path=Path(sys.argv[2]);path.write_text(head+declarations+'\n'+code+'\n'+wrapper)
    Path(sys.argv[3]).write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
