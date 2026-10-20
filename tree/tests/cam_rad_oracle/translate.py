"""Closed whole-module CAM v8.4.2 translation generator.

Parser primitives adapted from the pinned engine's lake oracle translator.
No radiation kernel or WRF radiation implementation is reused. Unsupported
Fortran must raise, never silently disappear. Generated code is unqualified
until whole-module oracle and forecast proofs pass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import argparse

def split(text, delim=","):
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == delim and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def statements(path):
    stack, active, carry, start = [], True, "", 0
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            if line.startswith("#if"):
                condition = line.split(maxsplit=1)[1]
                condition = re.sub(r'defined\s*\(\s*(\w+)\s*\)',
                    lambda m: '1' if m[1]=='mpas' else '0', condition)
                condition = re.sub(r'defined\s+(\w+)',
                    lambda m: '1' if m[1]=='mpas' else '0', condition)
                condition = re.sub(r'!(?!=)', ' not ', condition)
                condition = condition.replace('NMM_CORE', '0')
                take = bool(eval(condition, {'__builtins__':{}}, {}))
                stack.append((active, take))
                active = active and take
            elif line.startswith("#else"):
                parent, take = stack[-1]
                active = parent and not take
            elif line.startswith("#endif"):
                active, _ = stack.pop()
            elif active:
                raise ValueError((lineno, line))
            continue
        if not active:
            continue
        # Strip comments only outside quoted character strings.
        quote=None
        for i,c in enumerate(line):
            if c in ('\"', "'"):
                quote = None if quote==c else c if quote is None else quote
            if c=='!' and quote is None:
                line=line[:i]; break
        line=line.strip().lower()
        if not line:
            continue
        if not carry:
            start = lineno
        carry += " " + line.lstrip("&").rstrip("&").strip()
        if not line.endswith("&"):
            for item in split(carry.strip(), ";"):
                yield start, item
            carry = ""
    assert not carry and not stack


@dataclass
class Variable:
    name: str
    typ: str
    dims: list
    intent: str = ""
    initial: str = ""
    const: bool = False
    optional: bool = False


def declaration(line):
    if '::' not in line:
        m = re.match(r'(real(?:\s*\([^)]*\))?|integer|logical|character(?:\s*\([^)]*\))?)\s+(.+)', line)
        if m and not m[2].startswith('function '):
            line = m[1] + '::' + m[2]
    if "::" not in line:
        return None
    attrs, decls = line.split("::", 1)
    typ = ("double" if attrs.startswith("real(r8)") else
           "float" if attrs.startswith("real") else
           "int" if attrs.startswith("integer") else
           "bool" if attrs.startswith("logical") else "char")
    dim = re.search(r"dimension\s*\((.*?)\)", attrs)
    intent = re.search(r"intent\s*\((.*?)\)", attrs)
    out = []
    for decl in split(decls):
        name_init = decl.split("=", 1)
        m = re.fullmatch(r"(\w+)\s*(?:\((.*)\))?", name_init[0].strip())
        if not m:
            raise ValueError(decl)
        dims = m[2] if m[2] is not None else dim[1] if dim else None
        out.append(Variable(m[1], typ, split(dims) if dims else [],
                            intent[1].strip() if intent else "",
                            name_init[1].strip() if len(name_init) == 2 else "",
                            "parameter" in attrs, "optional" in attrs))
    return out


TOKEN = re.compile(r"\s*(\d+(?:\.\d*)?(?:[ed][+-]?\d+)?(?:_r8)?|\.\d+(?:[ed][+-]?\d+)?(?:_r8)?|\.[a-z]+\.|[a-z_]\w*|\*\*|<=|>=|==|/=|[^\s])")
PREC = {".or.": 1, ".and.": 2, ".eqv.": 2, ".neqv.": 2,
        "==": 3, "/=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
        ".eq.": 3, ".ne.": 3, ".lt.": 3, ".gt.": 3, ".le.": 3, ".ge.": 3,
        "+": 4, "-": 4, "*": 5, "/": 5, "**": 7}
OPS = {".or.": "||", ".and.": "&&", ".eqv.": "==", ".neqv.": "!=",
       ".eq.": "==", ".ne.": "!=", ".lt.": "<", ".gt.": ">",
       ".le.": "<=", ".ge.": ">=", "/=": "!="}


class Expression:
    def __init__(self, source, symbols, slices=None):
        self.tokens = TOKEN.findall(source)
        self.i, self.symbols, self.slices = 0, symbols, slices or {}

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else ""

    def parse(self, minimum=0):
        tok = self.take()
        if tok in ("+", "-", ".not."):
            # Fortran's leading sign applies to its whole multiplicative
            # term. MPAS module_ra_cam.F:7308 negates after division in the
            # strict GNU object; moving it into the coefficient changes NaN
            # sign words on explicit negative-aerosol diagnostic inputs.
            left = "(" + ("!" if tok == ".not." else tok) + self.parse(3 if tok == ".not." else 5) + ")"
        elif tok == "(":
            left = self.parse()
            assert self.take() == ")"
            left = "(" + left + ")"
        elif re.match(r"\d|\.\d", tok):
            if "_r8" in tok:
                left = tok.replace("_r8", "").replace("d", "e")
                if not any(c in left for c in ".e"):
                    left += ".0"
            elif any(c in tok for c in ".ed"):
                left = tok.replace("d", "e") + ("" if "d" in tok else "f")
            else:
                left = tok
        elif tok in (".true.", ".false."):
            left = tok[1:-1]
        else:
            left = tok
            if self.peek() == "(":
                self.take()
                args, rawargs, begin, depth = [], [], self.i, 0
                while True:
                    t = self.take()
                    if t == "(" : depth += 1
                    elif t == ")" and depth: depth -= 1
                    elif (t == "," or t == ")") and depth == 0:
                        rawargs.append("".join(self.tokens[begin:self.i-1]))
                        begin = self.i
                        if t == ")": break
                for j, arg in enumerate(rawargs):
                    if tok == 'maxval' and arg.startswith('dim='):
                        args.append(expr(arg[4:],self.symbols)); continue
                    if ":" in arg and tok in self.symbols and self.symbols[tok].dims:
                        lo = arg.split(":")[0]
                        var = self.symbols[tok]
                        lo = lo or bounds(var.dims[j])[0]
                        hi = arg.split(':')[1] or bounds(var.dims[j])[1]
                        args.append('cam_section(' + expr(lo,self.symbols) + ',' + expr(hi,self.symbols) + ')')
                    else:
                        args.append(expr(arg, self.symbols, None if tok in ('maxval','sum','count','size','present') else self.slices))
                if tok in self.symbols and self.symbols[tok].dims:
                    if any(':' in a for a in rawargs):
                        left = tok + '.section(' + ','.join(args) + ')'
                        if self.slices: left += '.linear(_slice)'
                    else:
                        left = tok + "(" + ",".join(args) + ")"
                elif tok in ("min", "max"):
                    left = "cam_" + tok + "(" + ",".join(args) + ")"
                elif tok in ("real", "dble", "int", "nint"):
                    if tok == 'nint': left = 'cam_nint(' + args[0] + ')'
                    else: left = "((" + {"real":"float", "dble":"double", "int":"int"}[tok] + ")(" + args[0] + "))"
                elif tok in ("sum",'product'):
                    left = 'cam_'+tok+'(' + args[0] + ')'
                elif tok == 'maxval':
                    left = 'cam_maxval(' + ','.join(args) + ',_slice)'
                elif tok in ('sign','abs','sqrt','log','log10','exp','sin','cos','asin','atan','mod','size','count','float','min0','max0','amin1','amax1','present','aint'):
                    left = 'cam_' + tok + '(' + ','.join(args) + ')'
                else: left = tok + "(" + ",".join(args) + ")"
            elif tok in self.symbols and self.symbols[tok].dims and tok in self.slices:
                left = tok + '.linear(_slice)'
        while self.peek() in PREC and PREC[self.peek()] >= minimum:
            op = self.take()
            right = self.parse(PREC[op] + (0 if op == "**" else 1))
            if op in ('+','-','*','/'):
                left = 'cam_' + {'+':'add','-':'sub','*':'mul','/':'div'}[op] + '(' + left + ',' + right + ')'
            elif op == "**": left = "cam_pow(" + left + "," + right + ")"
            else: left = "(" + left + OPS.get(op, op) + right + ")"
        return left


def expr(source, symbols, slices=None):
    parser = Expression(source, symbols, slices)
    result = parser.parse()
    if parser.i != len(parser.tokens):
        raise ValueError((source, parser.tokens[parser.i:]))
    return result


def bounds(dim):
    parts = dim.split(":")
    return parts if len(parts) == 2 else ["1", dim]


@dataclass
class Routine:
    name: str
    args: list
    result: str | None
    file: str
    start: int
    variables: dict = field(default_factory=dict)
    body: list = field(default_factory=list)
    end: int = 0


def parse(paths):
    module, routines = {}, {}
    for path in paths:
        current = None
        for n,line in statements(path):
            start = re.match(r'(?:(real\s*\([^)]*\)|logical|integer)\s+)?(subroutine|function)\s+(\w+)\s*\((.*?)\)\s*(?:result\((\w+)\))?$',line)
            if start:
                current = Routine(start[3],split(start[4]),start[5] or start[3] if start[2]=='function' else None,path.name,n)
                routines[current.name] = current
                if start[1]:
                    v = declaration(start[1]+'::'+current.result)[0]
                    current.variables[v.name] = v
                continue
            if re.match(r'end\s+(subroutine|function)',line):
                current.end=n; current=None; continue
            if line.startswith(('module ','end module','contains','implicit ','use ','save')):
                continue
            dest = current.variables if current else module
            declared = declaration(line)
            if declared is not None:
                for v in declared: dest[v.name]=v
                continue
            if line.startswith('parameter'):
                for p in split(line[line.index('(')+1:-1]):
                    name,value=p.split('=',1)
                    name=name.strip(); dest[name].initial=value.strip(); dest[name].const=True
                continue
            if line.startswith('data '):
                name,values = line[5:].strip().split('/',1)
                dest[name.strip()].initial='DATA:'+values.rsplit('/',1)[0]
                continue
            if current:
                current.body.append((n,line))
            else:
                raise ValueError((path.name,n,line))
    for r in routines.values():
        for a in r.args:
            if a not in r.variables: raise ValueError((r.name,'undeclared argument',a))
        if r.result and r.result not in r.variables:
            raise ValueError((r.name,'undeclared result',r.result))
    return module,routines


def dimensions(variable, symbols):
    return ','.join(expr(b,symbols) for d in variable.dims for b in bounds(d))


def scalar_initial(value,symbols):
    value=value.removeprefix('DATA:').strip()
    if value=="int(o'17777777777')": return '2147483647'
    return expr(value,symbols)


def initializer(v,symbols):
    if not v.initial: return ''
    if not v.dims: return scalar_initial(v.initial,symbols)
    value=v.initial
    if value.startswith('reshape'):
        # Flatten nested Fortran constructors, preserving column-major order.
        value=value[len('reshape('):].rsplit(', (/',1)[0]
        value=value.replace('(/','').replace('/)','')
    else: value=value.removeprefix('DATA:').replace('(/','').replace('/)','')
    return ','.join(expr(s,symbols) for s in split(value.strip()))


def assignment(line,symbols):
    lhs,rhs=line.split('=',1); lhs=lhs.strip(); rhs=rhs.strip()
    base=lhs.split('(')[0].strip()
    if symbols.get(base) and symbols[base].dims and ('(' not in lhs or ':' in lhs):
        dest=expr(lhs,symbols)
        val=expr(rhs,symbols,{n:True for n,v in symbols.items() if v.dims})
        return '{ auto _lhs='+dest+'; for(int _slice=0;_slice<_lhs.count;++_slice) _lhs.linear(_slice)='+val+'; }'
    return expr(lhs,symbols)+'='+expr(rhs,symbols)+';'


def action(n,line,symbols,routine):
    if line.startswith(('write','print')): return '/* MPAS diagnostic only */'
    if line.startswith('call '):
        target=line[5:].strip().split('(')[0].strip()
        if target in ('physics_message',): return '// MPAS logging only'
        if target=='physics_error_fatal':
            return 'error='+str(n)+'; return'+(' '+routine.result if routine.result else '')+';'
        raw=split(line[line.index('(')+1:-1])
        callee=ROUTINES[target]
        converted=[]
        for arg,name in zip(raw,callee.args,strict=True):
            v=callee.variables[name]
            lowered=expr(arg,symbols)
            base=arg.split('(')[0]
            if v.dims and '(' in arg and ':' not in arg and base in symbols and symbols[base].dims:
                lowered='CamArray<'+v.typ+'>::from(&'+lowered+')'
            converted.append(lowered)
        return target+'('+','.join(converted)+'); if(error) return'+(' '+routine.result if routine.result else '')+';'
    if line.startswith('stop'):
        return 'error='+str(n)+'; return'+(' '+routine.result if routine.result else '')+';'
    if line=='return': return 'return'+(' '+routine.result if routine.result else '')+';'
    if line=='cycle': return 'continue;'
    if line=='exit': return 'break;'
    if re.match(r'go\s*to\s+\d+',line): return 'goto label_'+re.findall(r'\d+',line)[0]+';'
    if '=' in line: return assignment(line,symbols)
    raise ValueError((routine.name,n,line))


def body(n,line,symbols,routine):
    prefix=''
    label=re.match(r'^(\d+)\s+(.*)',line)
    if label: prefix='label_'+label[1]+': ; '; line=label[2]
    if line=='continue': return prefix
    if line.startswith(('write','print','format')): return '// MPAS diagnostic only'
    if line in ('enddo','end do','endif','end if'): return prefix+'}'
    if line=='else': return prefix+'} else {'
    if line=='do': return prefix+'while(true) {'
    if line.startswith('do while'):
        return prefix+'while('+expr(line[line.index('(')+1:-1],symbols)+') {'
    if line.startswith('do '):
        name,values=line[3:].split('=',1); name=name.strip(); values=split(values)
        step=values[2] if len(values)>2 else '1'
        return prefix+'for('+name+'='+expr(values[0],symbols)+';'+name+('>=' if step.startswith('-') else '<=')+expr(values[1],symbols)+';'+name+'+='+expr(step,symbols)+') {'
    if line.startswith(('if ','if(','else if','elseif')):
        start=line.index('('); depth=0
        for end in range(start,len(line)):
            depth+=(line[end]=='(')-(line[end]==')')
            if not depth: break
        cond=expr(line[start+1:end],symbols)
        tail=line[end+1:].strip()
        return prefix+('} else if' if line.startswith('else') else 'if')+'('+cond+') {'+('' if tail=='then' else action(n,tail,symbols,routine)+'}')
    return prefix+action(n,line,symbols,routine)


def generate(paths):
    global ROUTINES
    module,routines=parse(paths)
    ROUTINES=routines
    lines=['// Whole MPAS CAM module transcription. No acceptance claim.',
           'struct CamColumn {','CamArena &arena; int error=0;',
           '// Partition metadata: preserve the native chunk first-column cache test.',
           'int native_chunk_its=0;']
    lines.append('static constexpr int r8kind=8;')
    # Parameter scalar declarations are compile-time Fortran expressions.
    for v in module.values():
        if v.typ=='char': continue
        if v.dims: lines.append('CamArray<'+v.typ+'> '+v.name+';')
        elif v.const:
            lines.append('static constexpr '+v.typ+' '+v.name+'='+scalar_initial(v.initial,module)+';')
        else: lines.append(v.typ+' '+v.name+';')
    lines+=['CAM_HD CamColumn(CamArena &a):arena(a) {}']
    for r in routines.values():
        symbols=module|r.variables
        statement_functions={}
        for n,line in r.body:
            m=re.match(r'(\w+)\s*\(([^()]*)\)\s*=(?!=)(.*)',line)
            if m and m[1] in symbols and not symbols[m[1]].dims:
                statement_functions[m[1]]=(split(m[2]),m[3])
        ret=r.variables[r.result].typ if r.result else 'void'
        args=[]
        for name in r.args:
            v=r.variables[name]
            # MPAS passes array descriptors by reference. Keep that ABI here;
            # each routine still rebounds its own descriptor below, while the
            # underlying storage and aliases remain unchanged. Passing every
            # descriptor by value inflated the CUDA solver call stack.
            typ='const CamArray<'+v.typ+'>&' if v.dims else ('const ' if v.intent=='in' else '')+v.typ+'&'
            args.append(typ+' '+('_'+name if v.dims else name))
        lines += ['// MPAS '+r.file+':'+str(r.start)+'-'+str(r.end),
                  'CAM_ROUTINE '+ret+' '+r.name+'('+','.join(args)+') {','CamFrame _frame(arena);']
        if r.name=='camrad':
            lines.append('const int cam_cache_its=native_chunk_its?native_chunk_its:its;')
            symbols['cam_cache_its']=Variable('cam_cache_its','int',[])
        for v in r.variables.values():
            if v.const and not v.dims and v.name not in r.args:
                lines.append('const '+v.typ+' '+v.name+'='+scalar_initial(v.initial,symbols)+';')
        for name in r.args:
            v=r.variables[name]
            if v.dims:
                lines.append('CAM_TRACE("'+r.name+'.'+name+'");')
                if ':' in v.dims:
                    lines.append('auto '+name+'=_'+name+';')
                else:
                    lines.append('auto '+name+'=_'+name+'.rebound('+dimensions(v,symbols)+');')
        for v in r.variables.values():
            if v.name in r.args or v.name in statement_functions or v.typ=='char' or (v.const and not v.dims): continue
            if v.dims:
                lines.append('auto '+v.name+'=arena.array<'+v.typ+'>('+dimensions(v,symbols)+');')
                if v.initial:
                    vals=initializer(v,symbols)
                    lines.append('{ const '+v.typ+' _init[]={'+vals+'}; '+v.name+'.init(_init,sizeof(_init)/sizeof(_init[0])); }')
            else:
                const='const ' if v.const else ''
                value='='+scalar_initial(v.initial,symbols) if v.initial else ''
                lines.append(const+v.typ+' '+v.name+value+';')
        for name,(args,value) in statement_functions.items():
            signature=','.join(symbols[a].typ+' '+a for a in args)
            lines.append('auto '+name+'=[&]('+signature+') { return '+expr(value,symbols)+'; };')
        for n,line in r.body:
            if any(re.match(name+r'\s*\(',line) for name in statement_functions): continue
            if r.name=='camrad' and 'abstot_3d(its,kts,kts,jts)' in line:
                line=line.replace('abstot_3d(its,kts,kts,jts)','abstot_3d(cam_cache_its,kts,kts,jts)')
            try: out=body(n,line,symbols,r)
            except Exception as exc: raise ValueError((r.file,r.name,n,line)) from exc
            lines += ['// MPAS '+r.file+':'+str(n),out]
        if r.result: lines.append('return '+r.result+';')
        lines.append('}')
    lines.append('};')
    # C++ keywords used as Fortran locals.
    result='\n'.join(lines)+'\n'
    for keyword in ('new','class','delete','private','default'):
        result=re.sub(r'\b'+keyword+r'\b','cam_'+keyword,result)
    inventory={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'routines':[{ 'name':r.name,'source':r.file,'first_line':r.start,'last_line':r.end} for r in routines.values()]}
    return result,inventory,module,routines


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    args=p.parse_args()
    text,inventory,_,_=generate([args.source/'module_ra_cam_support.F',args.source/'module_ra_cam.F'])
    args.output.write_text(text)
    args.inventory.write_text(json.dumps(inventory,indent=2)+'\n')


