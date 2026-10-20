"""Fortran source inventory with original line spans (no numeric data processing).

MPAS sources: src/core_atmosphere/physics/mpas_atmphys_driver*.F.
This lexer handles quoted comments, continuations and nested actual expressions.
It intentionally records unsupported declarations rather than guessing them.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import re

# Executable array/component assignments can precede the first CALL. Hooks
# must declare their counters before these statements as well as scalar ones.
EXECUTABLE = re.compile(r'(?:call\b|if\s*\(|do\b|select\b|\w+(?:\s*\([^=]*\))?(?:\s*%\s*\w+(?:\s*\([^=]*\))?)*\s*(?:=>|=)|write\s*\(|read\s*\(|allocate\s*\(|deallocate\s*\(|nullify\s*\(|return\b|stop\b|error\s+stop\b|where\s*\(|forall\s*\(|associate\s*\()', re.I)


def first_executable(routine):
    """Keep statement functions in the declaration section (NSSL uses them)."""
    for st in routine.body:
        text=re.sub(r'^\d+\s+','',st.text)
        # An initialized array declaration can match the broad assignment
        # expression through its type's parentheses. Native declarations must
        # all precede entry hooks (SMOKE ssalt_mod.F90:210-266).
        if declarations(text):continue
        if not EXECUTABLE.match(text):continue
        function=re.match(r'^(\w+)\s*\(\s*([a-z]\w*(?:\s*,\s*[a-z]\w*)*)\s*\)\s*=',text,re.I)
        variable=routine.variables.get(function[1].lower()) if function else None
        if variable and variable.supported and not variable.dims:continue
        return st
    return None


def split_top(text, delimiter=','):
    result, start, depth, quote = [], 0, 0, ''
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 1
                else:
                    quote = ''
        elif c in "'\"":
            quote = c
        elif c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        elif c == delimiter and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
        i += 1
    result.append(text[start:].strip())
    return result


@dataclass
class Statement:
    text: str
    start: int
    end: int


def statements(source):
    lines = source.splitlines(keepends=True)
    buffer, start = '', 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            continue
        quote, body = '', ''
        j = 0
        while j < len(line):
            c = line[j]
            if quote:
                body += c
                if c == quote:
                    if j + 1 < len(line) and line[j + 1] == quote:
                        body += c
                        j += 1
                    else:
                        quote = ''
            elif c in "'\"":
                quote = c
                body += c
            elif c == '!':
                break
            else:
                body += c
            j += 1
        body = body.strip()
        if not body:
            continue
        if not buffer:
            start = i
        buffer += ' ' + body.lstrip('&').rstrip('&').strip()
        if body.endswith('&'):
            continue
        for part in split_top(buffer, ';'):
            if part:
                yield Statement(part.strip(), start, i + 1)
        buffer = ''
    if buffer:
        raise ValueError('unterminated continuation')


@dataclass
class Variable:
    name: str
    type: str
    attrs: str
    dims: list[str]
    intent: str
    optional: bool
    initialized: bool

    @property
    def supported(self):
        return self.type.startswith(('real', 'complex', 'integer', 'logical', 'character', 'double precision', 'double complex')) and len(self.dims) <= 5


def declarations(statement):
    m = re.match(r'\s*(double\s+precision|double\s+complex|complex(?:\s*\([^)]*\)|\*\d+)?|real(?:\s*\([^)]*\)|\*\d+)?|integer(?:\s*\([^)]*\)|\*\d+)?|logical(?:\s*\([^)]*\)|\*\d+)?|character(?:\s*\([^)]*\)|\*\d+)?|type\s*\([^)]*\))(?=\s|::|,|$)\s*(.*)$', statement, re.I)
    if not m:
        return []
    typ, rest = m.groups()
    if '::' in rest:
        attrs, entities = rest.split('::', 1)
    else:
        attrs, entities = '', rest
    dm = re.search(r'dimension\s*\(([^)]*)\)', attrs, re.I)
    dims = split_top(dm[1]) if dm else []
    im = re.search(r'intent\s*\(\s*(inout|in|out)\s*\)', attrs, re.I)
    result = []
    for entity in split_top(entities):
        m = re.match(r'([A-Za-z]\w*)\s*(?:\(([^=]*)\))?\s*(.*)$', entity)
        if not m:
            continue
        name, own_dims, tail = m.groups()
        result.append(Variable(name.lower(), typ.lower(), attrs.lower(), split_top(own_dims) if own_dims else dims,
                               im[1].lower() if im else 'inout', 'optional' in attrs.lower(), '=' in tail))
    return result


@dataclass
class Routine:
    name: str
    module: str
    path: str
    args: list[str]
    start: int
    end: int
    body: list[Statement]
    variables: dict[str, Variable]


def routines(path, root):
    source = active_source(path.read_text(errors='replace'))
    module, current, depth = '', None, 0
    for st in statements(source):
        mm = re.match(r'module\s+(?!procedure\b)(\w+)', st.text, re.I)
        if mm:
            module = mm[1].lower()
        m = re.match(r'(?:recursive\s+|pure\s+|elemental\s+)*subroutine\s+(\w+)\s*(?:\((.*)\))?', st.text, re.I)
        if not m:
            m=re.match(r'(?!end\b)(?:[\w*]+(?:\([^)]*\))?\s+)*function\s+(\w+)\s*\((.*?)\)(?:\s+result\s*\(\w+\))?',st.text,re.I)
        if m:
            if current:
                depth += 1
                current.body.append(st)
                continue
            current = Routine(m[1].lower(), module, str(path.relative_to(root)),
                              [x.lower() for x in split_top(m[2] or '') if x], st.start, 0, [], {})
        elif current and re.match(r'end\s*(?:subroutine|function)\b', st.text, re.I):
            if depth:
                depth -= 1
                current.body.append(st)
            else:
                current.end = st.end
                yield current
                current = None
        elif current:
            current.body.append(st)
            for v in declarations(st.text):
                current.variables[v.name] = v


def call(statement):
    m = re.match(r'(?:if\s*\(.*\)\s*)?call\s+(\w+)\s*(?:\((.*)\))?\s*$', statement.text, re.I)
    if m:
        return m[1].lower(), [x for x in split_top(m[2] or '') if x]
    return None


def bind(callee, actuals):
    result, pos, named = {}, 0, False
    for actual in actuals:
        m = re.match(r'^(\w+)\s*=(?!=)(.*)$', actual, re.S)
        if m:
            name, expression = m[1].lower(), m[2].strip()
            named = True
        else:
            if named or pos >= len(callee.args):
                raise ValueError(f'invalid positional argument: {actual}')
            name, expression = callee.args[pos], actual
            pos += 1
        if name not in callee.args or name in result:
            raise ValueError(f'invalid/duplicate dummy: {name}')
        result[name] = expression
    missing = [a for a in callee.args if a not in result and not (callee.variables.get(a) and callee.variables[a].optional)]
    if missing:
        raise ValueError(f'missing required dummies: {missing}')
    return result


def active_source(source, macros=None):
    """Select MPAS non-chemistry branches while preserving original line offsets.

    Unsupported preprocessor expressions are rejected, never guessed.
    """
    import ast
    macros = dict(macros or {'mpas': 1, 'DO_PHYSICS': 1, 'SINGLE_PRECISION': 1, '_MPI': 1, 'WRF_CHEM': 0, 'tempo_mpas': 1})
    stack, active, result = [], True, []
    def test(expr):
        expr = re.sub(r'defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)',
                      lambda m: str(int((m[1] or m[2]) in macros)), expr)
        expr = re.sub(r'\b[A-Za-z_]\w*\b', lambda m: str(macros.get(m[0], 0)), expr)
        expr = expr.replace('&&', ' and ').replace('||', ' or ')
        expr = re.sub(r'!(?!=)', ' not ', expr).strip()
        tree = ast.parse(expr, mode='eval')
        allowed = (ast.Expression, ast.Constant, ast.BoolOp, ast.And, ast.Or,
                   ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Eq, ast.NotEq,
                   ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.BinOp, ast.Add, ast.Sub,
                   ast.Mult, ast.Div, ast.Mod, ast.BitAnd, ast.BitOr)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise ValueError(f'unsupported preprocessor condition {expr}')
        return bool(eval(compile(tree, '<preprocessor>', 'eval'), {'__builtins__': {}}, {}))
    for line in source.splitlines(keepends=True):
        m = re.match(r'\s*#\s*(\w+)\s*(.*)', line)
        if not m:
            result.append(line if active else '\n')
            continue
        directive, body = m.groups()
        if directive in {'ifdef', 'ifndef', 'if'}:
            yes = (body.strip() in macros) if directive == 'ifdef' else ((body.strip() not in macros) if directive == 'ifndef' else test(body))
            stack.append([active, yes])
            active = active and yes
        elif directive in {'else', 'elif'}:
            parent, taken = stack[-1]
            yes = not taken and (directive == 'else' or test(body))
            stack[-1][1] = taken or yes
            active = parent and yes
        elif directive == 'endif':
            active = stack.pop()[0]
        elif directive == 'define' and active:
            parts = body.split(None, 1)
            if parts and '(' not in parts[0]:
                try:
                    macros[parts[0]] = int(parts[1]) if len(parts) > 1 else 1
                except ValueError:
                    macros[parts[0]] = 1
        elif directive == 'undef' and active:
            macros.pop(body.strip(), None)
        result.append('\n')
    if stack:
        raise ValueError('unclosed preprocessing conditional')
    return ''.join(result)
