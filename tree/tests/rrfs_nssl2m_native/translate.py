"""Build-time translation of the pinned complete RRFS NSSL source to CUDA.

Descriptor/parser scaffolding adapted from lane p12; no MYNN physics is reused.

This tool handles the Fortran subset present in the pinned NSSL module. It
does not execute physics in Python. Unsupported syntax is an error with its
original source location. Generated routines retain file:line citations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re


def split_top(text, separator=","):
    result, start, depth, quote = [], 0, 0, None
    for index, char in enumerate(text):
        if char in "'\"":
            quote = None if quote == char else char if quote is None else quote
        if quote is None:
            depth += (char == "(") - (char == ")")
            if char == separator and depth == 0:
                result.append(text[start:index].strip())
                start = index + 1
    result.append(text[start:].strip())
    return result


def statements(path):
    """Preserve the first physical source line of every logical statement."""
    pending, start, enabled, stack = "", 0, True, []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line.startswith("#"):
            if line.startswith("#if"):
                stack.append((enabled, False))
                enabled = False
            elif line.startswith("#else"):
                parent, take = stack[-1]
                enabled = parent and not take
            elif line.startswith("#endif"):
                enabled, _ = stack.pop()
            elif enabled:
                raise ValueError((path, lineno, line))
            continue
        if not enabled:
            continue
        quote = None
        for index, char in enumerate(line):
            if char in "'\"":
                quote = None if quote == char else char if quote is None else quote
            if char == "!" and quote is None:
                line = line[:index]
                break
        line = line.strip().lstrip("&").strip()
        if not line:
            continue
        if not pending:
            start = lineno
        pending += " " + line.removesuffix("&").strip()
        if not line.endswith("&"):
            for stmt in split_top(pending, ";"):
                if stmt:
                    yield start, "".join(part if part.startswith(("'", '\"')) else part.lower() for part in re.split(r"('(?:[^']|'')*'|\"[^\"]*\")", stmt))
            pending = ""
    assert not pending and not stack


@dataclass
class Variable:
    name: str
    kind: str
    dimensions: list[str] = field(default_factory=list)
    intent: str = ""
    optional: bool = False
    parameter: bool = False
    initial: str | None = None
    line: int = 0


def declaration(stmt, line):
    match = re.match(r"(double\s+precision|real(?:\s*\([^)]*\))?|integer(?:\s*\([^)]*\))?|logical|character\s*\([^)]*\))\s*(.*)", stmt)
    if not match:
        return None
    dtype, rest = match.groups()
    kind = "double" if dtype.startswith("double") or "kind_phys8" in dtype or "kind=8" in dtype else "float" if dtype.startswith("real") else "int" if dtype.startswith("integer") else "bool" if dtype.startswith("logical") else "char"
    if "::" in rest:
        attrs, names = rest.split("::", 1)
    else:
        attrs, names = "", rest
    dim = re.search(r"dimension\s*\(([^)]*)\)", attrs)
    intent = re.search(r"intent\s*\(([^)]*)\)", attrs)
    output = []
    for item in split_top(names):
        initial = None
        if "=" in item:
            item, initial = item.split("=", 1)
        var = re.fullmatch(r"\s*(\w+)\s*(?:\((.*)\))?\s*", item)
        if not var:
            raise ValueError((line, stmt, item))
        dimensions = split_top(var[2]) if var[2] is not None else split_top(dim[1]) if dim else []
        output.append(Variable(var[1], kind, dimensions, intent[1] if intent else "", "optional" in attrs, "parameter" in attrs, initial.strip() if initial else None, line))
    return output


@dataclass
class Routine:
    name: str
    arguments: list[str]
    function: bool
    source: str
    start: int
    end: int = 0
    variables: dict = field(default_factory=dict)
    body: list = field(default_factory=list)
    statement_functions: dict = field(default_factory=dict)


def parse_modules(paths):
    globals_, routines, current = {}, {}, None
    for path in paths:
        for line, stmt in statements(path):
            begin = re.match(r"(?:(real|double precision|integer)\s+)?(subroutine|function)\s+(\w+)\s*\((.*)\)$", stmt)
            end = re.match(r"end\s*(subroutine|function)\b", stmt)
            if begin:
                assert current is None
                current = Routine(begin[3], split_top(begin[4]), begin[2] == "function", path.name, line)
                if current.function:
                    kind = "double" if begin[1] == "double precision" else "int" if begin[1] == "integer" else "float"
                    current.variables[current.name] = Variable(current.name, kind, line=line)
                routines[current.name] = current
            elif end:
                assert current is not None
                current.end = line
                current = None
            elif stmt.startswith("parameter"):
                table = current.variables if current else globals_
                for assignment in split_top(stmt[stmt.index("(")+1:-1]):
                    name, value = assignment.split("=", 1)
                    table[name.strip()].parameter = True
                    table[name.strip()].initial = value.strip()
            elif stmt.startswith("data "):
                table = current.variables if current else globals_
                match = re.fullmatch(r"data\s+(.+?)\s*/\s*(.*?)\s*/", stmt)
                assert match, (line, stmt)
                lhs, values = match.groups()
                base = lhs.split("(",1)[0].strip()
                var = table[base]
                if not hasattr(var, "data_initializers"):
                    var.data_initializers = []
                var.data_initializers.append((lhs, split_top(values)))
            elif (decl := declaration(stmt, line)) is not None:
                table = current.variables if current else globals_
                table.update((var.name, var) for var in decl)
            elif current:
                if not stmt.startswith(("implicit ", "use ", "save", "external ", "logical, external")):
                    current.body.append((line, stmt))
    assert current is None
    for routine in routines.values():
        assert set(routine.arguments) <= routine.variables.keys(), (routine.name, set(routine.arguments)-routine.variables.keys())
        for line, stmt in routine.body:
            if match := re.fullmatch(r"(\w+)\(([^()]*)\)\s*=\s*(.*)", stmt):
                var = routine.variables.get(match[1])
                if var is not None and not var.dimensions:
                    routine.statement_functions[match[1]] = split_top(match[2])
    return globals_, routines


TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?(?:[ed][+-]?\d+)?(?:_\w+)?|\.\d+(?:[ed][+-]?\d+)?(?:_\w+)?)|(\.[a-z]+\.)|(\w+)|('[^']*'|\"[^\"]*\")|(\*\*|<=|>=|==|/=|[+*/(),:\-=<>]))", re.I)
PRECEDENCE = {".or.": 1, ".and.": 2, ".eqv.": 0, ".neqv.": 0, "==": 3, "/=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3, ".eq.": 3, ".ne.": 3, ".lt.": 3, ".gt.": 3, ".le.": 3, ".ge.": 3, "+": 4, "-": 4, "*": 5, "/": 5, "**": 7}


class Expression:
    def __init__(self, source):
        self.tokens, pos = [], 0
        while pos < len(source):
            match = TOKEN.match(source, pos)
            if not match:
                raise ValueError(f"Cannot tokenize {source[pos:]!r} in {source!r}")
            self.tokens.append(next(s for s in match.groups() if s is not None))
            pos = match.end()
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def pop(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise ValueError((expected, token, self.tokens))
        self.index += 1
        return token

    def parse(self, minimum=0):
        token = self.pop()
        if token in ("+", "-", ".not."):
            lhs = ("unary", token, self.parse(6 if token != ".not." else 3))
        elif token == "(":
            lhs = self.parse()
            self.pop(")")
        elif token == ":":
            lhs = ("range", None, self.parse() if self.peek() not in (",", ")", "") else None)
        elif re.match(r"(?:\d|\.\d|'|\")", token) or token in (".true.", ".false."):
            lhs = ("literal", token)
        else:
            lhs = ("name", token)
            if self.peek() == "(":
                self.pop()
                args = []
                if self.peek() != ")":
                    while True:
                        args.append(self.parse())
                        if self.peek() != ",":
                            break
                        self.pop()
                self.pop(")")
                lhs = ("call", token, args)
        while True:
            op = self.peek()
            if op == ":" and minimum == 0:
                self.pop()
                rhs = self.parse() if self.peek() not in (",", ")", "") else None
                lhs = ("range", lhs, rhs)
                continue
            level = PRECEDENCE.get(op, -1)
            if level < minimum:
                break
            self.pop()
            lhs = ("binary", op, lhs, self.parse(level if op == "**" else level+1))
        return lhs

    @classmethod
    def tree(cls, source):
        parser = cls(source.strip())
        result = parser.parse()
        if parser.peek():
            raise ValueError((source, parser.tokens[parser.index:]))
        return result


def promote(a, b):
    return "double" if "double" in (a, b) else "float" if "float" in (a, b) else "int"


class Emitter:
    def __init__(self, globals_, routines, routine=None):
        self.globals = globals_
        self.routines = routines
        self.routine = routine
        self.variables = {**globals_, **(routine.variables if routine else {})}
        self.scalar_index = None
        self.constant = False
        self.selects = []
        self.do_labels = []
        self.loop_number = 0

    def expression(self, source):
        return self.emit(Expression.tree(source))

    def emit(self, tree):
        tag = tree[0]
        if tag == "literal":
            token = tree[1]
            if token in (".true.", ".false."):
                return ("true" if token == ".true." else "false"), "bool"
            if token[0] in "'\"":
                return json.dumps(token[1:-1]), "char"
            base, _, suffix = token.partition("_")
            kind = "double" if "d" in base or suffix == "kind_phys8" else "float" if "." in base or "e" in base or suffix == "kind_phys" else "int"
            value = base.replace("d", "e")
            if kind == "float":
                if "." not in value and "e" not in value:
                    value += ".0"
                value += "f"
            return value, kind
        if tag == "name":
            var = self.variables[tree[1]]
            code = tree[1]
            if self.scalar_index is not None and var.dimensions:
                code += ".linear(" + self.scalar_index + ")"
            return code, var.kind
        if tag == "unary":
            code, kind = self.emit(tree[2])
            op = "!" if tree[1] == ".not." else tree[1]
            return f"({op}({code}))", "bool" if op == "!" else kind
        if tag == "binary":
            _, op, left, right = tree
            a, ak = self.emit(left)
            b, bk = self.emit(right)
            kind = promote(ak, bk)
            comparisons = {".eq.": "==", ".ne.": "!=", ".gt.": ">", ".lt.": "<", ".ge.": ">=", ".le.": "<=", ".and.": "&&", ".or.": "||", ".eqv.": "==", ".neqv.": "!=", "/=": "!="}
            if op in comparisons or op in ("==", "<", ">", "<=", ">="):
                return f"(({a}) {comparisons.get(op, op)} ({b}))", "bool"
            if op == "**":
                if bk == "int" and right[0] == "literal":
                    power = int(right[1])
                    if self.constant:
                        def chain(n):
                            if n == 0:
                                return "1"
                            if n == 1:
                                return a
                            half = chain(n // 2)
                            square = f"(({half}) * ({half}))"
                            return f"(({square}) * ({a}))" if n % 2 else square
                        return chain(power), ak
                    return f"nssl_ipow<{power}>({a})", ak
                return f"nssl_pow(({kind})({a}), ({kind})({b}))", kind
            if self.constant or kind == "int":
                return f"(({a}) {op} ({b}))", kind
            helper = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]
            prefix = "nssl_d" if kind == "double" else "nssl_f"
            return f"{prefix}{helper}({a}, {b})", kind
        if tag == "call":
            _, name, arguments = tree
            name = {"dexp":"exp", "dlog":"log", "alog":"log", "alog10":"log10", "dlog10":"log10", "dsqrt":"sqrt", "dabs":"abs", "dmin1":"min", "amin1":"min", "dmax1":"max", "amax1":"max", "dsin":"sin", "dcos":"cos", "datan":"atan"}.get(name, name)
            if name == "minloc":
                def array_name(node):
                    if node[0] == "name" and self.variables[node[1]].dimensions:
                        return node[1]
                    children = node[2:] if node[0] in ("binary", "unary") else node[2] if node[0] == "call" else []
                    for child in children:
                        if isinstance(child, tuple) and (found := array_name(child)):
                            return found
                    return None
                array = array_name(arguments[0])
                assert array and arguments[1] == ("literal", "1")
                previous = self.scalar_index
                self.scalar_index = "nssl_element"
                value, _ = self.emit(arguments[0])
                self.scalar_index = previous
                return f"nssl_minloc({array}.size(), [&] (int nssl_element) {{ return {value}; }})", "int"
            var = self.variables.get(name)
            if var is not None and var.kind == "char":
                assert len(arguments) == 1 and arguments[0][0] == "range"
                lo = self.emit(arguments[0][1])[0]
                hi = self.emit(arguments[0][2])[0]
                return f"nssl_substring({name}, {lo}, {hi})", "char"
            if var is not None and var.dimensions:
                indices, ranges = [], False
                for dim, arg in enumerate(arguments):
                    if arg[0] == "range":
                        ranges = True
                        lo = self.emit(arg[1])[0] if arg[1] is not None else f"{name}.lo[{dim}]"
                        hi = self.emit(arg[2])[0] if arg[2] is not None else f"{name}.hi[{dim}]"
                        indices.append(f"NsslRange{{{lo}, {hi}}}")
                    else:
                        indices.append(self.emit(arg)[0])
                code = name + (".slice(" if ranges else "(") + ", ".join(indices) + ")"
                if ranges and self.scalar_index is not None:
                    code += f".linear({self.scalar_index})"
                return code, var.kind
            args = [self.emit(arg) for arg in arguments if not (arg[0] == "name" and arg[1] == "kind_phys")]
            if self.routine and name in self.routine.statement_functions:
                return name + "(" + ", ".join(v for v,k in args) + ")", self.variables[name].kind
            if name == "present":
                return f"({args[0][0]}.present())", "bool"
            if name in ("real", "float", "sngl", "int", "ifix", "dble", "dfloat"):
                kind = {"real": "float", "float": "float", "sngl": "float", "int": "int", "ifix": "int", "dble": "double", "dfloat": "double"}[name]
                return f"(({kind})({args[0][0]}))", kind
            if name in ("maxval", "minval"):
                return f"nssl_{name}({args[0][0]})", args[0][1]
            if name in ("ceiling", "floor", "nint"):
                return f"nssl_{name}({args[0][0]})", "int"
            if name in self.routines:
                target = self.routines[name]
                for missing in target.arguments[len(args):]:
                    assert target.variables[missing].optional
                    args.append(("{}", target.variables[missing].kind))
                return "f_" + name + "(" + ", ".join(x[0] for x in args) + ")", target.variables[name].kind
            if name in ("min", "max"):
                kind = args[0][1]
                result = args[0][0]
                for value, vk in args[1:]:
                    kind = promote(kind, vk)
                    result = f"nssl_{name}(({kind})({result}), ({kind})({value}))"
                return result, kind
            if name in ("abs", "sqrt", "exp", "log", "log10", "atan", "sin", "cos", "tanh", "erf", "sign", "mod"):
                return f"nssl_{name}(" + ", ".join(x[0] for x in args) + ")", args[0][1]
            raise ValueError(("unknown function", name, tree))
        raise ValueError(tree)

    def is_array(self, tree):
        if tree[0] == "name":
            return bool(self.variables[tree[1]].dimensions)
        return tree[0] == "call" and tree[1] in self.variables and any(arg[0] == "range" for arg in tree[2])

    def assignment(self, lhs, rhs):
        tree = Expression.tree(lhs)
        if tree[0] == "call" and tree[1] in self.routine.statement_functions:
            names = self.routine.statement_functions[tree[1]]
            args = ", ".join(self.variables[n].kind + " " + n for n in names)
            return "auto " + tree[1] + " = [&] (" + args + ") { return " + self.expression(rhs)[0] + "; };"
        if self.is_array(tree):
            view = self.emit(tree)[0]
            self.scalar_index = "nssl_element"
            a, _ = self.emit(tree)
            b, _ = self.expression(rhs)
            self.scalar_index = None
            return f"for (int nssl_element=0; nssl_element < ({view}).size(); ++nssl_element) {{ {a} = {b}; }}"
        return self.expression(lhs)[0] + " = " + self.expression(rhs)[0] + ";"

    def statement(self, stmt):
        stmt = re.sub(r"^\w+:\s*(do\b)", r"\1", stmt)
        stmt = re.sub(r"^(enddo|end do)\s+\w+$", r"\1", stmt)
        if match := re.match(r"select\s*case\s*\((.*)\)$", stmt):
            self.selects.append(False)
            return "switch (" + self.expression(match[1])[0] + ") {"
        if stmt.startswith("case"):
            prefix = "break; } " if self.selects[-1] else ""
            self.selects[-1] = True
            if stmt == "case default":
                return prefix + "default: {"
            match = re.fullmatch(r"case\s*\((.*)\)", stmt)
            return prefix + " ".join("case " + self.expression(x)[0] + ":" for x in split_top(match[1])) + " {"
        if stmt in ("endselect", "end select"):
            return ("break; } " if self.selects.pop() else "") + "}"
        if match := re.match(r"(\d+)\s+(.*)", stmt):
            code = "nssl_label_" + match[1] + ": " + self.statement(match[2])
            while self.do_labels and self.do_labels[-1] == match[1]:
                code += " }"
                self.do_labels.pop()
            return code
        if match := re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", stmt):
            return "} else if (" + self.expression(match[1])[0] + ") {"
        if stmt in ("else",):
            return "} else {"
        if stmt in ("endif", "end if", "enddo", "end do"):
            return "}"
        if re.match(r"if\s*[(]", stmt):
            parser = Expression(stmt[2:].lstrip())
            parser.pop("(")
            condition = parser.parse()
            parser.pop(")")
            tail = " ".join(parser.tokens[parser.index:])
            # Reconstruct from the raw string to preserve quoted text and kinds.
            depth = 0
            for index, char in enumerate(stmt[stmt.index("("):], stmt.index("(")):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    tail = stmt[index+1:].strip()
                    break
            code = "if (" + self.emit(condition)[0] + ") "
            return code + ("{" if tail == "then" else "{ " + self.statement(tail) + " }")
        if match := re.match(r"do\s*while\s*\((.*)\)$", stmt):
            return "while (" + self.expression(match[1])[0] + ") {"
        if labeled := re.match(r"do\s+(\d+)\s+(.*)", stmt):
            self.do_labels.append(labeled[1])
            stmt = "do " + labeled[2]
        if match := re.match(r"do\s+(\w+)\s*=\s*(.*)$", stmt):
            var, bounds = match[1], split_top(match[2])
            start, end = (self.expression(x)[0] for x in bounds[:2])
            
            step = self.expression(bounds[2])[0] if len(bounds) == 3 else "1"
            self.loop_number += 1
            index = self.loop_number
            end_name, step_name = f"nssl_end_{index}", f"nssl_stride_{index}"
            return (f"for (int nssl_first_{index}=(({var}={start}),0), "
                    f"{end_name}=({end}), {step_name}=({step}); "
                    f"{step_name}>0 ? {var}<={end_name} : {var}>={end_name}; "
                    f"{var}+={step_name}) {{")
        if match := re.match(r"call\s+(\w+)\s*\((.*)\)$", stmt):
            if match[1] == "wrf_error_fatal":
                return "native_error = 2; return;"
            target = self.routines[match[1]]
            supplied, positional = {}, 0
            for arg in split_top(match[2]):
                if keyword := re.match(r"(\w+)\s*=([^=].*)$", arg):
                    supplied[keyword[1]] = keyword[2]
                else:
                    supplied[target.arguments[positional]] = arg
                    positional += 1
            values = []
            for name in target.arguments:
                var = target.variables[name]
                if name in supplied:
                    arg = supplied[name]
                    arg = re.sub(r",\s*kind\s*=\s*kind_phys(?=\))", ",kind_phys", arg)
                    tree = Expression.tree(arg)
                    value = self.emit(tree)[0]
                    if var.dimensions and tree[0] == "call" and tree[1] in self.variables and not any(v[0] == "range" for v in tree[2]):
                        lo, hi = Emitter(self.globals, self.routines, target).bounds(var)
                        for dummy, actual in supplied.items():
                            lo = re.sub(rf"\b{dummy}\b", "(" + actual + ")", lo)
                            hi = re.sub(rf"\b{dummy}\b", "(" + actual + ")", hi)
                        value = f"nssl_sequence<{var.kind}, {len(var.dimensions)}>(&({value}), {lo}, {hi})"
                    if var.dimensions and tree[0] == "name" and len(self.variables[tree[1]].dimensions) != len(var.dimensions):
                        # Original Fortran sequence association; in this pin
                        # the main driver passes its full 2D rthraten to DDMF.
                        lo, hi = Emitter(self.globals, self.routines, target).bounds(var)
                        for dummy, actual in supplied.items():
                            lo = re.sub(rf"\b{dummy}\b", actual, lo)
                            hi = re.sub(rf"\b{dummy}\b", actual, hi)
                        value += f".sequence<{len(var.dimensions)}>({lo}, {hi})"
                    values.append(value)
                elif var.optional:
                    values.append("{}")
                else:
                    raise ValueError((target.name, "missing argument", name, stmt))
            return f"f_{target.name}(" + ", ".join(values) + ");"
        if stmt == "return":
            return "return" + (" " + self.routine.name if self.routine.function else "") + ";"
        if stmt == "exit":
            return "break;"
        if stmt == "cycle":
            return "continue;"
        if stmt == "continue":
            return ";"
        if match := re.match(r"go\s*to\s+(\d+)$", stmt):
            return "goto nssl_label_" + match[1] + ";"
        if stmt.startswith(("open(", "close(", "rewind(")):
            return "; /* File management belongs to namelist orchestration. */"
        if stmt.startswith("read("):
            assert "nml=nssl_mp_params" in stmt
            return "istat = 0; /* Values are supplied before CUDA initialization. */"
        if stmt == "stop":
            return "native_error = 1; return" + (" " + self.routine.name if self.routine.function else "") + ";"
        if stmt.startswith(("print", "write")):
            return "; /* Native diagnostic print; no field arithmetic. */"
        if match := re.match(r"(.+?)\s*=\s*(.*)$", stmt):
            rhs = re.sub(r",\s*kind\s*=\s*kind_phys(?=\))", ",kind_phys", match[2])
            return self.assignment(match[1], rhs)
        raise ValueError((self.routine.name, stmt))

    def variable(self, var, argument=False):
        if var.dimensions:
            return f"NsslArray<{var.kind}, {len(var.dimensions)}> {var.name}"
        kind = "NsslText" if var.kind == "char" else var.kind
        if var.optional:
            kind = f"NsslOptional<{kind}>"
        written = var.intent in ("out", "inout") or (not var.intent and self.routine and any(re.match(rf"{var.name}\s*=", stmt) for _, stmt in self.routine.body))
        return kind + ("& " if argument and written else " ") + var.name

    def bounds(self, var):
        lows, highs = [], []
        for dim, bounds in enumerate(var.dimensions):
            edges = split_top(bounds, ":")
            if len(edges) == 1:
                lo, hi = "1", edges[0]
            else:
                lo, hi = edges
                lo = lo or "1"
                hi = hi or f"{var.name}.extent({dim}) + ({lo}) - 1"
            lows.append(self.expression(lo)[0])
            # Descriptor methods in assumed-shape upper bounds need no AST.
            highs.append(hi if ".extent(" in hi else self.expression(hi)[0])
        return "{" + ", ".join(lows) + "}", "{" + ", ".join(highs) + "}"

    def routine_code(self):
        routine = self.routine
        result = routine.variables[routine.name].kind if routine.function else "void"
        args = ", ".join(self.variable(routine.variables[name], True) for name in routine.arguments)
        output = [f"// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/{routine.source}:{routine.start}-{routine.end} at 1d6e370e", f"__device__ __noinline__ {result} f_{routine.name}({args}) {{", "NsslFrame nssl_frame(arena);"]
        # Bounds can reference a PARAMETER declared later in the Fortran
        # declaration section. Resolve that section before emitting any view.
        for var in ordered_parameters(routine.variables):
            self.constant = True
            initial = parameter_value(routine.name, var, self)
            self.constant = False
            output.append("constexpr " + self.variable(var) + " = " + initial + ";")
        for name in routine.arguments:
            var = routine.variables[name]
            if var.dimensions:
                lo, _ = self.bounds(var)
                # Absent optional arguments have no descriptor bounds in
                # Fortran. Inspecting an empty view would read indeterminate
                # C++ bounds before PRESENT protects the original body.
                prefix = f"if ({name}.present()) " if var.optional else ""
                output.append(prefix + f"{name}.rebase({lo});")
        for name, var in routine.variables.items():
            if name in routine.arguments or name in routine.statement_functions or var.parameter:
                continue
            if var.dimensions:
                lo, hi = self.bounds(var)
                output.append(self.variable(var) + f" = nssl_array<{var.kind}, {len(var.dimensions)}>(arena, {lo}, {hi});")
                if var.initial:
                    output.append(f"for (int di=0; di<{var.name}.size(); ++di) {var.name}.linear(di)=" + self.expression(var.initial)[0] + ";")
            else:
                self.constant = var.parameter
                initial = " = " + self.expression(var.initial)[0] if var.initial is not None else ""
                self.constant = False
                output.append(("constexpr " if var.parameter else "") + self.variable(var) + initial + ";")
            for lhs, values in getattr(var, "data_initializers", []):
                view = self.expression(lhs)[0]
                if var.dimensions:
                    output.append("{ const " + var.kind + " data_values[] = {" + ", ".join(self.expression(v)[0] for v in values) + "}; " + f"for (int di=0; di<{len(values)}; ++di) ({view}).linear(di)=data_values[di]; }}")
                else:
                    assert len(values)==1
                    output.append(var.name + " = " + self.expression(values[0])[0] + ";")
        for line, stmt in routine.body:
            try:
                code = self.statement(stmt)
            except Exception as error:
                raise ValueError(f"{routine.source}:{line}: {stmt}") from error
            output.append(f"// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/{routine.source}:{line}\n{code}")
        if routine.function:
            output.append(f"return {routine.name};")
        output.append("}")
        return "\n".join(output)


PARAMETER_WORDS = {}


def ordered_parameters(variables):
    pending = {name: var for name, var in variables.items() if var.parameter}
    result = []
    while pending:
        ready = [var for var in pending.values()
                 if not (set(re.findall(r"\b[a-z_]\w*\b", var.initial or "")) & pending.keys())]
        assert ready, "cyclic parameters"
        for var in ready:
            result.append(var)
            del pending[var.name]
    return result


def parameter_value(scope, var, emitter):
    key = scope + "." + var.name
    if key in PARAMETER_WORDS:
        import struct
        word = int(PARAMETER_WORDS[key], 16)
        if var.kind == "float":
            return struct.unpack("<f", word.to_bytes(4, "little"))[0].hex() + "f"
        if var.kind == "double":
            return struct.unpack("<d", word.to_bytes(8, "little"))[0].hex()
    return emitter.expression(var.initial)[0]


def generate(paths, out, receipt, parameter_words=None):
    global PARAMETER_WORDS
    PARAMETER_WORDS = parameter_words or {}
    globals_, routines = parse_modules(paths)
    emitter = Emitter(globals_, routines)
    pieces = ["// Generated from pinned RRFS NSSL; see translation receipt.", "struct NsslState {", "int native_error = 0;"]
    ordered = ordered_parameters(globals_)
    ordered.extend(var for var in globals_.values() if not var.parameter)
    for var in ordered:
        emitter.constant = True
        initial = " = " + (parameter_value("module", var, emitter) if var.parameter else emitter.expression(var.initial)[0]) if var.initial is not None else "{}"
        pieces.append(("static constexpr " if var.parameter else "") + emitter.variable(var) + initial + ";")
    pieces.extend(["};", "struct NsslColumn : NsslState {", "NsslArena& arena;"])
    pieces.append("// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:174-1102; static module storage.\n__device__ void allocate_globals() {")
    for var in globals_.values():
        if not var.dimensions:
            continue
        lo, hi = emitter.bounds(var)
        pieces.append(f"{var.name} = nssl_array<{var.kind}, {len(var.dimensions)}>(arena, {lo}, {hi});")
        pieces.append(f"for (int di=0; di<{var.name}.size(); ++di) {var.name}.linear(di)=0;")
        for lhs, values in getattr(var, "data_initializers", []):
            view = emitter.expression(lhs)[0]
            pieces.append("{ const " + var.kind + " data_values[] = {" + ", ".join(emitter.expression(v)[0] for v in values) + "}; " + f"for (int di=0; di<{len(values)}; ++di) ({view}).linear(di)=data_values[di]; }}")
    pieces.append("}")
    pieces.append("// RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:174-1102 at 1d6e370e; original module storage.\n__device__ void copy_globals(const NsslState& other) {")
    for var in globals_.values():
        if not var.parameter:
            pieces.append(f"{var.name} = other.{var.name};")
    pieces.append("}")
    for routine in routines.values():
        pieces.append(Emitter(globals_, routines, routine).routine_code())
    pieces.append("};")
    text = "\n".join(pieces) + "\n"
    out.write_text(text)
    result = {"scope": "whole-module mechanical translation; compilation and numerical proof separate",
              "preprocessor": {"WRF_CHEM": 0},
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "routines": [{"name": r.name, "file": r.source, "lines": [r.start, r.end],
                            "arguments": r.arguments, "statements": len(r.body)} for r in routines.values()]}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return text, routines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    generate([args.source], args.out, args.receipt)
