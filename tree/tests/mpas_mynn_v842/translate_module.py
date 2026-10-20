"""Build-time translation of the pinned complete MMM MYNN source to CUDA.

This tool handles the Fortran subset present in the three pinned modules. It
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
                assert "WRF" in line or "HARDCODE_VERTICAL" in line, (path, lineno, line)
                stack.append(enabled)
                enabled = False
            elif line.startswith("#else"):
                enabled = stack[-1] and not enabled
            elif line.startswith("#endif"):
                enabled = stack.pop()
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
                    yield start, re.sub(r"\benum\b", "mynn_enum", stmt.lower())
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
        output.append(Variable(var[1], kind, dimensions, intent[1] if intent else "", "optional" in attrs, "parameter" in attrs, initial, line))
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


def parse_modules(paths):
    globals_, routines, current = {}, {}, None
    for path in paths:
        for line, stmt in statements(path):
            begin = re.match(r"(subroutine|function)\s+(\w+)\s*\((.*)\)$", stmt)
            end = re.match(r"end\s*(subroutine|function)\b", stmt)
            if begin:
                assert current is None
                current = Routine(begin[2], split_top(begin[3]), begin[1] == "function", path.name, line)
                routines[current.name] = current
            elif end:
                assert current is not None
                current.end = line
                current = None
            elif (decl := declaration(stmt, line)) is not None:
                table = current.variables if current else globals_
                table.update((var.name, var) for var in decl)
            elif current:
                if not stmt.startswith(("implicit ", "use ", "save", "external ")):
                    current.body.append((line, stmt))
    assert current is None
    for routine in routines.values():
        assert set(routine.arguments) <= routine.variables.keys(), (routine.name, set(routine.arguments)-routine.variables.keys())
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

    def expression(self, source):
        return self.emit(Expression.tree(source))

    def convert(self, code, original, target):
        if not self.constant and (original, target) in (("float", "double"), ("double", "float")):
            helper = "mynn_real64" if target == "double" else "mynn_real32"
            return f"{helper}({code})"
        return f"(({target})({code}))"

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
            if op == "-" and kind in ("float", "double") and not self.constant:
                return f"mynn_neg({code})", kind
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
                    return f"mynn_ipow<{power}>({a})", ak
                if (ak in ("float", "double") and right[0] == "literal"
                        and float(right[1].partition("_")[0].replace("d", "e")) == 2):
                    # Strict GNU lowers real-valued **2. to a single multiply,
                    # including bl_mynn_subroutines.F90:4270,6376,6428. Calling
                    # libm here introduces a rounding that native never takes.
                    return f"mynn_ipow<2>({a})", ak
                return f"mynn_pow({self.convert(a, ak, kind)}, {self.convert(b, bk, kind)})", kind
            if self.constant or kind == "int":
                return f"(({a}) {op} ({b}))", kind
            helper = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op]
            prefix = "mynn_d" if kind == "double" else "mpas_"
            if kind == "double":
                if ak == "float": a = self.convert(a, ak, kind)
                if bk == "float": b = self.convert(b, bk, kind)
            return f"{prefix}{helper}({a}, {b})", kind
        if tag == "call":
            _, name, arguments = tree
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
                self.scalar_index = "mynn_element"
                value, _ = self.emit(arguments[0])
                self.scalar_index = previous
                return f"mynn_minloc({array}.size(), [&] (int mynn_element) {{ return {value}; }})", "int"
            var = self.variables.get(name)
            if var is not None and var.dimensions:
                indices, ranges = [], False
                for dim, arg in enumerate(arguments):
                    if arg[0] == "range":
                        ranges = True
                        lo = self.emit(arg[1])[0] if arg[1] is not None else f"{name}.lo[{dim}]"
                        hi = self.emit(arg[2])[0] if arg[2] is not None else f"{name}.hi[{dim}]"
                        indices.append(f"MynnRange{{{lo}, {hi}}}")
                    else:
                        indices.append(self.emit(arg)[0])
                code = name + (".slice(" if ranges else "(") + ", ".join(indices) + ")"
                if ranges and self.scalar_index is not None:
                    code += f".linear({self.scalar_index})"
                return code, var.kind
            args = [self.emit(arg) for arg in arguments if not (arg[0] == "name" and arg[1] == "kind_phys")]
            if name == "present":
                return f"({args[0][0]}.present())", "bool"
            if name in ("real", "int", "dble"):
                kind = {"real": "float", "int": "int", "dble": "double"}[name]
                return self.convert(args[0][0], args[0][1], kind), kind
            if name in ("maxval", "minval"):
                return f"mynn_{name}({args[0][0]})", args[0][1]
            if name in ("ceiling", "floor", "nint"):
                return f"mynn_{name}({args[0][0]})", "int"
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
                    previous_kind = kind
                    kind = promote(kind, vk)
                    result = f"mynn_{name}({self.convert(result, previous_kind, kind)}, {self.convert(value, vk, kind)})"
                return result, kind
            if name in ("abs", "sqrt", "exp", "log", "log10", "atan", "tanh", "erf", "sign", "mod"):
                return f"mynn_{name}(" + ", ".join(x[0] for x in args) + ")", args[0][1]
            raise ValueError(("unknown function", name, tree))
        raise ValueError(tree)

    def is_array(self, tree):
        if tree[0] == "name":
            return bool(self.variables[tree[1]].dimensions)
        return tree[0] == "call" and tree[1] in self.variables and any(arg[0] == "range" for arg in tree[2])

    def assignment(self, lhs, rhs):
        tree = Expression.tree(lhs)
        if self.is_array(tree):
            view = self.emit(tree)[0]
            self.scalar_index = "mynn_element"
            a, ak = self.emit(tree)
            b, bk = self.expression(rhs)
            if (bk, ak) in (("float", "double"), ("double", "float")):
                b = self.convert(b, bk, ak)
            self.scalar_index = None
            return f"for (int mynn_element=0; mynn_element < ({view}).size(); ++mynn_element) {{ {a} = {b}; }}"
        a, ak = self.expression(lhs)
        b, bk = self.expression(rhs)
        if (bk, ak) in (("float", "double"), ("double", "float")):
            b = self.convert(b, bk, ak)
        return a + " = " + b + ";"

    def statement(self, stmt):
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
            return "mynn_label_" + match[1] + ": " + self.statement(match[2])
        if match := re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", stmt):
            return "} else if (" + self.expression(match[1])[0] + ") {"
        if stmt in ("else",):
            return "} else {"
        if stmt in ("endif", "end if", "enddo", "end do"):
            return "}"
        if stmt.startswith("if"):
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
        if match := re.match(r"do\s+(\w+)\s*=\s*(.*)$", stmt):
            var, bounds = match[1], split_top(match[2])
            start, end = (self.expression(x)[0] for x in bounds[:2])
            if self.routine.name == "bl_mynn_run" and var == "i" and bounds[:2] == ["its", "ite"]:
                start = "(active_column >= 0 ? active_column : its)"
                end = "(active_column >= 0 ? active_column : ite)"
            step = self.expression(bounds[2])[0] if len(bounds) == 3 else "1"
            return f"for ({var}={start}; ({step}) > 0 ? {var} <= ({end}) : {var} >= ({end}); {var} += ({step})) {{"
        if match := re.match(r"call\s+(\w+)\s*\((.*)\)$", stmt):
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
            return "goto mynn_label_" + match[1] + ";"
        if stmt.startswith(("print", "write")):
            return "; // Native diagnostic print; no field arithmetic."
        if match := re.match(r"(.+?)\s*=\s*(.*)$", stmt):
            rhs = re.sub(r",\s*kind\s*=\s*kind_phys(?=\))", ",kind_phys", match[2])
            return self.assignment(match[1], rhs)
        raise ValueError((self.routine.name, stmt))

    def variable(self, var, argument=False):
        if var.dimensions:
            return f"MynnArray<{var.kind}, {len(var.dimensions)}> {var.name}"
        kind = "MynnText" if var.kind == "char" else var.kind
        if var.optional:
            kind = f"MynnOptional<{kind}>"
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
        output = [f"// MPAS physics_mmm/{routine.source}:{routine.start}-{routine.end}", f"__device__ __noinline__ {result} f_{routine.name}({args}) {{", "MynnFrame mynn_frame(arena);"]
        for name in routine.arguments:
            var = routine.variables[name]
            if var.dimensions:
                lo, _ = self.bounds(var)
                output.append(f"{name}.rebase({lo});")
        for name, var in routine.variables.items():
            if name in routine.arguments:
                continue
            if var.dimensions:
                lo, hi = self.bounds(var)
                output.append(self.variable(var) + f" = mynn_array<{var.kind}, {len(var.dimensions)}>(arena, {lo}, {hi});")
            else:
                self.constant = var.parameter
                initial = " = " + self.expression(var.initial)[0] if var.initial is not None else ""
                self.constant = False
                output.append(("constexpr " if var.parameter else "") + self.variable(var) + initial + ";")
        for line, stmt in routine.body:
            try:
                code = self.statement(stmt)
                # The uninstrumented strict GNU module is the referee. Its
                # MINSS/MAXSS source operand is returned for unordered/equal
                # inputs; register allocation reverses these specific pairs.
                # The two turbulence CFG paths lower line1854 differently.
                # Evidence: native-full-assembly-v1, lines45398/47178/49098
                # and37530-37715, never inferred from instrumented traces.
                if routine.source == "bl_mynn_subroutines.F90":
                    if routine.name == "mym_level2" and line == 751:
                        # Strict GNU unrolls this loop by two. The second
                        # remainder iteration returns DUZ on unordered MAX;
                        # all other iterations return the constant operand.
                        # Pristine executable: level2-max-order-native-v14,
                        # MAXSS at 1a10/1c37/1ea0 versus 2113.
                        assert stmt == "ri = -gh(k)/max( duz, 1.0e-10 )"
                        reversed_code = self.statement("ri = -gh(k)/max(1.0e-10, duz)")
                        code = "if (k == kte && (kte-kts)%2 == 0) { " + reversed_code + " } else { " + code + " }"
                    elif routine.name == "mym_level2" and line == 755:
                        # Both unrolled iterations and the second remainder
                        # iteration preserve RI's NaN. The first remainder
                        # iteration instead returns zero. This is measured
                        # strict executable CFG behaviour, not a NaN filter.
                        # MAXSS: 1a72/1c9d/214e versus 1eff in the same receipt.
                        assert stmt == "a2fac = 1./(1. + max(ri,0.0))"
                        reversed_code = self.statement("a2fac = 1./(1. + max(0.0,ri))")
                        code = "if (k == kts + 2*((kte-kts-1)/2) + 1) { " + code + " } else { " + reversed_code + " }"
                    elif routine.name == "mym_turbulence" and line == 1853:
                        assert stmt == "shb   = max(sh(k), 0.002)"
                        code = self.statement("shb = max(0.002, sh(k))")
                    elif routine.name == "mym_turbulence" and line == 1854:
                        assert stmt == "sm(k) = min(sm(k), prlimit*shb)"
                        reversed_code = self.statement("sm(k) = min(prlimit*shb, sm(k))")
                        code = "if (q3sq < q2sq) { " + code + " } else { " + reversed_code + " }"
                    elif routine.name == "mym_predict" and line in (2409, 2457):
                        # The unchanged GNU loop keeps EL(k) in the destination
                        # register and adds EL(k+1): ADDSS 1bd10/1bf50 in the
                        # finite-edge executable. Preserve that operand order.
                        assert stmt == "b2l = b2*0.5*( el(k+1)+el(k) )"
                        code = self.statement("b2l = b2*0.5*(el(k)+el(k+1))")
                    elif routine.name == "mym_predict" and line in (2411, 2459):
                        # GNU's rolling load puts PDT/PDC(k) in ADDSS's
                        # destination and the next level in its source:
                        # unchanged instructions 1bd2f/1bf6f. This matters when
                        # both inputs are NaNs with different signs/payloads.
                        field = "pdt" if line == 2411 else "pdc"
                        assert stmt == "rp(k) = " + field + "(k+1) + " + field + "(k)"
                        code = self.statement("rp(k) = " + field + "(k) + " + field + "(k+1)")
                    elif routine.name == "mynn_tendencies" and line in (3530, 3532, 3536, 3538):
                        # The original strict GNU executable negates the
                        # completed product, after both MULSS instructions.
                        # Surface coefficients are reused from lines3277/3279;
                        # interior coefficients are computed at3536/3538.
                        # finite-edge-condensation-v15's original source map
                        # records MULSS/MULSS/XORPS in both paths. Moving the
                        # sign flip across an invalid multiplication changes
                        # the NaN sign, despite equal finite results.
                        factor = "khdz(k)" if line in (3530, 3536) else "khdz(k+1)"
                        assert stmt.replace(" ", "") == ("a(k)" if line in (3530, 3536) else "c(k)") + "=-dtz(k)*" + factor + "*rhoinv(k)"
                        lhs = "a(k)" if line in (3530, 3536) else "c(k)"
                        code = self.statement(lhs + " = -(dtz(k)*" + factor + "*rhoinv(k))")
                    elif routine.name == "mym_condensation" and line == 2739:
                        assert "max(sh(k),0.03)" in stmt
                        code = self.statement(stmt.replace("max(sh(k),0.03)", "max(0.03,sh(k))"))
                    elif routine.name == "mym_condensation" and line == 2788:
                        # All four unchanged GNU paths use RHMAX as MINSS's
                        # source operand: 16650/17697/19507/19e3b in the
                        # original species executable. The inner QSAT MAX
                        # retains its original operand order.
                        assert stmt == "rh(k)  = max(min(rhmax, qw(k)/max(1.e-10,qsat_tk)),0.001_kind_phys)"
                        code = self.statement("rh(k) = max(min(qw(k)/max(1.e-10,qsat_tk),rhmax),0.001_kind_phys)")
                    elif routine.name == "mym_condensation" and line == 2813:
                        # Original MINSS 167ad/17812/19682/19f93 returns the
                        # preceding SQRT result for unordered/tied operands.
                        assert stmt == "sgm(k) = min( sgm(k), qsat_tk*0.666 )"
                        code = self.statement("sgm(k) = min(qsat_tk*0.666, sgm(k))")
                    elif routine.name == "mym_condensation" and line == 2823:
                        # Original MAXSS 1682d/17894/196f7/1a018 returns the
                        # inflated previous SGM for unordered/tied operands.
                        assert stmt == "sgm(k) = max( sgm(k), qsat_tk*qpct )"
                        code = self.statement("sgm(k) = max(qsat_tk*qpct, sgm(k))")
                    elif routine.name == "mym_condensation" and line == 2856:
                        # Both original GNU CFG paths put constant one in
                        # MINSS's source operand, then constant zero in MAXSS's
                        # source operand. Preserve their unordered/tie results.
                        # Unchanged executable: finite-edge-condensation-v15,
                        # 17378/1737c and 191c3/191c7; constant load 16d7d.
                        assert stmt == "cldfra_bl1d(k) = max(0., min(1., 0.5+0.36*atan(1.8*(q1(k)+0.2))))"
                        code = self.statement("cldfra_bl1d(k) = max(min(0.5+0.36*atan(1.8*(q1(k)+0.2)),1.),0.)")
                    elif routine.name == "mym_condensation" and line in (2866, 2869):
                        # Unchanged strict GNU paths put the calculated liquid
                        # water in MINSS's source operand, preserving it on
                        # unordered/tied comparisons with MAXQC. Supersaturated:
                        # 173b5/1a6e9; slightly saturated: 18c3a/19225.
                        # Source/register mapping: finite-edge-condensation-v15.
                        product = ("sgm(k)*q1k" if line == 2866 else
                                   "sgm(k)*(exp(-1.) + 0.66*q1k + 0.086*q1k**2)")
                        assert stmt == "ql_water = min(" + product + ", maxqc)"
                        code = self.statement("ql_water = min(maxqc, " + product + ")")
                    elif routine.name == "ddmf_jpl" and line == 6079:
                        # The pristine strict executable computes POWF into
                        # xmm13 and uses zero as MAXSS's second operand. A
                        # negative surface buoyancy flux therefore gives zero
                        # after the NaN intermediate, as observed in complete
                        # radiative-cloud bank v2. Evidence: ddmf-max-order-v13.
                        assert stmt == "wstar   = max(0.,(grav/thv(1)*wthv*pblh)**(onethird))"
                        code = self.statement("wstar = max((grav/thv(1)*wthv*pblh)**(onethird),0.)")
            except Exception as error:
                raise ValueError(f"{routine.source}:{line}: {stmt}") from error
            output.append(f"// {routine.source}:{line}\n{code}")
        if routine.function:
            output.append(f"return {routine.name};")
        output.append("}")
        return "\n".join(output)


def generate(paths, out, receipt):
    globals_, routines = parse_modules(paths)
    emitter = Emitter(globals_, routines)
    pieces = ["// Generated from pinned MPAS MMM; see translation receipt.", "struct MynnColumn {", "MynnArena& arena;", "int active_column = -1;"]
    for var in globals_.values():
        emitter.constant = True
        initial = " = " + emitter.expression(var.initial)[0] if var.initial is not None else ""
        pieces.append(("static constexpr " if var.parameter else "") + emitter.variable(var) + initial + ";")
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


def splice(module, text, routines):
    run = routines["bl_mynn_run"]
    schema = {name: {"kind": var.kind, "dimensions": var.dimensions, "intent": var.intent, "optional": var.optional}
              for name in run.arguments for var in (run.variables[name],)}
    init = routines["bl_mynn_init"]
    parameters = ["const int batch_first", "const int batch_count", "const float* constants", "unsigned char* workspace", "const unsigned long long workspace_stride", "int* error_flags"]
    for name in run.arguments:
        var = run.variables[name]
        if var.kind == "char" or name == "errflg":
            continue
        kind = "int" if var.kind == "bool" else var.kind
        parameters.append(kind + ("* " if var.dimensions else " ") + name)
    wrapper = '// MPAS physics_mmm/bl_mynn.F90:24-107 and 129-1242, one original horizontal column per CUDA thread.\nextern "C" __global__ void mynn_mmm_run(' + ", ".join(parameters) + ") {\n"
    wrapper += "const int worker = blockIdx.x*blockDim.x+threadIdx.x;\nfor (int thread = worker; thread < batch_count; thread += blockDim.x*gridDim.x) {\nMynnArena arena{workspace+workspace_stride*(worker & ~31),0,32,worker & 31};\nMynnColumn column{arena};\ncolumn.active_column = its + batch_first + thread;\nMynnText errmsg;\nint errflg;\n"
    wrapper += "column.f_bl_mynn_init(" + ", ".join(f"constants[{i}]" for i in range(18)) + ",errmsg,errflg);\n"
    values = []
    for name in run.arguments:
        var = run.variables[name]
        if var.kind == "char":
            values.append("errmsg")
        elif var.dimensions:
            rank = len(var.dimensions)
            wrapper += f"MynnArray<{var.kind},{rank}> a_{name}; a_{name}.data={name};\n"
            wrapper += f"a_{name}.lo[0]=1; a_{name}.hi[0]=ite-its+1; a_{name}.stride[0]=1;\n"
            if rank == 2:
                upper = "kme" if name == "w" else "kte"
                wrapper += f"a_{name}.lo[1]=1; a_{name}.hi[1]={upper}-kts+1; a_{name}.stride[1]=ite-its+1;\n"
            values.append("a_"+name)
        else:
            values.append(name)
    wrapper += "column.f_bl_mynn_run(" + ", ".join(values) + ");\nerror_flags[batch_first+thread]=errflg;\n}\n}\n"
    marker = "# BEGIN GENERATED MPAS MMM MYNN\n"
    coefficient, fixed, arrays = 0, 0, 0
    extents = {"kts:kte": (1,0), "kts:kte+1": (1,1), "kts:kme": (1,1),
               "kts+1:kte+1": (1,0), "n": (1,0), "nchem": (0,1),
               "ndvel": (0,1), "1:nup": (0,8), "1:ndown": (0,5)}
    for routine in routines.values():
        for var in routine.variables.values():
            if var.name in routine.arguments or not var.dimensions:
                continue
            a,b = 0,1
            for dim in var.dimensions:
                c,d = extents[dim]
                assert not (a and c), "Only one vertical dimension in this pin"
                a,b = a*d+b*c,b*d
            itemsize = 8 if var.kind == "double" else 4
            coefficient += itemsize*a
            fixed += itemsize*b
            arrays += 1
    fixed += 7*arrays  # upper bound on every eight-byte allocator alignment
    source = module.read_text().split(marker, 1)[0]
    module.write_text(source + marker + "MYNN_MODULE_CUDA = r'''\n" + text + wrapper + "'''\n\nMYNN_ARGUMENTS = " + repr(schema) + f"\nMYNN_WORKSPACE_LEVEL_BYTES = {coefficient}\nMYNN_WORKSPACE_FIXED_BYTES = {fixed}\n# END GENERATED MPAS MMM MYNN\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpas-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--splice", type=Path)
    args = parser.parse_args()
    root = args.mpas_root / "src/core_atmosphere/physics/physics_mmm"
    text, routines = generate([root / name for name in ("mynn_shared.F90", "bl_mynn_subroutines.F90", "bl_mynn.F90")], args.out, args.receipt)
    if args.splice:
        splice(args.splice, text, routines)
