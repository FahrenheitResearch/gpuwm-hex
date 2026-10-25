"""Persist exact compiler results, including CuPy's empty preprocessing probes.

The ordinary CuPy disk cache does not cache its NVRTC preprocessing probe, and
in-memory physics compilers do not necessarily write CuPy's cache at all. This
layer serves those exact requests from binary artifacts. It changes no source,
options, launch operands or numerical defaults. The compiler family/build, run
configuration, architecture, source, headers and final options bind each entry.
Cold writes run beside compilation on an IO pool; warm reads address one entry,
never scan other plans. Unreadable entries fall back to the original compiler.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import warnings

SCHEMA = "gpuwm-hex.kernel-warm/v2"
DIRECTORY_ENVIRONMENT = "GPUWM_HEX_WARM_PACK_DIR"
_MAGIC = b"HEXWARM2"
_MAX_HEADER = 1024 * 1024
_STATE = None
_LOCK = threading.RLock()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _copy(value):
    if isinstance(value, tuple):
        return value[0], None if value[1] is None else dict(value[1])
    return value


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".warm-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _include_metadata(source):
    """Read object-like definitions without evaluating conditional branches.

    Every definition is retained, so conditional alternatives over-bind rather
    than omit a possible file input. Function-like computed include names keep
    ordinary compilation until their dependency closure can be bound.
    """
    source = re.sub(r"\\\r?\n", "", source)
    tokens = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*'
    source = re.sub(tokens, lambda m: m[0] if m[0][0] in "\"'" else
                    " " + "\n" * m[0].count("\n"), source, flags=re.S)
    macros = {}
    for name, rest in re.findall(r'^[^\S\n]*#[^\S\n]*define[^\S\n]+([A-Za-z_]\w*)([^\n]*)', source, re.M):
        macros.setdefault(name, set()).add(None if rest.startswith("(") else rest.strip())
    includes = re.findall(r'^[^\S\n]*#[^\S\n]*include[^\S\n]+([^\n]+)', source, re.M)
    return macros, [value.strip() for value in includes]


def _expanded_headers(operand, macros):
    """Conservatively expand finite object-like header alternatives."""
    if operand.startswith(('"', '<')):
        values = {operand}
    else:
        values = {operand}
        token_pattern = r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|.'
        for _ in range(32):
            expanded = set()
            for value in values:
                pieces = {""}
                for token in re.findall(token_pattern, value):
                    replacements = macros.get(token, {token})
                    if None in replacements:
                        raise ValueError("function-like computed include cannot be bound")
                    pieces = {left + right for left in pieces for right in replacements}
                    if len(pieces) > 256:
                        raise ValueError("computed include has too many alternatives")
                expanded.update(pieces)
            if expanded == values:
                break
            values = expanded
            if len(values) > 256:
                raise ValueError("computed include has too many alternatives")
        else:
            raise ValueError("recursive computed include cannot be bound")
    names = []
    for value in values:
        match = re.fullmatch(r'"([^"\\\n]+)"|<([^>\n]+)>', value.strip())
        if match is None:
            raise ValueError("unresolved computed include cannot be bound")
        names.append(match[1] or match[2])
    return names


class Store:
    def __init__(self, root, identity, tag, workers=None):
        self.identity = identity
        self.tag = str(tag)
        self.root = Path(root)
        # Content-addressed: an artifact's key digests its whole request (source, options, arch, filename and
        # the bound header bytes) and get() refuses any file whose header does not name that key, so one
        # directory per compiler identity serves every run.  THE BREAKAGE THIS PREVENTS: the directory was also
        # keyed by the run's argv (the plan tag), so every new storm-following member (a new --mesh/--grid)
        # missed every artifact and recompiled the MPAS suite's fast-mode kernels inline, 15 s at the start of
        # each moving segment on an RTX 4090 host (MPAS-ROUTE-SPEED-2026-10-24).
        self.directory = (self.root / ("cuda" + str(identity["nvrtc"][0])) /
                          _digest(identity) / "content")
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        self.workers = workers or cores
        self.pool = concurrent.futures.ThreadPoolExecutor(self.workers, thread_name_prefix="hex-warm-io")
        self.lock = threading.RLock()
        self.pending = {}
        self.headers = {}
        self.bindings = {}
        self.warned_bypasses = set()
        self.counts = dict(hits=0, misses=0, invalid=0, writes=0, write_failures=0,
                           bypasses=0, lookup_seconds=0.0)

    def _includes(self, options, source):
        """Bind explicit include trees once per process, with parallel byte hashing.

        Existing NVRTC caches assume installed compiler/header inputs stay fixed
        while a process runs. We make the same assumption, but rehash headers on
        every fresh process, including edited files with unchanged timestamps.
        """
        roots = []
        source_macros, source_includes = _include_metadata(source)
        iterator = iter(options)
        for option in iterator:
            option = str(option)
            if option in ("-D", "--define-macro"):
                definition = str(next(iterator))
                name, _, value = definition.partition("=")
                source_macros.setdefault(name, set()).add(value or "1")
            elif option.startswith(("-D", "--define-macro=")):
                definition = option[2:] if option.startswith("-D") else option.split("=", 1)[1]
                name, _, value = definition.partition("=")
                source_macros.setdefault(name, set()).add(value or "1")
            elif option in ("-I", "--include-path"):
                roots.append(str(next(iterator)))
            elif option.startswith("-I"):
                roots.append(option[2:])
            elif option.startswith("--include-path="):
                roots.append(option.split("=", 1)[1])
            elif option.startswith(("--pre-include", "-include", "-pch", "--pch")):
                # These additional file inputs must keep the compiler path until
                # their dependency closure can be bound explicitly.
                raise ValueError("external preinclude/PCH input")
        # Bind source-local literal or object-macro headers before any artifact
        # lookup. The explicit trees supply definitions used by nested headers.
        for operand in source_includes:
            try:
                names = _expanded_headers(operand, source_macros)
            except ValueError:
                continue  # Retry once explicit header definitions are known.
            for name in names:
                path = Path(name)
                if path.is_file():
                    roots.append(str(path.resolve().parent))
        # Different generated kernels commonly have the same include directives
        # and definitions. Installed headers are already fixed for this Store;
        # reuse their validated closure, still binding every request's roots.
        # The full source remains in request(), so executable keys do not change.
        binding = _digest({"roots": roots, "cwd": str(Path.cwd()),
                           "macros": {name: sorted(values, key=str)
                                      for name, values in source_macros.items()},
                           "includes": source_includes})
        with self.lock:
            future = self.bindings.get(binding)
            owner = future is None
            if owner:
                if len(self.bindings) >= 1024:
                    self.bindings.clear()
                future = concurrent.futures.Future()
                self.bindings[binding] = future
        if owner:
            try:
                future.set_result(self._bind_includes(roots, source_macros, source_includes))
            except BaseException as error:
                future.set_exception(error)
                with self.lock:
                    if self.bindings.get(binding) is future:
                        del self.bindings[binding]
                raise
        return list(future.result())

    def _bind_includes(self, roots, source_macros, source_includes):
        macros = {name: set(values) for name, values in source_macros.items()}
        indirect = []
        result = []
        bounded_paths = tuple(Path(root).resolve() for root in dict.fromkeys(roots))
        for root in dict.fromkeys(roots):
            path = Path(root).resolve()
            token = str(path)
            with self.lock:
                future = self.headers.get(token)
                owner = future is None
                if owner:
                    future = concurrent.futures.Future()
                    self.headers[token] = future
            if owner:
                try:
                    files = []
                    def failed(error):
                        raise error
                    for base, dirs, names in os.walk(path, onerror=failed):
                        if any((Path(base) / name).is_symlink() for name in dirs):
                            raise ValueError("include tree contains a directory symlink")
                        files.extend(Path(base) / name for name in names
                                     if (Path(base) / name).is_file())
                    files.sort()
                    def hashed(file):
                        content = file.read_bytes()
                        definitions, includes = _include_metadata(content.decode("utf-8", "replace"))
                        computed = []
                        for operand in includes:
                            if not operand.startswith(('"', '<')):
                                computed.append((str(file.parent), operand))
                                continue
                            for name in _expanded_headers(operand, {}):
                                include = Path(name)
                                if include.is_absolute() or ".." in include.parts:
                                    # Root metadata is shared by different option sets.
                                    # Revalidate these operands against each request's
                                    # bound roots, not only the first request's roots.
                                    computed.append((str(file.parent), operand))
                        return ((str(file.relative_to(path)), hashlib.sha256(content).hexdigest()),
                                definitions, computed)
                    rows = list(self.pool.map(hashed, files))
                    future.set_result((_digest([row[0] for row in rows]),
                                       [row[1] for row in rows],
                                       [entry for row in rows for entry in row[2]]))
                except BaseException as error:
                    future.set_exception(error)
                    raise
            digest, definitions, computed = future.result()
            result.append((token, digest))
            indirect.extend(computed)
            for definition in definitions:
                for name, values in definition.items():
                    macros.setdefault(name, set()).update(values)
        # Computed header operands can otherwise escape every hashed tree.
        # No compiler option is changed; unsupported cases retain compilation.
        checks = [(str(Path.cwd()), operand) for operand in source_includes]
        checks.extend(indirect)
        for parent, operand in checks:
            for name in _expanded_headers(operand, macros):
                include = Path(name)
                if include.is_absolute() or ".." in include.parts:
                    target = (Path(parent) / include).resolve()
                    if not any(target.is_relative_to(base) for base in bounded_paths):
                        raise ValueError("computed include escapes the bound header trees")
                else:
                    local = Path(parent) / include
                    if local.is_file() and not any(local.resolve().is_relative_to(base)
                                                   for base in bounded_paths):
                        raise ValueError("computed local include is outside bound header trees")
        return result

    def request(self, route, arguments):
        # All words remain in order: duplicate compiler flags can affect the
        # final value, and names can occur in __FILE__ and diagnostic metadata.
        bound = dict(arguments)
        options = tuple(bound.get("options") or ())
        bound["headers_digest"] = self._includes(options, bound.get("source", ""))
        return _digest({"schema": SCHEMA, "route": route, "arguments": bound})

    def path(self, key):
        return self.directory / key[:2] / (key + ".bin")

    def get(self, key):
        started = time.perf_counter()
        with self.lock:
            pending = self.pending.get(key)
        if pending is not None:
            with self.lock:
                self.counts["hits"] += 1
            return _copy(pending[0])
        try:
            data = self.path(key).read_bytes()
            if data[:8] != _MAGIC:
                raise ValueError("artifact magic")
            size = struct.unpack("<I", data[8:12])[0]
            if size > _MAX_HEADER or len(data) < 12 + size:
                raise ValueError("artifact header")
            header = json.loads(data[12:12 + size])
            if not isinstance(header, dict):
                raise ValueError("artifact header type")
            payload = data[12 + size:]
            # Bind the mapping and type too, not only executable bytes.
            proof = header.pop("sha256")
            if (header.get("schema") != SCHEMA or header.get("request") != key or
                    hashlib.sha256(_json(header) + payload).hexdigest() != proof):
                raise ValueError("artifact digest")
            if header["kind"] == "text":
                value = payload.decode("utf-8")
            elif header["kind"] == "image":
                mapping = header["mapping"]
                if mapping is not None and (not isinstance(mapping, dict) or
                        not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())):
                    raise ValueError("artifact mapping")
                value = payload, mapping
            else:
                raise ValueError("artifact kind")
            with self.lock:
                self.counts["hits"] += 1
            return value
        except FileNotFoundError:
            with self.lock:
                self.counts["misses"] += 1
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            with self.lock:
                self.counts["invalid"] += 1
        finally:
            with self.lock:
                self.counts["lookup_seconds"] += time.perf_counter() - started
        return None

    def put(self, key, value):
        value = _copy(value)
        if isinstance(value, str):
            payload, kind, mapping = value.encode("utf-8"), "text", None
        elif isinstance(value, tuple) and isinstance(value[0], bytes):
            payload, mapping = value
            kind = "image"
        else:
            return
        header = {"schema": SCHEMA, "request": key, "kind": kind, "mapping": mapping}
        header["sha256"] = hashlib.sha256(_json(header) + payload).hexdigest()
        encoded = _json(header)
        content = _MAGIC + struct.pack("<I", len(encoded)) + encoded + payload
        def write():
            try:
                _atomic(self.path(key), content)
                with self.lock:
                    self.counts["writes"] += 1
            except OSError as error:
                with self.lock:
                    self.counts["write_failures"] += 1
                warnings.warn(f"hex warm artifact write failed at {self.path(key)}: {error}; "
                              "ordinary compilation retained")
        with self.lock:
            if key not in self.pending:
                self.pending[key] = (value, self.pool.submit(write))

    def call(self, route, arguments, function):
        try:
            key = self.request(route, arguments)
        except (OSError, ValueError, TypeError, StopIteration) as error:
            reason = str(error) or type(error).__name__
            with self.lock:
                self.counts["bypasses"] += 1
                warn = reason not in self.warned_bypasses
                self.warned_bypasses.add(reason)
            if warn:
                warnings.warn(f"hex warm artifact bypass: {reason}; ordinary compilation retained")
            return function()
        value = self.get(key)
        if value is not None and not (route == "compiler-banner" and value == ""):
            return value
        value = function()
        if not (route == "compiler-banner" and value == ""):
            self.put(key, value)
        return value

    def flush(self):
        with self.lock:
            futures = [row[1] for row in self.pending.values()]
        for future in futures:
            future.result()

    def receipt(self):
        with self.lock:
            return {"schema": SCHEMA, "directory": str(self.directory),
                    "compiler": self.identity, "configuration": self.tag,
                    "io_workers": self.workers, **self.counts}


def _root():
    explicit = os.environ.get(DIRECTORY_ENVIRONMENT)
    if explicit:
        return Path(explicit)
    from . import prep_cache
    root = prep_cache.cache_root()
    return None if root is None else root / "kernel-packs"


def install(*, root=None, tag=None, identity=None):
    """Install before the plan funnel, preserving any numerical option wrapper."""
    global _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE
        root = _root() if root is None else root
        if root is None:
            return None
        from cupy.cuda import compiler
        from . import kernel_plan
        from .cuda_backend.aot import compiler_identity, effective_options
        from gpuwm.certify import compile_platform
        identity = compiler_identity() if identity is None else identity
        tag = kernel_plan.default_tag_text() if tag is None else tag
        store = Store(root, identity, tag, workers=kernel_plan._workers())
        original = compiler._compile_using_nvrtc_no_warning
        def compile_cached(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
                           log_stream=None, cache_in_memory=False, jitify=None, method=None):
            options = effective_options(options)
            def call():
                return original(source, options, arch, filename, name_expressions,
                                log_stream, cache_in_memory, jitify, method)
            if log_stream is not None or jitify or not isinstance(source, str):
                return call()
            # CuPy embeds a random temporary program path in these images.
            # Retaining a previous path would change observable __FILE__ data.
            if not cache_in_memory and kernel_plan._name_sensitive(source, options):
                with store.lock:
                    store.counts["bypasses"] += 1
                return call()
            selected = str(arch or compiler._get_arch())
            arguments = dict(source=source, options=list(options), arch=selected, filename=filename,
                             name_expressions=name_expressions, cache_in_memory=bool(cache_in_memory),
                             method=method)
            return store.call("nvrtc", arguments, call)
        compile_cached.__wrapped__ = original
        compiler._compile_using_nvrtc_no_warning = compile_cached
        preprocess = compiler._preprocess
        def preprocess_cached(source, options, arch, backend):
            if backend != "nvrtc":
                return preprocess(source, options, arch, backend)
            options = effective_options(options)
            return store.call("preprocess", dict(source=source, options=list(options), arch=str(arch)),
                              lambda: preprocess(source, options, arch, backend))
        preprocess_cached.__wrapped__ = preprocess
        compiler._preprocess = preprocess_cached
        banner = compile_platform.nvrtc_banner
        def banner_cached():
            return store.call("compiler-banner", {}, banner)
        banner_cached.__wrapped__ = banner
        compile_platform.nvrtc_banner = banner_cached
        _STATE = store
        atexit.register(store.flush)
        return store


def worker_config():
    return None if _STATE is None else dict(root=str(_STATE.root), tag=_STATE.tag, identity=_STATE.identity)


def restore(arguments, arch):
    """Load a plan entry directly, before creating any compiler processes."""
    if _STATE is None:
        return None
    from . import kernel_plan
    from .cuda_backend.aot import effective_options
    options = effective_options(arguments["options"])
    if (not arguments["cache_in_memory"] and
            kernel_plan._name_sensitive(arguments["source"], options)):
        return None
    bound = dict(arguments, arch=str(arch), options=list(options))
    try:
        return _STATE.get(_STATE.request("nvrtc", bound))
    except (OSError, ValueError, TypeError, StopIteration):
        return None


def raw_image(source, options, arch, filename, compile_image):
    """Cache raw NVRTC without adding CuPy's implicit compilation options."""
    store = _STATE
    if store is None:
        try:
            store = install()
        except (ImportError, OSError, ValueError) as error:
            warnings.warn(f"hex raw warm pack unavailable: {error}; ordinary compilation retained")
    if store is None:
        return compile_image()
    arguments = dict(source=source, options=list(options), arch=str(arch), filename=filename)
    image, _ = store.call("raw-nvrtc", arguments, lambda: (compile_image(), None))
    return image


def receipt():
    return None if _STATE is None else _STATE.receipt()
