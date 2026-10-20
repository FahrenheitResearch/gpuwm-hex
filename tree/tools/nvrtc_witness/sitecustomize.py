"""Untimed CuPy NVRTC witness. Enabled only by HEX_NVRTC_WITNESS_DIR.

Place this directory first on PYTHONPATH in a fresh Python process. Children
inherit it. No source text or option values are logged. Use the accompanying
summarizer with strace exec receipts to reject missing Python processes.
"""
import os

if os.environ.get("HEX_NVRTC_WITNESS_DIR"):
    import atexit
    import hashlib
    import json
    from pathlib import Path
    import threading
    import time

    _root = Path(os.environ["HEX_NVRTC_WITNESS_DIR"]).resolve()
    _root.mkdir(parents=True, exist_ok=True)
    _lock = threading.RLock()
    _programs = {}
    _calls = 0
    _closed = False
    _session = str(time.time_ns())

    def _digest(value):
        if not isinstance(value, bytes):
            value = str(value).encode("utf-8")
        return hashlib.sha256(value).hexdigest(), len(value)

    def _event(kind, **payload):
        row = dict(event=kind, pid=os.getpid(), ppid=os.getppid(),
                   session=_session, monotonic_ns=time.monotonic_ns(), **payload)
        with _lock:
            with (_root / (str(os.getpid()) + ".jsonl")).open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    _event("starting", executable=os.path.realpath(__import__("sys").executable))
    try:
        from cupy_backends.cuda.libs import nvrtc as _nvrtc
        _original_create = _nvrtc.createProgram
        _original_compile = _nvrtc.compileProgram
        _original_destroy = _nvrtc.destroyProgram

        def _create(source, name, headers, include_names):
            program = _original_create(source, name, headers, include_names)
            source_sha, source_bytes = _digest(source)
            with _lock:
                _programs[int(program)] = dict(source_sha256=source_sha,
                    source_bytes=source_bytes, name_sha256=_digest(name)[0])
            return program

        def _compile(program, options):
            global _calls
            with _lock:
                _calls += 1
                call = _calls
                info = dict(_programs.get(int(program), {}))
            _event("compile_start", call=call, program=int(program),
                   options_sha256=_digest(tuple(options))[0], **info)
            started = time.monotonic_ns()
            try:
                result = _original_compile(program, options)
            except BaseException as error:
                _event("compile_end", call=call, ok=False,
                       exception_type=type(error).__name__,
                       elapsed_ns=time.monotonic_ns() - started)
                raise
            _event("compile_end", call=call, ok=True,
                   elapsed_ns=time.monotonic_ns() - started)
            return result

        def _destroy(program):
            try:
                return _original_destroy(program)
            finally:
                with _lock:
                    _programs.pop(int(program), None)

        _nvrtc.createProgram = _create
        _nvrtc.compileProgram = _compile
        _nvrtc.destroyProgram = _destroy

        def _finish(reason="atexit"):
            global _closed
            if not _closed:
                _closed = True
                _event("finished", calls=_calls, reason=reason,
                       hook_intact=_nvrtc.compileProgram is _compile)

        def _after_fork():
            global _lock, _calls, _closed, _session
            _lock = threading.RLock()
            _calls = 0
            _closed = False
            _session = str(time.time_ns())
            _event("ready", origin="fork", hook_intact=_nvrtc.compileProgram is _compile)

        _original_exit = os._exit
        def _exit(status):
            _finish("os._exit")
            _original_exit(status)
        os._exit = _exit
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_after_fork)
        atexit.register(_finish)
        _event("ready", origin="interpreter", hook_intact=True)
    except BaseException as _error:
        _event("hook_error", exception_type=type(_error).__name__)
