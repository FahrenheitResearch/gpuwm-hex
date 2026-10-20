"""In-process wall-clock stack sampler for a hex forecast (no ptrace, no privileges).

Usage: python hex_stack_sampler.py OUT.jsonl.gz SCRIPT|-m MODULE [ARGS...]

Starts one daemon thread that reads sys._current_frames() every
HEX_SAMPLER_INTERVAL seconds (default 0.25), then runs SCRIPT (or MODULE) as
__main__.  Output rows (gzip JSON lines, flushed every HEX_SAMPLER_FLUSH
seconds so a running forecast can be read mid-run):
  {"d": [stack_id, "file:func:line;..."]}   first sighting of a stack
  {"t": monotonic, "u": epoch, "s": [[thread_name, stack_id], ...]}
Threads parked in a wait (queue get, lock acquire, Event.wait, select) are
recorded as IDLE with their innermost three frames.  t is time.monotonic(),
the clock the time ledger writes as t_mono.

Why it exists: forecast.construct took 418-796 s and each history capture
about 27 s on the step path, and the timeline has no marks inside them.  py-spy
needs ptrace, which the rented containers do not grant.
"""
import gzip
import json
import os
import runpy
import sys
import threading
import time

IDLE_FUNCS = {"wait", "get", "acquire", "select", "poll", "_wait_for_tstate_lock", "join", "sleep",
              "accept", "recv", "read", "readline", "epoll", "communicate", "_try_wait", "_communicate"}


def _stack(frame, limit=60):
    out = []
    while frame is not None and len(out) < limit:
        code = frame.f_code
        out.append(f"{os.path.basename(code.co_filename)}:{code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    out.reverse()
    return out


def main():
    out_path, script, *args = sys.argv[1:]
    interval = float(os.environ.get("HEX_SAMPLER_INTERVAL", "0.25"))
    flush_every = float(os.environ.get("HEX_SAMPLER_FLUSH", "20"))
    ids = {}
    stop = threading.Event()
    stream = gzip.open(out_path, "wt", compresslevel=3)
    lock = threading.Lock()
    sampler_ident = [None]

    def sample():
        sampler_ident[0] = threading.get_ident()
        names = {}
        last_flush = time.monotonic()
        while not stop.wait(interval):
            for thread in threading.enumerate():
                names[thread.ident] = thread.name
            row, defs = [], []
            for ident, frame in sys._current_frames().items():
                if ident == sampler_ident[0]:
                    continue
                stack = _stack(frame)
                if stack and stack[-1].split(":")[1] in IDLE_FUNCS:
                    key = "IDLE;" + ";".join(stack[-3:])
                else:
                    key = ";".join(stack)
                sid = ids.get(key)
                if sid is None:
                    sid = ids[key] = len(ids)
                    defs.append([sid, key])
                row.append([names.get(ident, str(ident)), sid])
            now = time.monotonic()
            with lock:
                for d in defs:
                    stream.write(json.dumps({"d": d}) + "\n")
                stream.write(json.dumps({"t": now, "u": time.time(), "s": row}) + "\n")
                if now - last_flush > flush_every:
                    stream.flush()
                    last_flush = now

    thread = threading.Thread(target=sample, name="hex-stack-sampler", daemon=True)
    thread.start()
    code = 0
    try:
        if script == "-m":
            sys.argv = [args[0], *args[1:]]
            runpy.run_module(args[0], run_name="__main__", alter_sys=True)
        else:
            sys.argv = [script, *args]
            sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
            runpy.run_path(script, run_name="__main__")
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    finally:
        stop.set()
        thread.join(timeout=5)
        with lock:
            stream.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
