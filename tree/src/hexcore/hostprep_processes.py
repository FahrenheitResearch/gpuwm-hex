"""Row-parallel host preparation across processes, bit-identical to the serial loop.

THE COST THIS REMOVES (measured 2026-10-16, 2x RTX PRO 6000 box, x1.163842,
every cache empty, the kernel pack loaded): ``prep.device_upload`` took 78.5 s
of a 98 s start, and about 71 s of it was ``initialize_deformation_weights_v841``
running its per-cell scalar code.  That code ran on a thread pool of 4, but
each cell is NumPy-scalar and ctypes work under the GIL, so the pool used one
core while the box had 128.  A 30 km mesh has four times the cells.

WHAT THIS DOES.  ``run_rows`` writes the inputs to ``.npy`` files in a private
temporary directory and starts N worker processes of this module.  Each
worker maps the inputs read-only, builds the SAME per-row function the serial
path uses (the kernel named by ``entry``), runs it over one contiguous slice of
the rows, in order, and writes its rows of each output.  The parent scatters
them back.  Every output word is therefore produced by the same code on the
same input bytes with the same C library as the serial loop, so the result is
byte-identical; the only change is which process computed which row.

Errors keep the serial contract: a kernel's ``ValueError`` in a worker is
returned by name, and the parent raises the one from the lowest row slice,
which is the first failing row of the serial loop.

``GPUWM_HEX_HOSTPREP_PROCESSES`` sets the process count (default: every core
this process may run on, at most one per ``MIN_ROWS_PER_PROCESS`` rows);
``1`` keeps the old in-process path.  Concurrent callers (several ranks
constructing at once) take turns on one lock, so the box is never asked for
ranks x cores processes at once.
"""
from __future__ import annotations

import threading

ENVIRONMENT = "GPUWM_HEX_HOSTPREP_PROCESSES"
#: Below this many rows per process the start-up cost (an interpreter plus
#: NumPy, about 0.2 s) outweighs the work.
MIN_ROWS_PER_PROCESS = 2048
#: One parallel job at a time per process; each job already uses every core.
_TURN = threading.Lock()
_VALUE_ERROR_EXIT = 3


def process_count(n_rows: int) -> int:
    """How many worker processes ``n_rows`` rows get (1 means: stay in process)."""
    import os

    declared = os.environ.get(ENVIRONMENT, "").strip()
    if declared:
        try:
            workers = int(declared)
        except ValueError:
            raise ValueError(f"{ENVIRONMENT} must be a positive integer, got {declared!r}") from None
        if workers < 1:
            raise ValueError(f"{ENVIRONMENT} must be a positive integer, got {declared!r}")
    else:
        try:
            workers = len(os.sched_getaffinity(0))
        except AttributeError:
            workers = os.cpu_count() or 1
    return max(1, min(workers, int(n_rows) // MIN_ROWS_PER_PROCESS))


def run_rows(entry: str, inputs: dict, outputs: dict, rows, workers: int) -> None:
    """Fill ``outputs[name][rows]`` with ``entry``'s per-row function, ``workers`` processes wide."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if not sys.executable:
        raise RuntimeError("no interpreter path to start host-prep workers from (sys.executable is empty)")
    package_parent = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = package_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    with _TURN, tempfile.TemporaryDirectory(prefix="hex-hostprep-") as tmp:
        root = Path(tmp)
        for name, value in inputs.items():
            np.save(root / f"in-{name}.npy", np.asarray(value), allow_pickle=False)
        np.save(root / "rows.npy", rows, allow_pickle=False)
        spec = {"entry": entry, "inputs": sorted(inputs),
                "outputs": {name: [np.asarray(value).dtype.str, list(np.shape(value))]
                            for name, value in outputs.items()}}
        (root / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        bounds = [(rows.size * i) // workers for i in range(workers + 1)]
        procs = []
        try:
            for index in range(workers):
                with open(root / f"err-{index}.txt", "wb") as err:
                    procs.append(subprocess.Popen(
                        [sys.executable, "-m", "hexcore.hostprep_processes", str(root), str(index),
                         str(bounds[index]), str(bounds[index + 1])],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, env=env))
            codes = [proc.wait() for proc in procs]
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        for index, code in enumerate(codes):
            message = (root / f"err-{index}.txt").read_text(encoding="utf-8", errors="replace").strip()
            if code == _VALUE_ERROR_EXIT:
                raise ValueError(message.splitlines()[-1] if message else "host-prep row refused")
            if code != 0:
                tail = "\n".join(message.splitlines()[-20:])
                raise RuntimeError(f"host-prep worker {index} of {workers} ({entry}) exited {code}:\n{tail}")
        for index in range(workers):
            mine = rows[bounds[index]:bounds[index + 1]]
            for name, value in outputs.items():
                value[mine] = np.load(root / f"out-{index}-{name}.npy", allow_pickle=False)


def _worker(root: str, index: int, lo: int, hi: int) -> int:
    import importlib
    import json
    import sys
    from pathlib import Path

    import numpy as np

    base = Path(root)
    spec = json.loads((base / "spec.json").read_text(encoding="utf-8"))
    inputs = {}
    for name in spec["inputs"]:
        value = np.load(base / f"in-{name}.npy", mmap_mode="r", allow_pickle=False)
        inputs[name] = np.asarray(value) if value.ndim else np.asarray(value)[()]
    outputs = {name: np.zeros(tuple(shape), dtype=np.dtype(dtype))
               for name, (dtype, shape) in spec["outputs"].items()}
    module, function = spec["entry"].split(":")
    one_row = getattr(importlib.import_module(module), function)(inputs, outputs)
    rows = np.asarray(np.load(base / "rows.npy", mmap_mode="r", allow_pickle=False)[lo:hi])
    try:
        for row in rows:
            one_row(int(row))
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return _VALUE_ERROR_EXIT
    for name, value in outputs.items():
        np.save(base / f"out-{index}-{name}.npy", value[rows], allow_pickle=False)
    return 0


if __name__ == "__main__":
    import sys

    root_arg, index_arg, lo_arg, hi_arg = sys.argv[1:5]
    sys.exit(_worker(root_arg, int(index_arg), int(lo_arg), int(hi_arg)))
