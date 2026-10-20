# Untimed NVRTC witness

This is diagnostic instrumentation, not a production startup path. It counts the
CuPy low-level `cupy_backends.cuda.libs.nvrtc.compileProgram` boundary. It leaves
source, options, images and exceptions unchanged. Native code calling NVRTC
outside that Python boundary is not covered.

`sitecustomize.py` installs before the target imports its model. Its directory
is inherited by Python subprocesses, including compiler workers and the bare
`-c` device probe. Forked children reset their counters. The wrapper records
normal exits and `os._exit`; a killed or uninstrumented process invalidates
coverage. Program source and option values are hashed rather than logged.

`strace -ff -e trace=execve` provides a separate executed-Python PID census.
The summarizer rejects missing Python PIDs, missing end events, incomplete
compiler calls and replaced hooks. The census recognizes executable basenames
`python`, `python3`, and versioned forms such as `python3.13`. Other launcher
names or native NVRTC callers require explicit coverage before using a zero
result as whole-process evidence. Do not reuse an output directory. The runner
rejects an existing sitecustomize instead of silently replacing it.

Run with the same interpreter, source paths, model argv and cache directory as
the candidate. The runner prepends only the witness directory to PYTHONPATH.
GPU work must already be inside the canonical admission helper. Example within
that lease (substitute the actual candidate Python arguments after `--`):

```sh
PYTHONPATH=/absolute/candidate/tree/src:/absolute/candidate/engine python -B tree/tools/nvrtc_witness/run.py --output-dir /absolute/owned/receipt --require-zero -- -B /absolute/model-launch.py
```

Read `summary.json` only after the target and its children have exited. Acceptance
requires `check_passed`, `coverage_valid` and `zero_nvrtc_observed` all true, plus
the separate model completion checks. This trace imports the NVRTC binding early
and adds Python and ptrace overhead. `elapsed_seconds` is never a startup or
step-performance measurement. Keep the timed run uninstrumented.

CPU self-check `receipts/run-witness-selfcheck.py` measured five calls across the
parent, multiprocessing spawn, bare `-c`, multiprocessing fork and failed
compilation. A separate zero-call process tree passes. Deliberate `-S` bypass and
hook replacement are rejected. No model or GPU run was made for these checks.
