"""A device command's exit runs threading's exit hooks before atexit, so no pool worker holds it.

THE BREAKAGE (the hl-3 headline run, 2026-10-24): device_process_exit ran atexit alone and then os._exit.
threading's hooks never ran, so concurrent.futures never stopped the kernel plan's spawn compile pool,
and multiprocessing's exit function joined its idle worker: the finished forecast never exited.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_an_idle_pool_worker_does_not_hold_a_device_command_exit(tmp_path):
    script = tmp_path / "door.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time, concurrent.futures, multiprocessing
        sys.path.insert(0, {str(SRC)!r})
        if __name__ == "__main__":
            from hexcore.cli import device_process_exit
            pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
            pool.submit(time.sleep, 0).result(60)   # the worker is up and idle, the pool never shut down
            print("forecast done", flush=True)
            device_process_exit(0)
    """))
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90)
    assert done.returncode == 0, done.stderr
    assert "forecast done" in done.stdout
