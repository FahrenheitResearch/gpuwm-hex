"""hexcore.file_lock: the doors' locks work without fcntl (Windows) and exclude a second holder."""
import subprocess
import sys

from hexcore import file_lock


def test_a_held_lock_refuses_a_second_non_blocking_holder(tmp_path):
    path = tmp_path / "locks" / "door.lock"
    probe = ("import sys\nfrom hexcore import file_lock\n"
             "with open(sys.argv[1], 'a') as s:\n"
             "    try:\n        file_lock.lock(s, blocking=False)\n    except BlockingIOError:\n        sys.exit(3)\n")
    with file_lock.held(path):
        held = subprocess.run([sys.executable, "-c", probe, str(path)])
    free = subprocess.run([sys.executable, "-c", probe, str(path)])
    assert (held.returncode, free.returncode) == (3, 0)
