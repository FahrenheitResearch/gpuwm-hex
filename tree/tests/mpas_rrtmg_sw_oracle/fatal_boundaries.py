"""Run pristine native fatal branches; retain messages rather than outputs."""
import hashlib
from pathlib import Path
import subprocess

CASES = (
    "INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA",
    "ICE RADIUS OUT OF BOUNDS", "ICE RADIUS OUT OF BOUNDS",
    "ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS",
    "ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS",
    "SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS",
    "SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS",
    "LIQUID EFFECTIVE RADIUS OUT OF BOUNDS",
    "LIQUID EFFECTIVE RADIUS OUT OF BOUNDS",
)


def native_witness(directory):
    directory = Path(directory).resolve()
    executable = directory/"oracle_fatal"
    rows = []
    for case, expected in enumerate(CASES):
        run = subprocess.run([str(executable), str(case)], cwd=directory,
                             capture_output=True, text=True)
        message = run.stdout+run.stderr
        assert expected.lower() in message.lower(), (case, expected, message)
        assert "native unexpectedly returned" not in message
        rows.append(dict(case=case, expected=expected, native_exit=run.returncode,
                         native_message=message))
    return dict(authority="Pristine MPAS cldprmc_sw fatal paths",
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                rows=rows, whole_model_proof=False)
