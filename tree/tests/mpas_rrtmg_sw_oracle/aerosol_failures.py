"""Native aerosol rejection witnesses; no output words are excluded."""
import hashlib
from pathlib import Path
import subprocess

MESSAGES = (
    "aer_aod550_val must be positive", "Expected gridded total AOD", "AOD@550 must be positive",
    "Expected gridded aerosol Angstrom exponent", "aer_ssa_val must be within", "aer_ssa_val must be within",
    "Expected gridded aerosol single-scattering albedo", "Aerosol single-scattering albedo must be within",
    "Aerosol single-scattering albedo must be within", "aer_asy_val must be withing", "aer_asy_val must be withing",
    "Expected gridded aerosol asymmetry parameter", "Aerosol asymmetry parameter must be within",
    "Aerosol asymmetry parameter must be within", "Expected aer_aod550_opt", "Expected aer_angexp_opt",
    "Expected aer_ssa_opt", "Expected aer_asy_opt",
)


def native_failures(oracle):
    executable = Path(oracle).resolve()/"oracle_aerosol_fatal"
    rows = []
    for case, expected in enumerate(MESSAGES):
        process = subprocess.run([str(executable), str(case)], cwd=executable.parent,
                                 capture_output=True, text=True)
        message = process.stdout + process.stderr
        assert process.returncode != 0 and expected in message, (case, message)
        assert "native unexpectedly returned" not in message
        rows.append(dict(case=case, native_exit=process.returncode, expected=expected, native_message=message))
    return dict(authority="Pristine MPAS aerosol module:70-451 fatal paths",
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                rows=rows, whole_model_proof=False)
