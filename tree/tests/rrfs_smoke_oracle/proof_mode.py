"""Select the pristine native comparison mode for this lane's forecast tools."""
from pathlib import Path

from hexcore.verify_mpas import (
    VerificationRefusal, command_physics_mode, file_ref, proof_physics_mode,
)
from hexcore.mpasphys.forecast_v842 import split_physics_mode


def exact_forecast_arguments(arguments, *, namelist_mode=None):
    """Make the actual CLI vector explicit without changing pinned inputs.

    Refuse a contradictory mode: running defect-corrected physics against the
    pristine native referee would invalidate the intended parity comparison.
    Malformed or repeated flags are refused by the landed verifier contract.
    """
    arguments = list(arguments)
    mode = command_physics_mode(arguments)
    if mode == 'correct' or namelist_mode not in (None, 'exact'):
        raise VerificationRefusal('r14 native parity requires --physics-mode exact')
    return arguments if mode == 'exact' else [*arguments, '--physics-mode', 'exact']


def require_exact_verifier_deck(deck, base: Path):
    """Check the authenticated deck's effective mode before the verifier runs.

    The verifier itself appends exact when neither its command nor its namelist
    names a mode. Preserve that shared launcher rather than changing a deck or
    implementing another route. Contradictory CLI/namelist modes refuse at the
    actual door; refuse them here before spending a forecast GPU slot.
    """
    namelist = file_ref(deck['inputs']['namelist'], base)
    _, namelist_mode, _ = split_physics_mode(namelist.read_text())
    if namelist_mode not in (None, 'exact'):
        raise VerificationRefusal('r14 native parity requires exact physics mode')
    _, record = proof_physics_mode(deck['hex_command'], namelist)
    if record['mode'] != 'exact':
        raise VerificationRefusal('r14 native parity requires exact physics mode')
    return record
