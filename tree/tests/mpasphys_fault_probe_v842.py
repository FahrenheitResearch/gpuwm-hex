"""Install the deliberately broken LW switch witness in an owned source copy.

This is a test fixture, never a selectable production scheme. The native LW
entry runs unchanged; CUDA flips exactly one diagnostic's low bit afterwards.
The full forecast must fail at diagnostics/olrtoa, first hourly step 3.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


FAULT_SOURCE = '''"""Intentional switch fault; not a port or scientific implementation."""
def corrupt_lw_diagnostic(call):
    # MPAS driver_radiation_lw.F:804-960; unchanged native driver first.
    owner = call.pools
    owner.session.hooks.invoke('driver_radiation_lw', call)
    # Native LW globals copy out at driver_radiation_lw.F:631-676.
    target = owner.field('diag_physics', 'olrtoa')
    kernel = owner.cp.RawKernel(
        'extern "C" __global__ void inject_lw_fault(unsigned int *words) {'
        'if (blockIdx.x == 0 && threadIdx.x == 0) words[0] ^= 1u; }',
        'inject_lw_fault', options=('--std=c++17', '--fmad=false'))
    kernel((1,), (1,), (target,))
'''


def install(checkout, destination):
    """Prepare a reviewable isolated test tree; no shared file is modified."""
    checkout, destination = checkout.resolve(), destination.resolve()
    if destination.exists() or destination == checkout:
        raise ValueError('fault destination must be a new owned directory')
    destination.mkdir(parents=True)
    subprocess.run(['cp', '-a', '--reflink=auto', str(checkout / 'tree'),
                    str(destination / 'tree')], check=True)
    table = destination / 'tree/src/hexcore/data/profiles/mpas-physics-suites.v1.json'
    baseline = table.read_bytes()
    data = json.loads(baseline)
    data['mpasphys_v842']['slots']['radiation_lw']['cuda_entries']['rrtmg_lw'] = {
        'module': '_fault_slot_v842', 'call': 'corrupt_lw_diagnostic'}
    table.write_text(json.dumps(data, indent=2) + '\n')
    module = destination / 'tree/src/hexcore/mpasphys/_fault_slot_v842.py'
    module.write_text(FAULT_SOURCE)
    return dict(schema='mpas-f4.intentional-switch-fault/v1',
                baseline_table_sha256=hashlib.sha256(baseline).hexdigest(),
                fault_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                fault_module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                expected_field='diagnostics/olrtoa', expected_first_step=3,
                backend_override="config_radt_lw_backend='hex-cuda'",
                fault='CUDA XOR of bit 0 in first committed olrtoa word after native LW')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    args.receipt.write_text(json.dumps(install(args.checkout, args.destination), indent=2)+'\n')
