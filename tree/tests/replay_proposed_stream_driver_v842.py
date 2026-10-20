"""Whole private owner-proposal experiment; MPAS mpas_atm_core.F:109-150,589-705.

Apply the committed exact reconciliation to private complete module copies.
The actual CLI, CUDA and native physics still run. Protected source is unchanged;
this explicitly marked experiment cannot qualify production dispatch.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from replay_native_physics_restart_v842 import digest
from replay_hex_restart_forecast_v842 import main as run_six_arms


PREIMAGES = {
    'hexcore.mesh': ('tree/src/hexcore/mesh.py',
        'e6d7ea9cceeb17532ed3b4d10e3336fb6686b62949081e6c880dccddbee11885'),
    'hexcore.mpasphys.runtime_v842': ('tree/src/hexcore/mpasphys/runtime_v842.py',
        '9bf2c00377455647a8c48984db631bafdb3b273f90b796f59df6b983f3aeb562'),
}


def prepare_fixture(checkout, stage):
    """Private complete driver/mesh copies; MPAS mpas_atm_core.F:109-150,589-705;
    framework/mpas_bootstrapping.F:310-331.
    """
    stage.mkdir(parents=True, exist_ok=False)
    patch = Path(__file__).with_name('proposed_stream_driver_v842.patch')
    receipt = dict(production_dispatch=False, process_count=1,
        scope='Private exact owner proposals; actual CLI/CUDA/native physics',
        launcher_sha256=digest(__file__), patch_sha256=digest(patch), modules={})
    for name, (relative, expected) in PREIMAGES.items():
        source = checkout / relative
        actual = digest(source)
        if actual != expected:
            raise ValueError(f'owner proposal preimage changed: {relative}: {actual}')
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        receipt['modules'][name] = dict(source_sha256=actual, relative=relative)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=stage, check=True)
    for name, (relative, _) in PREIMAGES.items():
        receipt['modules'][name]['proposed_sha256'] = digest(stage / relative)
        if digest(checkout / relative) != receipt['modules'][name]['source_sha256']:
            raise ValueError('protected source changed during private proposal staging')
    (stage / 'proposal.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def run_proposed_cli(argv):
    """Run the real door with explicit private proposals; MPAS mpas_atm_core.F:589-705."""
    checkout = Path(__file__).resolve().parents[2]
    output = Path(argv[argv.index('--out')+1]).resolve()
    stage = output.parent / 'private-owner-proposal'
    receipt = prepare_fixture(checkout, stage)
    original = {}
    try:
        for name, (relative, _) in PREIMAGES.items():
            parent_name, member = name.rsplit('.', 1)
            parent = importlib.import_module(parent_name)
            original[name] = (sys.modules.get(name), parent, member,
                              getattr(parent, member, None))
            spec = importlib.util.spec_from_file_location(name, stage / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            setattr(parent, member, module)
            spec.loader.exec_module(module)
        runtime = sys.modules['hexcore.mpasphys.runtime_v842']
        original_run = runtime.run_native_dynamics

        def proposed_run(*arguments, **keywords):
            """Retain proposal scope on the actual forecast; MPAS mpas_atm_core.F:589-705."""
            result = original_run(*arguments, **keywords)
            path = output / 'native-physics-receipt.json'
            model = json.loads(path.read_text())
            model['private_owner_proposal'] = receipt
            model['production_dispatch'] = False
            path.write_text(json.dumps(model, indent=2)+'\n')
            return result

        runtime.run_native_dynamics = proposed_run
        from hexcore import cli
        return cli.main(argv)
    finally:
        for name, (module, parent, member, attribute) in reversed(list(original.items())):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
            if attribute is None:
                delattr(parent, member)
            else:
                setattr(parent, member, attribute)


def main(argv=None):
    """Six complete forecast arms; MPAS mpas_atm_core.F:109-150,589-705."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--cli':
        return run_proposed_cli(argv[1:])
    return run_six_arms([*argv, '--stream-test-binding', '--proposed-driver-binding'])


if __name__ == '__main__':
    raise SystemExit(main())
