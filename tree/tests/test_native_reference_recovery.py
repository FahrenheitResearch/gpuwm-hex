"""Cached reference bytes must survive validation; cleaned runs must regenerate."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/native_dry_reference'
spec = importlib.util.spec_from_file_location('native_run_receipt', TOOLS / 'run_receipt.py')
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


@pytest.fixture
def completed(tmp_path):
    for name in ('atmosphere_model', 'namelist.atmosphere', 'streams.atmosphere',
                 'x1.2562.init.nc', 'tracer.0.nc', 'tracer.1.nc'):
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / 'log.atmosphere.0000.out').write_text('Finished running the atmosphere core\n')
    assert receipt.main(['write', str(tmp_path)]) == 0
    return tmp_path


@pytest.mark.parametrize('name', ['atmosphere_model', 'namelist.atmosphere', 'streams.atmosphere',
                                 'x1.2562.init.nc', 'tracer.1.nc'])
def test_changed_reference_bytes_refused(completed, name):
    assert receipt.main(['verify', str(completed), '--expected-frames', '2']) == 0
    (completed / name).write_bytes(b'changed')
    assert receipt.main(['verify', str(completed), '--expected-frames', '2']) == 2


def test_partial_and_unfinished_reference_refused(completed):
    (completed / 'tracer.1.nc').unlink()
    assert receipt.main(['verify', str(completed), '--expected-frames', '2']) == 2
    (completed / 'tracer.1.nc').write_bytes(b'restored')
    (completed / 'log.atmosphere.0000.out').write_text('aborted')
    assert receipt.main(['write', str(completed)]) == 2


@pytest.fixture
def native_source(tmp_path):
    """An actual Git index, with dummy bytes and no compiler or model run."""
    name = 'v8.4.2-gnu-o0-atmosphere'
    source = tmp_path / 'native' / name
    source.mkdir(parents=True)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True)

    git('init', '-q')
    # This disposable fixture is not an engine checkout. Its local identity
    # avoids depending on the test host's global Git configuration.
    git('config', 'user.name', 'Native reference fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    (source / 'Makefile').write_text('FLAGS=-O3\n')
    (source / 'dynamics.F').write_text('original numerics\n')
    git('add', 'Makefile', 'dynamics.F')
    git('commit', '-qm', 'fixture source')
    executable = source / 'atmosphere_model'
    executable.write_bytes(b'not an executable; provenance fixture only')
    (tmp_path / 'native' / (name + '.sha256')).write_text(hashlib.sha256(executable.read_bytes()).hexdigest())
    flags = tmp_path / 'native' / (name + '.makefile.diff')
    flags.write_text('')
    helper = tmp_path / 'helpers'
    helper.mkdir()
    (helper / 'validate_reference.py').write_bytes((TOOLS / 'validate_reference.py').read_bytes())
    (helper / 'REFERENCE-PINS.json').write_text(json.dumps({'upstream': {'v8.4.2': git('rev-parse', 'HEAD').strip()}}))

    def validate():
        return subprocess.run([sys.executable, str(helper / 'validate_reference.py'), str(tmp_path), name],
                              capture_output=True, text=True)

    assert validate().returncode == 0
    return source, git, flags, validate


@pytest.mark.parametrize('staged', [False, True])
def test_modified_native_numerics_refused_even_when_staged(native_source, staged):
    source, git, _, validate = native_source
    (source / 'dynamics.F').write_text('altered numerics\n')
    if staged:
        git('add', 'dynamics.F')
    result = validate()
    assert result.returncode != 0
    assert 'FAIL native source has changes beyond the recorded build flags' in result.stderr


def test_staged_flags_checked_against_full_build_receipt(native_source):
    source, git, flags, validate = native_source
    (source / 'Makefile').write_text('FLAGS=-O0 -ffp-contract=off\n')
    git('add', 'Makefile')
    flags.write_text(git('diff', 'HEAD', '--', 'Makefile'))
    assert validate().returncode == 0
    (source / 'Makefile').write_text('FLAGS=-ffast-math\n')
    git('add', 'Makefile')
    result = validate()
    assert result.returncode != 0
    assert 'FAIL native Makefile differs from its build receipt' in result.stderr


def test_condensate_variant_registry_checked_against_its_build_receipt(tmp_path):
    # The condensate native is a declared variant: its Registry.xml edit is
    # admitted only when it equals the recorded registry diff.
    name = 'v8.4.2-gnu-o0-condensate-atmosphere'
    source = tmp_path / 'native' / name
    (source / 'src/core_atmosphere').mkdir(parents=True)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True)

    git('init', '-q')
    git('config', 'user.name', 'Native reference fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    registry = source / 'src/core_atmosphere/Registry.xml'
    (source / 'Makefile').write_text('FLAGS=-O3\n')
    registry.write_text('<var name="qv"/>\n')
    git('add', 'Makefile', 'src/core_atmosphere/Registry.xml')
    git('commit', '-qm', 'fixture source')
    exe = source / 'atmosphere_model'
    exe.write_bytes(b'fixture')
    native = tmp_path / 'native'
    (native / (name + '.sha256')).write_text(hashlib.sha256(exe.read_bytes()).hexdigest())
    (native / (name + '.makefile.diff')).write_text('')
    helper = tmp_path / 'helpers'
    helper.mkdir()
    (helper / 'validate_reference.py').write_bytes((TOOLS / 'validate_reference.py').read_bytes())
    (helper / 'REFERENCE-PINS.json').write_text(json.dumps({'upstream': {'v8.4.2': git('rev-parse', 'HEAD').strip()}}))

    def validate():
        return subprocess.run([sys.executable, str(helper / 'validate_reference.py'), str(tmp_path), name],
                              capture_output=True, text=True)

    registry.write_text('<var name="qv"/>\n<var name="qc"/>\n')
    (native / (name + '.registry.diff')).write_text(git('diff', 'HEAD', '--', 'src/core_atmosphere/Registry.xml'))
    assert validate().returncode == 0, validate().stderr
    registry.write_text('<var name="qv"/>\n<var name="qr"/>\n')
    result = validate()
    assert result.returncode != 0
    assert 'FAIL native condensate Registry.xml differs from its build receipt' in result.stderr


@pytest.mark.parametrize('staged', [False, True])
def test_native_build_refuses_modified_sources_before_compiling(native_source, staged):
    source, git, _, _ = native_source
    root = source.parents[1]
    # Existing dummy dependencies bypass network setup. Refusal must happen
    # before any compiler or model is invoked.
    (root / 'pnetcdf/lib').mkdir(parents=True)
    (root / 'pnetcdf/lib/libpnetcdf.a').touch()
    (root / 'MPAS-Model/.git').mkdir(parents=True)
    (source / 'dynamics.F').write_text('altered numerics\n')
    if staged:
        git('add', 'dynamics.F')
    result = subprocess.run(['bash', str(TOOLS / 'build_native_reference.sh'), str(root),
                             'v8.4.2', 'gnu-o0', 'atmosphere', '1'], capture_output=True, text=True)
    assert result.returncode == 2
    expected = 'staged changes' if staged else 'changes beyond build flags'
    assert 'FAIL native source has ' + expected in result.stderr


@pytest.fixture
def reference_door(tmp_path):
    # Execute the shipped orchestrator AND namelist-producing helper, replacing
    # only MPI/model execution. No compiler, model or mesh download is needed.
    helpers = tmp_path / 'helpers'
    helpers.mkdir()
    (helpers / 'check_all.sh').write_bytes((TOOLS / 'check_all.sh').read_bytes())
    for name in ('run_receipt.py', 'run_native_jw.sh'):
        (helpers / name).write_bytes((TOOLS / name).read_bytes())
    (helpers / 'validate_reference.py').write_text('pass\n')
    for name in ('compare_dry_identity.py', 'assess_agreement.py', 'compare_native_spread.py', 'reference_digest.py'):
        (helpers / name).write_text('pass\n')
    root = tmp_path / 'reference'
    for recipe, core in (('gnu-o0', 'init_atmosphere'), ('gnu-o0', 'atmosphere'), ('gnu-o3', 'atmosphere')):
        exe = root / f'native/v8.4.2-{recipe}-{core}/{core}_model'
        exe.parent.mkdir(parents=True)
        exe.write_text(f'#!/bin/sh\n# {recipe}-{core}\nexit 0\n')
        exe.chmod(0o755)
    mesh = root / 'mesh'
    mesh.mkdir()
    (mesh / 'x1.2562.grid.nc').write_bytes(b'grid fixture, not simulation data')
    (mesh / 'x1.2562.graph.info').write_text('graph fixture\n')
    (helpers / 'REFERENCE-PINS.json').write_text(json.dumps({
        'grid_sha256_file': hashlib.sha256((mesh / 'x1.2562.grid.nc').read_bytes()).hexdigest()}))
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    mpi = bin_dir / 'mpirun'
    mpi.write_text('''#!/bin/bash
set -eu
test "$1" = -np
test "$2" = 1
shift 2
case "$1" in
  *init_atmosphere_model) echo initial > x1.2562.init.nc ;;
  *)
    echo zero > tracer.2000-01-01_00.00.00.nc
    echo one > tracer.2000-01-01_00.30.00.nc
    echo 'Finished running the atmosphere core' > log.atmosphere.0000.out ;;
esac
''')
    mpi.chmod(0o755)
    command = ['bash', str(helpers / 'check_all.sh'), str(root), str(tmp_path), '1', 'cpu', 'gnu']
    environment = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], PYTHON=sys.executable)
    environment.update({name: 'conflicting-forecast-setting' for name in
                        ('MESH', 'LEVELS', 'START', 'INIT_CASE', 'INIT_SUBDIR', 'VERTICAL_GRID',
                         'GRID_FILE', 'INIT_FILE', 'DT', 'DUR', 'OUT', 'MIXING', 'SMDIV', 'XNUTR', 'RANKS')})

    def run():
        return subprocess.run(command, capture_output=True, text=True, env=environment)

    return root, run


def test_cleaned_run_recovery_preserves_receipts(reference_door):
    root, run = reference_door
    init = root / 'jw/init/x1.2562.init.nc'
    init.parent.mkdir(parents=True)
    # Leave the init absent so the shipped orchestrator exercises both its
    # initialization and run arms in a deliberately conflicting environment.
    stale = root / 'jw/v8.4.2-gnu-o0/rk2-s4'
    stale.mkdir(parents=True)
    (stale / 'old.log').write_text('prior receipt')
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(stale.glob('tracer.*.nc'))) == 2
    assert [p.read_text() for p in (root / 'archived-receipts').glob('*/run/old.log')] == ['prior receipt']
    assert run().returncode == 0
    (stale / 'tracer.2000-01-01_00.30.00.nc').write_bytes(b'changed output')
    refusal = run()
    assert refusal.returncode == 2
    assert 'native run bytes differ' in refusal.stdout


@pytest.mark.parametrize('before,after', [
    ('config_init_case = 2', 'config_init_case = 7'),
    ('config_nvertlevels = 26', 'config_nvertlevels = 55'),
    ('config_vertical_grid = false', 'config_vertical_grid = true'),
])
def test_receipted_custom_initialization_refused(reference_door, before, after):
    root, run = reference_door
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    directory = root / 'jw/init'
    deck = directory / 'namelist.init_atmosphere'
    assert before in deck.read_text()
    deck.write_text(deck.read_text().replace(before, after))
    # The generic helper can record a custom case. A self-consistent receipt
    # must not authorize it as the public standard verifier's case.
    assert receipt.main(['write-init', str(directory)]) == 0
    refusal = run()
    assert refusal.returncode == 2
    assert 'not the standard dry JW case' in refusal.stdout


@pytest.mark.parametrize('before,after', [
    ('config_dt = 1800.0', 'config_dt = 360.0'),
    ('config_number_of_sub_steps = 4', 'config_number_of_sub_steps = 2'),
    ('config_horiz_mixing = "2d_fixed"', 'config_horiz_mixing = "2d_smagorinsky"'),
    ('config_physics_suite = "none"', 'config_physics_suite = "mesoscale_reference"'),
])
def test_receipted_custom_run_refused(reference_door, before, after):
    root, run = reference_door
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    directory = root / 'jw/v8.4.2-gnu-o0/rk2-s4'
    deck = directory / 'namelist.atmosphere'
    assert before in deck.read_text()
    deck.write_text(deck.read_text().replace(before, after))
    assert receipt.main(['write', str(directory)]) == 0
    refusal = run()
    assert refusal.returncode == 2
    assert 'not the standard dry JW case' in refusal.stdout


def test_wrong_reference_producer_refused(reference_door):
    root, run = reference_door
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    directory = root / 'jw/v8.4.2-gnu-o0/rk2-s4'
    executable = directory / 'atmosphere_model'
    executable.unlink()
    executable.symlink_to(root / 'native/v8.4.2-gnu-o3-atmosphere/atmosphere_model')
    assert receipt.main(['write', str(directory)]) == 0
    refusal = run()
    assert refusal.returncode == 2
    assert 'differs from the selected atmosphere_model' in refusal.stdout


def test_initialization_without_receipt_refused(reference_door):
    root, run = reference_door
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    (root / 'jw/init/init-receipt.json').unlink()
    refusal = run()
    assert refusal.returncode == 2
    assert 'init-receipt.json' in refusal.stdout and 'fresh --work-root' in refusal.stdout
