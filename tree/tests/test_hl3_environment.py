"""A header/compiler/environment mismatch cannot become a source regression."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hexcore import verification_environment as env
from hexcore import verify_mpas as gate


def record():
    return dict(schema=env.SCHEMA, errors=[], pins=dict(
        engine_version='2.9.2', engine_git_sha='a'*40, engine_record_sha256='b'*64,
        math_headers={'glibc_flt32.cuh':'c'*64}, cuda_toolkit='nvcc release 13.0',
        cuda_runtime=13000, nvrtc=[13,0], cupy_version='13.0',
        driver='610.57', gpu_uuid='GPU-example', numerics='strict'))


def test_exact_environment_accepts_relocated_sources():
    before = record(); before['engine_root'] = '/before'
    after = record(); after['engine_root'] = '/after'
    assert env.judge(record(), before, after)['verdict'] == 'PINNED'


@pytest.mark.parametrize('key', env.KEYS)
def test_each_pin_changed_during_the_run_is_environment(key):
    changed = record(); changed['pins'][key] = 'different'
    verdict = env.judge(record(), record(), changed)
    assert verdict['verdict'] == 'ENVIRONMENT'
    assert verdict['failures']


OTHER = dict(engine_version='2.9.6', engine_git_sha='f'*40, engine_wheel_sha256='e'*64,
             engine_content_sha256='d'*64, math_headers={'glibc_flt32.cuh': 'f'*64},
             cuda_toolkit='nvcc release 13.4', cuda_runtime=13020, nvrtc=[13, 4],
             cupy_version='14.2.0', driver='580.95', gpu_uuid='GPU-other')


@pytest.mark.parametrize('key', sorted(OTHER))
def test_each_pin_differing_from_the_reference_is_listed_not_refused(key):
    reference = record(); reference['pins'][key] = OTHER[key]
    verdict = env.judge(reference, record(), record())
    assert verdict['verdict'] == 'DIFFERS' and verdict['failures'] == []
    assert verdict['reference_differences'] == [key]


def test_no_reference_is_recorded_not_refused():
    """A deck needs no hand-captured record: the receipt's own capture is the record."""
    verdict = env.judge(None, record(), record())
    assert verdict['verdict'] == 'RECORDED' and verdict['failures'] == []


@pytest.mark.parametrize('which', ['before','after'])
def test_missing_as_run_cannot_be_reconstructed(which):
    records = dict(reference=record(), before=record(), after=record())
    records[which] = None
    assert env.judge(records['reference'], records['before'], records['after'])['verdict'] == 'ENVIRONMENT'


def test_fast_numerics_is_environment():
    fast = record(); fast['pins']['numerics'] = 'fast'
    verdict = env.judge(None, fast, fast)
    assert verdict['verdict'] == 'ENVIRONMENT'
    assert 'before: strict landing requires numerics=strict' in verdict['failures']


def test_collector_hashes_actual_override_headers_and_git(tmp_path, monkeypatch):
    """Exercise capture with fake CUDA API, not a real card or imported engine."""
    import sys
    import hexcore.engine_pin as engine_pin
    root = tmp_path/'engine'; kernels=root/'gpuwm/core/kernels'
    kernels.mkdir(parents=True); (root/'.git').mkdir()
    (kernels/'first.cuh').write_bytes(b'first')
    (kernels/'second.cuh').write_bytes(b'second')
    class Distribution:
        version='wrong-installed-version'
        def read_text(self, name): return 'installed-record' if name == 'RECORD' else None
    monkeypatch.setattr(env.importlib.metadata, 'distribution', lambda _: Distribution())
    monkeypatch.setattr(env.importlib.util, 'find_spec', lambda _: SimpleNamespace(submodule_search_locations=[str(root/'gpuwm')]))
    monkeypatch.setattr(engine_pin, 'checkout_version', lambda _: '2.9.2')
    runtime = SimpleNamespace(runtimeGetVersion=lambda:13000, getDevice=lambda:0)
    cuda = SimpleNamespace(runtime=runtime, nvrtc=SimpleNamespace(getVersion=lambda:(13,0)),
                           Device=lambda _: SimpleNamespace(pci_bus_id='0000:01:00.0'))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(__version__='13.0', cuda=cuda))
    monkeypatch.setitem(sys.modules, 'cupy.cuda', cuda)
    def execute(argv, **kw):
        if argv[0] == 'git': value = 'a'*40 if argv[-1] == 'HEAD' else ''
        elif argv[0] == 'nvcc': value = 'nvcc release 13.0'
        else: value = 'GPU-example, 610.57'
        return SimpleNamespace(stdout=value, returncode=0)
    monkeypatch.setattr(env.subprocess, 'run', execute)
    observed=env.capture('strict', ['--gpuwm-checkout', str(root)])
    assert env.problems(observed) == []
    assert observed['pins']['engine_version'] == '2.9.2'
    assert observed['pins']['engine_record_sha256'] is None
    assert observed['pins']['math_headers'] == {
        name:hashlib.sha256((kernels/name).read_bytes()).hexdigest()
        for name in ['first.cuh','second.cuh']}


def test_unpinned_gate_does_not_start_a_forecast(tmp_path, monkeypatch):
    inputs = {}
    for name in ['namelist','streams','init']:
        p=tmp_path/name; p.write_text('')
        inputs[name] = dict(path=str(p),sha256=hashlib.sha256(b'').hexdigest())
    deck=tmp_path/'deck.json'
    deck.write_text(json.dumps(dict(schema='mpas-verification-deck-v1',
        pin_set='example',mesh='mesh',date='date',id='deck',process_count=1,
        inputs=inputs,native={},hex_command=['gpuwm-hex','forecast'])))
    monkeypatch.setattr(gate, 'comparator_path', lambda _: tmp_path/'comparator')
    monkeypatch.setattr(gate, 'native_manifest', lambda *a:(tmp_path/'native.json',{}))
    monkeypatch.setattr(gate, 'proof_physics_mode', lambda c,n:(c,{}))
    fast = record(); fast['pins']['numerics'] = 'fast'
    monkeypatch.setattr(env, 'capture', lambda *a:fast)
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a,**kw:pytest.fail('forecast started'))
    result=gate.verify(SimpleNamespace(deck=deck,comparator=None,work_root=tmp_path/'work',hex_output=None))
    assert result['verdict'] == 'ENVIRONMENT'
    assert result['model_execution'] is False and result['passed'] is False


def test_deck_without_reference_starts_the_forecast(tmp_path, monkeypatch):
    """hex 0.4.1-0.4.4 refused every shipped deck here ('reference: missing as-run strict environment')."""
    inputs = {}
    for name in ['namelist','streams','init']:
        p=tmp_path/name; p.write_text('<streams/>' if name == 'streams' else '')
        inputs[name] = dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    deck=tmp_path/'deck.json'
    deck.write_text(json.dumps(dict(schema='mpas-verification-deck-v1',
        pin_set='example',mesh='mesh',date='date',id='deck',process_count=1,
        inputs=inputs,native={},hex_command=['gpuwm-hex','forecast','--physics-namelist','{namelist}',
        '--streams','{streams}','--init','{init}','--out','{out}'])))
    monkeypatch.setattr(gate, 'comparator_path', lambda _: tmp_path/'comparator')
    monkeypatch.setattr(gate, 'native_manifest', lambda *a:(tmp_path/'native.json',{}))
    monkeypatch.setattr(gate, 'proof_physics_mode', lambda c,n:(c,{}))
    monkeypatch.setattr(env, 'capture', lambda *a:record())
    class Started(Exception): pass
    def started(*a, **kw): raise Started
    # The model-launch validation runs only after the environment check admits the arm.
    monkeypatch.setattr(gate, 'validate_model_command', started)
    with pytest.raises(Started):
        gate.verify(SimpleNamespace(deck=deck,comparator=None,work_root=tmp_path/'work',hex_output=None))


def test_refusal_still_records_capture_status(tmp_path, monkeypatch):
    def refuse(_): raise gate.VerificationRefusal('bad deck')
    monkeypatch.setattr(gate,'verify',refuse)
    args=SimpleNamespace(receipt=tmp_path/'receipt.json')
    assert gate.run(args) == 1
    assert json.loads(args.receipt.read_text())['environment']['verdict'] == 'ENVIRONMENT'


@pytest.mark.parametrize('changed,words,verdict', [
    (False,0,'GREEN'), (False,1,'RED'), (True,0,'GREEN'), (True,1,'ENVIRONMENT')])
def test_receipt_verdict_attributes_only_in_the_reference_environment(tmp_path, monkeypatch, changed, words, verdict):
    """Identical words are the proof in any strict environment; a difference is a
    source regression only in the reference environment."""
    """Use comparison fixtures to exercise receipt/ratchet policy end to end."""
    def ref(name, value):
        p=tmp_path/name; p.write_text(json.dumps(value))
        return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    out=tmp_path/'output'; out.mkdir(); (out/'a.nc').write_bytes(b'fixture output')
    observed=record()
    if changed: observed['pins']['driver']='595.71'
    as_run=dict(schema='mpas-run-environment-v1',before=observed,after=observed,
                output_sha256={'a.nc':hashlib.sha256(b'fixture output').hexdigest()})
    inputs={key:ref(key,{}) for key in ['namelist','streams','init']}
    deck=tmp_path/'deck.json'
    deck.write_text(json.dumps(dict(schema='mpas-verification-deck-v1', pin_set='pin',
        mesh='mesh',date='date',id='deck',process_count=1,inputs=inputs,native={},
        hex_command=['gpuwm-hex','forecast'],strict_environment=ref('expected.json',record()),
        hex_environment=ref('as-run.json',as_run))))
    native=dict(resolved_output_root=tmp_path/'native',outputs=[dict(path='a.nc',stream='output',steps=[1],times=['t'])])
    (tmp_path/'fake-comparator').write_bytes(b'fixture comparator identity')
    monkeypatch.setattr(gate,'native_manifest',lambda *a:(Path(inputs['init']['path']),native))
    monkeypatch.setattr(gate,'comparator_path',lambda _:tmp_path/'fake-comparator')
    monkeypatch.setattr(gate,'proof_physics_mode',lambda c,n:(c,{}))
    monkeypatch.setattr(gate,'source_provenance',lambda:{})
    monkeypatch.setattr(gate,'stream_inventory',lambda *a:{out/'a.nc'})
    monkeypatch.setattr(gate,'compare_sample_headers',lambda *a,**kw:[])
    monkeypatch.setattr(gate,'maths_library_witness',lambda *a:None)
    monkeypatch.setattr(gate,'acceptance_tier',lambda *a:('BITWISE',[]))
    monkeypatch.setattr(gate,'tier',lambda *a:('BITWISE',[]))
    monkeypatch.setattr(gate,'ratchet_failures',lambda *a:['theta words differ'] if words else [])
    def compare(argv,**kw):
        Path(argv[-1]).write_text(json.dumps(dict(fields=[dict(stream='output',field='theta',
            step=1,status='compared',compared_words=2,differing_words=words,
            max_ulp=1 if words else 0,first_differing_index=0 if words else None)])))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess,'run',compare)
    result=gate.verify(SimpleNamespace(deck=deck,comparator=None,work_root=tmp_path/'work',
                                       hex_output=out,ratchet=None,render=False))
    assert result['verdict'] == verdict
    assert result['data_words_differing'] == words
    assert result['passed'] == (verdict == 'GREEN')
    assert result['ratchet_eligible'] == (not changed)
    if changed:
        assert result['ratchet_regressions'] == []
        assert result['environment']['reference_differences'] == ['driver']


def test_nvrtc_wheel_toolkit_is_pinned_without_nvcc(tmp_path, monkeypatch):
    # Reuse the complete collector fixture, changing only toolkit discovery.
    test_collector_hashes_actual_override_headers_and_git(tmp_path, monkeypatch)
    execute = env.subprocess.run
    def wheel_only(argv, **kwargs):
        if argv[0] == 'nvcc':
            raise FileNotFoundError('nvcc')
        return execute(argv, **kwargs)
    monkeypatch.setattr(env.subprocess, 'run', wheel_only)
    monkeypatch.setattr(env.importlib.metadata, 'version', lambda name: '13.4.92')
    observed = env.capture('strict', ['--gpuwm-checkout', str(tmp_path/'engine')])
    assert env.problems(observed) == []
    assert observed['pins']['cuda_toolkit'] == {name:'13.4.92' for name in
        ('cuda-toolkit','nvidia-cuda-nvrtc','nvidia-cuda-runtime')}


def installed_wheel_capture(tmp_path, monkeypatch, installed_record='installed-record'):
    # Reuse mocked CUDA and subprocess capture, then point at a wheel install
    # outside the checkout so no source commit can impersonate archive bytes.
    test_collector_hashes_actual_override_headers_and_git(tmp_path, monkeypatch)
    root = tmp_path/'site-packages'
    kernels = root/'gpuwm/core/kernels'
    kernels.mkdir(parents=True)
    (kernels/'first.cuh').write_bytes(b'first')
    class Distribution:
        version = '2.9.3'
        def read_text(self, name):
            if name == 'RECORD': return installed_record
            if name == 'direct_url.json':
                return json.dumps(dict(archive_info=dict(hashes=dict(sha256='d'*64))))
    monkeypatch.setattr(env.importlib.metadata, 'distribution', lambda _: Distribution())
    monkeypatch.setattr(env.importlib.util, 'find_spec', lambda _: SimpleNamespace(
        submodule_search_locations=[str(root/'gpuwm')]))
    return env.capture('strict')


def test_installed_wheel_archive_and_record_identity_is_accepted(tmp_path, monkeypatch):
    observed = installed_wheel_capture(tmp_path, monkeypatch)
    assert observed['pins']['engine_git_sha'] is None
    assert observed['pins']['engine_wheel_sha256'] == 'd'*64
    assert observed['pins']['engine_record_sha256'] == hashlib.sha256(b'installed-record').hexdigest()
    assert env.problems(observed) == []
    assert env.judge(observed, observed, deepcopy(observed))['verdict'] == 'PINNED'


def test_installed_wheel_without_record_is_refused(tmp_path, monkeypatch):
    observed = installed_wheel_capture(tmp_path, monkeypatch, installed_record=None)
    assert observed['pins']['engine_wheel_sha256'] == 'd'*64
    assert any(p.startswith('engine identity needs') for p in env.problems(observed))
    assert env.judge(observed, observed, observed)['verdict'] == 'ENVIRONMENT'


def pypi_install(tmp_path, monkeypatch, tamper=False):
    """A wheel installed the way pip installs from PyPI: RECORD, no direct_url.json."""
    import base64
    import importlib.metadata as metadata
    import zipfile
    test_collector_hashes_actual_override_headers_and_git(tmp_path, monkeypatch)
    site = tmp_path/'pypi-site'
    files = {'gpuwm/__init__.py': b'', 'gpuwm/core/kernels/first.cuh': b'first',
             'gpuwm/libexec/bridges/rw_mpas_static': b'binary'}
    def row(name, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        return f'{name},sha256={digest},{len(data)}'
    rows = [row(n, d) for n, d in files.items()]
    wheel = tmp_path/'gpuwm-2.9.6-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr('gpuwm-2.9.6.dist-info/METADATA', 'Name: gpuwm\nVersion: 2.9.6\n')
        archive.writestr('gpuwm-2.9.6.dist-info/RECORD',
                         '\n'.join(rows + ['gpuwm-2.9.6.dist-info/RECORD,,']) + '\n')
    for name, data in files.items():
        (site/name).parent.mkdir(parents=True, exist_ok=True)
        (site/name).write_bytes(data)
    info = site/'gpuwm-2.9.6.dist-info'
    info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: gpuwm\nVersion: 2.9.6\n')
    (info/'INSTALLER').write_text('pip\n')
    script = b'#!/venv/bin/python3\nimport gpuwm\n'
    (info/'RECORD').write_text('\n'.join(rows + [row('../../../bin/gpuwm', script),
        row('gpuwm-2.9.6.dist-info/INSTALLER', b'pip\n'), 'gpuwm-2.9.6.dist-info/RECORD,,',
        'gpuwm/__pycache__/__init__.cpython-314.pyc,,']) + '\n')
    if tamper:
        (site/'gpuwm/core/kernels/first.cuh').write_bytes(b'edited after install')
    dist = metadata.PathDistribution(info)
    monkeypatch.setattr(env.importlib.metadata, 'distribution', lambda _: dist)
    monkeypatch.setattr(env.importlib.util, 'find_spec', lambda _: SimpleNamespace(
        submodule_search_locations=[str(site/'gpuwm')]))
    return env.capture('strict'), wheel


def test_pypi_install_identity_is_its_verified_files(tmp_path, monkeypatch):
    """hex 0.4.1-0.4.4 refused every PyPI-installed engine: pip records no wheel hash."""
    observed, wheel = pypi_install(tmp_path, monkeypatch)
    assert observed['pins']['engine_wheel_sha256'] is None
    assert observed['pins']['engine_git_sha'] is None
    assert env.problems(observed) == []
    # Recomputable from the published wheel alone, without installing it.
    assert observed['pins']['engine_content_sha256'] == env.wheel_content_sha256(wheel)
    assert env.judge(None, observed, observed)['verdict'] == 'RECORDED'


def test_pypi_install_with_an_edited_file_is_refused(tmp_path, monkeypatch):
    observed, _ = pypi_install(tmp_path, monkeypatch, tamper=True)
    assert ('installed engine file differs from its RECORD: gpuwm/core/kernels/first.cuh'
            in env.problems(observed))
    assert env.judge(None, observed, observed)['verdict'] == 'ENVIRONMENT'


def test_checkout_identity_and_legacy_reference_are_unchanged(tmp_path, monkeypatch):
    test_collector_hashes_actual_override_headers_and_git(tmp_path, monkeypatch)
    root = tmp_path/'engine'
    class Distribution:
        version = 'wrong-installed-version'
        def read_text(self, name):
            if name == 'RECORD': return 'installed-record'
            if name == 'direct_url.json':
                return json.dumps(dict(archive_info=dict(hashes=dict(sha256='d'*64))))
    monkeypatch.setattr(env.importlib.metadata, 'distribution', lambda _: Distribution())
    observed = env.capture('strict', ['--gpuwm-checkout', str(root)])
    assert observed['pins']['engine_git_sha'] == 'a'*40
    assert observed['pins']['engine_wheel_sha256'] is None
    assert observed['pins']['engine_record_sha256'] is None
    assert env.problems(observed) == []
    legacy = deepcopy(observed)
    del legacy['pins']['engine_wheel_sha256']
    assert env.judge(legacy, observed, observed)['verdict'] == 'PINNED'
    execute = env.subprocess.run
    def dirty(argv, **kwargs):
        if argv[0] == 'git' and 'status' in argv:
            return SimpleNamespace(stdout=' M gpuwm/core/kernels/first.cuh', returncode=0)
        return execute(argv, **kwargs)
    monkeypatch.setattr(env.subprocess, 'run', dirty)
    changed = env.capture('strict', ['--gpuwm-checkout', str(root)])
    assert 'engine checkout has tracked modifications' in env.problems(changed)


def test_comparator_loader_failure_names_first_stderr_line(tmp_path, monkeypatch):
    path = tmp_path/'rw_mpas_verify'
    monkeypatch.setattr(gate.shutil, 'which', lambda _: str(path))
    error = 'error while loading shared libraries: libnetcdf.so.22: cannot open shared object file'
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=127, stdout='', stderr=error+'\nsecond diagnostic line\n'))
    with pytest.raises(gate.VerificationRefusal) as refusal:
        gate.comparator_path(path)
    assert str(refusal.value) == 'Rust comparator ABI probe failed (exit 127): '+error


def test_installed_multicall_comparator_is_found(tmp_path, monkeypatch):
    """rw_mpas_verify ships only as rw_mpas_static's verify tool; a PyPI install must find it."""
    from hexcore import engines
    package = tmp_path/'gpuwm'
    bridges = package/'libexec/bridges'
    bridges.mkdir(parents=True)
    tool = bridges/engines.executable_name('rw_mpas_static')
    tool.write_bytes(b'x')
    monkeypatch.setattr(engines, '_gpuwm_package_dir', lambda: package)
    monkeypatch.delenv('GPUWM_HEX_RW_MPAS_VERIFY', raising=False)
    monkeypatch.setattr(gate.shutil, 'which', lambda name: None)
    monkeypatch.setattr(gate.os, 'access', lambda path, mode: True)
    seen = []
    def probe(argv, **kw):
        seen.append(argv)
        # hex's own older static builder has no verify tool: it prints its door ABI.
        older = 'hexcore' in Path(argv[0]).parts
        return SimpleNamespace(returncode=0, stdout='GRID\tWROTE' if older else gate.ABI, stderr='')
    monkeypatch.setattr(gate.subprocess, 'run', probe)
    assert gate.comparator_path(None) == tool.resolve()
    assert seen[-1] == [str(tool.resolve()), 'verify', '--abi']
    assert gate.comparator_argv(tool) == [str(tool), 'verify']
    assert gate.comparator_argv(tmp_path/'rw_mpas_verify') == [str(tmp_path/'rw_mpas_verify')]
