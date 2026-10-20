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
def test_each_changed_pin_is_environment(key):
    changed = record(); changed['pins'][key] = 'different'
    verdict = env.judge(record(), record(), changed)
    assert verdict['verdict'] == 'ENVIRONMENT'
    assert verdict['failures']


@pytest.mark.parametrize('which', ['reference','before','after'])
def test_missing_as_run_cannot_be_reconstructed(which):
    records = dict(reference=record(), before=record(), after=record())
    records[which] = None
    assert env.judge(records['reference'], records['before'], records['after'])['verdict'] == 'ENVIRONMENT'


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
    monkeypatch.setattr(env, 'capture', lambda *a:record())
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a,**kw:pytest.fail('forecast started'))
    result=gate.verify(SimpleNamespace(deck=deck,comparator=None,work_root=tmp_path/'work',hex_output=None))
    assert result['verdict'] == 'ENVIRONMENT'
    assert result['model_execution'] is False and result['passed'] is False


def test_refusal_still_records_capture_status(tmp_path, monkeypatch):
    def refuse(_): raise gate.VerificationRefusal('bad deck')
    monkeypatch.setattr(gate,'verify',refuse)
    args=SimpleNamespace(receipt=tmp_path/'receipt.json')
    assert gate.run(args) == 1
    assert json.loads(args.receipt.read_text())['environment']['verdict'] == 'ENVIRONMENT'


@pytest.mark.parametrize('changed,words,verdict', [
    (False,0,'GREEN'), (False,1,'RED'), (True,0,'ENVIRONMENT'), (True,1,'ENVIRONMENT')])
def test_receipt_verdict_requires_same_environment(tmp_path, monkeypatch, changed, words, verdict):
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
    if changed: assert result['ratchet_regressions'] == []


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
    assert 'installed engine RECORD SHA256 is missing or invalid' in env.problems(observed)
    assert env.judge(observed, observed, observed)['verdict'] == 'ENVIRONMENT'


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
