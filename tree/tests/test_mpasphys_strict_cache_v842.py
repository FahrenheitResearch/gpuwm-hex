"""Persistent NVRTC image identity and original compile evidence, without CUDA."""
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

from hexcore.mpasphys import strict_cache_v842 as cache


def install_compiler(monkeypatch):
    calls = []
    cp, cuda = ModuleType('cupy'), ModuleType('cupy.cuda')
    def compile(source, **kwargs):
        calls.append((source, kwargs))
        return ('image:' + source + repr(kwargs)).encode(), {}
    cuda.compiler = SimpleNamespace(compile_using_nvrtc=compile)
    cuda.runtime = SimpleNamespace(getDevice=lambda: 0)
    cuda.Device = lambda device: SimpleNamespace(compute_capability='89')
    cp.cuda = cuda
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setitem(sys.modules, 'cupy.cuda', cuda)
    monkeypatch.setattr(cache, '_platform', lambda device: {'sha256': 'compiler-A'})
    class Capture:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def evidence(self, source):
            return dict(status='resolved', observations=[dict(source_sha256=source,
                effective_flags=['--ftz=false'], compiled_image={'sha256': 'observed'})])
    monkeypatch.setattr(cache, 'NvrtcCompileCapture', Capture)
    return calls


def test_disk_hit_retains_original_evidence_without_recompiling(monkeypatch, tmp_path):
    calls = install_compiler(monkeypatch)
    first, evidence = cache.compile_image('source', options=('--ftz=false',), cache_dir=tmp_path)
    second, hit = cache.compile_image('source', options=('--ftz=false',), cache_dir=tmp_path)
    assert second == first and len(calls) == 1
    assert not evidence['persistent_cache_hit'] and hit['persistent_cache_hit']
    assert hit['observations'] == evidence['observations']
    assert hit['persistent_image_sha256'] == hashlib.sha256(first).hexdigest()
    assert calls[0][1]['options'] == ('--ftz=false',)


def test_source_options_arch_platform_and_corruption_refuse_reuse(monkeypatch, tmp_path):
    calls = install_compiler(monkeypatch)
    cache.compile_image('source', options=(), cache_dir=tmp_path)
    cache.compile_image('changed', options=(), cache_dir=tmp_path)
    cache.compile_image('source', options=('--ftz=false',), cache_dir=tmp_path)
    cache.compile_image('source', options=(), arch='90', cache_dir=tmp_path)
    monkeypatch.setattr(cache, '_platform', lambda device: {'sha256': 'compiler-B'})
    cache.compile_image('source', options=(), cache_dir=tmp_path)
    assert len(calls) == 5
    for record_path in (tmp_path / 'mpas-strict-v842').glob('*.json'):
        record = json.loads(record_path.read_text())
        if record['identity']['platform']['sha256'] == 'compiler-B':
            record_path.with_suffix('.bin').write_bytes(b'corrupt')
    _, evidence = cache.compile_image('source', options=(), cache_dir=tmp_path)
    assert len(calls) == 6 and not evidence['persistent_cache_hit']
