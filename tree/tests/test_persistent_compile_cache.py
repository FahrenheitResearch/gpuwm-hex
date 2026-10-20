"""Cross-process images persist, with source, dimension, flags and compiler keys."""
from types import SimpleNamespace

from hexcore.cuda_backend.runtime import KernelCache, _cache_directory


def test_default_cache_survives_process_and_card_assignment(monkeypatch, tmp_path):
    monkeypatch.delenv('CUPY_CACHE_DIR', raising=False)
    monkeypatch.delenv('MPAS_PORT_CUDA_CACHE_DIR', raising=False)
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    first = _cache_directory(0, (12, 0), None)
    monkeypatch.delenv('CUPY_CACHE_DIR', raising=False)
    monkeypatch.setattr('os.getpid', lambda: 987654)
    assert _cache_directory(7, (12, 0), None) == first
    monkeypatch.delenv('CUPY_CACHE_DIR', raising=False)
    assert _cache_directory(0, (8, 9), None) != first


def test_compiler_input_changes_cannot_reuse_a_stale_module():
    cache = KernelCache.__new__(KernelCache)
    cache.capability = SimpleNamespace(sm='sm_120')
    cache.base_options = ('--std=c++17', '--fmad=false')
    cache._compile_manifest = SimpleNamespace(fingerprint_sha256='a'*64)
    original = cache._module_key('#define NCELLS 32768\nsource', ())
    assert cache._module_key('#define NCELLS 32769\nsource', ()) != original
    assert cache._module_key('#define NCELLS 32768\nchanged', ()) != original
    assert cache._module_key('#define NCELLS 32768\nsource', ('--use_fast_math',)) != original
    cache._compile_manifest.fingerprint_sha256 = 'b'*64
    assert cache._module_key('#define NCELLS 32768\nsource', ()) != original
