"""Direct NVRTC keeps its strict options while reusing authenticated binaries."""
import sys
from types import SimpleNamespace
import pytest
from hexcore.mpasphys import gf_mpas_v842 as gf


@pytest.mark.parametrize('numerics', ['strict', 'fast'])
def test_direct_cache_reuses_only_matching_complete_images(monkeypatch, tmp_path, numerics):
    monkeypatch.setattr(gf._hex_numerics, '_mode', numerics)
    monkeypatch.setattr(gf._hex_numerics, '_bound', False)
    from hexcore import kernel_warm
    # The process-wide warm store (kernel_warm._STATE) serves raw NVRTC images it already holds; an earlier
    # test that ran real CuPy with GPUWM_HEX_PREP_CACHE set installs it, and this test then never reaches
    # its own cache or pack logic (G0 order: tests/test_contract_hostprep.py first). Measure without it.
    monkeypatch.setattr(kernel_warm, "_STATE", None)
    monkeypatch.setattr(kernel_warm, "install", lambda **_: None)
    compiled=[];loaded=[]
    class Device:
        compute_capability='120'
        def __init__(self,device):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Module:
        def load(self,binary):loaded.append(binary)
    version=[13,2]
    nvrtc=SimpleNamespace(getVersion=lambda:tuple(version),
        createProgram=lambda source,name,*args:(source,name),
        compileProgram=lambda program,options:compiled.append((program,options)),
        getCUBIN=lambda program:program[0].encode(),destroyProgram=lambda program:None)
    cp=SimpleNamespace(__version__='test',cuda=SimpleNamespace(Device=Device,function=SimpleNamespace(Module=Module)))
    monkeypatch.setitem(sys.modules,'cupy',cp)
    monkeypatch.setitem(sys.modules,'cupy_backends.cuda.libs',SimpleNamespace(nvrtc=nvrtc))
    monkeypatch.setenv('CUPY_CACHE_DIR',str(tmp_path))
    gf._compile_module('source',0);gf._compile_module('source',0)
    assert len(compiled)==1 and loaded==[b'source',b'source']
    expected = gf._hex_numerics.options(gf.COMPILER_OPTIONS, numerics=numerics)
    assert compiled[0]==(('source','gf_mpas_v842.cu'),expected+('--gpu-architecture=sm_120',))
    path=next(tmp_path.rglob('*.cubin'));path.write_bytes(b'corrupt')
    gf._compile_module('source',0)
    assert len(compiled)==2 and loaded[-1]==b'source'
    gf._compile_module('changed source',0)
    assert len(compiled)==3
    Device.compute_capability='89';gf._compile_module('source',0)
    assert len(compiled)==4
    version[1]+=1;gf._compile_module('source',0)
    assert len(compiled)==5
    monkeypatch.setattr(gf,'COMPILER_OPTIONS',gf.COMPILER_OPTIONS+('--extra-test-option',))
    gf._compile_module('source',0)
    assert len(compiled)==6
    blocked=tmp_path/'not-a-directory';blocked.write_text('file')
    monkeypatch.setenv('CUPY_CACHE_DIR',str(blocked))
    gf._compile_module('uncacheable',0)
    assert len(compiled)==7 and loaded[-1]==b'uncacheable'


@pytest.mark.parametrize('columns,processors,blocks,free,expected',[
    (50,170,3,8*1024**3,128),(10000,170,3,32*1024**3,10112),
    (10000,170,3,0,128),(10000,10,3,32*1024**3,3840),
    (300000,188,3,8*1024**3,35712),
])
def test_scratch_tile_bounds(columns,processors,blocks,free,expected):
    # Whole 128-column slabs, every resident block, within 1/8 of free memory.
    assert gf._tile_columns(columns,30000,processors,blocks,free)==expected
