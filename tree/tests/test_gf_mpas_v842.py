"""Whole-module source and import contract; numerical checks use the oracle tool."""
import site_support
from pathlib import Path
import hashlib
import re
import subprocess
import sys

from hexcore.mpasphys import gf_mpas_v842 as gf

PROCEDURES={"cu_grell_freitas","cup_gf","cup_dd_edt","cup_dd_moisture_new","cup_env","cup_env_clev","cup_forcing_ens_3d","cup_kbcon","cup_ktop","cup_maximi","cup_minimi","cup_up_aa0","g3init","neg_check","cup_output_ens_3d","cup_up_moisture","gdinit","satvap","cup_gf_sh"}


def test_complete_module_cited_and_no_wrf_kernel_reused():
    definitions=re.findall(r'// MPAS module_cu_gf.mpas.F:(\d+)-(\d+)\n__device__ (?:void|GFReal) (\w+) ',gf._SCHEME_SOURCE)
    assert {r[2] for r in definitions}==PROCEDURES
    assert len(definitions)==19
    assert all(1<=int(lo)<int(hi)<=5085 for lo,hi,_ in definitions)
    assert 'tgamma' not in gf._SCHEME_SOURCE
    assert 'gpuwm.core.gf' not in Path(gf.__file__).read_text()


def test_float32_rounding_and_glibc_source_contract():
    assert '--fmad=false' in gf.COMPILER_OPTIONS
    assert '--ftz=false' in gf.COMPILER_OPTIONS
    for operation in ('__fadd_rn','__fsub_rn','__fmul_rn','__fdiv_rn','__fsqrt_rn'):
        assert operation in gf._SUPPORT
    assert 'gf_log (' in gf._SCHEME_SOURCE and 'gf_pow (' in gf._SCHEME_SOURCE
    assert '__builtin_' not in gf._SCHEME_SOURCE


def test_import_does_not_load_cuda_runtime():
    code='import sys; import hexcore.mpasphys.gf_mpas_v842; assert "cupy" not in sys.modules; assert "gpuwm.core.physics" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)


def test_authority_pin_when_campaign_source_available():
    p=(site_support.mpas_source('src/core_atmosphere/physics/physics_wrf/module_cu_gf.mpas.F') / 'src/core_atmosphere/physics/physics_wrf/module_cu_gf.mpas.F')
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==gf.SOURCE_SHA256
