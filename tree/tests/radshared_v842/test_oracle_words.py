"""Exercise the typed transport faults that could invalidate replay proof."""
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest


@pytest.fixture
def helper():
    path=os.environ.get('P21_ORACLE_WORDS')
    if not path:pytest.skip('build the Rust transport under the build queue and set P21_ORACLE_WORDS')
    return str(Path(path).resolve())


def record(path):
    header=[1,4,3,2,3,1,1,1,-1,5,1,1,1,7,9,0,2,1,2,256,0]
    strings=b'synthetic'.ljust(256,b' ')+b'pre'.ljust(256,b' ')+b'cldfrac'.ljust(256,b' ')
    # Float32 bit patterns include signed zero and distinct NaN payloads.
    payload=struct.pack('<6I',0x00000000,0x80000000,0x7fc00001,0x7fc00002,0x3f800000,0xbf800000)
    path.write_bytes(b'ORCL0001'+struct.pack('<21i',*header)+strings+struct.pack('<2i',41,73)+payload)
    return payload


def run(helper,*args):
    return subprocess.run([helper,*map(str,args)],capture_output=True)


def test_profile_preserves_nonfinite_words_and_native_layout(helper,tmp_path):
    source=tmp_path/'pre.bin';payload=record(source);out=tmp_path/'profile.raw'
    assert run(helper,'profile',source,out).returncode==0
    words=struct.unpack('<6I',payload)
    assert out.read_bytes()==struct.pack('<6I',*(words[i] for i in (0,2,4,1,3,5)))


def test_replace_preserves_ids_bounds_and_changes_only_phase_payload(helper,tmp_path):
    source=tmp_path/'pre.bin';payload=record(source);raw=tmp_path/'replacement.raw';raw.write_bytes(payload[::-1])
    out=tmp_path/'post.bin';assert run(helper,'replace',source,raw,out).returncode==0
    old=json.loads(run(helper,'header',source,'unused').stdout)
    new=json.loads(run(helper,'header',out,'unused').stdout)
    assert old.pop('phase')=='pre';assert new.pop('phase')=='post';assert old==new
    assert out.read_bytes()[-len(payload):]==payload[::-1]
    assert run(helper,'replace',source,raw,out).returncode!=0


@pytest.mark.parametrize('fault',('truncated','trailing','width','ids','axes','version'))
def test_invalid_records_cannot_become_profile_proof(helper,tmp_path,fault):
    source=tmp_path/'bad.bin';record(source);raw=bytearray(source.read_bytes())
    if fault=='truncated':raw=raw[:-1]
    elif fault=='trailing':raw+=b'\0'
    else:
        index,value={'width':(1,8),'ids':(16,1),'axes':(18,1),'version':(20,1)}[fault]
        struct.pack_into('<i',raw,8+4*index,value)
    source.write_bytes(raw);out=tmp_path/'out.raw'
    assert run(helper,'profile',source,out).returncode!=0
    assert not out.exists()


def test_replacement_length_is_exact(helper,tmp_path):
    source=tmp_path/'pre.bin';record(source);raw=tmp_path/'short.raw';raw.write_bytes(b'\0'*4)
    out=tmp_path/'post.bin';assert run(helper,'replace',source,raw,out).returncode!=0
    assert not out.exists()


def test_witness_prefix_preserves_every_unaffected_word_and_rejects_overrun(helper,tmp_path):
    source=tmp_path/'baseline.raw';words=(0x80000000,0x7fc00001,1,0x7fffff)
    source.write_bytes(struct.pack('<4I',*words));out=tmp_path/'masked.raw'
    assert run(helper,'raw-prefix',source,out,'1,2147483648').returncode==0
    assert out.read_bytes()==struct.pack('<4I',1,0x80000000,*words[2:])
    assert run(helper,'raw-prefix',source,out,'0').returncode!=0
    bad=tmp_path/'overrun.raw'
    assert run(helper,'raw-prefix',source,bad,'0,0,0,0,0').returncode!=0
    assert not bad.exists()


def test_raw_comparison_counts_nan_payloads_signed_zero_and_subnormals(helper,tmp_path):
    left=tmp_path/'left';right=tmp_path/'right'
    words=(0x80000000,0x7fc00001,1,0x7fffff)
    left.write_bytes(struct.pack('<4I',*words));right.write_bytes(left.read_bytes())
    assert json.loads(run(helper,'compare-raw',left,right).stdout)['different_words']==0
    right.write_bytes(struct.pack('<4I',0,0x7fc00002,0,0x7fffff))
    result=json.loads(run(helper,'compare-raw',left,right).stdout)
    assert result['words']==4 and result['different_words']==3
    assert not result['bitwise_equal'] and result['comparison_complete']
    right.write_bytes(right.read_bytes()[:-1])
    assert run(helper,'compare-raw',left,right).returncode!=0
