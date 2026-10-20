"""Real complete native control boundary and active forced-surface witness."""
import json,os
from pathlib import Path
import pytest
from rrfs_driver_control_replay import replay


@pytest.fixture
def packet():
    bank = os.environ.get('RRFS_DRIVER_CONTROL_BANK')
    library = os.environ.get('RRFS_DRIVER_RECORD_LIBRARY')
    if not bank or not library:pytest.skip('requires sealed pristine-native real control bank and Rust codec')
    bank=Path(bank);return bank,next(iter(json.loads((bank/'manifest.json').read_text())['captures'])),Path(library)


def test_complete_real_control(packet,tmp_path):
    bank,prefix,library=packet
    result=replay(bank/'dump',prefix,tmp_path,library,bank/'manifest.json')
    assert result['output_records']==13 and result['words']>0
    assert result['bitwise_equal']


def test_real_control_rejects_wrong_forced_surface(packet,tmp_path,monkeypatch):
    from dataclasses import replace
    from types import MappingProxyType
    from hexcore.mpasphys.rrfs import driver_rrfs
    original=driver_rrfs.resolve_namelist
    def wrong_surface(text):
        resolved=original(text);options=dict(resolved.options)
        options['config_sfclayer_scheme']='sf_mynn'
        return replace(resolved,options=MappingProxyType(options))
    monkeypatch.setattr(driver_rrfs,'resolve_namelist',wrong_surface)
    bank,prefix,library=packet
    result=replay(bank/'dump',prefix,tmp_path,library,bank/'manifest.json')
    assert not result['bitwise_equal'] and result['different_words']>0
    assert [k for k,c in result['comparisons'].items() if not c['bitwise_equal']]==['block.configs.config_sfclayer_scheme']
