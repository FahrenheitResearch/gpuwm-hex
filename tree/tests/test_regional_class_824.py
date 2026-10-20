"""An additional measured class must retain exact admission and real evidence."""
from dataclasses import replace
import json
from pathlib import Path

from hexcore.cuda_backend.regional_admission import ADMITTED_CLASSES


def test_new_exact_class_names_independent_word_identical_mints():
    root=Path(__file__).resolve().parents[1]
    klass=ADMITTED_CLASSES['graded-824m-dt5-z7']
    assert klass.finest_edge_band_relative==0
    assert not klass.key.finest_edge_within(replace(klass.key,finest_edge_mm=824452),0)
    a,b=[json.loads((root/path).read_text()) for path in klass.mint_receipts]
    for key in ('source_sha256','config_sha256','config_dt','config_len_disp','steps','start_time'):
        assert a[key]==b[key]
    assert a['minted_without_anchor_gate'] and b['minted_without_anchor_gate']
    assert a['steps']==2160 and a['config_dt']==5 and a['n_cells_solve']==43642
    assert len(a['frames'])==len(b['frames'])==2
    for left,right in zip(a['frames'],b['frames']):
        assert left['step']==right['step']
        assert left['masked_sha256']==right['masked_sha256']
        assert left['field_sha256']==right['field_sha256']
        assert all(left['finite'].values()) and left['sha256']!=right['sha256']
        words=json.loads((root/f'docs/proofs/regional-class-824/words-step{left["step"]}.json').read_text())
        assert words['equal']
        assert any(row['base']['dtype']=='S1' for row in words['variables'])
        assert all(row['base']==row['candidate'] for row in words['variables'])
