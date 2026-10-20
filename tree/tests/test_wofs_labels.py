"""Aggregate identity changes must not alter the numerical product."""
import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('labels',Path(__file__).parents[1]/'verification/wofs-ensemble/labels.py')
labels=importlib.util.module_from_spec(spec); spec.loader.exec_module(labels)

def test_label_calls_rust_with_explicit_identity_and_preserves_product(tmp_path,monkeypatch):
    root=tmp_path/'aggregate/d01'; root.mkdir(parents=True)
    product=root/'product.nc'; product.write_bytes(b'immutable product fixture')
    receipt=dict(frames=[dict(status='complete',valid_time='hour',products=['d01/product.nc'])])
    manifest=root/'ensemble-manifest.json'; manifest.write_text(json.dumps(receipt))
    def render(renderer,path,**kwargs):
        assert kwargs['source_label'].startswith('gpuwm-hex ')
        assert kwargs['products']==('mean','max','prob')
        return [Path('mean.png'),Path('max.png'),Path('prob.png')],'rust transcript'
    monkeypatch.setattr(labels.rustwx,'run_ensemble_product_renderer',render)
    labels.label_aggregates(tmp_path,Path('renderer'))
    assert product.read_bytes()==b'immutable product fixture'
    receipt['frames'][0]['status']='pending'; manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='close before'): labels.label_aggregates(tmp_path,Path('renderer'))
