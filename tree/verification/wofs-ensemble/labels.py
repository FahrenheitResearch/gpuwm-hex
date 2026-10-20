"""Publish the hex identity using the Rust aggregate-render API."""
from pathlib import Path
import hashlib,json,os,sys
from gpuwm import rustwx
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))

def label_aggregates(out,renderer):
    root=out/'aggregate'
    path=root/'d01/ensemble-manifest.json'
    if not path.exists(): return
    receipt=json.loads(path.read_text())
    label='gpuwm-hex | static | LTS'
    rendered=[]
    for frame in receipt['frames']:
        if frame['status']!='complete': raise ValueError('aggregate must close before labeling')
        for relative in frame['products']:
            product=root/relative
            if not product.resolve().is_relative_to(root.resolve()):
                raise ValueError('aggregate product escapes owned output')
            before=hashlib.sha256(product.read_bytes()).hexdigest()
            maps,transcript=rustwx.run_ensemble_product_renderer(renderer,product,
                out_dir=root/'maps',fields=('refl',),domain='d01',
                products=('mean','max','prob'),source_label=label)
            if len(maps)!=3: raise ValueError('aggregate needs mean, maximum and probability maps')
            if hashlib.sha256(product.read_bytes()).hexdigest()!=before:
                raise ValueError('rendering changed aggregate values')
            frame['renderer_transcript']=transcript
            rendered.append(dict(valid_time=frame['valid_time'],product_sha256=before,
                maps=[str(p) for p in maps],source_label=label))
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'ensemble-labels.json').write_text(json.dumps(rendered,indent=2)+'\n')

if __name__=='__main__':
    label_aggregates(W/'products'/sys.argv[1],Path(os.environ['GPUWM_RW_WRFBATCH']))
