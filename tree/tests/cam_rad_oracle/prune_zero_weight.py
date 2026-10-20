"""Build a whole-module candidate that skips provably rejected permutations.

MPAS module_ra_cam.F:6798-6847 multiplies one nonnegative stream weight per
region in fixed order, then accepts only weights >=areamin. IEEE nearest
multiplication is monotone for finite nonnegative operands. Multiplying each
region's largest weight in that same order gives a conservative upper bound.
When that bound is strictly below the native threshold, no permutation can
modify ccon/wgtv/ptrc/totwgt. The final mixed-radix istr words are preserved.
Malformed weights/dimensions use the original native loop without refusal.
This tool creates a diagnostic candidate; identity requires full replay.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source,output,receipt):
    text=source.read_text()
    start='// MPAS module_ra_cam.F:6806\nfor(iconfig=1;iconfig<=nconfigm;iconfig+=1) {'
    stop='// MPAS module_ra_cam.F:6852\nif((totwgt>0.f)) {'
    assert text.count(start)==1 and text.count(stop)==1
    a=text.index(start);b=text.index(stop,a)
    original=text[a:b]
    before='''// MPAS module_ra_cam.F:6798-6847: exact upper bound for rejected permutations.
{
bool _weights_valid=(nrgn>0 && nrgn<=nstr.count && nrgn<=wstr.extent[1]);
double _largest_weight=1.0;
if(_weights_valid) {
  for(int _region=1;_region<=nrgn;++_region) {
    int _streams=nstr(_region);
    if(_streams<1 || _streams>wstr.extent[0]) {_weights_valid=false;break;}
    double _largest=0.0;
    for(int _stream=1;_stream<=_streams;++_stream) {
      double _weight=wstr(_stream,_region);
      if(!(_weight>=0.0 && _weight<=1.0)) {_weights_valid=false;break;}
      _largest=cam_max(_largest,_weight);
    }
    if(!_weights_valid) break;
    _largest_weight=cam_mul(_largest_weight,_largest);
  }
}
if(!_weights_valid || !(_largest_weight<areamin)) {
'''
    after='''} else if(nconfigm>0) {
  // Preserve the native permutation counter's final automatic-array words.
  int _remaining=nconfigm;
  for(int _region=nrgn;_region>=2;--_region) {
    istr(_region)=cam_add(1,cam_mod(_remaining,nstr(_region)));
    _remaining=cam_div(_remaining,nstr(_region));
  }
  istr(1)=cam_add(1,_remaining);
}
}
'''
    candidate=text[:a]+before+original+after+text[b:]
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(candidate)
    receipt.write_text(json.dumps(dict(scope='whole-module exact-pruning candidate; unqualified',
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        native_citation='module_ra_cam.F:6798-6847',whole_model_acceptance=False,
        exclusions=0,source_transformations=1),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','receipt'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output,a.receipt)
