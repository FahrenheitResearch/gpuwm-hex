"""Regenerate checked-in CUDA and boundary metadata after translation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[4]
p=ROOT/'hex/tree/src/hexcore/mpasphys/rrfs/ruc_lsm_8c9f1a8.py'
s=p.read_text();head=s[:s.index('ARGUMENTS =')];tail=s[s.index('\ndef shared_math_source():'):]
args=json.loads((ROOT/'scratch/argument-layout.json').read_text());tables=json.loads((ROOT/'scratch/table-layout.json').read_text())
source=(ROOT/'scratch/generated.cu').read_text()+'\n'+(ROOT/'scratch/launch.cu').read_text()
p.write_text(head+'ARGUMENTS = '+repr(args)+'\nTABLE_LAYOUT = '+repr(tables)+"\nCUDA_SOURCE = r'''\n"+source+"\n'''\n"+tail)
