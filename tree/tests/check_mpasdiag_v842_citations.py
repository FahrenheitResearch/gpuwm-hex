"""Check this lane's static v8.4.2 source anchors using the existing scanner.

The legacy checker remains bound to v8.2.3. This companion never substitutes a
source release or updates its pin ledger; the integrator must route these three
modules to this check when adding the v8.4.2 campaign source registry.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

TREE=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("legacy_mpas_citations",TREE/"tools/check_mpas_citations.py")
LEGACY=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=LEGACY
SPEC.loader.exec_module(LEGACY)
FILES={"cloud_v842.py","pv_v842.py","soundings_v842.py"}
LEDGER=Path(__file__).with_name("mpasdiag_v842_citations.json")


def inventory(source_root):
    found=[r for r in LEGACY.citation_occurrences(TREE/"src/hexcore/mpasdiag")
           if Path(r.python_path).name in FILES]
    rows=[]
    for citation,count in sorted(Counter(r.citation for r in found).items()):
        name,first,last=LEGACY._split_citation(citation)
        path=LEGACY.resolve_cited_path(name,source_root)
        lines=(source_root/path).read_bytes().splitlines(keepends=True)
        if not 1<=first<=last<=len(lines):
            raise ValueError(f"citation outside source: {citation}")
        rows.append(dict(citation=citation,source_path=path,first=first,last=last,
            occurrences=count,window_sha256=hashlib.sha256(b"".join(lines[first-1:last])).hexdigest()))
    return {"release":"v8.4.2","commit":"f34984b2c89353cf31c235b08d3b4acd96b37bb3","rows":rows}


def check(source_root):
    current=inventory(source_root)
    expected=json.loads(LEDGER.read_text())
    assert current==expected,"v8.4.2 source citation range, occurrences, or digest changed"
    return dict(kind="v8.4.2-static-source-anchors",passed=True,citations=len(current["rows"]),
        occurrences=sum(r["occurrences"] for r in current["rows"]),
        ledger_sha256=hashlib.sha256(LEDGER.read_bytes()).hexdigest(),whole_forecast_proof=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source_root",type=Path)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=check(args.source_root)
    if args.receipt: args.receipt.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
