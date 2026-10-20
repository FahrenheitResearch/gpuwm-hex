"""Verify immutable v8.4.2 pins through the repository's existing citation checker."""
from pathlib import Path
import json
import os
import runpy

base=Path(__file__).resolve().parent;tree=base.parents[1]
checker=runpy.run_path(str(tree/'tools/check_mpas_citations.py'))
inventory=json.loads((base/'citation_pins.json').read_text())
assert inventory['mpas_commit']=='f34984b2c89353cf31c235b08d3b4acd96b37bb3'
pins={row['citation']:checker['CitationPin'](row['source_path'],row['first_line'],row['last_line'],row['window_sha256'],row['occurrences']) for row in inventory['pins']}
found={key: tuple(item for item in items if item.python_path.endswith('/noah_v842.py'))
       for key,items in checker['citations'](tree/'src/hexcore/mpasphys').items()}
found={key: items for key,items in found.items() if items}
failures=checker['check_pins'](found,pins=pins,vendor_root=Path(os.environ['MPAS_REF']))
result={'mpas_commit':inventory['mpas_commit'],'citation_windows':len(pins),'published_occurrences':sum(row['occurrences'] for row in inventory['pins']),'failures':failures}
output=Path('n1out/citations.json');output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if failures:raise SystemExit(1)
