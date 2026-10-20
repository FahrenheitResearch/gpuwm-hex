"""Build an isolated diagnostic artifact from the committed owner proposal.

The production checkout and seam files remain untouched. This fixture can
exercise the exact integration proposal in a whole forecast; it cannot qualify
as the landed production door or a positive acceptance reproducer.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--proposal', type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
proposal_path = args.proposal or here / 'integrator_iau_landed_hooks.json'
proposal = json.loads(proposal_path.read_text())
patch = proposal_path.with_suffix('.patch').read_bytes()
if hashlib.sha256(patch).hexdigest() != proposal['patch_sha256']:
    raise SystemExit('committed owner proposal hash differs')
for path, expected in proposal['source_sha256'].items():
    relative = Path(path).relative_to('tree/src')
    if hashlib.sha256((args.base / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit('base artifact does not match owner proposal: ' + path)
out = args.out.resolve()
shutil.copytree(args.base, out, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
installed_patch = patch.decode().replace('a/tree/src/hexcore/', 'a/hexcore/').replace(
                                       'b/tree/src/hexcore/', 'b/hexcore/')
subprocess.run(['git', 'apply', '--no-index', '--directory=' + out.name, '-'],
               cwd=out.parent, input=installed_patch, text=True, check=True)
modules = {}
for path in proposal['source_sha256']:
    relative = Path(path).relative_to('tree/src')
    source = (out / relative).read_bytes()
    ast.parse(source)
    modules[str(relative)] = hashlib.sha256(source).hexdigest()
args.receipt.write_text(json.dumps(dict(
    schema='mpas.iau-owner-hook-diagnostic-artifact/v1', acceptance=False,
    scope='isolated committed proposal fixture; production seam not changed',
    patch_sha256=proposal['patch_sha256'], hooked_modules=modules,
    iau_module_sha256=hashlib.sha256((out / 'hexcore/iau_v842.py').read_bytes()).hexdigest(),
    artifact=str(out)), indent=2) + '\n')
print(args.receipt.read_text())
