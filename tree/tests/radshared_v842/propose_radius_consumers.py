"""Prepare exact p22/p23 owner proposals consuming the stable p21 radii.

Production sources are read only. The complete proposed LW module can be
loaded by the coupled native driver-pool replay, without modifying a seam.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lw', type=Path, required=True)
    parser.add_argument('--sw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    records = {}
    for kind, path in [('lw', args.lw), ('sw', args.sw)]:
        old = path.read_text()
        new = old
        def replace(anchor, replacement):
            nonlocal new
            assert new.count(anchor) == 1, anchor
            new = new.replace(anchor, replacement)
        if kind == 'lw':
            replace("    for target in ('re_cloud','re_ice','re_snow'):\n"
                    "        state[target]=(diag[target][:,:nc] if radii_enabled else\n"
                    "                       cp.zeros_like(state['p3d']))\n", """    # MPAS: mpas_atmphys_driver_radiation_lw.F:414-454.
    from .radshared_v842 import RADIUS_SCHEMES, radii_from_MPAS, radii_to_MPAS
    names = ('re_cloud', 're_ice', 're_snow')
    radii = None
    if str(call.options['config_microp_scheme']).strip() in RADIUS_SCHEMES:
        inputs = {name: (cp.ascontiguousarray(diag[name][:,:nc].T) if radii_enabled
                         else cp.empty_like(state['p3d'].T, order='C')) for name in names}
        radii = radii_from_MPAS(microp_scheme=call.options['config_microp_scheme'],
            microp_re=call.options['config_microp_re'], **inputs)
    for index, target in enumerate(names):
        state[target] = (radii[index].T if radii is not None else
            (diag[target][:,:nc] if radii_enabled else cp.zeros_like(state['p3d'])))
""")
            replace("    if str(call.options['config_microp_scheme']).strip() in RADIUS_SCHEMES:\n"
                    "        for name in ('rre_cloud','rre_ice','rre_snow'):\n"
                    "            diag[name][:,:nc]=out[name] if radii_enabled else np.float32(0)\n", """    # MPAS: mpas_atmphys_driver_radiation_lw.F:688-712.
    names = ('rre_cloud', 'rre_ice', 'rre_snow')
    radii = radii_to_MPAS(microp_scheme=call.options['config_microp_scheme'],
        microp_re=call.options['config_microp_re'],
        **{name: cp.ascontiguousarray(out[name].T) for name in names})
    if radii is not None:
        for index, name in enumerate(names):
            diag[name][:,:nc] = radii[index].T
""")
        else:
            replace('    for name in ("re_cloud", "re_ice", "re_snow"):\n'
                    '        sounding[name] = diag[name][:nl] if diagnosed else provider.cp.zeros_like(sounding["p3d"])\n', """    # MPAS: mpas_atmphys_driver_radiation_sw.F:495-531.
    from .radshared_v842 import RADIUS_SCHEMES, radii_from_MPAS
    nc = int(provider.session.layout.dimensions['nCellsSolve'])
    names = ('re_cloud', 're_ice', 're_snow')
    radii = None
    if str(call.options['config_microp_scheme']).strip() in RADIUS_SCHEMES:
        inputs = {name: (provider.cp.ascontiguousarray(diag[name][:nl,:nc].T) if diagnosed
                         else provider.cp.empty_like(sounding['p3d'][:,:nc].T, order='C'))
                  for name in names}
        radii = radii_from_MPAS(microp_scheme=call.options['config_microp_scheme'],
            microp_re=call.options['config_microp_re'], **inputs)
    for index, name in enumerate(names):
        sounding[name] = (radii[index].T if radii is not None else
            (diag[name][:nl] if diagnosed else provider.cp.zeros_like(sounding['p3d'])))
""")
        ast.parse(new)
        filename = 'rrtmg_' + kind + '_v842.py'
        (args.out / filename).write_text(new)
        relative = 'tree/src/hexcore/mpasphys/' + filename
        patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
            fromfile='a/' + relative, tofile='b/' + relative))
        (args.out / (kind + '.patch')).write_text(patch)
        records[kind] = dict(original_path=str(path.resolve()),
            original_sha256=hashlib.sha256(old.encode()).hexdigest(),
            proposed_sha256=hashlib.sha256(new.encode()).hexdigest(),
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            production_applied=False, forecast_proven=False)
    (args.out / 'proposal.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
