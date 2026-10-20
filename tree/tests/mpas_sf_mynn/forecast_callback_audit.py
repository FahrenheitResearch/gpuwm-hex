"""Diagnostic live MYNN boundary comparison, never forecast acceptance.

Run only inside an actual queued real-door forecast. Native storage has not
been changed when the CUDA callback returns: its transaction commits later.
That permits the pristine driver to consume the same inputs once and provide
an independent output comparison. CUDA counts words; Python writes metadata.
"""
import json
import os
from pathlib import Path


def install():
    from hexcore.mpasphys import sf_mynn_v842 as port
    original = port.launch_slot

    def audit(call):
        result = original(call)
        views = call.pools
        session, cp = views.session, views.cp
        sea = ({name: views['prepared'][name+'_sea'].copy()
                for name in port._SEA_HANDOFF}
               if call.options['config_frac_seaice'] else {})
        session.hooks.invoke('driver_sfclayer', call)
        count = session.layout.dimensions['nCellsSolve']
        rows = []
        for name in port.OUTPUTS:
            if name in ('qstar', 'wstar'):
                continue
            field = port._DIAG_NAMES.get(name, name)
            expected = cp.asarray(session.native.get_field('diag_physics', field)[:count])
            actual = result[field]
            different = expected.view(cp.uint32) != actual.view(cp.uint32)
            indexes = cp.flatnonzero(different)
            row = dict(field=field, compared_words=count,
                       differing_words=int(cp.count_nonzero(different).item()),
                       native_nonfinite=int(cp.count_nonzero(~cp.isfinite(expected)).item()),
                       cuda_nonfinite=int(cp.count_nonzero(~cp.isfinite(actual)).item()))
            if indexes.size:
                index = int(indexes[0].item())
                row.update(first_index=index,
                           native_word=int(expected.view(cp.uint32)[index].item()),
                           cuda_word=int(actual.view(cp.uint32)[index].item()))
            rows.append(row)
        for name, actual in sea.items():
            host, _ = session.native.get_variable(name+'_sea')
            expected = cp.asarray(host[:count,0])
            rows.append(dict(field=name+'_sea', compared_words=count,
                             differing_words=int(cp.count_nonzero(expected.view(cp.uint32) != actual.view(cp.uint32)).item()),
                             native_nonfinite=int(cp.count_nonzero(~cp.isfinite(expected)).item()),
                             cuda_nonfinite=int(cp.count_nonzero(~cp.isfinite(actual)).item())))
        record = dict(
            diagnostic_only=True, forecast_acceptance=False, process_count=1,
            itimestep=call.itimestep, columns=count, rows=rows,
            differing_words=sum(r['differing_words'] for r in rows))
        path = Path(os.environ['MYNN_BOUNDARY_AUDIT_RECEIPT'])
        if os.environ.get('MYNN_BOUNDARY_AUDIT_CONTINUE') == '1':
            # Native execution supplies its hidden side effects for diagnosis
            # only. This run can never support a CUDA forecast acceptance.
            with path.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            return result
        path.write_text(json.dumps(record, indent=2)+'\n')
        raise RuntimeError('MYNN diagnostic stopped after the live native/CUDA boundary comparison')

    port.launch_slot = audit
