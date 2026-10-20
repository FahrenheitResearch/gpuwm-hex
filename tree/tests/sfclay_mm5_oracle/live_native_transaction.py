"""Replay complete CUDA transactions beside an unchanged native forecast.

Native driver outputs remain authoritative and are never replaced. The CUDA
candidate uses detached copies of the exact incoming native words. This is
diagnostic evidence, not the required CUDA-slot forecast acceptance.
MPAS mpas_atmphys_driver_sfclayer.F:352-1088;
mpas_atmphys_sfc_diagnostics.F:27-123.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('forecast_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import cupy as cp
    import numpy as np
    from hexcore.cli import main as real_cli
    from hexcore.mpasphys import sfclay_mm5_v842 as mm5
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.driver_v842 import resolve_namelist

    command = args.forecast_arguments
    if command[:1] == ['--']:
        command = command[1:]
    namelist = Path(command[command.index('--physics-namelist') + 1])
    configuration = resolve_namelist(namelist.read_text())

    original = NativePhysics.call
    receipt = dict(scope=__doc__, process_count=1, forecast_acceptance=False,
                   cuda_source_sha256=hashlib.sha256(mm5.CUDA_SOURCE.encode()).hexdigest(),
                   module_sha256=hashlib.sha256(Path(mm5.__file__).read_bytes()).hexdigest(),
                   calls=[], excluded_words=[], native_outputs_replaced=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    def seal():
        temporary = args.receipt.with_suffix('.writing')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(args.receipt)

    # MPAS mpas_atmphys_constants.F:8-103: copy compiled scalar words.
    def constants(native):
        aliases = dict(g='gravity', rovcp='rcp', r='R_d', ep1='ep_1', ep2='ep_2', p1000mb='P0')
        return cp.asarray([native.get_constant(aliases.get(n, n)) for n in mm5.CONSTANTS], dtype=cp.float32)

    # MPAS mpas_atmphys_driver_sfclayer.F:352-592: detached pool transport.
    def field(native, pool, name, n):
        host = native.get_field(pool, name)
        if host.dtype != np.float32 or host.ndim != 1:
            raise ValueError(f'{pool}.{name} must be an RKIND cell vector')
        return cp.array(host[:n], copy=True)

    # MPAS mpas_atmphys_driver_sfclayer.F:595-796: compare every stored field.
    def compare(native, candidate, names, n):
        fields = {}
        for name in names:
            actual = cp.asnumpy(candidate[name])
            expected = native.get_field('diag_physics', name)[:n]
            a, e = actual.view(np.uint32), expected.view(np.uint32)
            different = np.flatnonzero(a != e)
            fields[name] = dict(words=int(a.size), differing_words=int(different.size),
                first_differing_index=int(different[0]) if different.size else None,
                cuda_first_word=int(a[different[0]]) if different.size else None,
                native_first_word=int(e[different[0]]) if different.size else None,
                cuda_nonfinite_words=int(np.count_nonzero(~np.isfinite(actual))),
                native_nonfinite_words=int(np.count_nonzero(~np.isfinite(expected))))
        return fields

    # MPAS mpas_atmphys_driver_sfclayer.F:839-1088 and sfc_diagnostics.F:27-123.
    def observed(native, name, **arguments):
        if name not in ('driver_sfclayer', 'atmphys_sfc_diagnostics'):
            return original(native, name, **arguments)
        if arguments['its'] != 1:
            raise ValueError('Diagnostic requires the declared first solve-cell bound')
        n = arguments['ite']
        names = tuple(mm5._MPAS_NAMES.get(k, k) for k in mm5.OUTPUTS)
        incoming = {k: field(native, 'diag_physics', k, n) for k in names}
        sfc = {k: field(native, 'sfc_input', k, n) for k in ('skintemp', 'xland', 'sst', 'xice')}
        c = constants(native)
        if name == 'driver_sfclayer':
            profiles = dict(ux='u_p', vx='v_p', t1d='t_p', qv1d='qv_p', p1d='pres_hyd_p', dz8w1d='dz_p')
            atmosphere = {}
            for alias, global_name in profiles.items():
                value, lower = native.get_variable(global_name)
                assert lower[0] == 1 and value.shape[-1] == 1
                atmosphere[alias] = cp.array(value[:n, 0, 0], copy=True)
            value, lower = native.get_variable('psfc_p')
            atmosphere['psfcpa'] = cp.array(value[:n, 0], copy=True)
            incoming.update({k: field(native, 'diag_physics', k, n) for k in ('hpbl', 'mavail')})
            mesh_density = field(native, 'mesh', 'meshDensity', n)
            scheme = configuration.selectors['surface_layer']
            frac = configuration.options['config_frac_seaice']
            threshold = native.get_variable('xice_threshold')[0][()]
            options = dict(scheme=scheme, config_len_disp=mm5._native_len_disp(native),
                           config_frac_seaice=frac, xice_threshold=threshold)
            sea_exchange = ({k: cp.empty(n, dtype=cp.float32) for k in mm5.SEA_EXCHANGE}
                            if frac else {})
            options['sea_exchange'] = sea_exchange
            candidate = lambda: mm5.run_surface_layer(atmosphere, incoming, sfc, mesh_density, c, **options)
        else:
            names = ('q2', 't2m', 'th2m')
            atmosphere = dict(psfcpa=field(native, 'diag', 'surface_pressure', n))
            scheme = configuration.selectors['land_surface']
            threshold = native.get_variable('xice_threshold')[0][()]
            extra = ({k: field(native, 'output_noahmp', k, n) for k in ('q2mxy', 't2mxy')}
                     if scheme == 'sf_noahmp' else {})
            candidate = lambda: mm5.run_surface_diagnostics(atmosphere, incoming, sfc, c,
                                      lsm_scheme=scheme, xice_threshold=threshold, **extra)
        result = original(native, name, **arguments)
        candidate()
        fields = compare(native, incoming, names, n)
        if name == 'driver_sfclayer' and frac:
            # MPAS driver_sfclayer.F:260-310; driver_seaice.F:390-399.
            for key, value in sea_exchange.items():
                actual = cp.asnumpy(value)
                expected = native.get_variable(key + '_sea')[0][:n, 0]
                a, e = actual.view(np.uint32), expected.view(np.uint32)
                different = np.flatnonzero(a != e)
                fields[key + '_sea'] = dict(words=int(a.size),
                    differing_words=int(different.size),
                    first_differing_index=int(different[0]) if different.size else None,
                    cuda_first_word=int(a[different[0]]) if different.size else None,
                    native_first_word=int(e[different[0]]) if different.size else None,
                    cuda_nonfinite_words=int(np.count_nonzero(~np.isfinite(actual))),
                    native_nonfinite_words=int(np.count_nonzero(~np.isfinite(expected))))
        receipt['calls'].append(dict(index=len(receipt['calls']), entry=name, scheme=scheme,
            itimestep=arguments.get('itimestep'), columns=n, fields=fields,
            differing_words=sum(v['differing_words'] for v in fields.values())))
        seal()
        return result

    NativePhysics.call = observed
    receipt['real_cli_arguments'] = command
    seal()
    try:
        result = real_cli(command)
        receipt['real_cli_exit'] = result
        return result
    except BaseException as error:
        receipt['real_cli_exception'] = repr(error)
        raise
    finally:
        NativePhysics.call = original
        seal()


if __name__ == '__main__':
    raise SystemExit(main())
