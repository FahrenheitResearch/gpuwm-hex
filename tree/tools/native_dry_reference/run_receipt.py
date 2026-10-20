"""Bind native init/run bytes to their producers and validate the standard case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(directory: Path, expected_frames: int | None = None) -> dict:
    frames = sorted(directory.glob('tracer.*.nc'))
    if not frames or (expected_frames is not None and len(frames) != expected_frames):
        raise ValueError(f'incomplete native reference: {len(frames)} frames; expected {expected_frames or "at least 1"}')
    log = directory / 'log.atmosphere.0000.out'
    if 'Finished running the atmosphere core' not in log.read_text():
        raise ValueError('native completion message absent')
    inputs = list(directory.glob('*.init.nc'))
    if len(inputs) != 1:
        raise ValueError(f'expected one native initial file, found {len(inputs)}')
    paths = [directory / name for name in ('atmosphere_model', 'namelist.atmosphere', 'streams.atmosphere')]
    paths += inputs + frames
    return {'schema': 'gpuwm-hex.native-run-receipt/v1', 'frames': len(frames),
            'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in paths}}


def collect_init(directory: Path) -> dict:
    inputs = list(directory.glob('*.grid.nc'))
    outputs = list(directory.glob('*.init.nc'))
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('expected one initialization grid and one init output')
    paths = [directory / name for name in ('init_atmosphere_model',
             'namelist.init_atmosphere', 'streams.init_atmosphere')] + inputs + outputs
    if any(path.stat().st_size == 0 for path in paths):
        raise ValueError('empty native initialization input or output')
    return {'schema': 'gpuwm-hex.native-init-receipt/v1',
            'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in paths}}


def standard_namelist(directory: Path, order: int | None, substeps: int | None) -> None:
    from hexcore.config import parse_fortran_namelist
    from hexcore.errors import ConfigurationRefusal

    init = order is None
    common = {'io': {'config_pio_num_iotasks': 0, 'config_pio_stride': 1},
              'decomposition': {'config_block_decomp_file_prefix': 'x1.2562.graph.info.part.'}}
    if init:
        expected = dict(common, nhyd_model={
            'config_init_case': 2, 'config_start_time': '2000-01-01_00:00:00',
            'config_stop_time': '2000-01-01_00:00:00', 'config_theta_adv_order': 3,
            'config_coef_3rd_order': 0.25},
            dimensions={'config_nvertlevels': 26, 'config_nsoillevels': 4,
                        'config_nfglevels': 38, 'config_nfgsoillevels': 4},
            data_sources={'config_fg_interval': 21600, 'config_use_spechumd': False},
            vertical_grid={'config_ztop': 30000.0, 'config_nsmterrain': 1,
                'config_smooth_surfaces': True, 'config_dzmin': 0.3, 'config_nsm': 30,
                'config_tc_vertical_grid': True, 'config_blend_bdy_terrain': False},
            preproc_stages={'config_static_interp': False, 'config_native_gwd_static': False,
                'config_vertical_grid': False, 'config_met_interp': False,
                'config_input_sst': False, 'config_frac_seaice': False})
    else:
        if (order, substeps) not in ((2, 4), (3, 6), (2, 2)):
            raise ValueError('unsupported standard reference schedule')
        expected = dict(common, nhyd_model={
            'config_time_integration_order': order, 'config_dt': 1800.0,
            'config_start_time': '2000-01-01_00:00:00', 'config_run_duration': '0_00:30:00',
            'config_split_dynamics_transport': True, 'config_number_of_sub_steps': substeps,
            'config_dynamics_split_steps': 3, 'config_horiz_mixing': '2d_fixed',
            'config_len_disp': 0.0, 'config_visc4_2dsmag': 0.05,
            'config_smagorinsky_coef': 0.125, 'config_del4u_div_factor': 10.0,
            'config_h_scalewithmesh': True, 'config_h_theta_eddy_visc2': 0.0,
            'config_v_theta_eddy_visc2': 0.0, 'config_h_mom_eddy_visc2': 0.0,
            'config_v_mom_eddy_visc2': 0.0, 'config_h_theta_eddy_visc4': 0.0,
            'config_h_mom_eddy_visc4': 0.0, 'config_scalar_advection': True,
            'config_monotonic': True, 'config_positive_definite': False,
            'config_scalar_adv_order': 3, 'config_scalar_vadv_order': 3,
            'config_coef_3rd_order': 0.25, 'config_apvm_upwinding': 0.5, 'config_smdiv': 0.0},
            damping={'config_zd': 22000.0, 'config_xnutr': 0.0},
            limited_area={'config_apply_lbcs': False}, restart={'config_do_restart': False},
            printout={'config_print_global_minmax_vel': True, 'config_print_detailed_minmax_vel': False},
            iau={'config_iau_option': 'off'}, physics={'config_sst_update': False,
                'config_sstdiurn_update': False, 'config_deepsoiltemp_update': False,
                'config_bucket_update': 'none', 'config_physics_suite': 'none'},
            soundings={'config_sounding_interval': 'none'})
    name = 'init_atmosphere' if init else 'atmosphere'
    try:
        parsed = parse_fortran_namelist((directory / ('namelist.' + name)).read_text())
        actual = {group.lower(): {key.lower(): value for key, value in options.items()}
                  for group, options in parsed.items()}
    except ConfigurationRefusal as error:
        raise ValueError('invalid standard namelist: ' + str(error)) from error
    if actual != expected:
        differences = [f'{group}.{key}' for group in set(actual) | set(expected)
                       for key in set(actual.get(group, {})) | set(expected.get(group, {}))
                       if actual.get(group, {}).get(key) != expected.get(group, {}).get(key)]
        raise ValueError('cached case is not the standard dry JW case: ' + ', '.join(sorted(differences)))
    streams = ET.parse(directory / ('streams.' + name)).getroot()
    stream = streams.find("immutable_stream[@name='input']")
    if stream is None or stream.get('filename_template') != ('x1.2562.grid.nc' if init else 'x1.2562.init.nc'):
        raise ValueError('standard reference input stream mismatch')
    stream = streams.find("immutable_stream[@name='output']" if init else "stream[@name='tracer_oracle']")
    wanted = {'filename_template': 'x1.2562.init.nc', 'output_interval': 'initial_only'} if init else {
        'filename_template': 'tracer.$Y-$M-$D_$h.$m.$s.nc', 'output_interval': '0_00:30:00'}
    if stream is None or any(stream.get(key) != value for key, value in wanted.items()):
        raise ValueError('standard reference output stream mismatch')
    if init:
        pin = json.loads(Path(__file__).with_name('REFERENCE-PINS.json').read_text())['grid_sha256_file']
        if sha256(directory / 'x1.2562.grid.nc') != pin:
            raise ValueError('standard public mesh checksum mismatch')
    elif {p.name for p in directory.glob('tracer.*.nc')} != {
            'tracer.2000-01-01_00.00.00.nc', 'tracer.2000-01-01_00.30.00.nc'}:
        raise ValueError('standard reference output times mismatch')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('write', 'verify', 'write-init', 'verify-init'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-frames', type=int)
    parser.add_argument('--standard-init', action='store_true')
    parser.add_argument('--standard-order', type=int)
    parser.add_argument('--standard-substeps', type=int)
    parser.add_argument('--expected-executable', type=Path)
    parser.add_argument('--expected-init', type=Path)
    args = parser.parse_args(argv)
    try:
        init = args.mode.endswith('-init')
        actual = collect_init(args.directory) if init else collect(args.directory, args.expected_frames)
        path = args.directory / ('init-receipt.json' if init else 'run-receipt.json')
        if args.mode.startswith('write'):
            path.write_text(json.dumps(actual, indent=2) + '\n')
        elif actual != json.loads(path.read_text()):
            raise ValueError('native run bytes differ from their receipt')
        if args.standard_init or args.standard_order is not None:
            if args.standard_init != init or (not init and args.standard_substeps is None):
                raise ValueError('standard case validation arguments mismatch')
            standard_namelist(args.directory, args.standard_order, args.standard_substeps)
        for supplied, name in ((args.expected_executable, 'init_atmosphere_model' if init else 'atmosphere_model'),
                               (args.expected_init, 'x1.2562.init.nc')):
            if supplied is not None and actual['files'].get(name, {}).get('sha256') != sha256(supplied):
                raise ValueError('cached reference differs from the selected ' + name)
    except (OSError, ValueError, ET.ParseError) as error:
        print(f'FAIL cached native run {args.directory}: {error}. Use a fresh --work-root to rebuild.')
        return 2
    detail = 'initialization' if init else str(actual['frames']) + ' frames'
    print(f'Native receipt {args.mode}: {args.directory}, {detail}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
