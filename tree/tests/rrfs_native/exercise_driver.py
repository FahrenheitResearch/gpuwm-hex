"""Execute the complete additive driver on a published initialized input.

This is execution evidence, not a native identity gate. The input decoder and
reconstruction are the existing production loaders; no new file data path is
implemented here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--init", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=30.)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--namelist", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    from hexcore.mesh import Mesh
    from hexcore.vector import initialize_reconstruction_coefficients
    from hexcore.driver import load_mpas_vertical_grid, load_mpas_initial_state
    from hexcore.config_rrfs import RRFSDryDycoreConfig
    from hexcore.cuda_driver_rrfs import CudaRRFSDycoreDriver
    from hexcore.cuda_compile_rrfs import RRFSKernelCache

    receipt = dict(scope="WHOLE DRIVER EXECUTION ONLY; no matched native comparison",
                   identity_checked=False, differing_words=None,
                   host=socket.gethostname(), process_count=1,
                   steps_requested=2*args.steps, steps_per_case=args.steps, steps_completed=0,
                   input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.grid, args.init)}, cases=[])
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        options = {}
        if args.registry is not None:
            import xml.etree.ElementTree as ET
            from native_source import read_pinned_source
            source = args.registry.resolve().parent.parent.parent
            registry_text = read_pinned_source(source, args.registry.resolve())
            definitions = ET.fromstring(registry_text).iter('nml_option')
            names = {field.name.lower():field.name for field in fields(RRFSDryDycoreConfig)}
            converters = {'real':float,'integer':int,'character':str,
                          'logical':lambda text:text.strip().lower() in ('true','.true.')}
            for definition in definitions:
                name = definition.attrib['name'].lower()
                if name in names:
                    options[names[name]] = converters[definition.attrib['type']](definition.attrib['default_value'])
            receipt['registry_sha256'] = hashlib.sha256(registry_text.encode()).hexdigest()
        if args.namelist is not None:
            from hexcore.config import parse_fortran_namelist
            names = {field.name.lower():field.name for field in fields(RRFSDryDycoreConfig)}
            for group in parse_fortran_namelist(args.namelist.read_text()).values():
                for name,value in group.items():
                    if name.lower() in names:
                        options[names[name.lower()]] = value
            receipt['namelist_sha256'] = hashlib.sha256(args.namelist.read_bytes()).hexdigest()
        mesh = Mesh.from_netcdf(args.grid, args.init)
        mesh.arrays['coeffs_reconstruct'] = initialize_reconstruction_coefficients(mesh, include_halos=True)
        # Retain the actual initialized deformation pool, just as the fork
        # atmosphere does (mpas_atm_core.F:373-614 at 1d6e370e). Rebuilding or
        # zeroing inherited regional weights changes valid native input.
        grid = load_mpas_vertical_grid(args.init, mesh)
        for split_transport in (True, False):
            state, reference, saved = load_mpas_initial_state(
                args.init, mesh, grid.vertical_grid, reference_path=args.init,
                scalar_names=('qv',),
                terrain_metrics=grid.terrain_metrics, return_saved_diagnostics=True,
                cold_start=True)
            config = RRFSDryDycoreConfig(**dict(options,config_dt=args.dt,
                                        config_split_dynamics_transport=split_transport))
            cache = RRFSKernelCache()
            driver = CudaRRFSDycoreDriver.from_host(
                mesh, state, grid.vertical_grid, reference, config,
                saved_diagnostics=saved, terrain_metrics=grid.terrain_metrics,
                kernel_cache=cache)
            case = dict(split_transport=split_transport, config=asdict(config),
                        steps_completed=0)
            receipt['cases'].append(case)
            for step in range(args.steps):
                result = driver.step_device()
                driver.atmosphere = result.atmosphere
                case['steps_completed'] = step + 1
                receipt['steps_completed'] += 1
            driver.cp.cuda.get_current_stream().synchronize()
            case['final_state_health'] = {
                name: dict(words=int(getattr(driver.atmosphere.state, name).size),
                           nonfinite_words=int(driver.cp.count_nonzero(
                               ~driver.cp.isfinite(getattr(driver.atmosphere.state, name))).item()))
                for name in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars')
            }
            case['compile_manifest'] = cache.compile_manifest()
        receipt['completed'] = True
    except Exception as error:
        receipt.update(completed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['hex_commit'] = subprocess.check_output(
            ['git', '-C', str(Path(__file__).parents[3]), 'rev-parse', 'HEAD'], text=True).strip()
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
