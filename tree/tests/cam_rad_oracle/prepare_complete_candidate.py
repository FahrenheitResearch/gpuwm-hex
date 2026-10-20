"""Reproduce the complete CAM qualification candidate from committed code.

The source remains a whole-module transcription. Each transformation is
receipted, and the resulting source is unqualified until the entire native
matrix, captured drivers and whole forecasts have been compared.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import fresh_workspace
import oracle_local_cuda
import parallel_columns
import prune_zero_weight


def prepare(native, output):
    output.mkdir(parents=True, exist_ok=False)
    # The adapters consume a flat, owned copy of the scheme and driver files.
    # The canonical MPAS tree remains read-only.
    if (native / 'src/core_atmosphere/physics').is_dir():
        physics = native / 'src/core_atmosphere/physics'
        native = output / 'native'
        native.mkdir()
        for name in ('module_ra_cam.F', 'module_ra_cam_support.F'):
            shutil.copyfile(physics / 'physics_wrf' / name, native / name)
        for name in ('mpas_atmphys_camrad_init.F',
                     'mpas_atmphys_driver_radiation_lw.F',
                     'mpas_atmphys_driver_radiation_sw.F'):
            shutil.copyfile(physics / name, native / name)
    base = output / 'base.py'
    saved = output / 'saved.py'
    parallel = output / 'parallel.py'
    fixed = output / 'fixed.py'
    complete = output / 'cam_rad_v842.py'
    product = Path(__file__).resolve().parents[2] / 'src/hexcore/mpasphys/cam_rad_v842.py'
    shutil.copyfile(product, base)
    spec = importlib.util.spec_from_file_location('cam_pinned_product', base)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, digest in module.NATIVE_SOURCE_PINS.items():
        if hashlib.sha256((native / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Native CAM source differs from committed pin: ' + name)
    if hasattr(module, 'ORACLE_LOCAL_FIELDS'):
        # The complete implementation now lives in the product. Preserve its
        # exact literal and launcher instead of applying historical adapters
        # a second time; this copy carries no execution or acceptance claim.
        shutil.copyfile(product, complete)
        preparation = 'unchanged complete committed product'
    else:
        oracle_local_cuda.prepare(native, base, saved, output / 'saved.json')
        parallel_columns.prepare(saved, parallel, output / 'parallel.json', 1024)
        fresh_workspace.prepare(parallel, fixed, output / 'fixed.json',
                                optimized=True, fixed_stride=True)
        prune_zero_weight.prepare(fixed, complete, output / 'pruning.json')
        preparation = 'historical whole-module transformations'
    spec = importlib.util.spec_from_file_location('cam_complete_source', complete)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record = dict(
        scope='complete reproducible CAM source candidate; execution unqualified',
        preparation=preparation,
        whole_model_acceptance=False, excluded_words=0,
        committed_product_sha256=hashlib.sha256(product.read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(complete.read_bytes()).hexdigest(),
        cuda_source_sha256=hashlib.sha256(module.CUDA_SOURCE.encode()).hexdigest(),
        compile_options=module.COMPILE_OPTIONS,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.iterdir()) if p.is_file()},
        transformation_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(fresh_workspace.__file__),
                                          Path(oracle_local_cuda.__file__),
                                          Path(parallel_columns.__file__),
                                          Path(prune_zero_weight.__file__))},
        native_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(native.glob('*.F'))})
    (output / 'source.json').write_text(json.dumps(record, indent=2) + '\n')
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.native, args.output)
