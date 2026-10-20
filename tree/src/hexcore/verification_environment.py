"""Strict landing environment, captured by the forecast's Python installation.

The deck's strict_environment is a hash-bound capture from the before arm.
Comparison-only output must carry its as-run before/after capture; inspecting
an installation after the fact cannot authenticate an old forecast.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess


SCHEMA = 'mpas-strict-environment-v1'
KEYS = ('engine_version', 'engine_git_sha', 'engine_record_sha256', 'engine_wheel_sha256',
        'math_headers', 'cuda_toolkit', 'cuda_runtime', 'nvrtc',
        'cupy_version', 'driver', 'gpu_uuid', 'numerics')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(numerics, command=()):
    """Read the same engine override/import search path as the model command.

    Pin every external .cuh, a conservative superset of the headers inlined
    by the hex kernels. Paths are provenance, relative names and bytes gate.
    No CUDA kernel is compiled or launched here.
    """
    record = dict(schema=SCHEMA, pins={'numerics': numerics}, errors=[])
    pins, errors = record['pins'], record['errors']
    try:
        from hexcore import engine_pin
        dist = importlib.metadata.distribution(engine_pin.DISTRIBUTION)
        pins['engine_version'] = dist.version
        installed_record = dist.read_text('RECORD')
        pins['engine_record_sha256'] = (hashlib.sha256(installed_record.encode()).hexdigest()
                                         if installed_record else None)
        spec = importlib.util.find_spec('gpuwm')
        root = Path(next(iter(spec.submodule_search_locations))).parent
        overrides = [i for i, value in enumerate(command) if value == '--gpuwm-checkout'
                     or value.startswith('--gpuwm-checkout=')]
        if len(overrides) > 1:
            raise ValueError('duplicate --gpuwm-checkout')
        if overrides:
            i = overrides[0]
            root = Path(command[i].split('=', 1)[1] if '=' in command[i] else command[i+1]).resolve()
            pins['engine_version'] = engine_pin.checkout_version(root)
            # The source override is the engine in use; pip RECORD describes
            # a different installation and must not impersonate its bytes.
            pins['engine_record_sha256'] = None
        kernels = root / 'gpuwm/core/kernels'
        pins['math_headers'] = {str(p.relative_to(kernels)): sha(p)
                                for p in sorted(kernels.rglob('*.cuh'))}
        record['engine_root'] = str(root.resolve())
        pins['engine_git_sha'] = None
        pins['engine_wheel_sha256'] = None
        # A wheel's source revision is accepted only from installed metadata,
        # never from an unrelated enclosing checkout or an environment string.
        direct = dist.read_text('direct_url.json')
        if direct and not overrides:
            metadata = json.loads(direct)
            pins['engine_git_sha'] = metadata.get('vcs_info', {}).get('commit_id')
            pins['engine_wheel_sha256'] = metadata.get('archive_info', {}).get('hashes', {}).get('sha256')
        if (root / '.git').exists():
            git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                 capture_output=True, text=True, check=True)
            pins['engine_git_sha'] = git.stdout.strip()
            dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                                    '--untracked-files=no'], capture_output=True, text=True, check=True)
            if dirty.stdout.strip():
                errors.append('engine checkout has tracked modifications')
    except Exception as error:
        errors.append('engine capture: ' + str(error))
    try:
        import cupy as cp
        from cupy.cuda import nvrtc, runtime
        pins['cupy_version'] = cp.__version__
        pins['cuda_runtime'] = runtime.runtimeGetVersion()
        pins['nvrtc'] = list(nvrtc.getVersion())
        device = runtime.getDevice()
        bus = cp.cuda.Device(device).pci_bus_id
        info = subprocess.run(['nvidia-smi', '--id=' + bus,
                               '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        uuid, driver = info.stdout.strip().split(',')
        pins['gpu_uuid'], pins['driver'] = uuid.strip(), driver.strip()
        try:
            nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            pins['cuda_toolkit'] = nvcc.stdout.strip()
        except FileNotFoundError:
            # CUDA Python wheels ship an NVRTC toolkit without the nvcc CLI.
            # Pin the installed toolkit and compiler/runtime package versions;
            # the loaded NVRTC and runtime API versions are pinned above too.
            pins['cuda_toolkit'] = {name: importlib.metadata.version(name) for name in
                ('cuda-toolkit', 'nvidia-cuda-nvrtc', 'nvidia-cuda-runtime')}
    except Exception as error:
        errors.append('CUDA capture: ' + str(error))
    return record


def problems(record):
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        return ['missing as-run strict environment']
    pins = record.get('pins', {})
    errors = list(record.get('errors', []))
    for key in KEYS:
        if not pins.get(key) and key not in ('engine_git_sha', 'engine_record_sha256', 'engine_wheel_sha256'):
            errors.append('missing environment pin: ' + key)
    if not re.fullmatch(r'[0-9a-f]{40}', str(pins.get('engine_git_sha', ''))):
        # An installed archive has no checkout commit. Bind both its archive
        # bytes and installed RECORD instead; capture never uses these for an override.
        if not re.fullmatch(r'[0-9a-f]{64}', str(pins.get('engine_wheel_sha256', ''))):
            errors.append('engine identity needs an authenticated source git SHA or installed wheel SHA256')
        if not re.fullmatch(r'[0-9a-f]{64}', str(pins.get('engine_record_sha256', ''))):
            errors.append('installed engine RECORD SHA256 is missing or invalid')
    headers = pins.get('math_headers', {})
    if not isinstance(headers, dict) or not headers or any(
            not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in headers.values()):
        errors.append('external math header hashes are incomplete')
    if pins.get('numerics') != 'strict':
        errors.append('strict landing requires numerics=strict')
    return errors


def judge(expected, before, after):
    """Unmatched or incomplete environments are ENVIRONMENT, never regression."""
    failures = []
    for label, record in (('reference', expected), ('before', before), ('after', after)):
        failures.extend(label + ': ' + error for error in problems(record))
    if not failures:
        for label, record in (('before', before), ('after', after)):
            for key in KEYS:
                if record['pins'].get(key) != expected['pins'].get(key):
                    failures.append(label + ': environment differs: ' + key)
    return dict(verdict='ENVIRONMENT' if failures else 'PINNED', failures=failures,
                reference=expected, before=before, after=after)
