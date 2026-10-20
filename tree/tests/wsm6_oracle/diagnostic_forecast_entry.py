"""Explicit, unaccepted whole-forecast measurement with shared diagnostic math.

Only an owned diagnostic checkout selects this entry. It injects an explicitly
named header into the existing SlotCall option supported by the production
launcher. It does not install a factory, change a native option, certify math,
or provide a fallback. The real forecast verifier still compares every output.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path


def invoke(call):
    from hexcore.mpasphys.wsm6_v842 import cuda_entry

    header_path = Path(os.environ['WSM6_DIAGNOSTIC_SHARED_HEADER']).resolve(strict=True)
    header = header_path.read_text()
    manifest = json.loads(header_path.with_suffix('.json').read_text())
    if manifest['acceptance'] or manifest['header_sha256'] != hashlib.sha256(header.encode()).hexdigest():
        raise ValueError('explicit unaccepted diagnostic math manifest differs from header')
    return cuda_entry(replace(call, options=dict(call.options, wsm6_device_math_header=header)))
