#!/usr/bin/env python
"""Replay every LW call in one admitted f1 single-site capture job.

MPAS: mpas_atmphys_driver_radiation_lw.F:865-884. This launches the complete
column-proven module; it never reads expected POST as a runtime input.
"""
import json
from pathlib import Path
import subprocess
import sys

dump, output = map(Path, sys.argv[1:3])
manifest = json.loads((dump / 'manifest.json').read_text())
output.mkdir(exist_ok=False)
worker = Path(__file__).with_name('f1_cuda_replay.py')
for prefix in manifest['captures']:
    subprocess.run([sys.executable, str(worker), str(dump / prefix), str(output / prefix)], check=True)
