"""Execute the supplied checkout's real forecast after queue admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

case = json.loads(Path(sys.argv[1]).read_text())
checkout = Path(case['checkout'])
pins = json.loads(Path(case['source_pins']).read_text())
for relative, expected in pins.items():
    if hashlib.sha256((checkout / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('Supplied forecast checkout changed while queued')
verifier = Path(case['provider_verifier'])
if hashlib.sha256(verifier.read_bytes()).hexdigest() != case['provider_verifier_sha256']:
    raise ValueError('Native provider recipe verifier changed while queued')
spec = importlib.util.spec_from_file_location('ysu_admitted_provider_verifier', verifier)
provider_verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider_verifier)
provider_receipt = provider_verifier.verify(case['provider_index'], checkout)
wait_start = time.time()
while True:
    available = next(int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()
                     if row.startswith('MemAvailable:'))
    # Inherited admission can span several decks. Recheck the official
    # queue's 2000 MiB card minimum before each one: case10 of the final
    # twelve-deck attempt exhausted device memory during initial allocation.
    device_available = int(subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
    ], text=True).splitlines()[0])
    if available >= 16 * 1024 * 1024 and device_available >= 2000:
        break
    time.sleep(20)
start = time.time()
model = subprocess.Popen(case['command'])
model_pid = model.pid
model_exit = model.wait()
Path(case['execution']).write_text(json.dumps({
    'command': case['command'], 'model_door_invoked': True,
    'model_door_exit': model_exit, 'model_pid': model_pid, 'process_count': 1,
    'model_elapsed_seconds': time.time() - start,
    'memory_wait_seconds': start - wait_start, 'host_available_kib_before_door': available,
    'device_available_mib_before_door': device_available,
    'model_start_unix_seconds': start,
    'source_pins_sha256': hashlib.sha256(Path(case['source_pins']).read_bytes()).hexdigest(),
    'native_provider_authentication': provider_receipt,
    'PYTHONPATH': os.environ.get('PYTHONPATH'),
}, indent=2) + '\n')
raise SystemExit(model_exit)
