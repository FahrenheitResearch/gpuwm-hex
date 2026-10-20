"""One whole-card lease: byte parity then the two-hour ensemble pilot."""
import os
from pathlib import Path
import subprocess,sys
from products import retire
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
V=W/'hex/tree/verification/wofs-ensemble'
subprocess.run([sys.executable,str(V/'parity.py')],check=True)
for path in (W/'parity').rglob('cuda-history.*.nc'): retire(path)
sys.exit(subprocess.call([sys.executable,str(V/'run.py'),'pilot']))
