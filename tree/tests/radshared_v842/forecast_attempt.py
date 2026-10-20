"""Run the real 24-hour door and record a failed attempt without a parity claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--request',type=Path,required=True)
p.add_argument('--receipt',type=Path,required=True)
p.add_argument('--log',type=Path,required=True)
a=p.parse_args()
request=json.loads(a.request.read_text())
argv=list(request['argv']);argv.remove('--preflight')
# Use the backend option supported by the merged native door. The native
# namelist file remains unchanged; the receipt records the explicit overlay.
with a.log.open('w') as output:
    run=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT)
result={'purpose':'real mixed-backend 24h forecast attempt on current artifact',
        'process_count':1,'argv':argv,'returncode':run.returncode,'forecast_identity_claim':False,
        'native_reference':request['native_reference'],'namelist_sha256':request['native_input_sha256'],
        'whole_model_word_diff':None,'log_sha256':hashlib.sha256(a.log.read_bytes()).hexdigest(),
        'hex_tip':subprocess.check_output(['git','-C',str(Path(__file__).parents[3]),'rev-parse','HEAD'],text=True).strip(),
        'pending':'successful complete execution and every-stream comparison are required'}
a.receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(run.returncode or 1)
