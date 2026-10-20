"""Receipt closed native outputs before retiring explicitly disposable streams.

No field transformations: a Rust/native-NetCDF tool reads raw word digests.
MPAS mpas_atm_core.F:827,862 advances/writes before the next :760 banner.
A native Begin timestep beyond an output clock establishes that earlier stream
writes finished; the final pass is allowed only after successful native exit.
"""
import json,re,subprocess
from pathlib import Path
from publish_cache import sha
from verify_products import expected_forecast


class RetireOutputs:
    def __init__(self,directory,runtime,streams,comparator,inactive):
        self.directory=directory;self.comparator=comparator.resolve()
        self.expected=expected_forecast(runtime,streams,inactive)
        self.receipt=directory/'CLOCKS.json';self.files={}
        self.tool_sha256=sha(self.comparator)

    def collect(self,finished=False):
        log=self.directory/'log.atmosphere.0000.out'
        clocks=re.findall(r'Begin timestep (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)',log.read_text()) if log.exists() else []
        current=clocks[-1] if clocks else None
        for file in sorted(self.directory.glob('*.nc')):
            if file.is_symlink():continue
            if file.name not in self.expected:
                if finished:raise ValueError('unexpected native output before retirement')
                continue
            expected=self.expected[file.name]
            if not finished and (not current or current<=max(expected)):continue
            before=sha(file)
            result=subprocess.run([str(self.comparator),'--field-digests',str(file)],capture_output=True,text=True,check=True)
            inventory=json.loads(result.stdout)
            if not inventory['variables']:raise ValueError('empty native output before retirement')
            if any('word_sha256' not in field for field in inventory['variables']):raise ValueError('raw field digest coverage absent')
            times=inventory['xtime']
            if times and times!=expected:raise ValueError('native stored clocks differ before retirement')
            shapes=[field['shape'][field['dimensions'].index('Time')] for field in inventory['variables'] if 'Time' in field['dimensions']]
            if not shapes or any(count!=len(expected) for count in shapes):raise ValueError('native record count differs before retirement')
            if sha(file)!=before:raise ValueError('native output changed while it was being receipted')
            self.files[file.name]={'sha256':before,'bytes':file.stat().st_size,'inventory':inventory,
                                  'expected_clocks':expected,'phase':'completed native stream before retirement'}
            self.write();file.unlink()

    def write(self):
        proof={'schema':'mpas.native-retired-output-clocks/v1','comparator_sha256':self.tool_sha256,
               'complete':set(self.files)==set(self.expected),'files':self.files,
               'retention':'original bytes retired; every field has a raw little-endian C-order SHA256'}
        temporary=self.receipt.with_suffix('.partial');temporary.write_text(json.dumps(proof,indent=2)+'\n');temporary.replace(self.receipt)
