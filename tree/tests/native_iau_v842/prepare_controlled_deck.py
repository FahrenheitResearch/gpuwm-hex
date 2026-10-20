"""Prepare a specified complete controlled experiment from frozen f0 metadata.

Only duration and IAU activation/window change in the original namelist. Rust
creates actual increment fields. Independent analysis provenance is never claimed.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--spec", type=Path, required=True)
args = parser.parse_args()
spec = json.loads(args.spec.read_text())
out = Path(spec["request"])
out.mkdir(parents=True, exist_ok=False)
original = Path(spec["original"])
manifest = json.loads((original / "MANIFEST.json").read_text())
for filename in ("namelist.atmosphere", "streams.atmosphere"):
    path = original / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["inputs"][filename]
seconds = spec["duration_seconds"]
assert seconds > 0 and seconds % 3600 == 0
days, remain = divmod(seconds, 86400)
duration = f"{days}_{remain // 3600:02}:00:00"
text = (original / "namelist.atmosphere").read_text()
text, count = re.subn(r"(?im)^(\s*config_run_duration\s*=\s*)'[^']*'",
                      lambda match: match[1] + repr(duration), text)
assert count == 1
assert not re.search(r"(?i)config_IAU_option\s*=", text)
text += f"\n&IAU\n config_IAU_option = 'on'\n config_IAU_window_length_s = {spec['window_seconds']}\n/\n"
(out / "namelist.atmosphere").write_text(text)
shutil.copy2(original / "streams.atmosphere", out / "streams.atmosphere")
increment = out / "increments.nc"
start = manifest["runtime"]["start"]
begin = datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
earlier = (begin - timedelta(seconds=spec["window_seconds"])).strftime("%Y-%m-%d_%H:%M:%S")
argv = [spec["fixture"], spec["init"], str(increment.resolve()), earlier]
subprocess.run(argv, check=True)
end = begin + timedelta(seconds=seconds)
expected = {}
for filename in manifest["outputs"]:
    _, stamp = filename.split(".", 1)
    at = datetime.strptime(stamp.removesuffix(".nc"), "%Y-%m-%d_%H.%M.%S")
    if begin <= at <= end:
        expected[filename] = [at.strftime("%Y-%m-%d_%H:%M:%S")]
assert expected
receipt = dict(schema="mpas.iau-controlled-deck-preparation/v1", acceptance=False,
               scope="controlled inputs, copied owner proposal, diagnostic only",
               start=start, stop=end.strftime("%Y-%m-%d_%H:%M:%S"),
               dt=manifest["runtime"]["dt"], duration_seconds=seconds,
               window_seconds=spec["window_seconds"], expected_clocks=expected,
               increment=str(increment.resolve()), fixture_argv=argv,
               input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (*original.iterdir(), out / "namelist.atmosphere",
                                out / "streams.atmosphere", increment, Path(spec["fixture"]))
                   if path.is_file()})
(out / "prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "expected_clocks"}))
