"""Check complete controlled native clocks against original published schedules.

Rust reads every native variable and hashes its stored words. Python deals only
with metadata. The diagnostic retains all missing/extra clocks as failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
for name in ("prepared", "run", "reader", "out"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
expected = json.loads(args.prepared.read_text())["expected_clocks"]
actual = {path.name: path for path in args.run.glob("*.nc") if not path.is_symlink()}
assert set(actual) == set(expected), dict(missing=sorted(set(expected) - set(actual)),
                                         extra=sorted(set(actual) - set(expected)))
records = {}
for filename, path in sorted(actual.items()):
    inventory = json.loads(subprocess.check_output(
        [str(args.reader), "--field-digests", str(path)], text=True))
    if inventory["xtime"]:
        assert inventory["xtime"] == expected[filename], filename
        clock_source = "native xtime"
    else:
        # Some native streams contain no xtime variable. Their authenticated
        # published filename supplies the clock; require the actual record count.
        assert not any(row["field"] == "xtime" for row in inventory["variables"]), filename
        timed = [row for row in inventory["variables"] if "Time" in row["dimensions"]]
        assert timed and all(row["shape"][row["dimensions"].index("Time")] ==
                             len(expected[filename]) for row in timed), filename
        clock_source = "authenticated published filename and native Time count"
    records[filename] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             bytes=path.stat().st_size, inventory=inventory,
                             expected_clocks=expected[filename], clock_source=clock_source)
receipt = dict(schema="mpas.iau-controlled-native-clocks/v1", acceptance=False,
               complete=True, prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
               comparator_sha256=hashlib.sha256(args.reader.read_bytes()).hexdigest(), files=records)
encoded = json.dumps(receipt, indent=2) + "\n"
assert len(encoded.encode()) < 10_000_000
args.out.write_text(encoded)
print(json.dumps(dict(files=len(records), complete=True)))
