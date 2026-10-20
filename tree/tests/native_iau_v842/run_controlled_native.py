"""Complete one specified strict native control and certify every output clock."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--spec", type=Path, required=True)
args = parser.parse_args()
spec = json.loads(args.spec.read_text())
helpers = Path(__file__).parent
subprocess.run([sys.executable, str(helpers / "prepare_controlled_deck.py"),
                "--spec", str(args.spec)], check=True)
prepared = json.loads((Path(spec["request"]) / "prepared.json").read_text())
argv = [sys.executable, str(helpers / "native_stream_control.py")]
for key, value in (("native", spec["native"]), ("tables", spec["tables"]),
                   ("case", spec["case"]), ("request", spec["request"]),
                   ("increment", prepared["increment"]), ("mpi", spec["mpi"]),
                   ("out", spec["native_output"]), ("receipt", spec["native_receipt"])):
    argv.extend(("--" + key, value))
subprocess.run(argv, check=True)
subprocess.run([sys.executable, str(helpers / "control_clock_inventory.py"),
                "--prepared", str(Path(spec["request"]) / "prepared.json"),
                "--run", spec["native_output"], "--reader", spec["word_reader"],
                "--out", spec["native_clocks"]], check=True)
