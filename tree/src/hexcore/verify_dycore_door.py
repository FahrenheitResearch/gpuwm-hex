"""Rebuildable native reference door; model execution stays on Linux."""
from pathlib import Path
import subprocess
import sys


def _run(args):
    tree = Path(__file__).resolve().parents[2]
    script = tree / "tools/native_dry_reference/check_all.sh"
    if not script.is_file():
        print("verify-dycore requires the source distribution, including tools/native_dry_reference/", file=sys.stderr)
        return 2
    return subprocess.run(["bash", str(script), str(args.work_root.resolve()), str(tree),
                           str(args.jobs), args.backend, args.compiler], check=False).returncode


def add_verify_dycore_parser(commands):
    parser = commands.add_parser("verify-dycore", help="build or locate pinned native MPAS and check one dry step")
    parser.add_argument("--work-root", type=Path, default=Path(".native-dry-reference"))
    parser.add_argument("--jobs", type=int, choices=range(1,33), default=8)
    parser.add_argument("--backend", choices=("cpu","cuda","both"), default="cuda")
    parser.add_argument("--compiler", choices=("gnu","ifx"), default="gnu",
                        help="GNU with MPI/PnetCDF, or Intel oneAPI; records the actual toolchain")
    parser.set_defaults(handler=_run)
