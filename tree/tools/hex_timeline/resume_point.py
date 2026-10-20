"""Resume stock point preparation after a completed mesh leg.

Only the already completed generator invocation is reused. Static generation,
admission, culling and vertical preparation use the original point door.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv):
    from hexcore.cli import build_parser
    from hexcore import mesh_point

    args = build_parser().parse_args(["mesh-plan", *argv])
    request = mesh_point.request_from_arguments(args)
    plan = mesh_point.plan_point(request)
    root = Path(args.out_dir)
    specs = list(root.glob("*.spec.json"))
    if len(specs) != 1 or json.loads(specs[0].read_text()) != plan["spec"]:
        raise ValueError("the completed mesh spec differs; a changed request cannot reuse it")
    grid = specs[0].with_name(specs[0].name.replace(".spec.json", ".grid.nc"))
    receipt = grid.with_name(grid.name.replace(".grid.nc", ".grid.receipt.json"))
    if not grid.is_file() or not receipt.is_file():
        raise ValueError("no completed generator output and receipt to reuse")
    before = digest(grid)
    json.loads(receipt.read_text())
    original = mesh_point._run

    def run(command, *, what, log=None):
        if what == "rw_mpas_mesh" and "--out" in command:
            output = Path(command[command.index("--out") + 1])
            if output == grid:
                if digest(grid) != before:
                    raise ValueError("the completed grid changed during preparation")
                print(f"REUSE completed grid sha256={before}", flush=True)
                result = subprocess.CompletedProcess(command, 0, "completed grid reused", "")
                result.elapsed = 0.0
                return result
        return original(command, what=what, log=log)

    mesh_point._run = run
    try:
        result = mesh_point.generate_point(request, plan, out_dir=root, geog=args.geog,
            vertical_spec=args.vertical_spec, clobber=True, static_exe=args.static_exe)
    finally:
        mesh_point._run = original
    if digest(grid) != before:
        raise ValueError("resumed preparation changed the completed grid")
    print(json.dumps({"grid_sha256": before, "grid_reused": True, "rows_file": result["rows_file"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
