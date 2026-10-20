"""Refuse an executable whose local build record or pinned source changed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, name = Path(sys.argv[1]), sys.argv[2]
source = root / "native" / name
core = "init_atmosphere" if name.endswith("-init_atmosphere") else "atmosphere"
exe = source / (core + "_model")
pins = json.loads((Path(__file__).parent / "REFERENCE-PINS.json").read_text())
commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
expected = pins["upstream"][name.split("-")[0]]
if commit != expected:
    raise SystemExit(f"FAIL pinned source: {commit} != {expected}")
# HEAD includes both staged and unstaged edits. Checking only the working tree
# would admit a modified source file after `git add`.
changed = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True).splitlines()
# The condensate build variant (build_native_reference.sh ... condensate) edits
# Registry.xml through registry_condensate.py and records that edit; it is a
# declared variant, so its Registry.xml diff must equal the recorded one.
REGISTRY = "src/core_atmosphere/Registry.xml"
variant = "-condensate-" in name
allowed = {"Makefile", REGISTRY} if variant else {"Makefile"}
if any(path not in allowed for path in changed):
    raise SystemExit("FAIL native source has changes beyond the recorded build flags")
if variant:
    actual_registry = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", REGISTRY], text=True)
    recorded_path = root / "native" / (name + ".registry.diff")
    recorded_registry = recorded_path.read_text() if recorded_path.exists() else None
    if not actual_registry or actual_registry != recorded_registry:
        raise SystemExit("FAIL native condensate Registry.xml differs from its build receipt")
actual_flags = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", "Makefile"], text=True)
recorded_flags = (root / "native" / (name + ".makefile.diff")).read_text()
if actual_flags != recorded_flags:
    raise SystemExit("FAIL native Makefile differs from its build receipt")
digest = hashlib.sha256(exe.read_bytes()).hexdigest()
recorded = (root / "native" / (name + ".sha256")).read_text().split()[0]
if digest != recorded:
    raise SystemExit("FAIL executable differs from its build receipt")
print(f"Located pinned native {name}: {commit}, executable {digest}")
