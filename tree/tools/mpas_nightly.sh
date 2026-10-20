#!/usr/bin/env bash
# G3: the native reference rebuilds from this tree's recipe, and hex still matches it.
#
# THE BREAKAGE IT PREVENTS (release gate g; the project rule that "a claim whose
# reference cannot be rebuilt does not exist"): G1 and G2 compare hex with outputs one
# strict native binary wrote once.  G3 rebuilds that binary and its Thompson
# tables from tools/native_physics_reference/build_reference.sh and the pinned
# MPAS-Model source, reruns every deck row of the G3 index on it, and requires
# every field word of every output to equal the pinned reference
# (native_physics_reference/g3_rebuild.py).  It then runs G1 and G2 on this tree
# (hex against those same pinned references), so hex equals the rebuilt
# reference by the same words.
#
# Usage:
#   mpas_nightly.sh --g3-index INDEX --g3-work DIR --native-comparator BIN --out DIR \
#                   [mpas_gate.sh options, e.g. --index/--ratchet/--comparator]
# The build, run and compare steps can also run on separate hosts with
# g3_rebuild.py build|run|compare (the compare needs the pinned references).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
index=""; work=""; native=""; out=""; rest=()
while [ $# -gt 0 ]; do
  case "$1" in
    --g3-index) index=$2; shift 2;;
    --g3-work) work=$2; shift 2;;
    --native-comparator) native=$2; shift 2;;
    --out) out=$2; shift 2;;
    *) rest+=("$1"); shift;;
  esac
done
[ -n "$index" ] && [ -n "$work" ] && [ -n "$native" ] && [ -n "$out" ] || {
  echo "G3 needs --g3-index, --g3-work, --native-comparator and --out; an unnamed reference cannot be rebuilt" >&2; exit 2; }
mkdir -p "$out"
g3="$here/native_physics_reference/g3_rebuild.py"
"${PYTHON:-python3}" "$g3" build --index "$index" --work "$work"
"${PYTHON:-python3}" "$g3" run --index "$index" --work "$work"
"${PYTHON:-python3}" "$g3" compare --index "$index" --work "$work" --comparator "$native" --receipt "$out/G3-rebuild.json"
exec bash "$here/mpas_gate.sh" --gates G1,G2 --out "$out/G1-G2" "${rest[@]}"
