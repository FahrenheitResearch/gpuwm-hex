#!/bin/bash
# The kernel-reservation arm.  $1 = mesh key (x1|x4), $2 = arm label.
#
# Same environment contract as run_arm.sh beside it, and for the same
# reason: this file hard-coded ten absolute paths under one person's home
# directory and has been public since 0.1.0
# (evidence/assembly-rehearsal-20260827/ §6).  Unset means refused by name,
# never a default that silently points somewhere wrong.
#
#   HEX_REPO        the gpuwm-hex checkout's tree/ directory
#   HEX_WORK        a scratch directory for this campaign's outputs
#   HEX_ASSETS      a directory holding the four mesh/static/init files below
#   HEX_PYTHON      the interpreter with cupy and the port's dependencies
#   ARWEN_CHECKOUT  the gpuwm SOURCE checkout at the pinned commit
set -u
: "${HEX_REPO:?set HEX_REPO to the gpuwm-hex checkout's tree/ directory}"
: "${HEX_WORK:?set HEX_WORK to a scratch directory for this campaign}"
: "${HEX_ASSETS:?set HEX_ASSETS to the directory holding the mesh, static and init files}"
: "${HEX_PYTHON:?set HEX_PYTHON to the interpreter that carries cupy}"
: "${ARWEN_CHECKOUT:?set ARWEN_CHECKOUT to the gpuwm source checkout at the pinned commit}"

MESH="$1"; ARM2="${2:-kern}"
BASE="$HEX_WORK"
REPO="$HEX_REPO"
if [ "$MESH" = "x1" ]; then
  NAME=x1.40962
  GRID="$HEX_ASSETS/x1.40962.grid.nc"
  STATIC="$HEX_ASSETS/x1.40962.static.nc"
  INIT="$HEX_ASSETS/x1.40962.init.x1-gfs-20260812-06z.nc"
  SRC="GFS 2026-08-12 06Z on the published MPAS-Dev x1.40962 120 km mesh"
else
  NAME=x4.163842
  GRID="$HEX_ASSETS/x4.163842.grid.nc"
  STATIC="$HEX_ASSETS/x4.163842.static.nc"
  INIT="$HEX_ASSETS/x4.163842.init.gfs-20260812-06z.nc"
  SRC="GFS 2026-08-12 06Z on the x4.163842 25 km mesh"
fi
OUT=$BASE/out-$MESH-$ARM2
CACHE=$BASE/cache-$MESH-$ARM2
rm -rf "$OUT" "$CACHE"
exec "$HEX_PYTHON" "$BASE/hex_kernel_probe.py" \
  --kernel-json "$BASE/kernels-$MESH-$ARM2.json" \
  --repo "$REPO" --mesh "$NAME" --grid "$GRID" --static "$STATIC" --init "$INIT" \
  --init-source "$SRC" --hours 0.2 --history-every-minutes 12 \
  --arwen-checkout "$ARWEN_CHECKOUT" \
  --cache-root "$CACHE" --output "$OUT" \
  --ledger-json "$BASE/ledger-$MESH-$ARM2.json" \
  --case-label "hex-ledger-$MESH-$ARM2" --arm full --seam-trace
