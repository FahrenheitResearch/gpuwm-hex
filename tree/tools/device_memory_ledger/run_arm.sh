#!/bin/bash
# One ledger arm.  $1 = mesh key (x1|x4), $2 = arm (clean|hook|full).
#
# Every path this script needs comes from the environment and is REFUSED BY
# NAME when it is unset.  THE BREAKAGE THAT CHANGE PREVENTS: this file used
# to hard-code ten absolute paths under one person's home directory, and it
# has been published on a public repository since 0.1.0 -- exposing a home
# directory, a private working-tree layout and a venv path, in a script that
# could not run for anybody else anyway
# (evidence/assembly-rehearsal-20260827/ §6).  A default would have kept the
# leak and hidden the refusal; a named variable does neither.
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

MESH="$1"; ARM="$2"; EXTRA="${3:-}"
REPO="$HEX_REPO"
BASE="$HEX_WORK"
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
OUT=$BASE/out-$MESH-$ARM
CACHE=$BASE/cache-$MESH-$ARM
rm -rf "$OUT" "$CACHE"
exec "$HEX_PYTHON" "$BASE/hex_ledger_probe.py" \
  --repo "$REPO" --mesh "$NAME" --grid "$GRID" --static "$STATIC" --init "$INIT" \
  --init-source "$SRC" --hours 0.2 --history-every-minutes 12 \
  --arwen-checkout "$ARWEN_CHECKOUT" \
  --cache-root "$CACHE" --output "$OUT" \
  --ledger-json "$BASE/ledger-$MESH-$ARM.json" \
  --case-label "hex-ledger-$MESH-$ARM" --arm "$ARM" $EXTRA
