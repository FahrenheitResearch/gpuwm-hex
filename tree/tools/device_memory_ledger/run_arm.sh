#!/bin/bash
# $1 = mesh key (x1|x4), $2 = arm (clean|hook|full)
set -u
MESH="$1"; ARM="$2"; EXTRA="${3:-}"
REPO=/home/drew/gpuwm-work/mpas-port-mg/tree
BASE=/home/drew/gpuwm-work/hexledger-20260820
if [ "$MESH" = "x1" ]; then
  NAME=x1.40962
  GRID=/home/drew/arwen-durable/2026-08-14/mesh-x1.40962/x1.40962.grid.nc
  STATIC=/home/drew/gpuwm-work/mpasinit/assets/x1.40962.static.nc
  INIT=/home/drew/gpuwm-work/mpasinit/out/x1.40962.init.x1-gfs-20260812-06z.nc
  SRC="GFS 2026-08-12 06Z on the published MPAS-Dev x1.40962 120 km mesh"
else
  NAME=x4.163842
  GRID=/home/drew/gpuwm-work/mpasinit/out/x4.163842.grid.nc
  STATIC=/home/drew/gpuwm-work/mpasinit/assets/x4.163842.static.nc
  INIT=/home/drew/gpuwm-work/mpasinit/out/x4.163842.init.gfs-20260812-06z.nc
  SRC="GFS 2026-08-12 06Z on the x4.163842 25 km mesh"
fi
OUT=$BASE/out-$MESH-$ARM
CACHE=$BASE/cache-$MESH-$ARM
rm -rf "$OUT" "$CACHE"
exec /home/drew/arwen-gpu/venv/bin/python "$BASE/hex_ledger_probe.py" \
  --repo "$REPO" --mesh "$NAME" --grid "$GRID" --static "$STATIC" --init "$INIT" \
  --init-source "$SRC" --hours 0.2 --history-every-minutes 12 \
  --arwen-checkout /home/drew/gpuwm-work/arwen-e594dc5c5 \
  --cache-root "$CACHE" --output "$OUT" \
  --ledger-json "$BASE/ledger-$MESH-$ARM.json" \
  --case-label "hex-ledger-$MESH-$ARM" --arm "$ARM" $EXTRA
