#!/usr/bin/env bash
# Five-day native build pair, dry with identical MPAS mixing and damping.
# usage: run_forecast_pair.sh WORK_ROOT INIT_FILE [x1.40962] [RANKS] [DT] [START]
# Use a fresh work root for each configuration; existing runs are not overwritten.
set -euo pipefail
W=$(readlink -m "$1"); INIT=$(readlink -f "$2"); MESH=${3:-x1.40962}
RANKS=${4:-24}; DT=${5:-360}; START=${6:-2023-03-31_12:00:00}
H=$(dirname "$(readlink -f "$0")")
mkdir -p "$W/native"
for recipe in gnu-strict gnu-o3; do
  name=v8.4.2-$recipe-atmosphere
  if [ ! -x "$W/native/$name/atmosphere_model" ]; then
    bash "$H/build_native_reference.sh" "$W" v8.4.2 "$recipe" atmosphere 8
  else
    python3 "$H/validate_reference.py" "$W" "$name"
  fi
  MESH="$MESH" START="$START" INIT_FILE="$INIT" RANKS="$RANKS" DT="$DT" \
    DUR=5_00:00:00 OUT=0_06:00:00 MIXING=smag SMDIV=0.1 XNUTR=0.2 \
    bash "$H/run_native_jw.sh" "$W" run "v8.4.2-$recipe" forecast5d 2 4
  if grep -q NaN "$W/jw/v8.4.2-$recipe/forecast5d/log.atmosphere.0000.out"; then
    echo "FAIL: native $recipe became non-finite; no five-day agreement claim" >&2
    exit 1
  fi
done
echo "Native pair complete. Run forecast_divergence.py with --mixing smag --smdiv 0.1 --xnutr 0.2 and this exact dt."
