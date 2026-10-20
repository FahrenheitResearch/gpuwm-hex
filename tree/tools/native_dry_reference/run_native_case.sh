#!/bin/bash
# Run one native MPAS v8.4.2 arm on a prepared identity case (dycore-match deck).
# usage: run_native_case.sh ROOT MESH INIT BUILD ARM RANKS DUR OUT [DT]
#   BUILD v8.4.2-gnu-strict | v8.4.2-gnu-o3 ; the 5-day dycore-match deck:
#   RK2, 4 acoustic substeps, 3 splits, 2d_smagorinsky, smdiv 0.1, xnutr 0.2.
set -euo pipefail
ROOT=$1; MESH=$2; INIT=$3; BUILD=$4; ARM=$5; RANKS=$6; DUR=$7; OUT=$8; DT=${9:-360.0}
HERE=$(dirname "$(readlink -f "$0")")
# IDENTITY_DEPS / IDENTITY_VENV relocate the MPI runtime and Python on another host.
DEPS=${IDENTITY_DEPS:-/work/hex-dycore-claim/deps}; VENV=${IDENTITY_VENV:-/work/hex-dycore-claim/venv}
export PATH=$VENV/bin:$DEPS/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/lib:${LD_LIBRARY_PATH:-}
START=${START:-2023-03-31_12:00:00}
t0=$(date +%s)
MESH=$MESH START=$START INIT_FILE="$INIT" MIXING=${MIXING:-smag} SMDIV=${SMDIV:-0.1} XNUTR=${XNUTR:-0.2} DT=$DT DUR=$DUR OUT=$OUT RANKS=$RANKS \
  bash "$HERE/run_native_jw.sh" "$ROOT" run "$BUILD" "$ARM" ${ORDER:-2} ${SUBSTEPS:-4}
echo "NATIVE_WALL_SECONDS $(( $(date +%s) - t0 )) $BUILD $ARM ranks=$RANKS"
