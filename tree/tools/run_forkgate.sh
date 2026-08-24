#!/bin/sh
# Fork-equivalence gate for the derived forecast driver.
# Detached, rc-markered.  Runs the AUTHORITY init through the new driver for
# 30 steps / 1 h and compares bitwise against the release-proof arm.
set -u

# Both roots are site-specific.  An empty REPO would write the log and the rc
# markers into "/work" at the filesystem root, so refuse by name instead.
REPO=${MPAS_PROOF_REPO:?set MPAS_PROOF_REPO to the release-proof repository root}
PY=/venv/main/bin/python
TAG=forkgate-20260812
W=$REPO/work
ARWEN=${ARWEN_CHECKOUT:?set ARWEN_CHECKOUT to the ArWen checkout the driver runs against}
INIT=$REPO/work/v841-vr-static/run-real-init-v841-conus-official-full-a/x4.163842.init.nc
PROOF_RECEIPT=$W/v841-releaseproof-20260812-out-d/cuda-v841-full-physics-x4-receipt.json

cd "$REPO" || exit 90

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) forkgate run start" >> "$W/$TAG.log"
"$PY" tools/run_cuda_v841_forecast.py \
  --init "$INIT" \
  --init-source 'AUTHORITY init (release-proof pinned) 2026-08-10 12Z' \
  --start-time 2026-08-10_12:00:00 \
  --hours 1 \
  --history-every-minutes 30 \
  --fingerprint-every 1 \
  --case-label forkgate-authority-init \
  --arwen-checkout "$ARWEN" \
  --cache-root "$W/$TAG-cache" \
  --output "$W/$TAG-out" \
  >> "$W/$TAG.log" 2>&1
rc=$?
echo "run_rc=$rc" > "$W/$TAG-marker-run.done"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) forkgate run rc=$rc" >> "$W/$TAG.log"
if [ "$rc" -ne 0 ]; then
  echo "gate_rc=skipped_run_failed" > "$W/$TAG-marker-gate.done"
  exit "$rc"
fi

"$PY" tools/gate_v841_forecast_fork_equivalence.py \
  --proof-receipt "$PROOF_RECEIPT" \
  --candidate "$W/$TAG-out" \
  --out "$W/$TAG-report.json" \
  >> "$W/$TAG.log" 2>&1
grc=$?
echo "gate_rc=$grc" > "$W/$TAG-marker-gate.done"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) forkgate compare rc=$grc" >> "$W/$TAG.log"
exit "$grc"
