#!/bin/sh
# Fork-equivalence gate, then showcase sim C -- but ONLY if the gate returns
# BITWISE-IDENTICAL.  A mismatch means the fork changed the model, and the
# chain stops there rather than burning the card on a run nobody can trust.
set -u

# Site-specific roots.  An empty REPO would write the chain log and the rc
# markers into "/work" at the filesystem root, so refuse by name instead.
REPO=${MPAS_PROOF_REPO:?set MPAS_PROOF_REPO to the release-proof repository root}
PY=/venv/main/bin/python
W=$REPO/work
ARWEN=${ARWEN_CHECKOUT:?set ARWEN_CHECKOUT to the ArWen checkout the driver runs against}
# Resolved up front: sim C's input must be named before the gate leg spends an
# hour of card time that a missing init would then throw away.
INIT=${MPAS_INIT_FILE:?set MPAS_INIT_FILE to the pinned showcase initial-condition NetCDF}
LOG=$W/showcase-chain-20260812.log

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

cd "$REPO" || exit 90

echo "$(stamp) chain start" >> "$LOG"
sh tools/run_forkgate.sh
grc=$?
echo "$(stamp) forkgate exit=$grc" >> "$LOG"
if [ "$grc" -ne 0 ]; then
  echo "chain_rc=stopped_forkgate_not_identical" > "$W/showcase-chain-20260812-marker-stop.done"
  echo "$(stamp) CHAIN STOPPED: fork-equivalence gate did not return BITWISE-IDENTICAL" >> "$LOG"
  exit "$grc"
fi

TAG=showcase-c
echo "$(stamp) sim C start" >> "$LOG"
"$PY" tools/run_cuda_v841_forecast.py \
  --init "$INIT" \
  --init-source 'ERA5 2025-03-14 12Z (CDS reanalysis-era5-pressure-levels + reanalysis-era5-single-levels)' \
  --start-time 2025-03-14_12:00:00 \
  --hours 24 \
  --history-every-minutes 60 \
  --case-label 'C - hindcast 2025-03-14 12Z (ERA5)' \
  --arwen-checkout "$ARWEN" \
  --cache-root "$W/$TAG-cache" \
  --output "$W/$TAG" \
  >> "$LOG" 2>&1
crc=$?
echo "sim_c_rc=$crc" > "$W/showcase-chain-20260812-marker-c.done"
echo "$(stamp) sim C rc=$crc" >> "$LOG"
if [ "$crc" -eq 0 ]; then
  ( cd "$W/$TAG" && sha256sum ./* > SHA256SUMS.node 2>/dev/null )
fi
exit "$crc"
