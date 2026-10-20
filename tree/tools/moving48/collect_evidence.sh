#!/bin/bash
# Collect the small receipts of a long run into one folder (CPU only; no history or carriers).
# usage: collect_evidence.sh RUNDIR OUTDIR
set -eu
R=$1; O=$2
mkdir -p "$O"
for f in controller.json controller.log summary.json summary-a3.json track-session.json; do
  [ -f "$R/$f" ] && cp "$R/$f" "$O/"
done
for s in "$R"/segment-*; do
  n=$(basename "$s"); mkdir -p "$O/$n"
  for f in command.json result.json control/progress.jsonl control/seals.jsonl fc/fullstep-lts-receipt.json \
           fc/moving-remap.json fc/fullstep-lts-restart-identity.jsonl fc/timeline.jsonl fc/fullstep-lts-first-failure.json \
           fc/forecast-traceback.log; do
    [ -f "$s/$f" ] && cp "$s/$f" "$O/$n/$(echo "$f" | tr / _)"
  done
  grep -v '^ \|^$\|CuPy\|----\|pip uninstall\|conda\|Follow these\|Refer to\|https://\|existing\|Install the\|warnings.warn\|radiation_chunk\|edge_hoist' \
    "$s/forecast.log" > "$O/$n/forecast.filtered.log" 2>/dev/null || true
done
echo "collected $(du -sh "$O" | cut -f1) into $O"
