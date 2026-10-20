#!/bin/bash
# Acceptance evidence for one moving run against its fixed-region control (CPU only; run after both finish).
# usage: acceptance.sh CASE.json CASEDIR MOVING_RUN FIXED_RUN OUTDIR [--no-render]
#   MOVING_RUN is a `gpuwm-hex moving run` folder, FIXED_RUN a `gpuwm-hex moving fixed` folder;
#   CASEDIR/obs/best-track.json comes from `gpuwm-hex moving stage --best-track`. PYTHON names the
#   interpreter (default python3), which must import hexcore.
#   FIXED_RUN may be "-" when no completed control exists (A3 then uses the run's own envelope only, A7 the best track).
#   A2  a2_check.py (tracker re-run over every output, offset against the region in effect)
#   A3/A4  summarize_run.py with the fixed run as the matched no-move control (post-move window 10 min)
#   A7  track_history.py over the fixed run, track_score.py (moving, fixed vs the case best track), track_chart.py
#   A5  render_run.py (one lattice, Rust frames and loops), unless --no-render
#   evidence: collect_evidence.sh receipts, wall times
set -eu
CASE=$1; C=$2; R=$3; F=$4; O=$5; RENDER=${6:-}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
mkdir -p "$O"
START=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['start_time'])" "$CASE")
BEST=$C/obs/best-track.json
CONTROL=(); TRACKS=(--track "moving=$R/track-session.json")
if [ "$F" != "-" ]; then
  CONTROL=(--control "$F"); TRACKS+=(--track "fixed=$O/track-fixed.json")
fi
"$PY" "$H/summarize_run.py" "$R" "${CONTROL[@]}" --out "$O/summary-a3.json" > /dev/null
"$PY" "$H/a2_check.py" "$R" --case "$CASE" --out "$O/a2.json" > "$O/a2.log"
"$PY" - "$O/a2.json" "$O/region-log.jsonl" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))['rows']
with open(sys.argv[2], 'w') as f:
    for r in rows:
        for c in r['regions']:
            f.write(json.dumps(dict(seconds=r['seconds'], region=c['centre'])) + '\n')
PY
if [ "$F" != "-" ]; then
  "$PY" "$H/track_history.py" --case "$CASE" --history-dir "$F/fc" --out "$O/track-fixed.json" --label fixed > /dev/null
fi
"$PY" "$H/track_score.py" --best "$BEST" --start-time "$START" "${TRACKS[@]}" \
  --out "$O/track-score.json" --markdown "$O/track-score.md"
"$PY" "$H/track_chart.py" --score "$O/track-score.json" --best "$BEST" --start-time "$START" \
  --title "$(basename "$CASE" .json): moving 3 km region$([ "$F" != "-" ] && echo " vs fixed region") vs best track" \
  --out "$O/track-error.png" --region-log "$O/region-log.jsonl"
"$PY" "$H/compare_shock.py" "$R" > "$O/shock-table.txt"
cp "$R/shock-series.json" "$O/shock-series-moving.json"
if [ "$F" != "-" ]; then
  "$PY" - "$F/fc/fullstep-lts-receipt.json" "$O/shock-series-fixed.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1])).get('remap_shock_series', [])
json.dump([dict(r, segment='fixed', remap=False) for r in rows], open(sys.argv[2], 'w'), indent=1)
PY
else
  echo '[]' > "$O/shock-series-fixed.json"
fi
"$PY" "$H/plot_shock.py" "$O/shock-series-moving.json" "$O/shock-series-fixed.json" "$O/transient.png" \
  "$(basename "$CASE" .json): surface-pressure tendency and max |w|$([ "$F" != "-" ] && echo ", moving vs fixed region")"
bash "$H/collect_evidence.sh" "$R" "$O/evidence"
if [ "$F" != "-" ]; then
  mkdir -p "$O/evidence/fixed"
  for f in command.txt process-wall.txt fc/fullstep-lts-receipt.json; do
    [ -f "$F/$f" ] && cp "$F/$f" "$O/evidence/fixed/$(echo "$f" | tr / _)"
  done
fi
cp "$R.process-wall.txt" "$O/moving-process-wall.txt" 2>/dev/null || true
if [ "$RENDER" != "--no-render" ]; then
  "$PY" "$H/render_run.py" "$R" "$O/products" --best-track "$BEST" --jobs 8
fi
echo "ACCEPTANCE_DONE $O"
