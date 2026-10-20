#!/usr/bin/env bash
# Run the native producer without depending on the installed wheel's CLI version.
set -euo pipefail
usage() {
  echo 'usage: run_s2s_surface_proof.sh ENGINE FORECAST_INIT_OR_STATIC OUT_DIR [CYCLE] [HOURS] [THREADS] [FETCH_JOBS]' >&2
}
if (( $# < 3 || $# > 7 )); then usage; exit 2; fi
engine=$(realpath -e -- "$1")
mesh=$(realpath -e -- "$2")
requested_out=$3
cycle=${4:-2023-03-31T00}
hours=${5:-240}
threads=${6:-$(getconf _NPROCESSORS_ONLN)}
jobs=${7:-32}
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
table=$source_root/src/hexcore/data/init-sources.v1.json
data=$source_root/src/hexcore/data
[[ -x "$engine" && -f "$mesh" && -f "$table" && -f "$data/vtables/Vtable.NCEP-surface" ]]
[[ $hours =~ ^[0-9]+$ && $threads =~ ^[1-9][0-9]*$ && $jobs =~ ^[1-9][0-9]*$ ]]
"$engine" --abi
mkdir -p -- "$requested_out"
out=$(cd -- "$requested_out" && pwd)
series=$out/surface_series.nc
if [[ -e "$series" || -e "$out/surface_series.receipt.json" ]]; then
  echo "refusing to replace existing series or receipt in $out; consume or inspect it first" >&2
  exit 2
fi
# The acceptance consumer still needs these files. Remove this marker only after
# consumption, then remove the prepared init/series and retain their receipts.
printf '%s\n' 'Surface-update acceptance is pending. Retain the series and its matching forecast init until the consumer confirms consumption.' > "$out/.keep"
command=("$engine" --table "$table" --source gfs --cycle "$cycle"
  --hours "$hours" --interval-hours 6 --static "$mesh" --out "$series"
  --receipt "$out/surface_series.receipt.json" --vtable-root "$data"
  --threads "$threads" --jobs "$jobs")
printf '%q ' "${command[@]}" > "$out/surface_series.command.txt"
printf '\n' >> "$out/surface_series.command.txt"
sha256sum -- "$mesh" "$engine" "$table" "$data/vtables/Vtable.NCEP-surface" > "$out/surface_series.inputs.sha256"
printf 'mesh_input\t%s\ncycle\t%s\nhours\t%s\ncompute_threads\t%s\nfetch_workers\t%s\n' \
  "$mesh" "$cycle" "$hours" "$threads" "$jobs" > "$out/surface_series.handoff.tsv"
export RAYON_NUM_THREADS=$threads
/usr/bin/time -f 'PRODUCER_WRAPPER_WALL_SECONDS=%e' -o "$out/surface_series.wall.txt" \
  "${command[@]}" > "$out/surface_series.log" 2>&1
sha256sum -- "$series" "$out/surface_series.receipt.json" > "$out/surface_series.outputs.sha256"
printf 'init_or_static=%s\nseries=%s\nreceipt=%s\n' "$mesh" "$series" "$out/surface_series.receipt.json"
