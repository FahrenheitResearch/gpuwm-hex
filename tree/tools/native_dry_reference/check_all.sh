#!/usr/bin/env bash
# Rebuild pinned v8.4.2 references and measure one dry step at three schedules.
# usage: check_all.sh WORK_ROOT HEX_TREE [JOBS] [cpu|cuda|both] [gnu|ifx]
# Place CUDA execution under the site's GPU allocation protocol.
set -euo pipefail
W=$1; TREE=$2; J=${3:-8}; BACKEND=${4:-cuda}; FAMILY=${5:-gnu}
H=$(dirname "$(readlink -f "$0")")
PYTHON=${PYTHON:-python3}
export PYTHONPATH="$TREE/src:$TREE/tools:${PYTHONPATH:-}"
# This product door checks one fixed public case. The lower-level helper also
# serves custom forecast experiments, whose environment must not change this
# check accidentally.
run_standard_native() {
  env MESH=x1.2562 LEVELS=26 START=2000-01-01_00:00:00 \
    INIT_CASE=2 INIT_SUBDIR=init VERTICAL_GRID=false \
    GRID_FILE="$W/mesh/x1.2562.grid.nc" INIT_FILE="$W/jw/init/x1.2562.init.nc" \
    DT=1800.0 DUR=0_00:30:00 OUT=0_00:30:00 MIXING=off SMDIV=0.0 XNUTR=0.0 RANKS=1 \
    bash "$H/run_native_jw.sh" "$@"
}
if [ "$FAMILY" = gnu ]; then STRICT=gnu-o0; OPT=gnu-o3
elif [ "$FAMILY" = ifx ]; then STRICT=ifx-valuesafe; OPT=ifx-o3
else echo "unknown compiler family: $FAMILY" >&2; exit 2; fi
mkdir -p "$W/out"
for recipecore in "$STRICT init_atmosphere" "$STRICT atmosphere" "$OPT atmosphere"; do
  read -r recipe core <<< "$recipecore"
  exe="$W/native/v8.4.2-$recipe-$core/${core}_model"
  if [ ! -x "$exe" ]; then
    bash "$H/build_native_reference.sh" "$W" v8.4.2 "$recipe" "$core" "$J"
  else
    "$PYTHON" "$H/validate_reference.py" "$W" "v8.4.2-$recipe-$core"
  fi
done
if [ -L "$W/jw/init" ]; then
  echo "FAIL native init directory is a link. Use a fresh --work-root." >&2; exit 2
fi
test -s "$W/jw/init/x1.2562.init.nc" || run_standard_native "$W" init "v8.4.2-$STRICT"
"$PYTHON" "$H/run_receipt.py" verify-init "$W/jw/init" --standard-init \
  --expected-executable "$W/native/v8.4.2-$STRICT-init_atmosphere/init_atmosphere_model"
status=0
for arm in 'rk2-s4 2 4' 'rk3-s6 3 6' 'rk2-s2 2 2'; do
  read -r name order sub <<< "$arm"
  for recipe in "$STRICT" "$OPT"; do
    dir="$W/jw/v8.4.2-$recipe/$name"
    if [ -L "$dir" ]; then
      echo "FAIL native run directory is a link: $dir. Use a fresh --work-root." >&2; exit 2
    fi
    # Scratch cleanup retains logs but removes frames. Preserve those receipts
    # outside the active reference tree, then regenerate without overwriting.
    if [ -d "$dir" ] && ! compgen -G "$dir/tracer.*.nc" > /dev/null; then
      mkdir -p "$W/archived-receipts"
      archive=$(mktemp -d "$W/archived-receipts/$recipe.$name.XXXXXXXX")
      mv -- "$dir" "$archive/run"
      echo "Raw native reference absent; preserved old receipts in $archive/run"
    fi
    if [ ! -d "$dir" ]; then
      run_standard_native "$W" run "v8.4.2-$recipe" "$name" "$order" "$sub"
    fi
    "$PYTHON" "$H/run_receipt.py" verify "$dir" --expected-frames 2 \
      --standard-order "$order" --standard-substeps "$sub" \
      --expected-executable "$W/native/v8.4.2-$recipe-atmosphere/atmosphere_model" \
      --expected-init "$W/jw/init/x1.2562.init.nc"
    "$PYTHON" "$H/compare_dry_identity.py" --tree "$TREE" --init "$W/jw/init/x1.2562.init.nc" \
      --native-dir "$dir" --order "$order" --substeps "$sub" --backend "$BACKEND" --steps 1 \
      --cache-root "$W/kernel-cache" --json "$W/out/$recipe.$name.$BACKEND.raw.json" \
      > "$W/out/$recipe.$name.$BACKEND.raw.log" 2>&1
    "$PYTHON" "$H/assess_agreement.py" "$W/out/$recipe.$name.$BACKEND.raw.json" || status=1
  done
  "$PYTHON" "$H/compare_native_spread.py" --tree "$TREE" --init "$W/jw/init/x1.2562.init.nc" \
    --strict "$W/jw/v8.4.2-$STRICT/$name" --optimized "$W/jw/v8.4.2-$OPT/$name" \
    --out "$W/out/native-spread.$name.json"
done
"$PYTHON" "$H/reference_digest.py" "$W/jw" > "$W/out/reference-digests.json"
echo "Agreement limits: 0 differing words against the strict GNU native (byte identity); against the optimised native 6 ULP thermodynamics and absolute bounds for near-zero fields."
echo "Native build spread is reported separately. Only the measured configuration is covered."
exit "$status"
