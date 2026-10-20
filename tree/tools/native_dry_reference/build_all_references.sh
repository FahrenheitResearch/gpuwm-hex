#!/usr/bin/env bash
# Historical Intel native reference matrix, start to finish:
# one init file, then each build and its three schedules.  Sequential and niced.
#
# usage: build_all_references.sh WORK_ROOT [JOBS]
set -euo pipefail
W=$1; J=${2:-10}; H=$(dirname "$(readlink -f "$0")")
mkdir -p "$W"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$W/drive.log"; }
build() {
  if nice -n 10 bash "$H/build_native_reference.sh" "$W" "$1" "$2" "$3" "$J" >> "$W/drive.log" 2>&1
  then log "ok build $1 $2 $3"; else log "FAILED build $1 $2 $3"; return 1; fi
}
build v8.4.1 ifx-valuesafe init_atmosphere
if bash "$H/run_native_jw.sh" "$W" init v8.4.1-ifx-valuesafe >> "$W/drive.log" 2>&1
then log "ok init"; else log "FAILED init"; exit 1; fi
for pair in "v8.4.1 ifx-o3" "v8.4.1 ifx-valuesafe" "v8.4.2 ifx-o3" "v8.4.2 ifx-valuesafe"; do
  set -- $pair; tag=$1; recipe=$2
  build "$tag" "$recipe" atmosphere
  for arm in "rk3-s6 3 6" "rk2-s2 2 2" "rk2-s4 2 4"; do
    set -- $arm
    if bash "$H/run_native_jw.sh" "$W" run "$tag-$recipe" "$1" "$2" "$3" >> "$W/drive.log" 2>&1
    then log "ok run $tag-$recipe $1"; else log "FAILED run $tag-$recipe $1"; exit 1; fi
  done
done
log ALL-DONE
