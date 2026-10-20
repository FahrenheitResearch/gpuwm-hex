#!/usr/bin/env bash
# Build the correct-mode native reference library from a PATCHED MPAS tree (config_physics_mode).
# usage: build_v3.sh PATCHED_MPAS_ROOT OUT_DIR
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2"); here=$(dirname "$(realpath "$0")")
mkdir -p "$out"
cp "$here"/{framework_v3.f90,wrapper_v3.f90,Makefile.v3} "$out"/
cp --remove-destination "$src/src/core_atmosphere/dynamics/mpas_atm_iau.F" "$out"/
cp --remove-destination "$src/src/framework/mpas_constants.F" "$out"/
cd "$out"
grep -q config_physics_mode mpas_atm_iau.F || { echo "not a correct-mode tree: $src" >&2; exit 2; }
sha256sum mpas_atm_iau.F mpas_constants.F framework_v3.f90 wrapper_v3.f90 Makefile.v3 > sources.sha256
gfortran --version | head -1 > toolchain.txt
make -f Makefile.v3 -j4 > build.log 2>&1
echo "BUILD-OK $out/libiau_reference_v3.so"
