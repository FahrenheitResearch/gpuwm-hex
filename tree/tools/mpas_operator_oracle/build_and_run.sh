#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
mesh="${repo_root}/data/meshes/x1.2562/x1.2562.static.nc"
output="${repo_root}/oracle/x1.2562"
source_file="${repo_root}/tools/mpas_operator_oracle/operator_oracle.F90"
executable="${repo_root}/tools/mpas_operator_oracle/operator_oracle"

mkdir -p "${output}"
gfortran -std=f2008 -O0 -fno-fast-math -ffp-contract=off \
  $(nf-config --fflags) "${source_file}" -o "${executable}" \
  $(nf-config --flibs)
"${executable}" "${mesh}" "${output}"
gfortran --version | head -n 1
nf-config --version
