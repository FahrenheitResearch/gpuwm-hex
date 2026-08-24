#!/usr/bin/env bash
set -euo pipefail

# Compile against the same NetCDF-Fortran stack used by the frozen model and
# extract both endpoints without text formatting or precision loss.
repo_root="${1:?usage: extract.sh REPOSITORY_ROOT [RUN_ROOT]}"
# RUN_ROOT may be passed explicitly; otherwise it is derived from the authority
# root, which is site-specific.  An empty value would read "/runs/..." at the
# filesystem root, so refuse by name instead.
run_root="${2:-${MPAS_AUTHORITY_ROOT:?set MPAS_AUTHORITY_ROOT to the MPAS authority checkout root, or pass RUN_ROOT}/runs/jw-v823-20260810}"
tool_dir="${repo_root}/tools/mpas_step_oracle"
output_dir="${repo_root}/oracle/jw-x1.2562-v8.2.3"
build_dir="${tool_dir}/build"

# shellcheck disable=SC1091
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${MPAS_ENV_SETUP:?set MPAS_ENV_SETUP to the MPAS dependency environment setup script}"
mkdir -p "${build_dir}" "${output_dir}"

compiler="${FC:-$(nf-config --fc)}"
"${compiler}" -O2 $(nf-config --fflags) \
  "${tool_dir}/extract_step_oracle.F90" \
  -o "${build_dir}/extract_step_oracle" \
  $(nf-config --flibs)

"${build_dir}/extract_step_oracle" \
  "${run_root}/step/history.2000-01-01_00.00.00.nc" \
  "${run_root}/step/history.2000-01-01_00.10.00.nc" \
  "${output_dir}"

"${compiler}" --version | head -2
sha256sum "${build_dir}/extract_step_oracle"
