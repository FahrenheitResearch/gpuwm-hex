#!/usr/bin/env bash
set -euo pipefail

# Build the exact vendored MPAS-A v8.2.3 archive as CPU-only executables.
# The existing NVIDIA-built MPI/PIO/NetCDF stack is reused.  Compatibility
# command aliases avoid modifying frozen source merely because PGI renamed its
# compilers to NVHPC.

repo_root="${1:?usage: build_frozen_authority.sh REPOSITORY_ROOT}"
# Both roots are site-specific.  An empty value would build the authority into
# "/source" and "/bin" at the filesystem root, so refuse by name instead.
authority_root="${MPAS_AUTHORITY_ROOT:?set MPAS_AUTHORITY_ROOT to the MPAS authority checkout root}"
archive="${repo_root}/vendor/MPAS_source_v8.2.3_group/MPAS-Model-v8.2.3.tar.gz"
expected_sha="bb3b02c30abffe9ff0318165b25724e6855fb69076fd89243f06a24e11912ee1"
env_setup="${MPAS_ENV_SETUP:?set MPAS_ENV_SETUP to the MPAS dependency environment setup script}"
receipt_dir="${repo_root}/receipts/frozen-v8.2.3"
jobs="${JOBS:-$(nproc)}"

actual_sha="$(sha256sum "${archive}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "frozen archive SHA-256 mismatch: ${actual_sha} != ${expected_sha}" >&2
  exit 2
fi
if [[ ! -f "${env_setup}" ]]; then
  echo "missing known MPAS dependency environment: ${env_setup}" >&2
  exit 2
fi

mkdir -p "${authority_root}/source" "${authority_root}/bin" \
  "${authority_root}/logs" "${authority_root}/compiler-compat" "${receipt_dir}"
if [[ ! -f "${authority_root}/source/Makefile" ]]; then
  tar -xzf "${archive}" --strip-components=1 -C "${authority_root}/source"
fi

# shellcheck disable=SC1091
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${env_setup}"
ln -sfn "$(command -v nvfortran)" "${authority_root}/compiler-compat/pgf90"
ln -sfn "$(command -v nvc)" "${authority_root}/compiler-compat/pgcc"
ln -sfn "$(command -v nvc++)" "${authority_root}/compiler-compat/pgc++"
export PATH="${authority_root}/compiler-compat:${PATH}"
export LD_LIBRARY_PATH="${PNETCDF}/lib:${PIO}/lib:${NETCDF}/lib:${LD_LIBRARY_PATH:-}"

build_core() {
  local core="$1"
  local executable="$2"
  local log_file="${authority_root}/logs/build_${core}.log"
  cd "${authority_root}/source"
  make clean CORE="${core}" >/dev/null 2>&1 || true
  make -j"${jobs}" pgi CORE="${core}" USE_PIO2=true PRECISION=single \
    AUTOCLEAN=true NETCDF="${NETCDF}" PNETCDF="${PNETCDF}" PIO="${PIO}" \
    >"${log_file}" 2>&1
  if [[ ! -x "${executable}" ]]; then
    tail -n 120 "${log_file}" >&2
    echo "${core} build did not produce ${executable}" >&2
    exit 3
  fi
  cp -f "${executable}" "${authority_root}/bin/${executable}"
}

build_core init_atmosphere init_atmosphere_model
build_core atmosphere atmosphere_model

{
  echo "authority_release=v8.2.3"
  echo "authority_tag_commit=ac3866c1e5b05f6d4f5bd41aeab7d3882bace514"
  echo "source_archive_sha256=${actual_sha}"
  echo "build_target=pgi (NVHPC compatibility aliases), CPU only"
  echo "precision=single"
  echo "use_pio2=true"
  echo "jobs=${jobs}"
  nvfortran --version | head -n 2 | sed 's/[[:space:]]*$//'
  "${NETCDF}/bin/nc-config" --version || true
  "${NETCDF}/bin/nf-config" --version || true
  sha256sum "${authority_root}/bin/init_atmosphere_model"
  sha256sum "${authority_root}/bin/atmosphere_model"
  file "${authority_root}/bin/init_atmosphere_model"
  file "${authority_root}/bin/atmosphere_model"
} >"${receipt_dir}/build-receipt.txt"

tail -n 40 "${authority_root}/logs/build_init_atmosphere.log" \
  >"${receipt_dir}/build-init-tail.log"
tail -n 40 "${authority_root}/logs/build_atmosphere.log" \
  >"${receipt_dir}/build-atmosphere-tail.log"
cat "${receipt_dir}/build-receipt.txt"
