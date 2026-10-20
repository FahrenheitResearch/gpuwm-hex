export GPUWM_HEX_RW_MPAS_HOSTPREP="$PWD/bridges/rw_mpas_hostprep"
export GPUWM_HEX_RW_MPAS_GEOMETRY="$PWD/bridges/rw_mpas_geometry"
export GPUWM_HEX_PREP_CACHE=off
export GPUWM_HEX_HOST_THREADS=1
export PYTHONPATH="$PWD/hex/tree/src:$PWD/experiments"
export CUPY_CACHE_DIR="$PWD/.cache/cupy"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
