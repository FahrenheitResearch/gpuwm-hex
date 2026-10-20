#!/usr/bin/env bash
# Run INSIDE publish.yml's pinned manylinux_2_28 image, before the native and
# packets phases of build_physics_release.py. PREFIX holds the static libraries.
# The caller supplies the pinned Rust toolchain and gpuwm 2.9.2 Python package.
set -euo pipefail
prefix=${1:?PREFIX}
[[ $(getconf GNU_LIBC_VERSION) == 'glibc 2.28' ]] || { echo 'requires release manylinux_2_28 image' >&2; exit 2; }
tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$prefix"
prefix=$(cd -- "$prefix" && pwd)
python=${PYTHON:-/opt/python/cp313-cp313/bin/python}
jobs=${GPUWM_BUILD_JOBS:-4}
"$python" - "$tree/release-assets/kernel-packs/kernel-pack-cuda13.json" <<'PY'
import json, subprocess, sys
compiler=json.load(open(sys.argv[1]))['compiler']
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',
                'numpy>=2.0','netCDF4>=1.6','scipy>=1.11',
                compiler['cupy_package'],compiler['nvrtc_package']],check=True)
PY
bash "$tree/tools/build_netcdf4_static.sh" "$prefix"
work="$prefix/fortran-source"
mkdir -p "$work"
archive="$work/netcdf-fortran-4.6.2.tar.gz"
curl -fsSL https://downloads.unidata.ucar.edu/netcdf-fortran/4.6.2/netcdf-fortran-4.6.2.tar.gz -o "$archive"
echo "df26b99d9003c93a8bc287b58172bf1c279676f8c10d6dd0daf8bc7204877096  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$work"
cd "$work/netcdf-fortran-4.6.2"
CPPFLAGS="-I$prefix/include" LDFLAGS="-L$prefix/lib" LIBRARY_PATH="$prefix/lib" \
  ./configure --prefix="$prefix" --disable-shared --enable-static --with-pic \
    --disable-doxygen --disable-examples --disable-testsets
make -j"$jobs"
make install
# The pinned native reference recipe requests -I/usr/include. This is the
# disposable release container, never an end user's installation.
cp "$prefix"/include/*.mod /usr/include/
cp COPYRIGHT "$prefix/netcdf-fortran-COPYRIGHT"
echo "use LIBRARY_PATH=$prefix/lib for the resource build phases"
