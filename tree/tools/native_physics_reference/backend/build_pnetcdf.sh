#!/usr/bin/env bash
# Offline PIC dependency build. Call through the build queue; no upstream download.
# usage: build_pnetcdf.sh LOCAL_TARBALL WORK_DIR
set -euo pipefail
archive=$(readlink -f "$1")
work=$(realpath -m "$2")
mkdir -p "$work/source" "$work/install"
sha256sum "$archive" > "$work/source.sha256"
if [ ! -e "$work/source/configure" ]; then
  tar --no-same-owner --strip-components=1 -xzf "$archive" -C "$work/source"
fi
cd "$work/source"
./configure --prefix="$work/install" --disable-fortran --disable-cxx --disable-shared \
  MPICC=mpicc CFLAGS='-O2 -fPIC -std=gnu17'
make -j4
make -j4 install
sha256sum "$work/install/lib/libpnetcdf.a" > "$work/library.sha256"
