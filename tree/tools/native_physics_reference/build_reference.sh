#!/usr/bin/env bash
# Offline full-physics MPAS build. Compile only a writable copy of the pinned tree.
# usage: build_reference.sh WORK SOURCE RECIPE CORE MPI_PREFIX PNETCDF_PREFIX [MATH_OBJECT]
# Invoke through the build queue; four compiler jobs; no MUSICA or OpenMP.
set -euo pipefail
W=$(realpath -m "$1"); SRC=$(realpath "$2"); RECIPE=$3; CORE=$4
MPI=$(realpath "$5"); PNETCDF=$(realpath "$6"); MATH=${7:-}
NETCDF_PREFIX=${NETCDF_PREFIX:-/usr}
HERE=$(dirname "$(realpath "$0")")
case "$RECIPE" in gnu-strict|gnu-o3) ;; *) echo 'unsupported recipe' >&2; exit 2;; esac
case "$CORE" in atmosphere|init_atmosphere) ;; *) echo 'unsupported core' >&2; exit 2;; esac
export PATH="$MPI/bin:$PATH" LD_LIBRARY_PATH="$MPI/lib:${LD_LIBRARY_PATH:-}"
export MPICH_FC=gfortran MPICH_CC=gcc MPICH_CXX=g++
D="$W/native/v8.4.2-$RECIPE-$CORE${MATH:+-pinned}"; mkdir -p "$W/native"
[ ! -e "$D" ] || { echo "refusing existing build $D" >&2; exit 2; }
test "$(git -C "$SRC" rev-parse HEAD)" = f34984b2c89353cf31c235b08d3b4acd96b37bb3
test "$(git -C "$SRC/src/core_atmosphere/physics/physics_mmm" rev-parse HEAD)" = a4baf7f3243d1db0dbc5f63473f895bdbdc05c30
test "$(git -C "$SRC/src/core_atmosphere/physics/physics_noaa/UGWP" rev-parse HEAD)" = c1c893edcf171af5639af60e3a3a528816f6cc2b
cp -a "$SRC" "$D"; chmod -R u+w "$D"
# Manage_externals with repo_url='.' uses local refs without remote access.
python3 - "$D/src/core_atmosphere/Externals.cfg" <<'PY'
import pathlib,sys,re
p=pathlib.Path(sys.argv[1]); p.write_text(re.sub(r'(?m)^repo_url = .*$', 'repo_url = .', p.read_text()))
PY
if [ "$RECIPE" = gnu-strict ]; then
  python3 "$HERE/../native_dry_reference/edit_makefile_flags.py" "$D/Makefile" gfortran '-O3 ' '-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations '
fi
if [ -n "$MATH" ]; then
  MATH=$(realpath "$MATH"); test -s "$MATH"
  # Insert into the GNU linker flags only, before all archives.
  python3 - "$D/Makefile" "$MATH" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); s=p.read_text(); head,tail=s.split('\ngfortran:',1)
a,b=tail.split('\n',1); lines=b.splitlines(keepends=True)
for i,l in enumerate(lines):
    if '"LDFLAGS_OPT = ' in l:
        lines[i]=l.replace('LDFLAGS_OPT = ', 'LDFLAGS_OPT = '+sys.argv[2]+' -lm ',1);break
else: raise SystemExit('missing linker flags')
p.write_text(head+'\ngfortran:'+a+'\n'+''.join(lines))
PY
fi
{
  echo "recipe=$RECIPE core=$CORE precision=single MUSICA=off"
  gfortran --version | head -1; "$MPI/bin/mpichversion"; getconf GNU_LIBC_VERSION
  grep -m1 'model name' /proc/cpuinfo; grep -m1 '^flags' /proc/cpuinfo
  sha256sum "$PNETCDF/lib/libpnetcdf.a"
  grep '#define PNETCDF_VERSION ' "$PNETCDF/include/pnetcdf.h"
  echo "netcdf_prefix=$NETCDF_PREFIX"
  [ -z "$MATH" ] || sha256sum "$MATH"
} > "$D/toolchain.txt"
git -C "$D" diff -- Makefile src/core_atmosphere/Externals.cfg > "$D/build-source.diff"
cd "$D"
make -j4 gfortran CORE="$CORE" PRECISION=single PNETCDF="$PNETCDF" NETCDF="$NETCDF_PREFIX" > build.log 2>&1
test -x "${CORE}_model"
if [ "$CORE" = atmosphere ]; then
  python3 "$HERE/audit_flags.py" "$D" --recipe "$RECIPE" --out "$D/flag-audit.json"
  test -x build_tables
fi
sha256sum "${CORE}_model" > executable.sha256
if [ -n "$MATH" ]; then
  ! nm -D "${CORE}_model" | awk '$1 == "U" {sub(/@.*/, "", $2); print $2}' | grep -xE 'asinf|acosf|atanf|atan2f|tanf|tanhf|erff|log10f|sinf|cosf|sincosf|expf|logf|powf'
fi
printf 'BUILD-OK %s\n' "$D"
