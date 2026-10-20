#!/usr/bin/env bash
# Build native MPAS-Atmosphere for the dry-dycore identity check.
#
# usage: build_native_reference.sh WORK_ROOT TAG RECIPE CORE [JOBS] [VARIANT]
#   VARIANT condensate: apply registry_condensate.py before building (physics-off
#           allocation of qc..qg); the build is named TAG-RECIPE-condensate-CORE and is
#           NOT the pinned reference binary.
#   TAG     an upstream MPAS-Dev/MPAS-Model tag (v8.4.1, v8.4.2)
#   RECIPE  ifx-o3         Intel ifx with the stock "intel-mpi" target flags.  This is
#                          the recipe of the x4 native authority (receipt
#                          native-authority-regen-20260820: "intel-mpi ifx, single
#                          precision, SMIOL").
#           ifx-valuesafe  the same plus -fp-model=source -no-fma
#                          -fimf-arch-consistency=true, the recipe of the
#                          CANDIDATE-REGIONAL-DRY-VALUESAFE reference (2026-08-26).
#           gnu-o3         GNU Fortran with the stock "gfortran" target flags (the
#                          recipe of the 2026-10-11 v8.4.2 re-pin).
#           gnu-o0         GNU Fortran at -O0 -ffp-contract=off: source order, no
#                          contraction.
#           gnu-strict     GNU Fortran at -O3 -ffp-contract=off
#                          -fno-tree-vectorize -fno-unsafe-math-optimizations.
#   CORE    atmosphere | init_atmosphere
#
# Needs Intel oneAPI (ifx, icx, Intel MPI) under /opt/intel/oneapi and, for the GNU
# recipes, gfortran.  PnetCDF (C library only, built with gcc so both
# compiler families link it; the SMIOL library inside MPAS does the I/O) is built once
# under WORK_ROOT/pnetcdf.  Single precision, CPU only, no OpenMP.
set -euo pipefail
W=$1; TAG=$2; RECIPE=$3; CORE=$4; JOBS=${5:-8}; VARIANT=${6:-}
[ -z "$VARIANT" ] || [ "$VARIANT" = condensate ] || { echo "unknown variant $VARIANT" >&2; exit 2; }
[ "$JOBS" -le 32 ] || { echo "at most 32 build jobs" >&2; exit 2; }
if [[ $RECIPE = ifx-* ]]; then
  set +u; source "${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}" >/dev/null 2>&1; set -u
fi
mkdir -p "$W/native" "$W/shim"
# oneAPI 2025 and later ship no ifort/icc; the intel-mpi target still names them.
for pair in ifort:ifx icc:icx icpc:icpx; do
  old=${pair%%:*}; new=${pair##*:}
  if [[ $RECIPE = ifx-* ]]; then
    command -v "$old" >/dev/null 2>&1 || ln -sfn "$(command -v "$new")" "$W/shim/$old"
  fi
done
export PATH=$W/shim:$PATH
if [ ! -f "$W/pnetcdf/lib/libpnetcdf.a" ]; then
  mkdir -p "$W/src"; cd "$W/src"
  [ -f pnetcdf-1.12.3.tar.gz ] || curl -fsSLO https://parallel-netcdf.github.io/Release/pnetcdf-1.12.3.tar.gz
  sha256sum pnetcdf-1.12.3.tar.gz > "$W/native/pnetcdf.tarball.sha256"
  PIN_FILE="$(dirname "$(readlink -f "$0")")/REFERENCE-PINS.json"
  PINNED_PNETCDF=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pnetcdf_tarball_sha256"])' "$PIN_FILE")
  echo "$PINNED_PNETCDF  pnetcdf-1.12.3.tar.gz" | sha256sum -c -
  [ -d pnetcdf-1.12.3 ] || tar xzf pnetcdf-1.12.3.tar.gz
  cd pnetcdf-1.12.3
  MPICC_BUILD=mpicc
  [[ $RECIPE != ifx-* ]] || MPICC_BUILD=mpigcc
  ./configure --prefix="$W/pnetcdf" --disable-fortran --disable-cxx --disable-shared \
      MPICC="$MPICC_BUILD" CFLAGS="-O2 -fPIC -std=gnu17" > "$W/native/pnetcdf.configure.log" 2>&1
  make -j"$JOBS" > "$W/native/pnetcdf.build.log" 2>&1
  make install >> "$W/native/pnetcdf.build.log" 2>&1
fi
[ -d "$W/MPAS-Model/.git" ] || git clone -q https://github.com/MPAS-Dev/MPAS-Model.git "$W/MPAS-Model"
N=$TAG-$RECIPE${VARIANT:+-$VARIANT}-$CORE; D=$W/native/$N
[ -d "$D" ] || git -C "$W/MPAS-Model" worktree add -q --detach "$D" "$TAG"
cd "$D"
# Never build altered numerics under an upstream source pin. A prior recipe's
# unstaged Makefile flags, and the declared condensate Registry.xml variant
# (regenerated below from registry_condensate.py), can be reset; any staged
# edit, or any other change, requires a fresh work root.
git diff --cached --quiet -- || { echo "FAIL native source has staged changes. Use a fresh --work-root." >&2; exit 2; }
git diff --quiet -- . ':(exclude)Makefile' ':(exclude)src/core_atmosphere/Registry.xml' || { echo "FAIL native source has changes beyond build flags. Use a fresh --work-root." >&2; exit 2; }
git checkout -q -- Makefile src/core_atmosphere/Registry.xml
if [ "$VARIANT" = condensate ]; then
  python3 "$(dirname "$(readlink -f "$0")")/registry_condensate.py" src/core_atmosphere/Registry.xml
  git diff -- src/core_atmosphere/Registry.xml > "$W/native/$N.registry.diff"
fi
test "$(git rev-parse HEAD)" = "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream"][sys.argv[2]])' "$(dirname "$(readlink -f "$0")")/REFERENCE-PINS.json" "$TAG")"
case $RECIPE in
  ifx-o3)
    # ifx cannot preprocess MPAS without the macro option the upstream "intel"
    # (oneAPI) target carries; it changes preprocessing only, no arithmetic.
    TARGET=intel-mpi; export I_MPI_F90=ifx I_MPI_CC=icx I_MPI_CXX=icpx
    python3 "$(dirname "$(readlink -f "$0")")/edit_makefile_flags.py" Makefile intel-mpi         "-O3 -convert big_endian -free -align array64byte"         "-O3 -convert big_endian -free -align array64byte -Qoption,fpp,-macro_expand=vc" ;;
  ifx-valuesafe)
    TARGET=intel-mpi; export I_MPI_F90=ifx I_MPI_CC=icx I_MPI_CXX=icpx
    python3 "$(dirname "$(readlink -f "$0")")/edit_makefile_flags.py" Makefile intel-mpi \
        "-O3 -convert big_endian -free -align array64byte" \
        "-O3 -convert big_endian -free -align array64byte -Qoption,fpp,-macro_expand=vc -fp-model=source -no-fma -fimf-arch-consistency=true" ;;
  gnu-o3)
    TARGET=gfortran; export I_MPI_F90=gfortran I_MPI_CC=gcc I_MPI_CXX=g++ ;;
  gnu-o0)
    TARGET=gfortran; export I_MPI_F90=gfortran I_MPI_CC=gcc I_MPI_CXX=g++
    python3 "$(dirname "$(readlink -f "$0")")/edit_makefile_flags.py" Makefile gfortran \
        "-O3 " "-O0 -ffp-contract=off " ;;
  gnu-strict)
    TARGET=gfortran
    python3 "$(dirname "$(readlink -f "$0")")/edit_makefile_flags.py" Makefile gfortran \
        "-O3 " "-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations " ;;
  *) echo "unknown recipe $RECIPE" >&2; exit 2 ;;
esac
git rev-parse HEAD > "$W/native/$N.commit"
git diff HEAD -- Makefile > "$W/native/$N.makefile.diff"
{ echo "recipe=$RECIPE target=$TARGET tag=$TAG core=$CORE precision=single io=SMIOL"
  "$([ "$TARGET" = gfortran ] && echo gfortran || echo ifx)" --version | head -1
  mpirun --version | head -1; grep -m1 "model name" /proc/cpuinfo; } > "$W/native/$N.toolchain.txt"
make clean CORE="$CORE" > /dev/null 2>&1 || true
make -j"$JOBS" "$TARGET" CORE="$CORE" PRECISION=single PNETCDF="$W/pnetcdf" > "$W/native/$N.build.log" 2>&1
test -x "$D/${CORE}_model"
# The reference must be single-threaded (DY1 thread-sentinel race; the breakage and the
# three checks are documented in check_single_threaded.sh).  Exit 2 refuses the build.
bash "$(dirname "$(readlink -f "$0")")/check_single_threaded.sh" "$W/native/$N.build.log" "$D/${CORE}_model"
sha256sum "$D/${CORE}_model" | tee "$W/native/$N.sha256"
grep -m2 "mpas_atm_time_integration" "$W/native/$N.build.log" | cut -c1-600 > "$W/native/$N.compile-lines.txt" || true
echo "BUILD-OK $N"
