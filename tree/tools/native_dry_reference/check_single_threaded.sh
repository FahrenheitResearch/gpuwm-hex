#!/usr/bin/env bash
# usage: check_single_threaded.sh BUILD_LOG BINARY
#
# Refuse (exit 2) a native MPAS reference that was built with OpenMP.
#
# Breakage this prevents (DY1, audit-mpas-dycore 2026-10-13): under OpenMP,
# atm_rk_integration_setup and atm_rk_dynamics_substep_finish zero
# theta_m(:,cellEnd+1) per thread, and for every thread but the last that is the
# next thread's first real cell (mpas_atm_time_integration.F:3139,7730); a
# 3,000-case concurrent run of the pristine routines gave 1,253-1,457 wrong
# temperature columns.  A threaded native build is therefore no referee.  The
# fix, if threading is ever wanted, is
# the upstream-format DY1 patch of the 2026-10-13 mpas-mixing-dycore correct-mode cluster.
#
# Three independent checks, any one refuses:
#   1. a compile line (one carrying " -c ") with an OpenMP flag of any MPAS
#      Makefile target: GNU -fopenmp, Intel -qopenmp / -fiopenmp / -openmp,
#      IBM -qsmp=omp, NVIDIA/PGI -mp.  Only compile lines count: every target
#      echoes "FFLAGS_OMP = <flag>" whether or not OPENMP=true.
#   2. an OpenMP runtime in the binary's dynamic dependencies (libgomp,
#      libiomp5, libomp, libnvomp).  ldd's output is captured first and matched
#      in the shell, never piped into "grep -q": under pipefail an early grep
#      exit can SIGPIPE ldd and turn a match into "no match" (fail open).
#   3. OpenMP runtime entry points in the binary itself (GOMP_parallel,
#      __kmpc_fork_call), which also catches a statically linked runtime.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 BUILD_LOG BINARY" >&2; exit 64; }
LOG=$1; BIN=$2
[ -r "$LOG" ] || { echo "FAIL cannot read build log $LOG" >&2; exit 2; }
[ -r "$BIN" ] || { echo "FAIL cannot read binary $BIN" >&2; exit 2; }

compile_lines=$(awk '
  /(^|[ \t])-c([ \t]|$)/ &&
  /(^|[ \t])(-fopenmp|-qopenmp|-fiopenmp|-openmp|-qsmp=omp|-mp)([ \t]|$)/ { n++ }
  END { print n + 0 }' "$LOG")

dependencies=$(ldd "$BIN" 2>&1) || dependencies=""
runtime=""
case $dependencies in
  *libgomp*) runtime=libgomp ;;
  *libiomp5*) runtime=libiomp5 ;;
  *libomp.so*) runtime=libomp ;;
  *libnvomp*) runtime=libnvomp ;;
esac

entry_points=$(grep -a -c -E 'GOMP_parallel|__kmpc_fork_call' "$BIN" || true)

if [ "$compile_lines" -gt 0 ] || [ -n "$runtime" ] || [ "${entry_points:-0}" -gt 0 ]; then
  echo "FAIL native reference was built with OpenMP (DY1 thread-sentinel race):" \
       "$compile_lines OpenMP compile line(s), runtime '${runtime:-none}'," \
       "${entry_points:-0} OpenMP entry point line(s); rebuild without OPENMP=true" >&2
  exit 2
fi
echo "OK single-threaded: 0 OpenMP compile lines, no OpenMP runtime linked, no OpenMP entry points"
