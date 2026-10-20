#!/usr/bin/env bash
# Strict/bounds standalone diagnostics through the same Rust record adapter.
# MPAS v8.4.2 physics_wrf/module_bl_ysu.F:16-475.
set -euo pipefail
exec taskset -c 8-23 python "$(dirname "${BASH_SOURCE[0]}")/replay_real.py" --backend "${YSU_REAL_BACKEND:-native}" "$@"
