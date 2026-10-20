#!/usr/bin/env bash
# Integrator 2026-10-14 03:00Z: landing uses real-input replay; no extra forecast.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: reproduce.sh <hex checkout>' >&2; exit 2; }
exec bash "$1/tree/tests/mpas_ysu_v842/replay.sh" "$1"
