#!/usr/bin/env bash
# 2026-10-14 ruling: unit acceptance is module replay plus existing reviewed sheets.
# Keep the older entry point compatible without starting any new forecasts.
set -euo pipefail
checkout=$(realpath "${1:?usage: reproduce.sh HEX_CHECKOUT}")
exec bash "$checkout/tree/tests/gsl_proof/replay.sh" "$checkout"
