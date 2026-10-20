#!/usr/bin/env bash
# G0 CPU/citations, G1 cached dry + suite-none physics, G2 landed suite/options.
# Usage: mpas_gate.sh --gates G1,G2 --out DIR
# MPAS_GATE_INDEX / MPAS_RATCHET override receipts/mpas-gate-index.json and
# receipts/ratchet.json; --index / --ratchet override those environment paths.
# --stage all (G0-G2) and individual --stage G0/G1/G2/G3 remain supported.
# A shared host names its GPU/build queues and the gate CPU set in site/runners.env or the
# environment (hexcore.site_runner); elsewhere the gate runs directly on every CPU.
set -euo pipefail
tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$tree/src:$tree/tools:${PYTHONPATH:-}"
cpuset=$("${PYTHON:-python3}" -c "from hexcore import site_runner; print(site_runner.gate_cpuset())")
if [ -n "$cpuset" ]; then
  exec taskset -c "$cpuset" "${PYTHON:-python3}" -m hexcore.verify_mpas "$@"
fi
exec "${PYTHON:-python3}" -m hexcore.verify_mpas "$@"
