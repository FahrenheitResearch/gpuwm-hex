#!/usr/bin/env bash
set -euo pipefail
exec python "$(dirname "$0")/replay_f1_host.py" "$@"
