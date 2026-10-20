#!/usr/bin/env bash
set -e
p21_adapter_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec taskset -c 8-23 python "$p21_adapter_dir/f1_adapter.py" "$@"
