#!/usr/bin/env bash
# Hash selected variable values without run-specific NetCDF global attributes.
# Nine and seventeen significant digits preserve finite float32 and float64
# values respectively. This hashes canonical ncdump text, not file bytes or
# NaN payload bits. Compare the schema separately when comparing datasets.
set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "usage: $0 FILE VARIABLE [VARIABLE ...]" >&2
    exit 2
fi
export LC_ALL=C
input=$1
shift
for variable in "$@"; do
    digest=$(ncdump -p 9,17 -v "$variable" "$input" | sed -n '/^data:/,$p' | sha256sum)
    printf '%s\t%s\n' "$variable" "${digest%% *}"
done
