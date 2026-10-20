#!/usr/bin/env bash
# Build the dependency-free reader without resolving engine path dependencies.
set -euo pipefail
out=$(mkdir -p "${1:?build output}" && cd "$1" && pwd)
shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$out/crate"
cp -a "$root/rustwx/crates/rw-surface/." "$out/crate/"
printf '\n[workspace]\n' >> "$out/crate/Cargo.toml"
# Preserve the shared release profile, which follows workspace settings.
sed -n '/^\[profile.release\]/,$p' "$root/rustwx/Cargo.toml" >> "$out/crate/Cargo.toml"
cargo generate-lockfile --offline --manifest-path "$out/crate/Cargo.toml"
cargo "${RW_SURFACE_CARGO_ACTION:-build}" --release --locked --offline \
    --manifest-path "$out/crate/Cargo.toml" --target-dir "$out/target" \
    -j "${CARGO_BUILD_JOBS:-2}" "$@"
