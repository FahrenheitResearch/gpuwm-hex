#!/usr/bin/env bash
# Build only in an owned output directory; dependencies stay read-only.
set -euo pipefail
engine=$(cd "${1:?engine source}" && pwd)
out=$(mkdir -p "${2:?build output}" && cd "$2" && pwd)
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 - "$root" "$engine" "$out" <<'PY'
from pathlib import Path
import shutil, sys
root, engine, out = map(Path, sys.argv[1:])
source = root / 'tools/rustwx/crates/rw-s2s-energy'
target = out / 'source'
shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('target'))
p = target / 'Cargo.toml'
s = p.read_text().replace('../../vendor/netcrust', str(engine / 'tools/rustwx/vendor/netcrust'))
s = s.replace('../rw-store', str(engine / 'tools/rustwx/crates/rw-store'))
p.write_text(s)
PY
# Load the engine's vendored dependency configuration through cwd.
cd "$engine/tools/rustwx"
cargo test --release --locked --manifest-path "$out/source/Cargo.toml" --target-dir "${CARGO_TARGET_DIR:-$out/target}" -j "${CARGO_BUILD_JOBS:-14}"
cargo build --release --locked --manifest-path "$out/source/Cargo.toml" --target-dir "${CARGO_TARGET_DIR:-$out/target}" -j "${CARGO_BUILD_JOBS:-14}"
cp "${CARGO_TARGET_DIR:-$out/target}/release/rw_s2s_energy" "$out/rw_s2s_energy"
"$out/rw_s2s_energy" --abi
python3 - "$root" "$engine" "$out" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root, engine, out = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'schema': 'gpuwm-hex.s2s-energy-build/v1', 'engine_source': str(engine),
          'sha256': sha(out / 'rw_s2s_energy'), 'bytes': (out / 'rw_s2s_energy').stat().st_size,
          'abi': subprocess.check_output([str(out / 'rw_s2s_energy'), '--abi'], text=True).strip(),
          'recipe_sha256': sha(root / 'tools/build_s2s_energy.sh'),
          'source_sha256': sha(out / 'source/src/main.rs'),
          'source_files_sha256': {str(p.relative_to(out / 'source')): sha(p)
                                 for p in sorted((out / 'source/src').rglob('*.rs'))},
          'cargo_lock_sha256': sha(out / 'source/Cargo.lock')}
(out / 'provenance.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
