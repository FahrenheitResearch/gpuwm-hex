"""Prepare, without editing owner files, a complete NEAREST transport proposal.

The copies are diagnostic fixtures. o1 owns timestamp metadata selection and
Rust owns every field read and permutation. MPAS mpas_atm_core.F:222-233.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hex', type=Path, required=True)
parser.add_argument('--rust-source', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)


def replace(source, before, after):
    if source.count(before) != 1:
        raise SystemExit('owner source changed; inspect before proposing: ' + before[:70])
    return source.replace(before, after)


rust = args.rust_source.read_text()
updated = replace(rust, '    path: String,\n}', '    path: String,\n    record: usize,\n}')
updated = replace(updated, '            path: path.into(),\n',
                  '            path: path.into(),\n            record: 0,\n')
updated = replace(updated, '    sources: Vec<String>,\n',
                  '    sources: Vec<String>,\n    #[serde(default)]\n'
                  '    source_records: std::collections::BTreeMap<String, usize>,\n')
updated = replace(updated, "    schema: &'static str,\n    fields: Vec<Record>,",
                  "    schema: &'static str,\n"
                  '    source_records: std::collections::BTreeMap<String, usize>,\n'
                  '    fields: Vec<Record>,')
updated = replace(updated, '        let starts = vec![0; counts.len()];',
                  '        let mut starts = vec![0; counts.len()];')
updated = replace(updated, '            counts[0] = 1;\n',
                  '            if f.record >= counts[0] {\n'
                  '                return Err(format!("{name}: Time record {} exceeds {}", f.record, counts[0]));\n'
                  '            }\n            starts[0] = f.record;\n            counts[0] = 1;\n')
updated = replace(updated, '        .map(|p| File::open(p))',
                  '        .map(|p| {\n            let mut file = File::open(p)?;\n'
                  '            file.record = r.source_records.get(p).copied().unwrap_or(0);\n'
                  '            Ok::<File, String>(file)\n        })')
updated = replace(updated, '    let files: Vec<File> = r\n',
                  '    if r.source_records.keys().any(|p| !r.sources.contains(p)) {\n'
                  '        return Err("selected record source absent from request sources".into());\n'
                  '    }\n    let files: Vec<File> = r\n')
updated = replace(updated, '        schema: "mpas-native-pool-payload-v1",\n',
                  '        schema: "mpas-native-pool-payload-v1",\n'
                  '        source_records: files.iter().map(|f| (f.path.clone(), f.record)).collect(),\n')

pool_path = 'tree/src/hexcore/mpasphys/pool_v842.py'
pool = (args.hex / pool_path).read_text()
new_pool = replace(pool, '                          namelist: str, sources: list[Path], output: Path) -> RegistryLayout:',
                   '                          namelist: str, sources: list[Path], output: Path, *,\n'
                   '                          stream_seek: Mapping[str, tuple[Path, str]] | None = None) -> RegistryLayout:')
new_pool = replace(new_pool, '    request.write_text(json.dumps(layout.request(sources), indent=2) + "\\n")',
                   '    transport = layout.request(sources)\n'
                   '    # MPAS mpas_atm_core.F:222-233; mpas_io_streams.F:153-339.\n'
                   '    definitions = {s.get("name"): s for s in layout.root.findall("./streams/stream")}\n'
                   '    selections = []\n'
                   '    for name, (path, whence) in (stream_seek or {}).items():\n'
                   '        stream = definitions[name]\n'
                   '        if not layout.enabled(stream):\n            continue\n'
                   '        if "input" not in stream.get("type", "").split(";"):\n'
                   '            raise ValueError(f"{name}: cannot seek a non-input stream")\n'
                   '        from ..restart_v842 import seek_stream\n'
                   '        source_path = str(path.resolve())\n'
                   '        if source_path not in transport["sources"]:\n'
                   '            raise ValueError(f"{name}: selected source absent from pool inputs")\n'
                   '        target = layout.configs["config_start_time"]\n'
                   '        record, stamp = seek_stream(path, valid_time=target, whence=whence,\n'
                   '                                    scratch=output / "stream-seek")\n'
                   '        records = transport.setdefault("source_records", {})\n'
                   '        if source_path in records and records[source_path] != record:\n'
                   '            raise ValueError("input streams select different records from one source")\n'
                   '        records[source_path] = record\n'
                   '        selections.append(dict(stream=name, source=source_path, target=target,\n'
                   '                               whence=whence, record=record, selected_time=stamp))\n'
                   '    transport["source_seek_witness"] = selections\n'
                   '    request.write_text(json.dumps(transport, indent=2) + "\\n")')
new_pool = replace(new_pool, '    subprocess.run([str(reader), str(request), str(output)], check=True)\n',
                   '    subprocess.run([str(reader), str(request), str(output)], check=True)\n'
                   '    # An older v1 reader silently ignores source_records; detect that defect.\n'
                   '    if selections:\n'
                   '        actual = json.loads((output / "manifest.json").read_text()).get("source_records", {})\n'
                   '        if any(actual.get(path) != record for path, record in transport["source_records"].items()):\n'
                   '            raise ValueError("Rust pool reader did not witness the selected input records")\n')
forecast_path = 'tree/src/hexcore/mpasphys/forecast_v842.py'
forecast = (args.hex / forecast_path).read_text()
new_forecast = replace(forecast, "    sources = list(dict.fromkeys((*sources,*bindings.values())))\n",
                       '    from ..iau_v842 import IAU_STREAM_ID, IAU_STREAM_WHENCE\n'
                       '    stream_seek = {IAU_STREAM_ID: (bindings[IAU_STREAM_ID], IAU_STREAM_WHENCE)} \\\n'
                       '                  if IAU_STREAM_ID in bindings else {}\n'
                       '    sources = list(dict.fromkeys((*sources,*bindings.values())))\n')
new_forecast = replace(new_forecast, "                                   scratch/'native-pools')",
                       "                                   scratch/'native-pools', stream_seek=stream_seek)")
rust_path = 'tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_physics_pool.rs'
changes = [(pool_path, pool, new_pool), (forecast_path, forecast, new_forecast),
           (rust_path, rust, updated)]
for path, _, value in changes:
    if path.endswith('.py'):
        ast.parse(value)
    (args.out / Path(path).name).write_text(value)
patch = ''.join(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                  fromfile='a/' + path, tofile='b/' + path)) for path, before, after in changes)
(args.out / 'integrator_iau_source_seek.patch').write_text(patch)
receipt = dict(schema='mpas.iau-source-seek-proposal/v1', applied=False, acceptance=False,
               scope='copied diagnostic proposal; production owners must reconcile',
               source_sha256={path: hashlib.sha256(before.encode()).hexdigest()
                              for path, before, _ in changes},
               proposed_sha256={path: hashlib.sha256(after.encode()).hexdigest()
                                for path, _, after in changes},
               patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
               requires='o1 restart_v842.seek_stream and its compiled Rust stream artifact')
(args.out / 'integrator_iau_source_seek.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
