"""Seal native table paths without duplicating the driver's source table names.

All scientific bytes remain pinned by the original forecast decks. Python
handles file identity and symlinks only; it does not decode weather data.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('plan', type=Path)
p.add_argument('view', type=Path)
p.add_argument('receipt', type=Path)
a = p.parse_args()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan = json.loads(a.plan.read_text())
rows, extra, matches = [], {}, {}
for row in plan['decks']:
    path = a.plan.parent / row['path']
    assert sha(path) == row['sha256'], path
    deck = json.loads(path.read_text())
    command = deck['hex_command']
    source = Path(command[command.index('--mpas-source') + 1])
    source_tables = source / 'src/core_atmosphere/physics/physics_wrf/files'
    supplied = Path(command[command.index('--native-physics-tables') + 1])
    tables = {f.name: dict(path=str(f.resolve()), sha256=sha(f))
              for f in supplied.iterdir() if f.is_file() and
              (f.suffix in ('.TBL', '.DBL') or f.name.startswith(('RRTMG_', 'CCN_')))}
    for name, pinned in deck['auxiliary_inputs'].items():
        if Path(name).suffix in ('.TBL', '.DBL') or name.startswith(('RRTMG_', 'CCN_')):
            assert sha(Path(pinned['path'])) == pinned['sha256'], name
            if name in tables:
                assert tables[name]['sha256'] == pinned['sha256'], name
            tables[name] = pinned
    for name, pinned in tables.items():
        original = source_tables / name
        if original.is_file():
            assert sha(original) == pinned['sha256'], ('different scientific table', name)
            matches[name] = dict(source=str(original.resolve()), **pinned)
        else:
            if name in extra:
                assert extra[name]['sha256'] == pinned['sha256'], name
            extra[name] = pinned
    rows.append(dict(path=path, deck=deck, old_sha256=sha(path)))
a.view.mkdir(parents=True, exist_ok=False)
for name, pinned in extra.items():
    (a.view / name).symlink_to(Path(pinned['path']).resolve())
for item, row in zip(rows, plan['decks']):
    deck = item['deck']
    command = deck['hex_command']
    command[command.index('--native-physics-tables') + 1] = str(a.view.resolve())
    item['path'].write_text(json.dumps(deck, indent=2) + '\n')
    row['sha256'] = sha(item['path'])
a.plan.write_text(json.dumps(plan, indent=2) + '\n')
a.receipt.write_text(json.dumps(dict(
    scope='identical pinned table content; source tables retained; only duplicate path staging removed',
    identical_source_tables=matches, additional_pinned_tables=extra,
    view=str(a.view.resolve()), decks=[dict(path=str(x['path']),
        original_sha256=x['old_sha256'], sha256=sha(x['path'])) for x in rows],
    plan_sha256=sha(a.plan), scientific_bytes_changed=False,
    native_reference_changed=False, forecast_proven=False), indent=2) + '\n')
print(len(matches), 'identical source tables;', len(extra), 'additional pinned tables')
