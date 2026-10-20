"""Seal complete real replay metadata; native records remain Rust-owned."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


BANKS = {
    'cloud': ('real-cloud-bank', '71e05f84562b5536dbe6d24c7c188d25fa5f462f811687e5aab6f0fc7008e13c'),
    'scheme': ('real-radiation-bank', 'a60ca0489b0fb452ee2651686b77f4717cb339d494807da072ad973d2a26c75b'),
    'driver': ('real-radiation-driver-bank', '9b752169b5ceacf9b164ebcc4683b1ee0033b930e6add76f1acb4a27a7182f41'),
}
REPLAYS = {
    'cloud': ('cloud-clean-replay.json', 'cloud_replay.py', 'r9.pristine-real-cloudiness-replay/v1'),
    'scheme': ('radiation-clean-scheme.json', 'radiation_replay.py', 'r9.pristine-real-radiation-replay/v1'),
    'driver': ('radiation-clean-driver.json', 'radiation_driver_replay.py', 'r9.pristine-real-radiation-slots/v1'),
}
PARENTS = {
    'radshared_v842': 'f6d6ee2b823aa102ccd6392a3250c37fd9dbd4a7860a0f5ba2355b0c83c4b902',
    'rrtmg_sw_v842': '2d73596d3d955880d5cb1cffabc16884b33dbe516a738ae20330f810c94f552b',
    'rrtmg_lw_v842': '7e9e004a7a4bc0b6d47aa0be004967d326ac5a8f57591cb50dd8cd46bfe3a5dc',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inventory(manifest):
    """Obtain the exact recorded case/field roster without decoding payloads."""
    files = manifest['files']
    require(bool(files), 'Native inventory is empty')
    require(all(Path(name).name == name for name in files), 'Native member escapes bank')
    before, after = set(), set()
    scopes = {name.removesuffix('.scope.txt') for name in files if name.endswith('.scope.txt')}
    for name in files:
        for phase, result in [('.pre.', before), ('.post.', after)]:
            if phase in name and name.endswith('.bin'):
                case, field = name.removesuffix('.bin').split(phase, 1)
                result.add((case, field))
    require(before and before == after, 'Native PRE/POST inventory is incomplete')
    require({case for case, _ in after} == scopes, 'Native case scopes differ from records')
    return after


def validate_receipt(receipt, *, tip, schema, bank_sha, replay_sha, production_sha, expected):
    require(receipt['schema'] == schema, 'Replay schema differs')
    require(receipt['checkout_tip'] == tip, 'Stale replay checkout')
    require(receipt['native_manifest_sha256'] == bank_sha, 'Replay bank authority differs')
    require(receipt['replay_source_sha256'] == replay_sha, 'Replay source differs')
    require(receipt['production_sha256'] == production_sha, 'Production source differs')
    require(receipt['bitwise_equal'] is True and receipt['comparison_complete'] is True,
            'Replay is unequal or incomplete')
    rows = receipt['comparisons']
    require(rows and len(rows) == receipt['arrays'], 'Replay array count differs')
    roster = [(row['case'], row['field']) for row in rows]
    require(len(set(roster)) == len(roster), 'Duplicate comparison record')
    require(set(roster) == expected, 'Not every native POST record was compared')
    cases = [case['case'] for case in receipt['cases']]
    require(len(cases) == len(set(cases)) and set(cases) == {case for case, _ in expected},
            'Replay case inventory differs')
    for row in rows:
        require(row.get('comparison_complete') is True and row.get('bitwise_equal') is True
                and not row.get('schema_mismatch'), 'Record comparison is incomplete or unequal')
        require(type(row['words']) is int and row['words'] > 0, 'Record word count is invalid')
        require(type(row['different_words']) is int and row['different_words'] == 0,
                'A recorded word differs or was not compared')
    require(type(receipt['words']) is int and receipt['words'] == sum(row['words'] for row in rows),
            'Replay word total differs')
    require(type(receipt['different_words']) is int and receipt['different_words'] == 0,
            'Replay difference total is invalid')
    preserved = receipt.get('preserved_words', receipt.get('preserved_input_words'))
    scientific = receipt['scientific_output_words']
    require(type(scientific) is int and scientific > 0 and type(preserved) is int
            and preserved >= 0 and scientific + preserved == receipt['words'],
            'Scientific and preserved word totals differ')


def loaded_source(entry, expected):
    require(Path(entry['path']).resolve() == expected.resolve(), 'Replay loaded another source path')
    require(entry['sha256'] == digest(expected), 'Loaded source hash differs')


def validate(checkout, lane):
    tip = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(checkout), 'diff', '--quiet', 'HEAD', '--',
                    'tree/src', 'tree/tests', 'tree/tools/battery'], check=True)
    production = checkout/'tree/src/hexcore/mpasphys/rrfs/radiation_rrfs.py'
    helper = checkout/'tree/tests/rrfs_radiation_oracle'
    validated = []
    for kind, (filename, source, schema) in REPLAYS.items():
        path = lane/'receipts'/filename
        receipt = json.loads(path.read_text())
        bank, bank_sha = BANKS[kind]
        manifest = lane/'scratch'/bank/'manifest.json'
        require(digest(manifest) == bank_sha, 'Pinned bank manifest changed')
        validate_receipt(receipt, tip=tip, schema=schema, bank_sha=bank_sha,
                         replay_sha=digest(helper/source), production_sha=digest(production),
                         expected=inventory(json.loads(manifest.read_text())))
        if kind == 'cloud':
            loaded_source(dict(path=receipt['production_source'], sha256=receipt['production_sha256']), production)
            entries = {'radshared_v842': dict(path=receipt['parent_source'], sha256=receipt['parent_sha256'])}
            require(receipt['observer_passivity'] is True, 'Cloud observer passivity is absent')
        else:
            require(receipt['native_process_count'] == 1, 'Native replay process count differs')
            modules = receipt['loaded_modules']
            loaded_source(modules['hexcore.mpasphys.rrfs.radiation_rrfs'], production)
            names = ['rrtmg_sw_v842', 'rrtmg_lw_v842']
            if kind == 'driver':
                names.append('radshared_v842')
                require(receipt['coefficient_manifest_sha256'] == BANKS['scheme'][1],
                        'Driver coefficient authority differs')
            entries = {name: modules['hexcore.mpasphys.'+name] for name in names}
        for name, entry in entries.items():
            parent = checkout/'tree/src/hexcore/mpasphys'/(name+'.py')
            if not parent.exists():
                parent = lane/'scratch/dependencies'/(name+'.py')
                require(digest(parent) == PARENTS[name], 'Unlanded parent authority differs')
            loaded_source(entry, parent)
        validated.append(dict(path=str(path), sha256=digest(path), arrays=receipt['arrays'],
                              words=receipt['words'], different_words=0))
    return dict(schema='r9.complete-replay-validation/v1', checkout_tip=tip,
                production_sha256=digest(production), receipts=validated,
                arrays=sum(row['arrays'] for row in validated),
                words=sum(row['words'] for row in validated), different_words=0,
                comparison_complete=True, bitwise_equal=True,
                scope='complete stated real module records; visual forecast acceptance is separate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'lane', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.checkout.resolve(), args.lane.resolve())
    except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        result = dict(schema='r9.complete-replay-validation/v1', bitwise_equal=False,
                      comparison_complete=False, error=str(error))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['bitwise_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
