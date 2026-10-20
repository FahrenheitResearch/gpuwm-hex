"""Map every remaining gcov gap to a condition only when native graphs agree.

This is complete-module compiler evidence, not a reachability classification.
An early CFG block number is insufficient: the original gcno checksum, block
and both successor arcs must agree with the later compiler profile CFG.
Every unmapped edge remains open and all scientific words remain required.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from audit_complete_native_driver_edges_v83 import gcov_edges


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def graphs(text):
    current = None
    result = {}
    for line in text.splitlines():
        m = re.search(r"FUNCTION ident=\d+, lineno_checksum=(\w+), cfg_checksum=(\w+), `([^']+)'", line)
        if m:
            current = m[3]
            result[current] = dict(lineno_checksum=m[1], cfg_checksum=m[2], arcs={})
        m = re.search(r'block (\d+): ((?:\d+:[0-9a-f]+[^ ]* ?)+)$', line)
        if m and current:
            pairs = re.findall(r'(\d+):([0-9a-f]+)', m[2])
            result[current]['arcs'][int(m[1])] = [dict(destination=int(d), flags=f) for d, f in pairs]
    return result


def profile_blocks(text):
    result = {}
    for chunk in text.split(';; Function ')[1:]:
        symbol = re.match(r'\S+ \(([^,]+),', chunk)[1]
        blocks = {}
        for m in re.finditer(r'^  <bb (\d+)>[^\n]*:\n(.*?)(?=^  <bb |^}\s*$|\Z)', chunk, re.M | re.S):
            number, body = int(m[1]), m[2]
            condition = re.search(r'^  (\[[^\n]+?\] )?if \(([^\n]+)\)\n\s+goto <bb (\d+)>[^\n]*\n\s+else\n\s+goto <bb (\d+)>', body, re.M)
            if condition:
                blocks[number] = dict(source=condition[1], expression=condition[2],
                    true_destination=int(condition[3]), false_destination=int(condition[4]),
                    full_block=body)
        result[symbol] = blocks
    return result


def main(a):
    a.output.mkdir(exist_ok=False)
    ledger = json.loads(a.classified.read_text())
    old_admission = json.loads((a.graph / 'admission.json').read_text())
    new_admission = json.loads((a.profile / 'admission.json').read_text())
    for name, digest in old_admission['native_sources'].items():
        assert new_admission['original_files'][name] == digest
    native_graphs, blocks, edges = {}, {}, {}
    for name in ledger['science_sources']:
        stem = Path(name).stem
        new = a.profile / ('run_adapter-' + stem + '.gcno')
        dump = a.output / (new.name + '.dump')
        with dump.open('w') as out:
            subprocess.run(['gcov-dump', '-l', str(new)], stdout=out, check=True)
        native_graphs[name] = graphs((a.graph / (new.name + '.dump')).read_text())
        assert native_graphs[name] == graphs(dump.read_text()), 'Native coverage graph changed: ' + name
        matches = list(a.profile.glob('run_adapter-' + name + '.*.profile_estimate'))
        assert len(matches) == 1
        blocks[name] = profile_blocks(matches[0].read_text())
        edges[name] = gcov_edges(a.graph / (name + '.gcov'))
    rows = []
    for gap in ledger['uncovered_edges']:
        if not gap['classification'].startswith('OPEN'):
            continue
        name = gap['file']
        symbol = '__' + ('module_bl_mynn' if name == 'module_bl_mynn.F' else Path(name).stem.lower()) + '_MOD_' + gap['routine']
        edge = edges[name][(symbol, gap['line'], gap['branch'])]
        block = edge['block']
        condition = blocks[name].get(symbol, {}).get(block)
        arcs = native_graphs[name][symbol]['arcs'].get(block)
        mapped = condition is not None and arcs is not None and len(arcs) == 2 and sorted(x['destination'] for x in arcs) == sorted([condition['true_destination'], condition['false_destination']])
        # gcov prints the fallthrough arm first; gcno's storage order can be
        # different. Identify the arm by GCC's actual ARC_FALLTHROUGH bit,
        # never by branch ordinal, parity, or successor-list position.
        direction = None
        if mapped:
            fallthrough = '(fallthrough)' in edge['arc']
            candidates = [x for x in arcs
                          if bool(int(x['flags'], 16) & 4) == fallthrough]
            if len(candidates) == 1:
                destination = candidates[0]['destination']
                direction = dict(destination=destination,
                    predicate_value=destination == condition['true_destination'],
                    gcov_fallthrough=fallthrough,
                    gcno_flags=candidates[0]['flags'])
        rows.append(dict(gap=gap, gcov_arc=edge, original_successors=arcs,
                         mapped=bool(mapped), original_profile_condition=condition,
                         untaken_arm=direction))
    result = dict(scope=__doc__, process_count=1, classified_ledger_sha256=sha(a.classified),
        profile_admission_sha256=sha(a.profile / 'admission.json'),
        original_graph_admission_sha256=sha(a.graph / 'admission.json'),
        identical_complete_native_graphs=True, rows=rows,
        mapped=sum(r['mapped'] for r in rows), unmapped=sum(not r['mapped'] for r in rows),
        arms_mapped=sum(r['untaken_arm'] is not None for r in rows),
        exclusions=[], gcov_branches_reclassified=[], numerical_proof=False)
    (a.output / 'mapping.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['mapped', 'unmapped']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('classified', 'profile', 'graph', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    main(p.parse_args())
