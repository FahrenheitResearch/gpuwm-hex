"""Write explicit unreached native gcov branches; no coverage is inferred."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inventory(gcov):
    text = gcov.read_text()
    routine, line, statement = None, 0, ''
    missed_lines, missed_edges, total_edges, taken_edges = [], [], 0, 0
    for record in text.splitlines():
        match = re.match(r'\s*([^:]+):\s*(\d+):(.*)', record)
        if match:
            count, line, statement = match[1].strip(), int(match[2]), match[3].strip()
            declaration = re.match(r'(?:[\w()=,*]+\s+)*?(?:subroutine|function)\s+(\w+)(?:\s|\(|$)', statement, re.I)
            if declaration:
                routine = declaration[1].lower()
            if count in ('#####', '====='):
                missed_lines.append(dict(routine=routine, line=line, statement=statement))
        elif record.startswith('branch '):
            total_edges += 1
            if 'never executed' in record or re.search(r'taken 0(?:\s|$)', record):
                missed_edges.append(dict(routine=routine, line=line, statement=statement,
                                         edge=record.strip()))
            else:
                taken_edges += 1
    invariants = {
        ('cumastr_new',983): {
            'classification':'unreachable under native whole-module dataflow',
            'source_windows':['module_cu_tiedtke.F:919-938','module_cu_tiedtke.F:952-983'],
            'proof':'ictop0 begins at kcbot-1 and can only decrease; active type-1 ihmin begins at kcbot. The only scan assignment sets ihmin=jk under jk>=ictop0. Thus ihmin<ictop0 cannot hold.'},
        ('cumastr_new',1103): {
            'classification':'unreachable under native whole-module dataflow',
            'source_windows':['module_cu_tiedtke.F:1332','module_cu_tiedtke.F:2016-2019',
                              'module_cu_tiedtke.F:2046-2074','module_cu_tiedtke.F:2158-2217'],
            'proof':'The missing short-circuit edge requires pmfd<0 with loddraf false. cuini initializes pmfd=0. cudlfs creates negative mass flux only when setting loddraf true; cuddraf modifies it only under loddraf. No intervening routine clears that flag before the closure test.'}}
    for edge in missed_edges:
        if (edge['routine'],edge['line']) in invariants:
            edge.update(invariants[(edge['routine'],edge['line'])])
    return dict(gcov_sha256=hashlib.sha256(gcov.read_bytes()).hexdigest(),
                branch_edges=total_edges, branch_edges_taken=taken_edges,
                unreached_edges=missed_edges, unexecuted_lines=missed_lines,
                note='Raw gcov denominator is unchanged. kcall=4, optional absence, empty tiles and downdraft-top rejection are direct boundary probes; whole-module columns do not reach every public routine arm. Only the explicitly classified native dataflow edges have a source proof; no other missing edge is waived.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('gcov', type=Path)
    p.add_argument('receipt', type=Path)
    args = p.parse_args()
    args.receipt.write_text(json.dumps(inventory(args.gcov), indent=2) + '\n')
