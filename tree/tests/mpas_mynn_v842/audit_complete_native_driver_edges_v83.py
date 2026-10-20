"""Bind all remaining native branch gaps to original CFG and gcov evidence.

Only branches impossible at the two pinned caller interfaces are classified.
Synthetic words, optional outputs and unclassified branches remain required.
The complete original coverage union is retained without changing counts.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import re


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def gcov_edges(p):
    routine = None
    source = None
    block = None
    rows = {}
    for line in p.read_text().splitlines():
        m = re.match(r'function (\S+) called ', line)
        if m:
            routine = m[1]
        m = re.match(r'\s*[^:]+:\s*(\d+)(?:-block (\d+))?(?::.*)?$', line)
        if m:
            source = int(m[1])
            block = int(m[2]) if m[2] else None
        m = re.match(r'branch\s+(\d+) (.*)', line)
        if m:
            key = (routine, source, int(m[1]))
            assert key not in rows, key
            rows[key] = dict(block=block, arc=m[2])
    return rows


def main(a):
    result = copy.deepcopy(json.loads(a.classified.read_text()))
    contract = json.loads(a.contract.read_text())
    mpas, wrapper = contract['calls']
    assert mpas['callee']['name'] == 'mynn_bl_driver'
    assert wrapper['callee']['name'] == 'bl_mynn_run'
    for name, h in contract['native_sources'].items():
        assert sha(Path(name)) == h
    cfgs = {}
    edges = {}
    for name in result['science_sources']:
        cfg = a.cfg / ('run_adapter-' + name + '.016t.cfg')
        cfgs[name] = cfg.read_text()
        gcov = a.graph / (name + '.gcov')
        edges[name] = gcov_edges(gcov)
    always_present = mpas['optional_arguments']
    mmm_present = wrapper['optional_arguments']
    # These absent chemistry inputs are never enabled by the MPAS caller.
    init_path = Path(mpas['path']).parent / 'mpas_atmphys_init.F'
    driver = init_path.read_text().splitlines()
    assert driver[505].split('!',1)[0].strip().lower() == 'f_qoz = .false.'
    assert driver[523].split('!',1)[0].strip().lower() == 'f_nbca = .false.'
    fields = {393:'qc',400:'qi',407:'qs',414:'nc',421:'ni',428:'nifa',
              435:'nwfa',442:'nbca',449:'qoz',636:'rqcblten',643:'rqiblten',
              650:'rqsblten',657:'rqozblten',664:'rncblten',671:'rniblten',
              678:'rnifablten',685:'rnwfablten',692:'rnbcablten'}
    reasons = {}
    for line, field in fields.items():
        present = always_present[field]['present']
        if not present:
            assert field in ('nbca','qoz','rnbcablten','rqozblten')
        statements = [s for s in cfgs['module_bl_mynn.F'].splitlines()
                      if re.search(r'\[module_bl_mynn.F:' + str(line) + r':',s)]
        conditions = [s for s in statements if '] if (' in s]
        assert len(conditions) == 2 and f'if ({field} != 0B)' in conditions[1]
        reasons[('module_bl_mynn.F',line,3)] = dict(
            reason=(f'Pinned MPAS driver supplies {field}; its null-pointer edge '
                    'cannot be taken.' if present else
                    f'Pinned MPAS driver omits {field} and fixes its chemistry '
                    'flag false at mpas_atmphys_init.F:506/524; the '
                    'second condition is never evaluated.'),
            cfg_conditions=conditions)
    for name, line, call in [('bl_mynn.F90',1152,mmm_present),
                             ('module_bl_mynn.F',730,always_present)]:
        assert all(call[n]['present'] for n in ('qwt','qbuoy','qshear','qdiss','dqke'))
        statements = [s for s in cfgs[name].splitlines()
                      if re.search(r'\['+re.escape(name)+':'+str(line)+':',s)]
        conditions = [s for s in statements if '] if (' in s]
        assert len(conditions) == 3
        branches = (1,3,5) if name == 'bl_mynn.F90' else (3,5,7)
        for branch in branches:
            reasons[(name,line,branch)] = dict(
                reason='Both pinned caller interfaces always supply all five budget arrays; optional absence is impossible.',
                cfg_conditions=conditions)
    assert mmm_present['dqke']['present']
    reasons[('bl_mynn.F90',439,1)] = dict(
        reason='Pinned WRF adapter always supplies dqke to MMM; optional absence is impossible.')
    dd = ('edmf_a_dd','edmf_w_dd','edmf_qt_dd','edmf_thl_dd','edmf_ent_dd','edmf_qc_dd')
    assert all(not mmm_present[n]['present'] for n in dd)
    statements = [s for s in cfgs['bl_mynn.F90'].splitlines()
                  if '[bl_mynn.F90:1216:' in s]
    assert len([s for s in statements if '] if (' in s]) == 3
    for branch in (0,2,3,4,5):
        reasons[('bl_mynn.F90',1216,branch)] = dict(
            reason='Pinned WRF adapter omits all six downdraft diagnostic arrays. First conjunction is false; later tests are never evaluated.',
            cfg_conditions=statements)
    for branch in (0,1):
        reasons[('bl_mynn.F90',1217,branch)] = dict(
            reason='The diagnostic copy loop is inside the conjunction that is always false because all six diagnostic arrays are omitted by the pinned WRF adapter.')
    audit_rows = []
    changed = []
    for row in result['uncovered_edges']:
        key = (row['file'],row['line'],row['branch'])
        symbol = '__' + ('module_bl_mynn' if row['file']=='module_bl_mynn.F' else
                         Path(row['file']).stem.lower()) + '_MOD_' + row['routine']
        graph = edges[row['file']][(symbol,row['line'],row['branch'])]
        proof = reasons.get(key)
        if proof and row['classification'].startswith('OPEN'):
            row['classification'] = 'UNREACHABLE in pinned MPAS caller interfaces: optional contract'
            row.update(proof)
            row['gcov_arc'] = graph
            changed.append(key)
        audit_rows.append(dict(file=row['file'],line=row['line'],branch=row['branch'],
                               routine=row['routine'],gcov_arc=graph,
                               classification=row['classification']))
    result['counts'] = {n:sum(row['classification'].startswith(n) for row in result['uncovered_edges'])
                        for n in ('OPEN','UNREACHABLE')}
    result['optional_contract_audit'] = dict(scope=__doc__,
        contract_sha256=sha(a.contract), previous_classified_sha256=sha(a.classified),
        graph_admission_sha256=sha(a.graph/'admission.json'),
        cfg_admission_sha256=sha(a.cfg/'admission.json'),
        witness_files={str(p):sha(p) for p in
                       [init_path]+[p for root in (a.cfg,a.graph)
                                   for p in root.iterdir() if p.is_file()]},
        all_original_gap_edges=audit_rows,newly_classified=changed,
        output_words_excluded=0,full_branch_coverage=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('classified','contract','cfg','graph','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
