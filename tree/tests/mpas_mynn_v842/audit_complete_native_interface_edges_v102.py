"""Audit the complete remaining native interface coverage ledger.

MPAS bl_mynn.F90:63-124,434; module_bl_mynn.F:306,462;
bl_mynn_pre.F:19-57,142; bl_mynn_post.F:19-57,152.
Only original caller contracts and uniquely identified character assignments
are classified. Descriptor and science branches remain open; no output word,
test profile, optional feature, or native read is removed.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from audit_complete_native_driver_edges_v83 import gcov_edges


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(a):
    result = copy.deepcopy(json.loads(a.classified.read_text()))
    contract = json.loads(a.contract.read_text())
    driver = a.native / 'src/core_atmosphere/physics/mpas_atmphys_driver_pbl.F'
    kinds = a.native / 'src/framework/mpas_kind_types.F'
    assert re.search(r'StrKIND\s*=\s*512', kinds.read_text())
    lines = driver.read_text().splitlines()
    assert lines[734].strip().lower() == 'character(len=strkind):: errmsg'
    assert lines[801].strip().lower() == 'character(len=strkind):: errmsg'
    assert contract['calls'][0]['optional_arguments']['pattern_spp']['present']
    cfgs = {n: (a.cfg / ('run_adapter-' + n + '.016t.cfg')).read_text()
            for n in result['science_sources']}
    graphs = {n: gcov_edges(a.graph / (n + '.gcov'))
              for n in result['science_sources']}
    unused = {'bl_mynn_pre_init', 'bl_mynn_pre_finalize',
              'bl_mynn_post_init', 'bl_mynn_post_finalize', 'bl_mynn_finalize'}
    # Search every Fortran source in the pin, including framework and drivers;
    # strip comments before finding calls. The file inventory is authenticated.
    inventory = {}
    calls = {name: [] for name in unused}
    for p in sorted(a.native.rglob('*')):
        if p.is_file() and p.suffix.lower() in ('.f', '.f90'):
            inventory[str(p.relative_to(a.native))] = sha(p)
            for no, line in enumerate(p.read_text(errors='strict').splitlines(), 1):
                code = line.split('!', 1)[0].lower()
                for name in unused:
                    if re.search(r'\bcall\s+' + name + r'\b', code):
                        calls[name].append(dict(path=str(p), line=no, code=line))
    assert not any(calls.values()), calls
    reasons = {}
    # Source-mapped bodies have one optimized gcov conditional at the blank
    # character assignment. Both early CFG predicates are the same length>0
    # test; no ordinal correspondence to the early CFG is assumed.
    for file, line in [('bl_mynn.F90', 434), ('module_bl_mynn.F', 306),
                       ('bl_mynn_pre.F', 142), ('bl_mynn_post.F', 152)]:
        at = [s for s in cfgs[file].splitlines()
              if re.search(r'\[' + re.escape(file) + ':' + str(line) + r':', s)]
        conditions = [s for s in at if '] if (' in s]
        assert len(conditions) == 2
        assert all(re.search(r'if \(D\.\d+ > 0\)', s) for s in conditions)
        assert any('= _errmsg;' in s for s in at)
        reason = dict(reason='Pinned MPAS driver declares errmsg with StrKIND=512; both adapters forward its length unchanged. The zero-length branch of this uniquely source-mapped blank assignment cannot be taken.',
                      original_cfg=at)
        reasons[(file, line, 1)] = reason
    # bl_mynn_init has no science conditional: its only CFG predicates are
    # duplicate character-length tests. The optimized graph has one pair.
    init = cfgs['bl_mynn.F90'].split(';; Function bl_mynn_init ', 1)[1]
    conditions = [s for s in init.splitlines() if '] if (' in s]
    assert len(conditions) == 2
    assert all(re.search(r'if \(D\.\d+ > 0\)', s) for s in conditions)
    assert 'call bl_mynn_init(' in driver.read_text().lower()
    reasons[('bl_mynn.F90', 109, 1)] = dict(
        reason='The initialization routine has only blank-error-string length predicates; the pinned initialization caller supplies StrKIND=512.',
        original_cfg=conditions)
    reasons[('module_bl_mynn.F', 462, 2)] = dict(
        reason='The pinned MPAS caller always supplies pattern_spp. The absent-pointer arm of the second conjunction predicate is impossible, including when stochastic forcing is enabled.',
        original_cfg=[s for s in cfgs['module_bl_mynn.F'].splitlines()
                      if '[module_bl_mynn.F:462:' in s and '] if (' in s])
    changed = []
    for row in result['uncovered_edges']:
        key = (row['file'], row['line'], row['branch'])
        proof = reasons.get(key)
        if row['routine'] in unused:
            proof = dict(reason='No call to this adapter lifecycle routine exists in any pinned MPAS Fortran source; the complete caller search is retained. The routine remains ported and no scientific word is excluded.')
        if proof and row['classification'].startswith('OPEN'):
            symbol = '__' + ('module_bl_mynn' if row['file'] == 'module_bl_mynn.F'
                            else Path(row['file']).stem.lower()) + '_MOD_' + row['routine']
            row['gcov_arc'] = graphs[row['file']][(symbol, row['line'], row['branch'])]
            row.update(proof)
            row['classification'] = 'UNREACHABLE in pinned MPAS caller interfaces: lifecycle or character contract'
            changed.append(key)
    result['counts'] = {n: sum(r['classification'].startswith(n) for r in result['uncovered_edges'])
                        for n in ('OPEN', 'UNREACHABLE')}
    result['complete_interface_audit'] = dict(
        scope=__doc__, previous_ledger_sha256=sha(a.classified),
        contract_sha256=sha(a.contract), native_fortran_inventory=inventory,
        original_lifecycle_calls=calls, newly_classified=changed,
        witness_hashes={str(p): sha(p) for p in [driver, kinds]},
        output_words_excluded=0, full_branch_coverage=False)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('classified', 'contract', 'cfg', 'graph', 'native', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    main(p.parse_args())
