"""Record each native branch not taken, including its routine and source line."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

SOURCES = ('module_ra_rrtmg_sw.F', 'module_ra_rrtmg_sw_aerosols.F',
           'module_mp_thompson_aerosols.F', 'module_ra_rrtmg_vinterp.F',
           'mpas_atmphys_rrtmg_swinit.F', 'oracle_solar.F90',
           'module_ra_rrtmg_lw.F', 'mpas_atmphys_functions.F')
# SW dependencies only; the unrelated longwave scheme is another lane.
SCOPES = {'module_ra_rrtmg_lw.F': ((1751,1820),(1878,1930),(12308,12538)),
          'mpas_atmphys_functions.F': ((156,186),)}


def coverage(directory, receipt):
    directory = Path(directory).resolve()
    process = subprocess.run(['taskset', '-c', '8-23', 'gcov', '-b', '-c', *SOURCES],
                             cwd=directory, text=True, capture_output=True, check=True)
    Path(receipt).with_suffix('.log').write_text(process.stdout + process.stderr)
    rows = []
    for name in SOURCES:
        routine, line, source, branches, taken, uncovered = '', 0, '', 0, 0, []
        for text in (directory / (name + '.gcov')).read_text().splitlines():
            match = re.match(r'\s*[^:]+:\s*(\d+):(.*)', text)
            if match:
                line, source = int(match[1]), match[2].strip()
                match = re.search(r'\b(?:subroutine|function)\s+(\w+)', source, re.I) if not source.lower().startswith(('!','end ')) else None
                if match:
                    routine = match[1]
            match = re.match(r'branch\s+(\d+)\s+(.*)', text)
            if match:
                if name in SCOPES and not any(first<=line<=last for first,last in SCOPES[name]):
                    continue
                branches += 1
                status = match[2]
                executed = re.match(r'taken\s+(\d+)', status)
                if executed and int(executed[1]):
                    taken += 1
                else:
                    uncovered.append({'routine': routine, 'line': line, 'branch': int(match[1]),
                                      'source': source, 'gcov': status})
        rows.append({'file': name, 'source_sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                     'scope': SCOPES.get(name, 'whole file'),
                     'branches': branches, 'branches_taken': taken, 'untaken': uncovered})
    result = {'authority': 'gfortran gcov of pristine pinned MPAS standalone fixtures',
              'whole_model_proof': False, 'sources': rows}
    Path(receipt).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory'); p.add_argument('receipt')
    result = coverage(**vars(p.parse_args()))
    print(json.dumps([{k:v for k,v in row.items() if k != 'untaken'} for row in result['sources']], indent=2))
