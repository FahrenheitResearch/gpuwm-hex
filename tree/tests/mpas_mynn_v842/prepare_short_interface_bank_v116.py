"""Prepare whole-driver active-depth probes, retaining all caller storage words.

MPAS module_bl_mynn.F:347-376,543-741 allocates and copies kts:kte
sections. MPAS bl_mynn_subroutines.F90:1116,2781,4543-4569,4901 has
depth-dependent loop arms. These additional probes change only the synthetic
driver's active extent, never the unchanged science or previous required banks.
"""
from pathlib import Path
import hashlib
import json
import shutil


def prepare():
    root = Path('cpuwork/short-interface-bank-v116')
    root.mkdir(exist_ok=False)
    previous = Path('cpuwork/absent-interface-bank-v114')
    for kind in ('native', 'host', 'coverage', 'bounds', 'snan'):
        target = root / kind
        target.mkdir()
        for p in (previous / kind).iterdir():
            if p.suffix in ('.F90', '.F', '.cpp', '.cuh') or p.name == 'Makefile':
                shutil.copy2(p, target / p.name)
        path = target / ('run_host_adapter.F90' if kind == 'host' else 'run_adapter.F90')
        text = path.read_text()
        assert text.count('do deck=4544,5055') == 1
        text = text.replace('do deck=4544,5055', 'do deck=5056,5119').replace('deck-4544', 'deck-5056')
        text = text.replace('  integer :: deck,step,c,k,unit\n',
            '  integer :: deck,step,c,k,unit,p12_active_depth\n'
            '  character(len=32) :: p12_depth_argument\n', 1)
        assert text.count('  call get_command_argument(1,outfile)') == 1
        text = text.replace('  call get_command_argument(1,outfile)',
            '  call get_command_argument(2,p12_depth_argument)\n'
            '  read(p12_depth_argument,*) p12_active_depth\n'
            '  if(p12_active_depth<1.or.p12_active_depth>nz) error stop "invalid synthetic extent"\n'
            '  call get_command_argument(1,outfile)', 1)
        assert text.count('    do step=1,2\n') == 1
        text = text.replace('    do step=1,2\n',
            '    ! Change the active slice only; retain every allocated storage word.\n'
            '    kte=p12_active_depth\n    do step=1,2\n', 1)
        path.write_text(text)
    files = {str(p.relative_to(root)): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
             for p in root.rglob('*') if p.is_file()}
    (root / 'scope.json').write_text(json.dumps(dict(scope=__doc__, process_count=1,
        active_depths=[1,2,3,4,8,16,55], profiles_per_depth=64, calls_per_depth=128,
        caller_storage_depth=55, previous_banks_retained=True,
        every_actual_output_storage_word_graded=True, files=files,
        excluded_words=[]), indent=2) + '\n')


if __name__ == '__main__':
    prepare()
