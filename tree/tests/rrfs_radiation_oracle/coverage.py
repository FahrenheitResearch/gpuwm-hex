"""Summarize authentic GNU gcov JSON; retain every unexecuted branch location."""
from pathlib import Path
import gzip
import json
import sys

root, destination = map(Path, sys.argv[1:3])
files = []
for path in sorted(root.glob('*.gcov.json.gz')):
    packet = json.loads(gzip.decompress(path.read_bytes()))
    for source in packet['files']:
        original = Path(packet['current_working_directory'])/source['file']
        text = original.read_text().splitlines() if original.is_file() else []
        branches = [(line['line_number'], index, branch['count'])
                    for line in source['lines']
                    for index, branch in enumerate(line.get('branches', []))]
        files.append({'source':source['file'], 'gcov_packet':path.name,
            'compile_directory':packet['current_working_directory'],
            'lines':len(source['lines']),
            'lines_executed':sum(line['count'] > 0 for line in source['lines']),
            'branches':len(branches),
            'branches_executed':sum(count > 0 for _, _, count in branches),
            'unexecuted_branches':[{'line':line, 'branch':index,
                'source_line':text[line-1].strip() if line <= len(text) else None}
                for line, index, count in branches if count == 0],
            'unexecuted_functions':[f['name'] for f in source.get('functions',[])
                                     if f['execution_count'] == 0]})
destination.write_text(json.dumps({'schema':'rrfs-radiation-gcov-v1',
    'scope':sys.argv[3] if len(sys.argv)>3 else 'standalone synthetic columns; no forecast coverage',
    'files':files}, indent=2)+'\n')
