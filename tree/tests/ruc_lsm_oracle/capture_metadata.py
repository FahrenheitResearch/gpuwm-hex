"""Validate capture paths and apply declared namelist/stream metadata changes.

This helper never decodes weather fields. Original and materialized metadata
hashes remain in each native producer receipt.
"""
from pathlib import Path


def reference_metadata(request):
    """Carry separately labelled input authority without changing numeric data."""
    return {name: request[name] for name in ('reference_label', 'input_authority')
            if name in request}


def input_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or any(p in ('.', '..') for p in relative.parts):
        raise ValueError('unsafe input member: ' + name)
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def materialize(work, members, request):
    duration = request.get('original_run_duration', '0_01:00:00')
    replacements = {
        'namelist.atmosphere': [dict(old="config_run_duration = '" + duration + "'",
                                    new="config_run_duration = '0_00:12:00'", count=1)],
        'streams.atmosphere': request.get('streams_replacements', []),
    }
    for name in request['inputs']:
        destination = input_path(work, name)
        edits = replacements.get(name, [])
        if edits:
            content = (members / name).read_text()
            for edit in edits:
                if content.count(edit['old']) != edit['count']:
                    raise ValueError('declared metadata replacement count differs: ' + name)
                content = content.replace(edit['old'], edit['new'])
            destination.write_text(content)
        else:
            destination.symlink_to(members / name)
