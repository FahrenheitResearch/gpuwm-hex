"""Publication integrity and resource checks shared by native matrix runners."""
import json
from pathlib import Path
from publish_cache import sha
from artifact_storage import content_sha


def free_memory_bytes():
    # The pool rule is literal free memory, rather than Linux MemAvailable.
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemFree'].split()[0]) * 1024


def published(destination, source):
    metadata = destination / 'PUBLICATION.json'
    if not metadata.is_file():
        return None
    receipt = json.loads(metadata.read_text())
    if receipt['source'] != str(source.resolve()):
        raise ValueError('publication came from another scratch directory')
    if not receipt['files']:
        raise ValueError('publication has no files')
    for name, digest in receipt['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('publication file escapes cache')
        if content_sha(destination / relative) != digest:
            raise ValueError(f'published file hash differs: {name}')
    return {'path': str(destination), 'publication_sha256': sha(metadata),
            'files': len(receipt['files']), 'existing': True}


def discard_copied_outputs(source, destination, names):
    """Delete only own regular dumps after their published copies match."""
    for name in names:
        own = source / name
        cached = destination / name
        if own.is_file() and not own.is_symlink():
            if sha(own) != sha(cached):
                raise ValueError(f'published dump differs: {name}')
            own.unlink()
