"""Hash access to native artifacts, including byte-verified cold archives."""
import json
from pathlib import Path
from publish_cache import sha

from hexcore import site_runner


def _archive_host_ok(host):
    """The archive host the site names (HEX_ARCHIVE_HOST), else any recorded host."""
    expected = site_runner.setting('HEX_ARCHIVE_HOST')
    return bool(host) and (not expected or host == expected)


def content_sha(path):
    path=Path(path)
    if path.is_file():return sha(path)
    metadata=path.parent/'ARCHIVE.json'
    if not metadata.is_file():raise FileNotFoundError(path)
    archive=json.loads(metadata.read_text())
    if (archive.get('schema')!='mpas.native-byte-archive/v1' or not _archive_host_ok(archive.get('host'))
            or not archive.get('restored_all_bytes_verified') or not archive.get('archive_sha256')):
        raise ValueError('native archive lacks a restored-byte witness')
    if archive['manifest_sha256']!=sha(path.parent/'MANIFEST.json'):
        raise ValueError('archive belongs to another native manifest')
    if archive['publication_sha256']!=sha(path.parent/'PUBLICATION.json'):
        raise ValueError('archive belongs to another publication')
    manifest=json.loads((path.parent/'MANIFEST.json').read_text())
    if archive['files']!=manifest['outputs']:raise ValueError('archive output coverage differs')
    if path.name not in archive['files']:raise FileNotFoundError(path)
    return archive['files'][path.name]
