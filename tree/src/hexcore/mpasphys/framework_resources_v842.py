"""Small pinned ozone inputs for framework initialization, without MPAS source."""
def ozone_resources(packet, levels, device):
    from .resources_build import default_cache, fetch, load_packet, MPAS_DATA_COMMIT
    import json
    from pathlib import Path
    resources = load_packet(packet)
    if 'ozone' in resources:
        row = dict(resources['ozone'], nOznLevels=levels)
    else:
        # Only three data files are needed, not the native source-tree resource.
        directory = default_cache() / 'mpas-data' / MPAS_DATA_COMMIT[:12]
        manifest = json.loads((Path(__file__).resolve().parents[1] / 'data' /
            'mpasphys-native' / 'mpas-source-v842.json').read_text())['tables']
        hashes = {}
        for name in ('OZONE_PLEV.TBL', 'OZONE_LAT.TBL', 'OZONE_DAT.TBL'):
            source = manifest[name]
            fetch(name, source['origin'], source['path'], source['sha256'], directory/name)
            hashes[name] = source['sha256']
        row = dict(table_directory=str(directory), table_sha256=hashes, nOznLevels=levels)
    return decode_ozone_tables(row, device)


def decode_ozone_tables(resources, device):
    """Formatted RKIND records, mpas_atmphys_o3climatology.F:102-140, in file order.

    The published Thompson decoder predates the optional ozone ABI. Parsing
    these three small decimal files here works with that packet and needs no
    native framework or source tree. Native reads one value per record.
    """
    from pathlib import Path
    import numpy as np
    from .table_resources_v842 import certified_path
    directory = Path(resources['table_directory'])
    levels = int(resources['nOznLevels'])
    if not 0 < levels <= 59:
        raise ValueError('ozone levels outside pinned table')
    arrays = {}
    for filename, name, shape in (('OZONE_PLEV.TBL','pin',(levels,)),
                                  ('OZONE_LAT.TBL','lat_ozone',(64,)),
                                  ('OZONE_DAT.TBL','ozmixin',(12,64,levels))):
        path = certified_path(dict(path=directory/filename,
                                   sha256=resources['table_sha256'][filename]))
        count = int(np.prod(shape))
        records = (line.split()[0] for line in path.read_text().splitlines() if line.strip())
        values = np.fromiter((np.float32(word.replace('D','e').replace('d','e'))
                              for word in records), dtype=np.float32, count=count)
        arrays[name] = device.asarray(values.reshape(shape))
    return arrays
