"""Source selection/provenance tables shared by the MPAS and fork oracle."""
import hashlib
import json
from pathlib import Path

TABLE = Path(__file__).with_suffix('.json')


def pin_set(name):
    table = json.loads(TABLE.read_text())
    if name not in table:
        raise ValueError(f'unknown oracle pin set: {name}')
    return table[name]


def reference_build(reference,name):
    """Authenticate the native publication before using its build plumbing."""
    config=pin_set(name)
    manifest=json.loads((reference/'MANIFEST.json').read_text())
    if manifest['recipe']!='gnu-strict' or manifest['core']!='atmosphere':
        raise ValueError('reference must be strict atmosphere')
    upstream=manifest
    for part in config.get('reference_upstream_path',['upstream']):upstream=upstream[part]
    if config.get('model_commit') and upstream!=config['model_commit']:
        raise ValueError('native reference model pin differs')
    native_pins=manifest.get('pin_set',{}).get('submodules')
    if native_pins is not None and native_pins!=config.get('submodules',{}):
        raise ValueError('native reference submodule pins differ')
    if 'pin_set_hash' in manifest and hashlib.sha256(json.dumps(manifest['pin_set'],sort_keys=True).encode()).hexdigest()!=manifest['pin_set_hash']:
        raise ValueError('native reference pin-set hash differs')
    if hashlib.sha256((reference/'atmosphere_model').read_bytes()).hexdigest()!=manifest['binary_sha256']:
        raise ValueError('native reference binary hash differs')
    plumbing=config.get('reference_plumbing')
    if plumbing:
        path=reference/plumbing['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[plumbing['sha256_field']]:
            raise ValueError('native reference build plumbing hash differs')
        if hashlib.sha256((reference/'flag-audit.json').read_bytes()).hexdigest()!=manifest['flags_sha256']:
            raise ValueError('native reference flag audit hash differs')
        if not json.loads((reference/'flag-audit.json').read_text())['all_ok']:
            raise ValueError('native reference object audit failed')
    return manifest,upstream


def source_files(root, name='mpas-v842', subtree='src'):
    excluded = set(pin_set(name)['exclude_source_parts'])
    return [p for p in sorted((root / subtree).rglob('*'))
            if p.suffix.lower() in {'.f', '.f90'}
            and not any(part.lower() in excluded for part in p.relative_to(root).parts)
            and 'musica' not in str(p).lower()]


def provenance(root, name):
    config = pin_set(name)
    import subprocess
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if config.get('model_commit') and actual != config['model_commit']:
        raise ValueError(f'source pin mismatch: {actual} != {config["model_commit"]}')
    submodules={}
    for path,expected in config.get('submodules',{}).items():
        head=subprocess.check_output(['git','-C',str(root/path),'rev-parse','HEAD'],text=True).strip()
        if head!=expected:raise ValueError(f'submodule pin mismatch: {path}: {head} != {expected}')
        submodules[path]=head
    pin_hash=hashlib.sha256(json.dumps({'model':actual,'submodules':submodules},sort_keys=True).encode()).hexdigest()
    return {'submodule_commits':submodules,'pin_set_sha256':pin_hash,'reference': config['reference'], 'pin_set': name, 'model_commit': actual,
            'pin_set_table_sha256': hashlib.sha256(TABLE.read_bytes()).hexdigest()}
