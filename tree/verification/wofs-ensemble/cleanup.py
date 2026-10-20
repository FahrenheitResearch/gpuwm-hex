"""Delete only owned, receipted scratch; do not follow directory links."""
from pathlib import Path
import csv,gzip,json,os,sys
W=Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble')); R=(W/'prepared').resolve()

def remove_one(path):
    parent=path.parent.resolve()
    if not (parent.is_relative_to(W.resolve()) or parent.is_relative_to(R)):
        raise ValueError(f'cleanup escapes owned roots: {path}')
    size=path.lstat().st_size
    with (W/'deleted-files.log').open('a') as f:
        f.write(json.dumps(dict(path=str(path),bytes=size,kind='link' if path.is_symlink() else 'file'))+'\n')
    path.unlink()

def remove_tree(root):
    if root.is_symlink(): remove_one(root); return
    if not root.exists(): return
    resolved=root.resolve()
    if not (resolved.is_relative_to(W.resolve()) or resolved.is_relative_to(R)):
        raise ValueError('unowned cleanup root')
    for current,dirs,files in os.walk(root,topdown=False,followlinks=False):
        current=Path(current)
        for name in files: remove_one(current/name)
        for name in dirs:
            path=current/name
            if path.is_symlink(): remove_one(path)
            else: path.rmdir()
    root.rmdir()

def import_native_retirements(manifest):
    """Keep the Rust reducer's exact deletion receipt in the lane ledger."""
    if not manifest.is_file(): return
    existing={row['path'] for row in map(json.loads,(W/'deleted-files.log').read_text().splitlines())}
    # Native paths are relative to the spool root, one level above d01.
    root=manifest.parent.parent
    for row in json.loads(manifest.read_text()).get('deleted_scratch',[]):
        path=(root/row['path']).absolute()
        if not path.resolve().is_relative_to(W.resolve()):
            raise ValueError('native deletion receipt escapes owned root')
        if path.exists(): raise ValueError('native deletion receipt names a surviving file')
        if str(path) not in existing:
            with (W/'deleted-files.log').open('a') as f:
                f.write(json.dumps(dict(path=str(path),bytes=row['bytes'],kind='file'))+'\n')
            existing.add(str(path))

def main(stage):
    if stage=='prepared-inputs':
        if not (W/'evidence/inputs-ready.json').is_file(): raise ValueError('initializations have not closed')
        remove_tree(R/'geog')
        for path in R.glob('*.grib2'): remove_one(path)
        for path in R.glob('*/MET:*'): remove_one(path)
        for path in (W/'vertical-cache').glob('*.nc'): remove_one(path)
        for path in (W/'vertical-cache').glob('*.npz'): remove_one(path)
    elif stage=='completed-model-data':
        if not (W/'products/showcase/complete.json').is_file(): raise ValueError('showcase products have not closed')
        for phase in ('pilot','showcase'):
            out=W/'products'/phase
            if not (out/'map-artifacts.json').is_file(): raise ValueError('map hashes not recorded')
            import_native_retirements(out/'aggregate/d01/ensemble-manifest.json')
            for path in out.rglob('*.nc'): remove_one(path)
            for path in out.rglob('*.npy'): remove_one(path)
            for path in out.rglob('*.npz'): remove_one(path)
            remove_tree(out/'stores')
        for path in R.glob('*/init.nc'): remove_one(path)
        remove_one(R/'mesh.grid.nc'); remove_one(R/'mesh.static.nc')
        for path in (R/'observations').rglob('*.f32'): remove_one(path)
        remove_tree(W/'hostprep-cache')
        for root in (W/'runs',W/'parity'):
            for path in root.glob('*/scratch'): remove_tree(path)
            for path in root.glob('*/*/scratch'): remove_tree(path)
        for name in ('tmp','cupy-cache','cuda-cache','vertical-cache'):
            remove_tree(W/name)
    else: raise ValueError('unknown cleanup stage')
    rows=[json.loads(line) for line in (W/'deleted-files.log').read_text().splitlines()]
    with (W/'evidence/deleted-files.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','kind']); writer.writeheader()
        for row in rows: writer.writerow({'kind':'file',**row})
    with (W/'evidence/deleted-files.csv').open('rb') as src, gzip.open(W/'evidence/deleted-files.csv.gz','wb') as dst:
        for block in iter(lambda:src.read(1024*1024),b''): dst.write(block)
    (W/'evidence/cleanup-summary.json').write_text(json.dumps(dict(stage=stage,
        deleted_files=len(rows),deleted_bytes=sum(row['bytes'] for row in rows),
        exact_list='deleted-files.csv'),indent=2)+'\n')
    with (W/'MANIFEST.txt').open('w') as f:
        for root in (W,R):
            for path in root.rglob('*'):
                if '.git' not in path.parts: f.write(str(path)+'\n')

if __name__=='__main__': main(sys.argv[1])
