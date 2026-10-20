"""Prepare a complete256-profile whole-driver bank and a native bounds witness.

The original1088 matrix remains unchanged and fully graded. Tall-column profiles
are additional defined inputs, not substitutions or exclusions. The bounds
copy runs all512 original added thin profiles with unchanged scientific source.
"""
from pathlib import Path
import json,shutil,hashlib

def copy(source,target):
 target.mkdir(exist_ok=False)
 for p in source.iterdir():
  if p.suffix in ['.F90','.F','.cpp','.cuh'] or p.name=='Makefile':shutil.copyfile(p,target/p.name)

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 root=Path('scratch/defined-branch-bank-v1');root.mkdir(exist_ok=False)
 override='''
    ! Additional tall-column inputs; original thin profiles are retained separately.
    if(deck>=1088) then
      do c=1,ncol
        pblh(c,1)=300.+real(mod(deck+c,25))*100.
        do k=1,nz+1
          dz(c,k,1)=500.+real(mod(13*k+17*deck,97))
        enddo
      enddo
    endif
'''
 for kind in ['native','host','bounds']:
  source=Path('scratch/native-branch-1088-v1' if kind!='host' else 'scratch/host-ieee-1088-v8');target=root/kind;copy(source,target)
  name='run_host_adapter.F90' if kind=='host' else 'run_adapter.F90';p=target/name;s=p.read_text();assert s.count('do deck=0,1087')==1
  if kind=='bounds':
   s=s.replace('do deck=0,1087','do deck=576,1087');m=target/'Makefile';m.write_text(m.read_text().replace('-O3 -cpp','-O3 -fcheck=bounds -fbacktrace -cpp'))
  else:
   s=s.replace('do deck=0,1087','do deck=1088,1343');assert s.count('    do step=1,2\n')==1;s=s.replace('    do step=1,2\n',override+'    do step=1,2\n')
  p.write_text(s)
 (root/'scope.json').write_text(json.dumps(dict(scope=__doc__,process_count=1,exclusions=[],files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}),indent=2)+'\n')
if __name__=='__main__':main()
