"""Extract MPAS precipitation and humidity slices without numeric edits."""
import argparse
from pathlib import Path
import re
from hexcore.mpasphys import thompson_v842 as port


def prepare(source,directory):
    lines=source.read_text().splitlines()
    def extract(ranges):
        result=[]
        for start,end in ranges:
            result.append(f'! MPAS mpas_atmphys_driver_microphysics.F:{start}-{end}')
            for line in lines[start-1:end]:
                if line.lstrip().startswith('!'):continue
                for i,name in enumerate(port.SURFACE_ARGUMENTS[:7],1):
                    line=re.sub(r'\b'+name+r'_p\(i,j\)',f's(i,{i})',line)
                for i,name in enumerate(port.PRECIP_ARGUMENTS,1):
                    line=re.sub(r'\b'+name+r'\(i\)',f'd(i,{i})',line)
                for i,name in enumerate(('qv_p','rho_p','dz_p','th_p','pi_p','pres_p'),1):
                    line=re.sub(r'\b'+name+r'\(i,k,j\)',f'f(i,k,{i})',line)
                line=line.replace('relhum(k,i)','out(i,k)')
                result.append(line.replace('_RKIND',''))
        return '\n'.join(result)
    text='''module diagnostics_oracle
use iso_c_binding
use mpas_atmphys_functions, only:rslf,rsif
implicit none
contains
subroutine precip_ref(nz,ncol,f,s,d,i_rainnc,phase,flag,config_bucket_rainnc) bind(C)
integer(c_int),value :: nz,ncol,phase,flag
real(c_float),value :: config_bucket_rainnc
real(c_float) :: f(ncol,nz,6),s(ncol,7),d(ncol,8)
integer(c_int) :: i_rainnc(ncol)
integer :: i,j,k,its,ite,kts,kte,jts,jte
logical :: l_acrain
real :: rho_a
its=1;ite=ncol;kts=1;kte=nz;jts=1;jte=1;l_acrain=flag/=0
if(phase==0)then
'''
    text+=extract(((531,540),(545,559)))+'\nelse\n'+extract(((605,632),(637,648)))+'\nendif\nend subroutine\n'
    text+='''subroutine relhum_ref(nz,ncol,f,out) bind(C)
integer(c_int),value :: nz,ncol
real(c_float) :: f(ncol,nz,6),out(ncol,nz)
real :: qv1d(nz),qvs1d(nz),t1d(nz),p1d(nz),tempc
integer :: i,j,k,its,ite,kts,kte,jts,jte
its=1;ite=ncol;kts=1;kte=nz;jts=1;jte=1
'''
    text+=extract(((830,843),))+'\nend subroutine\nend module\n'
    (directory/'diagnostics_oracle.F90').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('directory',type=Path)
    a=p.parse_args();prepare(a.source,a.directory)
