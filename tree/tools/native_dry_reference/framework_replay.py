#!/usr/bin/env python3
"""Compile pinned framework authority only for proof, never for a forecast.

Geometry routines and the complete ESMF/MPAS clock are unchanged. Stubs only
provide named pool pointers, logging and the one-process threading identity.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE/'src'))
from hexcore.mpasphys.framework_calendar_v842 import PhysicsClock
from hexcore.mpasphys.framework_init_v842 import initialize_directions

FLAGS = ['-O3','-ffp-contract=off','-fno-tree-vectorize','-fno-unsafe-math-optimizations',
         '-fPIC','-cpp','-DSINGLE_PRECISION','-DHIDE_MPI','-ffree-form','-ffree-line-length-none',
         '-fallow-argument-mismatch']


def routine(text, name):
    lines=text.splitlines(keepends=True)
    start=next(i for i,line in enumerate(lines) if line.strip().lower().startswith('subroutine '+name+'('))
    stop=next(i for i in range(start+1,len(lines)) if lines[i].strip().lower()=='end subroutine '+name)
    return ''.join(lines[start:stop+1])


def build(source, folder):
    folder.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((TREE/'src/hexcore/data/mpasphys-native/mpas-source-v842.json').read_text())
    verified={}
    for path in source.rglob('*'):
        if not path.is_file():continue
        relative=str(path.relative_to(source))
        if relative not in manifest['files']:continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=manifest['files'][relative]['sha256']:
            raise ValueError('source differs from pin: '+relative)
        verified[relative]=actual
    required=['src/framework/mpas_kind_types.F','src/framework/mpas_string_utils.F',
              'src/framework/mpas_timekeeping.F','src/framework/mpas_timekeeping_types.inc',
              'src/core_atmosphere/physics/mpas_atmphys_init.F',
              'src/core_atmosphere/physics/mpas_atmphys_manager.F']
    for relative in required:
        if relative not in verified:raise ValueError('unverified authority: '+relative)
    esmf=source/'src/external/esmf_time_f90'
    names=['ESMF_BaseMod','ESMF_BaseTimeMod','ESMF_CalendarMod','ESMF_FractionMod',
           'ESMF_ShrTimeMod','ESMF_Stubs','ESMF_TimeIntervalMod','ESMF_TimeMod',
           'ESMF_AlarmMod','ESMF_ClockMod','ESMF_AlarmClockMod','MeatMod','ESMF',
           'wrf_error_fatal','wrf_message']
    files=[source/'src/framework/mpas_kind_types.F',*[esmf/(n+'.F90') for n in names]]
    for path in esmf.iterdir():
        if path.is_file() and str(path.relative_to(source)) not in verified:
            raise ValueError('unverified ESMF authority: '+str(path))
    (folder/'stubs.f90').write_text('''module mpas_derived_types
 use ESMF
 use mpas_kind_types
 include 'mpas_timekeeping_types.inc'
 type MPAS_stream_list_type
 character(512) :: name
 type(MPAS_stream_list_type),pointer :: next=>null()
 end type
 end module
 module mpas_dmpar
 end module
 module mpas_threading
 contains
 integer function mpas_threading_get_thread_num()
 mpas_threading_get_thread_num=0
 end function
 end module
 module mpas_log
 integer,parameter :: MPAS_LOG_ERR=1,MPAS_LOG_WARN=2,MPAS_LOG_CRIT=3
 contains
 subroutine mpas_log_write(text,level,intArgs,logicArgs)
 character(*),intent(in) :: text
 integer,optional :: level,intArgs(:)
 logical,optional :: logicArgs(:)
 if(present(level)) then
 if(level==MPAS_LOG_ERR.or.level==MPAS_LOG_CRIT) then
 print *,text
 error stop 'framework authority error'
 endif
 endif
 end subroutine
 end module
''')
    files.extend([folder/'stubs.f90',source/'src/framework/mpas_string_utils.F',
                  source/'src/framework/mpas_timekeeping.F'])
    geometry=(source/'src/core_atmosphere/physics/mpas_atmphys_init.F').read_text()
    (folder/'geometry.f90').write_text('''module geometry_authority
 use iso_c_binding
 implicit none
 integer,parameter :: RKIND=c_float
 type mpas_pool_type
 integer :: dummy
 end type
 integer,target :: cells
 real(RKIND),pointer :: latitude(:),longitude(:),east_words(:,:),north_words(:,:)
 interface mpas_pool_get_array
 module procedure vector,matrix
 end interface
 contains
 subroutine mpas_pool_get_dimension(pool,name,pointer)
 type(mpas_pool_type) :: pool
 character(*) :: name
 integer,pointer :: pointer
 pointer=>cells
 end subroutine
 subroutine vector(pool,name,pointer)
 type(mpas_pool_type) :: pool
 character(*) :: name
 real(RKIND),pointer :: pointer(:)
 if(name=='latCell') pointer=>latitude
 if(name=='lonCell') pointer=>longitude
 end subroutine
 subroutine matrix(pool,name,pointer)
 type(mpas_pool_type) :: pool
 character(*) :: name
 real(RKIND),pointer :: pointer(:,:)
 if(name=='east') pointer=>east_words
 if(name=='north') pointer=>north_words
 end subroutine
'''+routine(geometry,'init_dirs_forphys')+routine(geometry,'r3_normalize')+'''
 subroutine capture_geometry(n,lat,lon,east,north) bind(c)
 integer(c_int),value :: n
 real(c_float),target :: lat(n),lon(n),east(3,n),north(3,n)
 type(mpas_pool_type) :: mesh
 cells=n; latitude=>lat; longitude=>lon; east_words=>east; north_words=>north
 call init_dirs_forphys(mesh)
 end subroutine
 end module
''')
    manager=(source/'src/core_atmosphere/physics/mpas_atmphys_manager.F').read_text()
    arithmetic=''.join(manager.splitlines(keepends=True)[203:213])
    assert 'curr_julday = real(julday-1)' in arithmetic
    (folder/'calendar.f90').write_text('''subroutine capture_clock(start,cal,dt,n,stamps,dates,julians) bind(c)
 use iso_c_binding
 use mpas_timekeeping
 use mpas_kind_types
 implicit none
 character(c_char) :: start(*),cal(*),stamps(512,n+1)
 integer(c_int),value :: n
 real(c_float),value :: dt
 integer(c_int) :: dates(5,n+1)
 real(c_float) :: julians(n+1)
 type(MPAS_Time_type) :: startTime,stopTime,currTime
 type(MPAS_TimeInterval_type) :: step
 type(MPAS_Clock_type) :: clock
 character(512) :: startString,calString,timeStamp
 integer :: i,j,ierr,yr,year,month,day,julday,h,m,s,s_n,s_d,DoY
 real(RKIND) :: utc_h,utc_s,curr_julday
 logical :: LeapYear
 startString='';calString=''
 do i=1,512
 if(start(i)==c_null_char) exit
 startString(i:i)=start(i)
 enddo
 do i=1,512
 if(cal(i)==c_null_char) exit
 calString(i:i)=cal(i)
 enddo
 call mpas_timekeeping_init(trim(calString))
 call mpas_set_time(startTime,dateTimeString=trim(startString),ierr=ierr)
 if(ierr/=0) error stop 'set time failed'
 call mpas_set_timeInterval(step,dt=dt,ierr=ierr)
 if(ierr/=0) error stop 'interval failed'
 stopTime=startTime+step*(n+1)
 call mpas_create_clock(clock,startTime,step,stopTime=stopTime,ierr=ierr)
 if(ierr/=0) error stop 'clock failed'
 do j=1,n+1
'''+arithmetic+'''
 dates(:,j)=[year,month,day,julday,merge(1,0,LeapYear)]
 julians(j)=curr_julday
 do i=1,512
 stamps(i,j)=timeStamp(i:i)
 enddo
 call mpas_advance_clock(clock,ierr=ierr)
 if(ierr/=0) error stop 'advance failed'
 enddo
 call mpas_destroy_clock(clock,ierr=ierr)
 call mpas_timekeeping_finalize()
 end subroutine
''')
    files.extend([folder/'geometry.f90',folder/'calendar.f90'])
    command=['gfortran',*FLAGS,'-shared','-I'+str(esmf.resolve()),
             '-I'+str((source/'src/framework').resolve()),
             *(str(f.resolve()) for f in files),'-o','framework_referee.so']
    result=subprocess.run(command,cwd=folder,text=True,capture_output=True)
    (folder/'build.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    return command,files


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--fixture',type=Path)
    args=parser.parse_args()
    command,files=build(args.source,args.scratch)
    lib=ct.CDLL(str((args.scratch/'framework_referee.so').resolve()))
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='F_CONTIGUOUS')
    iptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    lib.capture_geometry.argtypes=[ct.c_int,ptr,ptr,ptr,ptr]
    rng=np.random.default_rng(842)
    lat=rng.uniform(-np.pi/2,np.pi/2,4096).astype(np.float32)
    lon=rng.uniform(-np.pi,np.pi,4096).astype(np.float32)
    lat[:5]=[0,-0.,np.pi/2,-np.pi/2,.1]
    lon[:5]=[0,-0.,np.pi,-np.pi,.1]
    east=np.empty((3,len(lat)),np.float32,order='F');north=np.empty_like(east)
    lib.capture_geometry(len(lat),lat,lon,east,north)
    python_east,python_north=initialize_directions(lat,lon)
    different=sum(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))
                  for a,b in ((east,python_east),(north,python_north)))
    geometry_words=int(east.size+north.size)
    lib.capture_clock.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_float,ct.c_int,
        ct.c_void_p,iptr,ptr]
    cases=[];clock_words=0
    for start,calendar,dt,n in [
        ('2023-12-31_23:59:30','gregorian',30.,5),
        ('2024-02-28_23:59:59','gregorian',.1,40),
        ('2000-02-29_23:59:59','gregorian',.33333334,7),
        ('2100-02-28_23:59:59','gregorian',1.,5),
        ('2024-02-28_23:59:59','gregorian_noleap',1.,5)]:
        stamps=np.zeros((512,n+1),np.uint8,order='F')
        dates=np.zeros((5,n+1),np.int32,order='F')
        julian=np.zeros(n+1,np.float32)
        lib.capture_clock(start.encode(),calendar.encode(),dt,n,stamps.ctypes.data,dates,julian)
        clock=PhysicsClock(start=start,stop='2101-01-01_00:00:00',dt=dt,calendar=calendar)
        rows=[]
        for i in range(n+1):
            words=clock.calendar_words();t=clock.current_time
            stamp=stamps[:,i].tobytes().decode('ascii').rstrip(' ')
            expected=[t.year,t.month,t.day,t.day_of_year,int(t.leapyear)]
            assert stamp==words['current_date'],(start,i,stamp,words)
            assert dates[:,i].tolist()==expected,(start,i,dates[:,i],expected)
            assert julian[i].view(np.uint32)==words['julian'].view(np.uint32),(start,i,julian[i],words)
            rows.append(dict(stamp=stamp,dates=expected,julian_word=int(julian[i].view(np.uint32))))
            clock.advance();clock_words+=6
        cases.append(dict(start=start,calendar=calendar,dt=dt,steps=n,rows=rows))
    begin=time.perf_counter()
    initialize_directions(np.resize(lat,655362),np.resize(lon,655362))
    seconds=time.perf_counter()-begin
    receipt=dict(schema='mpas-framework-native-replay/v1',geometry_words=geometry_words,
        different_words=int(different),calendar_words=clock_words,calendar_cases=len(cases),
        directions_655362_cells_seconds=seconds,command=command,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        library_sha256=hashlib.sha256((args.scratch/'framework_referee.so').read_bytes()).hexdigest())
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    if args.fixture:
        args.fixture.write_text(json.dumps(dict(calendar=cases,
            geometry=dict(latitude=lat[:64].view(np.uint32).tolist(),longitude=lon[:64].view(np.uint32).tolist(),
                          east=east[:,:64].view(np.uint32).tolist(),north=north[:,:64].view(np.uint32).tolist())),indent=1)+'\n')
    print(json.dumps(receipt))
    assert different==0


if __name__=='__main__':main()
