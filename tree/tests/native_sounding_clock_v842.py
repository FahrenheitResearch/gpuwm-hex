"""Build copied pristine MPAS timekeeping/embedded ESMF with framework stubs."""
import argparse
from pathlib import Path
import shutil


def generate(root,d):
    d.mkdir(parents=True,exist_ok=True)
    esmf=d/'esmf'
    if esmf.exists():
        for item in [esmf,*esmf.rglob('*')]: item.chmod(item.stat().st_mode|0o700)
    shutil.copytree(root/'src/external/esmf_time_f90',esmf,dirs_exist_ok=True)
    # Source directories are read-only; these build copies belong to the lane.
    for item in [esmf,*esmf.rglob('*')]: item.chmod(item.stat().st_mode|0o700)
    for name in ('mpas_kind_types.F','mpas_string_utils.F','mpas_timekeeping.F','mpas_timekeeping_types.inc'):
        shutil.copyfile(root/'src/framework'/name,d/name)
    (d/'stubs.f90').write_text('''module mpas_derived_types
use mpas_kind_types
use ESMF
include 'mpas_timekeeping_types.inc'
type MPAS_stream_list_type
character(len=StrKIND) :: name
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
integer,parameter :: MPAS_LOG_ERR=1,MPAS_LOG_CRIT=2,MPAS_LOG_WARN=3
contains
subroutine mpas_log_write(message,level,intArgs,logicArgs)
character(len=*),intent(in)::message
integer,optional,intent(in)::level,intArgs(:)
logical,optional,intent(in)::logicArgs(:)
end subroutine
end module
''')
    (d/'oracle.f90').write_text('''program clock_oracle
use mpas_timekeeping
use mpas_derived_types
implicit none
type(MPAS_Time_type) :: start,now,previous,next
type(MPAS_TimeInterval_type) :: intv,step,zero
type(MPAS_Clock_type) :: clock
character(len=512) :: timeString,intervalString,previousString,nextString,resetString
logical :: ring
integer :: i,sn,sd
character(len=8) :: fractionString
call mpas_timekeeping_init('gregorian')
call get_command_argument(1,timeString)
call mpas_set_time(start,dateTimeString=trim(timeString))
call get_command_argument(2,intervalString)
call mpas_set_timeInterval(intv,timeString=trim(intervalString))
call mpas_set_timeInterval(step,S=1)
call mpas_set_timeInterval(zero,S=0)
call mpas_create_clock(clock,start,step,runDuration=zero)
call mpas_add_clock_alarm(clock,'soundingAlarm',start,intv)
call get_command_argument(3,timeString)
call mpas_set_time(now,dateTimeString=trim(timeString))
! Match mpas_advance_clock: change current time without restart calibration.
call ESMF_ClockSet(clock%c,CurrTime=now%t)
call get_command_argument(4,resetString)
ring=mpas_is_alarm_ringing(clock,'soundingAlarm')
if(trim(resetString)=='reset') call mpas_reset_clock_alarm(clock,'soundingAlarm')
previous=clock%alarmListHead%prevRingTime
next=previous+intv
call mpas_get_time(previous,dateTimeString=previousString,S_n=sn,S_d=sd)
if(sn/=0 .and. sd/=0) then
write(fractionString,'(i8.8)') int(int(sn,8)*100000000_8/int(sd,8))
previousString=trim(previousString)//'.'//fractionString
end if
call mpas_get_time(next,dateTimeString=nextString,S_n=sn,S_d=sd)
if(sn/=0 .and. sd/=0) then
write(fractionString,'(i8.8)') int(int(sn,8)*100000000_8/int(sd,8))
nextString=trim(nextString)//'.'//fractionString
end if
if(ring) then
write(*,'(a)') 'true'
else
write(*,'(a)') 'false'
end if
write(*,'(a)') trim(previousString)
write(*,'(a)') trim(nextString)
end program
''')
    flags='-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -cpp -DSINGLE_PRECISION -DHIDE_MPI'
    (d/'Makefile').write_text(f'''FC=gfortran
FFLAGS={flags}
all: clock_oracle
esmf/libesmf_time.a:
\t$(MAKE) -C esmf -j4 FC=$(FC) FFLAGS="$(FFLAGS)" CPPFLAGS="-DHIDE_MPI"
clock_oracle: esmf/libesmf_time.a stubs.f90 oracle.f90 mpas_timekeeping.F
\t$(FC) $(FFLAGS) -Iesmf -c mpas_kind_types.F mpas_string_utils.F stubs.f90 mpas_timekeeping.F oracle.f90
\t$(FC) $(FFLAGS) mpas_kind_types.o mpas_string_utils.o stubs.o mpas_timekeeping.o oracle.o esmf/libesmf_time.a -o $@
''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();generate(a.root,a.destination)
