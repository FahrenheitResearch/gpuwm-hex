"""Wrap unchanged mpas_atm_core.F:168-201 for filter-length control replay."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text().splitlines(keepends=True)
block=''.join(source[167:201])
assert 'if (config_len_disp > 0.0_RKIND)' in block
assert 'ierr = 1' in block
Path(sys.argv[2]).write_text('''module core_length_oracle
 implicit none
 integer,parameter::RKIND=kind(1.0),MPAS_LOG_WARN=1,MPAS_LOG_ERR=2
contains
 subroutine native_length(config_len_disp,nominalMinDc,ierr)
 real,intent(inout)::config_len_disp,nominalMinDc
 integer,intent(out)::ierr
 ierr=0
'''+block+''' end subroutine
 subroutine mpas_log_write(message,realArgs,messageType)
 character(len=*),intent(in)::message
 real,optional,intent(in)::realArgs(:)
 integer,optional,intent(in)::messageType
 end subroutine
end module
program length_deck
 use core_length_oracle
 implicit none
 real::configured(8),nominal(8),before(2,8),after(2,8)
 integer::errors(8),i
 configured=[-1.0,0.0,5000.0,5000.0,0.0,-1.0,-12000.0,2.e-50]
 nominal=[12000.0,12000.0,12000.0,-1.0,0.0,-1.0,12501.75,12000.0]
 before(1,:)=configured;before(2,:)=nominal
 do i=1,8
 call native_length(configured(i),nominal(i),errors(i))
 end do
 after(1,:)=configured;after(2,:)=nominal
 open(20,file='length.bin',access='stream',form='unformatted',convert='little_endian')
 write(20)before,after,errors
 close(20)
end program
''')
