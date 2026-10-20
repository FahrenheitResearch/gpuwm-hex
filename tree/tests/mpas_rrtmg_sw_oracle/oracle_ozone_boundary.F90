! Pristine MPAS module_ra_rrtmg_vinterp.F:21-173, mixed-column fallback.
program oracle_ozone_boundary
 use module_ra_rrtmg_vinterp,only:vinterp_ozn
 implicit none
 integer,parameter::nc=1,nl=3
 integer::nozn
 character(len=32)::mode
 character(len=256)::filename
 real,allocatable::pin(:),oz(:,:)
 real::pmid(nc,nl),out(nc,nl)
 integer::ou,c,k
 call get_command_argument(1,mode)
 nozn=1
 if(trim(mode)=='empty')nozn=0
 allocate(pin(nozn),oz(nc,nozn))
 pin=100.
 oz=1.e-6
 pmid(1,:)=[150.,100.,50.]
 call vinterp_ozn(1,nc,nc,nl,pmid,pin,nozn,oz,out)
 filename='ozone-boundary-'//trim(mode)//'.swd'
 open(newunit=ou,file=trim(filename),status='replace',access='stream',form='unformatted')
 write(ou)'SWD1'
 call wr1('pin',pin)
 call wr2('ozone',oz)
 call wr2('pmid',pmid)
 call wr2('output',out)
 close(ou)
contains
  subroutine wname(name, dtype, dims)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dtype
    integer, intent(in) :: dims(:)
    write(ou) int(len_trim(name),4)
    write(ou) trim(name)
    write(ou) int(dtype,4)
    write(ou) int(size(dims),4)
    write(ou) int(dims,4)
  end subroutine wname

  subroutine wr1(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr1

  subroutine wr2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr2
end program oracle_ozone_boundary
