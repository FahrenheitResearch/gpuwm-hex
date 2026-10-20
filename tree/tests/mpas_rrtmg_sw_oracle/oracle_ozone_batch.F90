! Pristine MPAS module_ra_rrtmg_vinterp.F:21-173, mixed-column fallback.
program oracle_ozone_batch
 use module_ra_rrtmg_vinterp,only:vinterp_ozn
 implicit none
 integer,parameter::nc=3,nl=5,nozn=4
 real::pin(nozn),oz(nc,nozn),pmid(nc,nl),out(nc,nl)
 integer::ou,c,k
 pin=[10.,100.,300.,600.]
 pmid(1,:)=[900.,500.,250.,75.,5.]
 pmid(2,:)=[500.,250.,75.,20.,5.]
 pmid(3,:)=[600.,300.,100.,10.,3.]
 do c=1,nc
 do k=1,nozn
   oz(c,k)=real(k)*1.e-6+real(c)*1.e-7
 enddo
 enddo
 call vinterp_ozn(1,nc,nc,nl,pmid,pin,nozn,oz,out)
 open(newunit=ou,file='ozone-batch.swd',status='replace',access='stream',form='unformatted')
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
end program oracle_ozone_batch
