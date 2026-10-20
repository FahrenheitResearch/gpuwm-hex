! MPAS: module_ra_rrtmg_vinterp.F:21-179. Multi-column native oracle.
program vinterp_columns
 use module_ra_rrtmg_vinterp, only: vinterp_ozn
 use lw_binio
 implicit none
 integer, parameter :: nc=3,nl=10,nz=7
 real :: p(nc,nl),pin(nz),oz(nc,nz),out(nc,nl)
 integer :: c,k,j,case
 character(len=3) :: key
 pin=[1.,2.,5.,10.,20.,50.,100.]
 call bio_open('native-vinterp.bin')
 do case=1,6
  do c=1,nc
   do j=1,nz
    oz(c,j)=real(c)*real(j*j+3)/1000000.
   enddo
   do k=1,nl
    p(c,nl-k+1)=real(k*k+c)/1.5
   enddo
  enddo
  select case(case)
  case(2)
   do k=1,nl
    p(:,nl-k+1)=pin(min(nz,max(1,k-2)))
   enddo
  case(3)
   p(1,nl:1:-1)=[15.,7.,30.,10.,0.5,150.,1.,100.,2.,6.]
   p(2,nl:1:-1)=[1.5,1.7,6.,9.,0.2,3.,7.,50.,20.,80.]
   p(3,nl:1:-1)=[3.,4.,8.,15.,40.,75.,120.,5.,2.,1.]
  case(4)
   p(1,nl:1:-1)=[0.,0.5,1.,1.5,3.,6.,15.,30.,75.,150.]
   p(2,nl:1:-1)=[0.2,1.,2.,5.,10.,20.,50.,100.,110.,150.]
   p(3,nl:1:-1)=[0.9,1.1,2.1,5.1,10.1,20.1,50.1,100.1,150.,175.]
  case(5)
   p(1,nl:1:-1)=[75.,30.,15.,7.,3.,1.5,0.5,100.,150.,1.]
   p(2,nl:1:-1)=[80.,40.,20.,8.,4.,2.,0.9,75.,125.,2.]
   p(3,nl:1:-1)=[60.,35.,17.,9.,5.,3.,1.,50.,150.,5.]
  case(6)
   p=p*1024.;pin=pin*1024.
   oz=oz/100.
  end select
  call vinterp_ozn(1,nc,nc,nl,p,pin,nz,oz,out)
  write(key,'(I3.3)')case
  call wr_r2(key//'/pmid',p)
  call wr_r1(key//'/pin',pin)
  call wr_r2(key//'/ozone',oz)
  call wr_r2(key//'/out',out)
 enddo
 call bio_close()
end program
