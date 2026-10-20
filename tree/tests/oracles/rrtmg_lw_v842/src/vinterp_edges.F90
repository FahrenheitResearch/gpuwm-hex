! MPAS: module_ra_rrtmg_vinterp.F:21-179. Complete pristine public boundary.
program vinterp_edges
 use mpas_kind_types, only: RKIND
 use module_ra_rrtmg_vinterp, only: vinterp_ozn
 use lw_binio
 implicit none
 real(kind=RKIND), allocatable :: p(:,:),pin(:),oz(:,:),out(:,:)
 integer :: nc,nl,nz,scenario,c,k,j
 character(len=512) :: arg,directory
 call get_command_argument(1,arg);read(arg,*) nc
 call get_command_argument(2,arg);read(arg,*) nl
 call get_command_argument(3,arg);read(arg,*) nz
 call get_command_argument(4,arg);read(arg,*) scenario
 call get_command_argument(5,directory)
 allocate(p(nc,nl),pin(nz),oz(nc,nz),out(nc,nl))
 do j=1,nz
  pin(j)=real(j*j+3,RKIND)*1000._RKIND
  do c=1,nc
   oz(c,j)=real(c*(j*j+3),RKIND)/1000000._RKIND
   if(scenario==5) oz(c,j)=real(c*(j*j+3),RKIND)*tiny(1._RKIND)/8._RKIND
  enddo
 enddo
 do k=1,nl
  do c=1,nc
   select case(scenario)
   case(0)
    p(c,k)=pin(1)/2._RKIND
   case(1)
    p(c,k)=pin(nz)*2._RKIND
   case(2)
    p(c,k)=pin(1)
   case(3)
    p(c,k)=pin(nz)
   case(4,5)
    select case(mod(c+k,3))
    case(0)
     p(c,k)=pin(1)/2._RKIND
    case(1)
     p(c,k)=pin(nz)*2._RKIND
    case(2)
     p(c,k)=pin(1)
    end select
   end select
  enddo
 enddo
 call vinterp_ozn(1,nc,nc,nl,p,pin,nz,oz,out)
 call bio_open(trim(directory)//'/native-vinterp.bin')
 call wr_r2('pmid',p)
 call wr_r1('pin',pin)
 call wr_r2('ozone',oz)
 call wr_r2('out',out)
 call bio_close()
end program
