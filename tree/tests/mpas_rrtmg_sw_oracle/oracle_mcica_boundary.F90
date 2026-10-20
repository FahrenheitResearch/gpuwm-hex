! MPAS module_ra_rrtmg_sw.F:1403-2012, including the unchecked KISS seed reads.
! Diagnostic only: this standalone fixture cannot certify native forecast garbage.
program oracle_mcica_boundary
 use mcica_subcol_gen_sw,only:mcica_subcol_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer :: nl,nc,icld,rng,u,i,k
 character(len=32) :: text
 real,allocatable :: p(:,:),x(:,:),optics(:,:,:),output(:,:,:,:),radius(:,:,:)
 call get_command_argument(1,text);read(text,*)nl
 call get_command_argument(2,text);read(text,*)icld
 call get_command_argument(3,text);read(text,*)rng
 call get_command_argument(4,text);read(text,*)nc
 call rrtmg_initsw_forMPAS(dminfo)
 allocate(p(nc,nl),x(nc,nl),optics(14,nc,nl),output(8,112,nc,nl),radius(3,nc,nl))
 do k=1,nl
 do i=1,nc
  p(i,k)=1000.125/(1.+real(k-1)*.23)+real(i)*.07125
 enddo
 enddo
 x=.4;optics=.5;output=-987.25;radius=-987.25
 call mcica_subcol_sw(1,nc,nl,icld,7,rng,p,x,x,x,x,x,x,x,optics,optics,optics,optics, &
   x,0,279,-41.25,output(1,:,:,:),output(2,:,:,:),output(3,:,:,:),output(4,:,:,:), &
   radius(1,:,:),radius(2,:,:),radius(3,:,:),output(5,:,:,:),output(6,:,:,:),output(7,:,:,:),output(8,:,:,:))
 open(newunit=u,file='mcica-boundary.bin',access='stream',form='unformatted',status='replace')
 write(u)nl,nc,icld,rng,output,radius
 close(u)
 print *, 'NATIVE MCICA RETURNED',nl,nc,icld,rng
end program oracle_mcica_boundary
