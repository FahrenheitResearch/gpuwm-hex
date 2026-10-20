! Genuine MPAS v8.4.2 aerosol driver slice; test serialization adapted from engine oracle.
! mpas_atmphys_driver_radiation_sw.F:529-583; module_mp_thompson_aerosols.F:23-210.
program oracle_aerosol_options
  use module_mp_thompson_aerosols, only: gt_aod
  use module_ra_rrtmg_sw_aerosols, only: calc_aerosol_rrtmg_sw
  implicit none
  integer, parameter :: nbndsw = 14
  integer :: taer_aod550_opt = 2
  integer :: taer_angexp_opt = 3
  integer :: taer_ssa_opt    = 3
  integer :: taer_asy_opt    = 3
  real :: aer_aod550_val = 0.12
  real :: aer_angexp_val = 1.3
  real :: aer_ssa_val    = 0.85
  real :: aer_asy_val    = 0.9
  integer :: has3d,type_override
  real, allocatable :: angexp(:,:),aerssa(:,:),aerasy(:,:)
  integer :: ncol, nz, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  integer :: i, j, k, n, u
  integer :: repeat_count,repeat_index
  character(len=80) :: repeat_argument,output_name
  real, allocatable, dimension(:,:,:) :: pres_hyd_p, t_p, qv_p, dz_p, nwfa_p, nifa_p, taod5503d_p
  real, allocatable, dimension(:,:,:,:) :: tauaer_p, ssaaer_p, asyaer_p
  real, allocatable, dimension(:,:) :: taod5502d_p, ht_p, xland_p, buf
  integer, allocatable, dimension(:,:) :: taer_type_p
  real, allocatable :: buf1(:)

  open(newunit=u,file='aerosol_control.txt',status='old')
  read(u,*) has3d,taer_aod550_opt,taer_angexp_opt,taer_ssa_opt,taer_asy_opt,type_override, &
            aer_aod550_val,aer_angexp_val,aer_ssa_val,aer_asy_val
  close(u)
  open(newunit=u, file='aer3_mpas_in.bin', access='stream', form='unformatted', status='old')
  read(u) ncol, nz
  ims = 1; ime = ncol; its = 1; ite = ncol
  jms = 1; jme = 1; jts = 1; jte = 1
  kms = 1; kme = nz + 1; kts = 1; kte = nz
  allocate(pres_hyd_p(ims:ime,kms:kme,jms:jme), t_p(ims:ime,kms:kme,jms:jme), &
           qv_p(ims:ime,kms:kme,jms:jme), dz_p(ims:ime,kms:kme,jms:jme), &
           nwfa_p(ims:ime,kms:kme,jms:jme), nifa_p(ims:ime,kms:kme,jms:jme), &
           taod5503d_p(ims:ime,kms:kme,jms:jme))
  allocate(tauaer_p(ims:ime,kms:kme,jms:jme,nbndsw), ssaaer_p(ims:ime,kms:kme,jms:jme,nbndsw), &
           asyaer_p(ims:ime,kms:kme,jms:jme,nbndsw))
  allocate(taod5502d_p(ims:ime,jms:jme), ht_p(ims:ime,jms:jme), xland_p(ims:ime,jms:jme), &
           taer_type_p(ims:ime,jms:jme))
  allocate(buf(ncol,nz), buf1(ncol))
  pres_hyd_p = 1.e5; t_p = 280.; qv_p = 0.; dz_p = 1.; nwfa_p = 0.; nifa_p = 0.
  taod5503d_p = 0.
  read(u) buf; pres_hyd_p(:,kts:kte,1) = buf
  read(u) buf; t_p(:,kts:kte,1) = buf
  read(u) buf; qv_p(:,kts:kte,1) = buf
  read(u) buf; dz_p(:,kts:kte,1) = buf
  read(u) buf; nwfa_p(:,kts:kte,1) = buf
  read(u) buf; nifa_p(:,kts:kte,1) = buf
  read(u) buf1; xland_p(:,1) = buf1
  read(u) buf1; ht_p(:,1) = buf1
  close(u)
  ! taer_type_p is allocated and not initialised by the MPAS driver; a
  ! sentinel stands for that, and the deck carries only xland 1 and 2.
  taer_type_p = -999

  allocate(angexp(ncol,1),aerssa(ncol,1),aerasy(ncol,1))
  do i=1,ncol
    angexp(i,1)=real(mod(i,7))*.5-.5
    aerssa(i,1)=real(mod(i,5))*.25
    aerasy(i,1)=real(mod(i,5))*.5-1.
  enddo
  ! driver :487-498
  do n = 1,nbndsw
  do j = jts,jte
  do k = kts,kte
  do i = its,ite
     tauaer_p(i,k,j,n) = 0.
     ssaaer_p(i,k,j,n) = 1.
     asyaer_p(i,k,j,n) = 0.
  enddo
  enddo
  enddo
  enddo
  ! driver :537-544 (ht_p is zgrid(1,i) there; it only feeds the absent
  ! 2-D profiler branch)
  do j = jts,jte
  do i = its,ite
     if(xland_p(i,j)==1.) then
        taer_type_p(i,j) = 1
     elseif(xland_p(i,j)==2.) then
        taer_type_p(i,j) = 3
     endif
  enddo
  enddo
  ! driver :549-554
  call gt_aod(p_phy = pres_hyd_p, dz8w = dz_p, t_phy = t_p, qvapor = qv_p, &
              nwfa = nwfa_p, nifa = nifa_p, taod5503d = taod5503d_p, &
              ims = ims, ime = ime, jms = jms, jme = jme, kms = kms, kme = kme, &
              its = its, ite = ite, jts = jts, jte = jte, kts = kts, kte = kte)
  ! driver :598-607
  do j = jts,jte
  do i = its,ite
     taod5502d_p(i,j) = 0.
     do k = kts,kte
        taod5502d_p(i,j) = taod5502d_p(i,j) + taod5503d_p(i,k,j)
     enddo
  enddo
  enddo
  ! driver :612-626
  if(type_override>0)taer_type_p=type_override
  call get_command_argument(1,repeat_argument)
  repeat_count=1
  if(len_trim(repeat_argument)>0)read(repeat_argument,*)repeat_count
  do repeat_index=1,repeat_count
  if(has3d/=0)then
  call calc_aerosol_rrtmg_sw(ht = ht_p, dz8w = dz_p, p = pres_hyd_p, t3d = t_p, &
              qv3d = qv_p, tauaer = tauaer_p, ssaaer = ssaaer_p, asyaer = asyaer_p, &
              angexp2d=angexp,aerssa2d=aerssa,aerasy2d=aerasy, &
              aod5502d = taod5502d_p, aod5503d = taod5503d_p, aer_type = taer_type_p, &
              aer_aod550_opt = taer_aod550_opt, aer_angexp_opt = taer_angexp_opt, &
              aer_ssa_opt = taer_ssa_opt, aer_asy_opt = taer_asy_opt, &
              aer_aod550_val = aer_aod550_val, aer_angexp_val = aer_angexp_val, &
              aer_ssa_val = aer_ssa_val, aer_asy_val = aer_asy_val, &
              ims = ims, ime = ime, jms = jms, jme = jme, kms = kms, kme = kme, &
              its = its, ite = ite, jts = jts, jte = jte, kts = kts, kte = kte)
  else
  call calc_aerosol_rrtmg_sw(ht = ht_p, dz8w = dz_p, p = pres_hyd_p, t3d = t_p, &
              qv3d = qv_p, tauaer = tauaer_p, ssaaer = ssaaer_p, asyaer = asyaer_p, &
              angexp2d=angexp,aerssa2d=aerssa,aerasy2d=aerasy, &
              aod5502d = taod5502d_p,  aer_type = taer_type_p, &
              aer_aod550_opt = taer_aod550_opt, aer_angexp_opt = taer_angexp_opt, &
              aer_ssa_opt = taer_ssa_opt, aer_asy_opt = taer_asy_opt, &
              aer_aod550_val = aer_aod550_val, aer_angexp_val = aer_angexp_val, &
              aer_ssa_val = aer_ssa_val, aer_asy_val = aer_asy_val, &
              ims = ims, ime = ime, jms = jms, jme = jme, kms = kms, kme = kme, &
              its = its, ite = ite, jts = jts, jte = jte, kts = kts, kte = kte)
  endif


  output_name='aerosol_options_out.bin'
  if(repeat_count>1)write(output_name,'("aerosol-repeat-",I4.4,".bin")')repeat_index
  open(newunit=u, file=trim(output_name), access='stream', form='unformatted', status='replace')
  write(u) tauaer_p(its:ite,kts:kte,1,:)
  write(u) ssaaer_p(its:ite,kts:kte,1,:)
  write(u) asyaer_p(its:ite,kts:kte,1,:)
  write(u) taod5503d_p(its:ite,kts:kte,1)
  write(u) taod5502d_p(its:ite,1)
  write(u) taer_type_p(its:ite,1)
  write(u) angexp(:,1),aerssa(:,1),aerasy(:,1)
  close(u)
  enddo
end program oracle_aerosol_options
