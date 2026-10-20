! Bind unchanged MPAS pool entry points; no numerical implementation here.
! ABI v2 separates state and increment extents; stale libraries fail symbol lookup.
subroutine run_iau_v2(nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,itime,dt,window,option_code, &
  zz,rho_zz,theta_m,scalars,rho_edge,u_amb,rho_amb,theta_amb,scalars_amb, &
  tend_ru,tend_rtheta,tend_rho,tend_scalars,weight) bind(C)
  use iso_c_binding
  use mpas_atm_iau
  implicit none
  integer(c_int), value, target :: nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,itime,option_code
  real(c_float), value, target :: dt,window
  real(c_float), target :: zz(nl,nc),rho_zz(nl,nc),theta_m(nl,nc),scalars(ns,nl,nc),rho_edge(nl,ne)
  real(c_float), target :: u_amb(nl,ne),rho_amb(nl,nc),theta_amb(nl,nc),scalars_amb(nsi,nl,nc)
  real(c_float), target :: tend_ru(nl,ne),tend_rtheta(nl,nc),tend_rho(nl,nc),tend_scalars(ns,nl,nc)
  real(c_float), intent(out) :: weight
  character(len=StrKIND), target :: option
  type(mpas_pool_type) :: configs,structs
  type(mpas_pool_type), target :: mesh,state,diag,tend,tend_iau
  select case(option_code)
  case(0); option='off'
  case(1); option='on'
  case(2); option='ON'
  case(3); option='other'
  end select
  configs%option=>option; configs%window=>window
  structs%mesh=>mesh; structs%state=>state; structs%diag=>diag
  structs%tend=>tend; structs%tend_iau=>tend_iau
  mesh%nc=>nc; mesh%ne=>ne; mesh%ncs=>ncs; mesh%nes=>nes; mesh%nl=>nl; mesh%zz=>zz
  state%ms=>ms; state%me=>me; state%qv=>qv
  state%theta_m=>theta_m; state%rho_zz=>rho_zz; state%scalars=>scalars
  diag%rho_edge=>rho_edge
  tend%scalars_tend=>tend_scalars
  tend_iau%u=>u_amb; tend_iau%theta=>theta_amb; tend_iau%rho=>rho_amb; tend_iau%scalars=>scalars_amb
  weight=atm_iau_coef(configs,itime,dt)
  call pre_add_tend_anal_incr(configs,structs)
  call atm_add_tend_anal_incr(configs,structs,itime,dt,tend_ru,tend_rtheta,tend_rho)
  call post_add_tend_anal_incr(configs,structs,tend_ru,tend_rtheta,tend_rho)
end subroutine
