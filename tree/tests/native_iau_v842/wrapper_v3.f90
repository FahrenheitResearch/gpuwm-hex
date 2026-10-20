! Bind the correct-mode (config_physics_mode) MPAS IAU module; no numerical implementation here.
! ABI v3 adds mode_code, state u, cellsOnEdge (one-based, Fortran (2,ne)) and the window position.
subroutine run_iau_v3(nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,itime,dt,window,option_code,mode_code, &
  zz,rho_zz,theta_m,scalars,rho_edge,u_amb,rho_amb,theta_amb,scalars_amb,u,cells_on_edge,elapsed, &
  tend_ru,tend_rtheta,tend_rho,tend_scalars,weight) bind(C)
  use iso_c_binding
  use mpas_atm_iau
  implicit none
  integer(c_int), value, target :: nl,nc,ne,ncs,nes,ns,nsi,ms,me,qv,itime,option_code,mode_code
  real(c_float), value, target :: dt,window
  real(c_float), target :: zz(nl,nc),rho_zz(nl,nc),theta_m(nl,nc),scalars(ns,nl,nc),rho_edge(nl,ne)
  real(c_float), target :: u_amb(nl,ne),rho_amb(nl,nc),theta_amb(nl,nc),scalars_amb(nsi,nl,nc)
  real(c_float), target :: u(nl,ne)
  integer(c_int), target :: cells_on_edge(2,ne)
  real(c_float), target :: elapsed
  real(c_float), target :: tend_ru(nl,ne),tend_rtheta(nl,nc),tend_rho(nl,nc),tend_scalars(ns,nl,nc)
  real(c_float), intent(out) :: weight
  character(len=StrKIND), target :: option, mode
  type(mpas_pool_type) :: configs,structs
  type(mpas_pool_type), target :: mesh,state,diag,tend,tend_iau
  select case(option_code)
  case(0); option='off'
  case(1); option='on'
  case(2); option='ON'
  case(3); option='other'
  end select
  select case(mode_code)
  case(0); mode='exact'
  case(1); mode='correct'
  end select
  configs%option=>option; configs%window=>window; configs%mode=>mode
  structs%mesh=>mesh; structs%state=>state; structs%diag=>diag
  structs%tend=>tend; structs%tend_iau=>tend_iau
  mesh%nc=>nc; mesh%ne=>ne; mesh%ncs=>ncs; mesh%nes=>nes; mesh%nl=>nl; mesh%zz=>zz
  mesh%cellsOnEdge=>cells_on_edge
  state%ms=>ms; state%me=>me; state%qv=>qv
  state%theta_m=>theta_m; state%rho_zz=>rho_zz; state%scalars=>scalars; state%u=>u
  diag%rho_edge=>rho_edge; diag%elapsed=>elapsed
  tend%scalars_tend=>tend_scalars
  tend_iau%u=>u_amb; tend_iau%theta=>theta_amb; tend_iau%rho=>rho_amb; tend_iau%scalars=>scalars_amb
  if (mode_code == 1) then
    weight=atm_iau_coef_window(configs,elapsed,dt)
  else
    weight=atm_iau_coef(configs,itime,dt)
  end if
  call pre_add_tend_anal_incr(configs,structs)
  call atm_add_tend_anal_incr(configs,structs,itime,dt,tend_ru,tend_rtheta,tend_rho)
  call post_add_tend_anal_incr(configs,structs,tend_ru,tend_rtheta,tend_rho)
end subroutine
