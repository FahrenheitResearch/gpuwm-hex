subroutine replay_les_models(les_model_opt,les_surface_opt,dynamics_substep,eddy_visc_horz,eddy_visc_vert,u,v,ucell,vcell,w,c_s,bv_freq2,zgrid,config_len_disp,deformation_coef_c2,deformation_coef_s2,deformation_coef_cs,deformation_coef_c,deformation_coef_s,prandtl_3d_inv,invdt,h_mom_eddy_visc4,config_visc4_2dsmag,h_theta_eddy_visc4,scalars,tend_scalars,index_tke,rho_zz,meshscalingdel2,cellstart,cellend,nedgesoncell,edgesoncell,cellsonedge,ncells,nedges,nvertlevels,maxedges,num_scalars) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
type(MPAS_Clock_type) :: clock
integer(c_int) :: les_model_opt
integer(c_int) :: les_surface_opt
integer(c_int) :: dynamics_substep
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_horz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_vert
real(c_float), dimension(nvertlevels,nedges+1) :: u
real(c_float), dimension(nvertlevels,nedges+1) :: v
real(c_float), dimension(nvertlevels,ncells+1) :: ucell
real(c_float), dimension(nvertlevels,ncells+1) :: vcell
real(c_float), dimension(nvertlevels+1,ncells+1) :: w
real(c_float) :: c_s
real(c_float), dimension(nvertlevels,ncells+1) :: bv_freq2
real(c_float), dimension(nvertlevels+1,ncells+1) :: zgrid
real(c_float) :: config_len_disp
real(c_float), dimension(maxedges,ncells+1) :: deformation_coef_c2
real(c_float), dimension(maxedges,ncells+1) :: deformation_coef_s2
real(c_float), dimension(maxedges,ncells+1) :: deformation_coef_cs
real(c_float), dimension(maxedges,ncells+1) :: deformation_coef_c
real(c_float), dimension(maxedges,ncells+1) :: deformation_coef_s
real(c_float), dimension(nvertlevels,ncells+1) :: prandtl_3d_inv
real(c_float) :: invdt
real(c_float) :: h_mom_eddy_visc4
real(c_float) :: config_visc4_2dsmag
real(c_float) :: h_theta_eddy_visc4
real(c_float), dimension(num_scalars,nvertlevels,ncells+1) :: scalars
real(c_float), dimension(num_scalars,nvertlevels,ncells+1) :: tend_scalars
integer(c_int) :: index_tke
real(c_float), dimension(nvertlevels,ncells+1) :: rho_zz
real(c_float), dimension(nedges+1) :: meshscalingdel2
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int), dimension(ncells+1) :: nedgesoncell
integer(c_int), dimension(maxedges,ncells+1) :: edgesoncell
integer(c_int), dimension(2,nedges+1) :: cellsonedge
integer(c_int) :: ncells
integer(c_int) :: nedges
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
call les_models(les_model_opt,les_surface_opt,dynamics_substep,eddy_visc_horz,eddy_visc_vert,u,v,ucell,vcell,w,c_s,bv_freq2,zgrid,config_len_disp,deformation_coef_c2,deformation_coef_s2,deformation_coef_cs,deformation_coef_c,deformation_coef_s,prandtl_3d_inv,invdt,h_mom_eddy_visc4,config_visc4_2dsmag,h_theta_eddy_visc4,scalars,tend_scalars,index_tke,rho_zz,meshscalingdel2,cellstart,cellend,nedgesoncell,edgesoncell,cellsonedge,ncells,nedges,nvertlevels,maxedges,num_scalars)
end subroutine replay_les_models

subroutine replay_calculate_n2(bn2,theta_m,exner,pressure_b,pp,zgrid,scalars,index_qv,index_qc,qtot,cellstart,cellend,ncells,nvertlevels,maxedges,num_scalars) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
type(MPAS_Clock_type) :: clock
real(c_float), dimension(nvertlevels,ncells+1) :: bn2
real(c_float), dimension(nvertlevels,ncells+1) :: theta_m
real(c_float), dimension(nvertlevels,ncells+1) :: exner
real(c_float), dimension(nvertlevels,ncells+1) :: pressure_b
real(c_float), dimension(nvertlevels,ncells+1) :: pp
real(c_float), dimension(nvertlevels+1,ncells+1) :: zgrid
real(c_float), dimension(num_scalars,nvertlevels,ncells+1) :: scalars
integer(c_int) :: index_qv
integer(c_int) :: index_qc
real(c_float), dimension(nvertlevels,ncells+1) :: qtot
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int) :: ncells
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
call calculate_n2(bn2,theta_m,exner,pressure_b,pp,zgrid,scalars,index_qv,index_qc,qtot,cellstart,cellend,ncells)
end subroutine replay_calculate_n2

subroutine replay_u_dissipation_3d(edgestart,edgeend,edgesolvestart,edgesolveend,vertexstart,vertexend,cellstart,cellend,ncells,nedges,nvertices,vertexdegree,cellsonedge,verticesonedge,edgesoncell,edgesonvertex,nedgesoncell,edgesoncell_sign,edgesonvertex_sign,invareacell,invareatriangle,invdvedge,invdcedge,angleedge,dcedge,dvedge,meshscalingdel2,meshscalingdel4,config_mix_full,h_mom_eddy_visc4,v_mom_eddy_visc2,config_del4u_div_factor,zgrid,eddy_visc_horz,eddy_visc_vert,zz,rdzu,rdzw,fzm,fzp,les_model_opt,les_surface_opt,config_surface_drag_coefficient,delsq_u,delsq_vorticity,delsq_divergence,u,v,divergence,vorticity,rho_edge,rho_zz,u_init,v_init,ustm,tend_u_euler,nvertlevels,maxedges,num_scalars) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
type(MPAS_Clock_type) :: clock
integer(c_int) :: edgestart
integer(c_int) :: edgeend
integer(c_int) :: edgesolvestart
integer(c_int) :: edgesolveend
integer(c_int) :: vertexstart
integer(c_int) :: vertexend
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int) :: ncells
integer(c_int) :: nedges
integer(c_int) :: nvertices
integer(c_int) :: vertexdegree
integer(c_int), dimension(2,nedges+1) :: cellsonedge
integer(c_int), dimension(2,nedges+1) :: verticesonedge
integer(c_int), dimension(maxedges,ncells+1) :: edgesoncell
integer(c_int), dimension(vertexdegree,nvertices+1) :: edgesonvertex
integer(c_int), dimension(ncells+1) :: nedgesoncell
real(c_float), dimension(maxedges,ncells+1) :: edgesoncell_sign
real(c_float), dimension(vertexdegree,nvertices+1) :: edgesonvertex_sign
real(c_float), dimension(ncells+1) :: invareacell
real(c_float), dimension(nvertices+1) :: invareatriangle
real(c_float), dimension(nedges+1) :: invdvedge
real(c_float), dimension(nedges+1) :: invdcedge
real(c_float), dimension(nedges+1) :: angleedge
real(c_float), dimension(nedges+1) :: dcedge
real(c_float), dimension(nedges+1) :: dvedge
real(c_float), dimension(nedges+1) :: meshscalingdel2
real(c_float), dimension(nedges+1) :: meshscalingdel4
integer(c_int) :: config_mix_full
logical :: l_config_mix_full
real(c_float) :: h_mom_eddy_visc4
real(c_float) :: v_mom_eddy_visc2
real(c_float) :: config_del4u_div_factor
real(c_float), dimension(nvertlevels+1,ncells+1) :: zgrid
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_horz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_vert
real(c_float), dimension(nvertlevels,ncells+1) :: zz
real(c_float), dimension(nvertlevels) :: rdzu
real(c_float), dimension(nvertlevels) :: rdzw
real(c_float), dimension(nvertlevels) :: fzm
real(c_float), dimension(nvertlevels) :: fzp
integer(c_int) :: les_model_opt
integer(c_int) :: les_surface_opt
real(c_float) :: config_surface_drag_coefficient
real(c_float), dimension(nvertlevels,nedges+1) :: delsq_u
real(c_float), dimension(nvertlevels,nvertices+1) :: delsq_vorticity
real(c_float), dimension(nvertlevels,ncells+1) :: delsq_divergence
real(c_float), dimension(nvertlevels,nedges+1) :: u
real(c_float), dimension(nvertlevels,nedges+1) :: v
real(c_float), dimension(nvertlevels,ncells+1) :: divergence
real(c_float), dimension(nvertlevels,nvertices+1) :: vorticity
real(c_float), dimension(nvertlevels,nedges+1) :: rho_edge
real(c_float), dimension(nvertlevels,ncells+1) :: rho_zz
real(c_float), dimension(nvertlevels) :: u_init
real(c_float), dimension(nvertlevels) :: v_init
real(c_float), dimension(ncells+1) :: ustm
real(c_float), dimension(nvertlevels,nedges+1) :: tend_u_euler
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
l_config_mix_full=(config_mix_full/=0)
call u_dissipation_3d(edgestart,edgeend,edgesolvestart,edgesolveend,vertexstart,vertexend,cellstart,cellend,ncells,nedges,nvertices,vertexdegree,cellsonedge,verticesonedge,edgesoncell,edgesonvertex,nedgesoncell,edgesoncell_sign,edgesonvertex_sign,invareacell,invareatriangle,invdvedge,invdcedge,angleedge,dcedge,dvedge,meshscalingdel2,meshscalingdel4,l_config_mix_full,h_mom_eddy_visc4,v_mom_eddy_visc2,config_del4u_div_factor,zgrid,eddy_visc_horz,eddy_visc_vert,zz,rdzu,rdzw,fzm,fzp,les_model_opt,les_surface_opt,config_surface_drag_coefficient,delsq_u,delsq_vorticity,delsq_divergence,u,v,divergence,vorticity,rho_edge,rho_zz,u_init,v_init,ustm,tend_u_euler)
end subroutine replay_u_dissipation_3d

subroutine replay_w_dissipation_3d(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,rdzw,rdzu,v_mom_eddy_visc2,h_mom_eddy_visc4,delsq_w,w,rho_edge,rho_zz,divergence,zz,eddy_visc_horz,eddy_visc_vert,les_model_opt,les_surface_opt,tend_w_euler,nvertlevels,maxedges,num_scalars) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
type(MPAS_Clock_type) :: clock
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int) :: cellsolvestart
integer(c_int) :: cellsolveend
integer(c_int) :: ncells
integer(c_int) :: nedges
integer(c_int), dimension(ncells+1) :: nedgesoncell
integer(c_int), dimension(maxedges,ncells+1) :: edgesoncell
integer(c_int), dimension(2,nedges+1) :: cellsonedge
real(c_float), dimension(maxedges,ncells+1) :: edgesoncell_sign
real(c_float), dimension(ncells+1) :: invareacell
real(c_float), dimension(nedges+1) :: invdcedge
real(c_float), dimension(nedges+1) :: dvedge
real(c_float), dimension(nedges+1) :: meshscalingdel2
real(c_float), dimension(nedges+1) :: meshscalingdel4
real(c_float), dimension(nvertlevels) :: rdzw
real(c_float), dimension(nvertlevels) :: rdzu
real(c_float) :: v_mom_eddy_visc2
real(c_float) :: h_mom_eddy_visc4
real(c_float), dimension(nvertlevels,ncells+1) :: delsq_w
real(c_float), dimension(nvertlevels+1,ncells+1) :: w
real(c_float), dimension(nvertlevels,nedges+1) :: rho_edge
real(c_float), dimension(nvertlevels,ncells+1) :: rho_zz
real(c_float), dimension(nvertlevels,ncells+1) :: divergence
real(c_float), dimension(nvertlevels,ncells+1) :: zz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_horz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_vert
integer(c_int) :: les_model_opt
integer(c_int) :: les_surface_opt
real(c_float), dimension(nvertlevels+1,ncells+1) :: tend_w_euler
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
call w_dissipation_3d(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,rdzw,rdzu,v_mom_eddy_visc2,h_mom_eddy_visc4,delsq_w,w,rho_edge,rho_zz,divergence,zz,eddy_visc_horz,eddy_visc_vert,les_model_opt,les_surface_opt,tend_w_euler)
end subroutine replay_w_dissipation_3d

subroutine replay_scalar_dissipation_3d_les(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,config_mix_full,t_init,zgrid,rdzw,rdzu,fzm,fzp,v_theta_eddy_visc2,h_theta_eddy_visc4,prandtl_inv,prandtl_3d_inv,delsq_theta,theta_m,rho_edge,rho_zz,zz,eddy_visc_horz,eddy_visc_vert,bv_freq2,config_len_disp,scalars,tend_scalars,index_tke,index_qv,num_scalars_dummy,mix_scalars,les_model_opt,les_surface_opt,dt,config_surface_heat_flux,config_surface_moisture_flux,ureconstructzonal,ureconstructmeridional,hfx,qfx,tend_theta_euler,dynamics_substep,nvertlevels,maxedges,num_scalars) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
type(MPAS_Clock_type) :: clock
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int) :: cellsolvestart
integer(c_int) :: cellsolveend
integer(c_int) :: ncells
integer(c_int) :: nedges
integer(c_int), dimension(ncells+1) :: nedgesoncell
integer(c_int), dimension(maxedges,ncells+1) :: edgesoncell
integer(c_int), dimension(2,nedges+1) :: cellsonedge
real(c_float), dimension(maxedges,ncells+1) :: edgesoncell_sign
real(c_float), dimension(ncells+1) :: invareacell
real(c_float), dimension(nedges+1) :: invdcedge
real(c_float), dimension(nedges+1) :: dvedge
real(c_float), dimension(nedges+1) :: meshscalingdel2
real(c_float), dimension(nedges+1) :: meshscalingdel4
integer(c_int) :: config_mix_full
logical :: l_config_mix_full
real(c_float), dimension(nvertlevels,ncells+1) :: t_init
real(c_float), dimension(nvertlevels+1,ncells+1) :: zgrid
real(c_float), dimension(nvertlevels) :: rdzw
real(c_float), dimension(nvertlevels) :: rdzu
real(c_float), dimension(nvertlevels) :: fzm
real(c_float), dimension(nvertlevels) :: fzp
real(c_float) :: v_theta_eddy_visc2
real(c_float) :: h_theta_eddy_visc4
real(c_float) :: prandtl_inv
real(c_float), dimension(nvertlevels,ncells+1) :: prandtl_3d_inv
real(c_float), dimension(nvertlevels,ncells+1) :: delsq_theta
real(c_float), dimension(nvertlevels,ncells+1) :: theta_m
real(c_float), dimension(nvertlevels,nedges+1) :: rho_edge
real(c_float), dimension(nvertlevels,ncells+1) :: rho_zz
real(c_float), dimension(nvertlevels,ncells+1) :: zz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_horz
real(c_float), dimension(nvertlevels,ncells+1) :: eddy_visc_vert
real(c_float), dimension(nvertlevels,ncells+1) :: bv_freq2
real(c_float) :: config_len_disp
real(c_float), dimension(num_scalars,nvertlevels,ncells+1) :: scalars
real(c_float), dimension(num_scalars,nvertlevels,ncells+1) :: tend_scalars
integer(c_int) :: index_tke
integer(c_int) :: index_qv
integer(c_int) :: num_scalars_dummy
integer(c_int) :: mix_scalars
logical :: l_mix_scalars
integer(c_int) :: les_model_opt
integer(c_int) :: les_surface_opt
real(c_float) :: dt
real(c_float) :: config_surface_heat_flux
real(c_float) :: config_surface_moisture_flux
real(c_float), dimension(nvertlevels,ncells+1) :: ureconstructzonal
real(c_float), dimension(nvertlevels,ncells+1) :: ureconstructmeridional
real(c_float), dimension(ncells+1) :: hfx
real(c_float), dimension(ncells+1) :: qfx
real(c_float), dimension(nvertlevels,ncells+1) :: tend_theta_euler
integer(c_int) :: dynamics_substep
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
l_config_mix_full=(config_mix_full/=0)
l_mix_scalars=(mix_scalars/=0)
call scalar_dissipation_3d_les(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,l_config_mix_full,t_init,zgrid,rdzw,rdzu,fzm,fzp,v_theta_eddy_visc2,h_theta_eddy_visc4,prandtl_inv,prandtl_3d_inv,delsq_theta,theta_m,rho_edge,rho_zz,zz,eddy_visc_horz,eddy_visc_vert,bv_freq2,config_len_disp,scalars,tend_scalars,index_tke,index_qv,num_scalars_dummy,l_mix_scalars,les_model_opt,les_surface_opt,clock,dt,config_surface_heat_flux,config_surface_moisture_flux,ureconstructzonal,ureconstructmeridional,hfx,qfx,tend_theta_euler,dynamics_substep)
end subroutine replay_scalar_dissipation_3d_les

subroutine replay_les_boundary(les_model_opt,les_surface_opt,dynamics_substep,eddy_visc_horz,eddy_visc_vert,u,v,ucell,vcell,w,c_s,bv_freq2,zgrid,config_len_disp,deformation_coef_c2,deformation_coef_s2,deformation_coef_cs,deformation_coef_c,deformation_coef_s,prandtl_3d_inv,invdt,h_mom_eddy_visc4,config_visc4_2dsmag,h_theta_eddy_visc4,scalars,tend_scalars,index_tke,rho_zz,meshscalingdel2,cellstart,cellend,nedgesoncell,edgesoncell,cellsonedge,ncells,nedges,nvertlevels,maxedges,num_scalars,bn2,theta_m,exner,pressure_b,pp,index_qv,index_qc,qtot,edgestart,edgeend,edgesolvestart,edgesolveend,vertexstart,vertexend,nvertices,vertexdegree,verticesonedge,edgesonvertex,edgesoncell_sign,edgesonvertex_sign,invareacell,invareatriangle,invdvedge,invdcedge,angleedge,dcedge,dvedge,meshscalingdel4,config_mix_full,v_mom_eddy_visc2,config_del4u_div_factor,zz,rdzu,rdzw,fzm,fzp,config_surface_drag_coefficient,delsq_u,delsq_vorticity,delsq_divergence,divergence,vorticity,rho_edge,u_init,v_init,ustm,tend_u_euler,cellsolvestart,cellsolveend,delsq_w,tend_w_euler,t_init,v_theta_eddy_visc2,prandtl_inv,delsq_theta,num_scalars_dummy,mix_scalars,dt,config_surface_heat_flux,config_surface_moisture_flux,ureconstructzonal,ureconstructmeridional,hfx,qfx,tend_theta_euler) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
use mpas_atm_dimensions, only: module_nv=>nVertLevels,module_me=>maxEdges,module_ns=>num_scalars
implicit none
integer :: k,iedge
type(MPAS_Clock_type) :: clock
logical :: l_config_mix_full,l_mix_scalars
integer(c_int) :: les_model_opt
integer(c_int) :: les_surface_opt
integer(c_int) :: dynamics_substep
real(c_float),dimension(nvertlevels,ncells+1) :: eddy_visc_horz
real(c_float),dimension(nvertlevels,ncells+1) :: eddy_visc_vert
real(c_float),dimension(nvertlevels,nedges+1) :: u
real(c_float),dimension(nvertlevels,nedges+1) :: v
real(c_float),dimension(nvertlevels,ncells+1) :: ucell
real(c_float),dimension(nvertlevels,ncells+1) :: vcell
real(c_float),dimension(nvertlevels+1,ncells+1) :: w
real(c_float) :: c_s
real(c_float),dimension(nvertlevels,ncells+1) :: bv_freq2
real(c_float),dimension(nvertlevels+1,ncells+1) :: zgrid
real(c_float) :: config_len_disp
real(c_float),dimension(maxedges,ncells+1) :: deformation_coef_c2
real(c_float),dimension(maxedges,ncells+1) :: deformation_coef_s2
real(c_float),dimension(maxedges,ncells+1) :: deformation_coef_cs
real(c_float),dimension(maxedges,ncells+1) :: deformation_coef_c
real(c_float),dimension(maxedges,ncells+1) :: deformation_coef_s
real(c_float),dimension(nvertlevels,ncells+1) :: prandtl_3d_inv
real(c_float) :: invdt
real(c_float) :: h_mom_eddy_visc4
real(c_float) :: config_visc4_2dsmag
real(c_float) :: h_theta_eddy_visc4
real(c_float),dimension(num_scalars,nvertlevels,ncells+1) :: scalars
real(c_float),dimension(num_scalars,nvertlevels,ncells+1) :: tend_scalars
integer(c_int) :: index_tke
real(c_float),dimension(nvertlevels,ncells+1) :: rho_zz
real(c_float),dimension(nedges+1) :: meshscalingdel2
integer(c_int) :: cellstart
integer(c_int) :: cellend
integer(c_int),dimension(ncells+1) :: nedgesoncell
integer(c_int),dimension(maxedges,ncells+1) :: edgesoncell
integer(c_int),dimension(2,nedges+1) :: cellsonedge
integer(c_int) :: ncells
integer(c_int) :: nedges
integer(c_int) :: nvertlevels
integer(c_int) :: maxedges
integer(c_int) :: num_scalars
real(c_float),dimension(nvertlevels,ncells+1) :: bn2
real(c_float),dimension(nvertlevels,ncells+1) :: theta_m
real(c_float),dimension(nvertlevels,ncells+1) :: exner
real(c_float),dimension(nvertlevels,ncells+1) :: pressure_b
real(c_float),dimension(nvertlevels,ncells+1) :: pp
integer(c_int) :: index_qv
integer(c_int) :: index_qc
real(c_float),dimension(nvertlevels,ncells+1) :: qtot
integer(c_int) :: edgestart
integer(c_int) :: edgeend
integer(c_int) :: edgesolvestart
integer(c_int) :: edgesolveend
integer(c_int) :: vertexstart
integer(c_int) :: vertexend
integer(c_int) :: nvertices
integer(c_int) :: vertexdegree
integer(c_int),dimension(2,nedges+1) :: verticesonedge
integer(c_int),dimension(vertexdegree,nvertices+1) :: edgesonvertex
real(c_float),dimension(maxedges,ncells+1) :: edgesoncell_sign
real(c_float),dimension(vertexdegree,nvertices+1) :: edgesonvertex_sign
real(c_float),dimension(ncells+1) :: invareacell
real(c_float),dimension(nvertices+1) :: invareatriangle
real(c_float),dimension(nedges+1) :: invdvedge
real(c_float),dimension(nedges+1) :: invdcedge
real(c_float),dimension(nedges+1) :: angleedge
real(c_float),dimension(nedges+1) :: dcedge
real(c_float),dimension(nedges+1) :: dvedge
real(c_float),dimension(nedges+1) :: meshscalingdel4
integer(c_int) :: config_mix_full
real(c_float) :: v_mom_eddy_visc2
real(c_float) :: config_del4u_div_factor
real(c_float),dimension(nvertlevels,ncells+1) :: zz
real(c_float),dimension(nvertlevels) :: rdzu
real(c_float),dimension(nvertlevels) :: rdzw
real(c_float),dimension(nvertlevels) :: fzm
real(c_float),dimension(nvertlevels) :: fzp
real(c_float) :: config_surface_drag_coefficient
real(c_float),dimension(nvertlevels,nedges+1) :: delsq_u
real(c_float),dimension(nvertlevels,nvertices+1) :: delsq_vorticity
real(c_float),dimension(nvertlevels,ncells+1) :: delsq_divergence
real(c_float),dimension(nvertlevels,ncells+1) :: divergence
real(c_float),dimension(nvertlevels,nvertices+1) :: vorticity
real(c_float),dimension(nvertlevels,nedges+1) :: rho_edge
real(c_float),dimension(nvertlevels) :: u_init
real(c_float),dimension(nvertlevels) :: v_init
real(c_float),dimension(ncells+1) :: ustm
real(c_float),dimension(nvertlevels,nedges+1) :: tend_u_euler
integer(c_int) :: cellsolvestart
integer(c_int) :: cellsolveend
real(c_float),dimension(nvertlevels,ncells+1) :: delsq_w
real(c_float),dimension(nvertlevels+1,ncells+1) :: tend_w_euler
real(c_float),dimension(nvertlevels,ncells+1) :: t_init
real(c_float) :: v_theta_eddy_visc2
real(c_float) :: prandtl_inv
real(c_float),dimension(nvertlevels,ncells+1) :: delsq_theta
integer(c_int) :: num_scalars_dummy
integer(c_int) :: mix_scalars
real(c_float) :: dt
real(c_float) :: config_surface_heat_flux
real(c_float) :: config_surface_moisture_flux
real(c_float),dimension(nvertlevels,ncells+1) :: ureconstructzonal
real(c_float),dimension(nvertlevels,ncells+1) :: ureconstructmeridional
real(c_float),dimension(ncells+1) :: hfx
real(c_float),dimension(ncells+1) :: qfx
real(c_float),dimension(nvertlevels,ncells+1) :: tend_theta_euler
module_nv=nvertlevels
module_me=maxedges
module_ns=num_scalars
l_config_mix_full=config_mix_full/=0
l_mix_scalars=mix_scalars/=0
if(les_model_opt==0)return
do iedge=edgestart,edgeend
do k=1,nvertlevels
tend_u_euler(k,iedge)=0.0
enddo
enddo
call calculate_n2(bn2,theta_m,exner,pressure_b,pp,zgrid,scalars,index_qv,index_qc,qtot,cellstart,cellend,ncells)
call les_models(les_model_opt,les_surface_opt,dynamics_substep,eddy_visc_horz,eddy_visc_vert,u,v,ucell,vcell,w,c_s,bn2,zgrid,config_len_disp,deformation_coef_c2,deformation_coef_s2,deformation_coef_cs,deformation_coef_c,deformation_coef_s,prandtl_3d_inv,invdt,h_mom_eddy_visc4,config_visc4_2dsmag,h_theta_eddy_visc4,scalars,tend_scalars,index_tke,rho_zz,meshscalingdel2,cellstart,cellend,nedgesoncell,edgesoncell,cellsonedge,ncells,nedges,nvertlevels,maxedges,num_scalars)
call u_dissipation_3d(edgestart,edgeend,edgesolvestart,edgesolveend,vertexstart,vertexend,cellstart,cellend,ncells,nedges,nvertices,vertexdegree,cellsonedge,verticesonedge,edgesoncell,edgesonvertex,nedgesoncell,edgesoncell_sign,edgesonvertex_sign,invareacell,invareatriangle,invdvedge,invdcedge,angleedge,dcedge,dvedge,meshscalingdel2,meshscalingdel4,l_config_mix_full,h_mom_eddy_visc4,v_mom_eddy_visc2,config_del4u_div_factor,zgrid,eddy_visc_horz,eddy_visc_vert,zz,rdzu,rdzw,fzm,fzp,les_model_opt,les_surface_opt,config_surface_drag_coefficient,delsq_u,delsq_vorticity,delsq_divergence,u,v,divergence,vorticity,rho_edge,rho_zz,u_init,v_init,ustm,tend_u_euler)
call w_dissipation_3d(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,rdzw,rdzu,v_mom_eddy_visc2,h_mom_eddy_visc4,delsq_w,w,rho_edge,rho_zz,divergence,zz,eddy_visc_horz,eddy_visc_vert,les_model_opt,les_surface_opt,tend_w_euler)
call scalar_dissipation_3d_les(cellstart,cellend,cellsolvestart,cellsolveend,ncells,nedges,nedgesoncell,edgesoncell,cellsonedge,edgesoncell_sign,invareacell,invdcedge,dvedge,meshscalingdel2,meshscalingdel4,l_config_mix_full,t_init,zgrid,rdzw,rdzu,fzm,fzp,v_theta_eddy_visc2,h_theta_eddy_visc4,prandtl_inv,prandtl_3d_inv,delsq_theta,theta_m,rho_edge,rho_zz,zz,eddy_visc_horz,eddy_visc_vert,bn2,config_len_disp,scalars,tend_scalars,index_tke,index_qv,num_scalars_dummy,l_mix_scalars,les_model_opt,les_surface_opt,clock,dt,config_surface_heat_flux,config_surface_moisture_flux,ureconstructzonal,ureconstructmeridional,hfx,qfx,tend_theta_euler,dynamics_substep)
end subroutine replay_les_boundary

! MPAS v8.4.2 mpas_atm_dissipation_models.F:57-78,95-116.
subroutine replay_les_selectors(text,n,model,surface) bind(C)
use iso_c_binding
use mpas_atm_dissipation_models
implicit none
integer(c_int), intent(in) :: n
character(c_char), intent(in) :: text(*)
integer(c_int), intent(out) :: model,surface
character(len=n) :: value
integer :: i
do i=1,n
   value(i:i)=text(i)
end do
model=les_model_from_string(value)
surface=les_surface_from_string(value)
end subroutine replay_les_selectors
