"""Wrap unchanged pinned native upper damping blocks, not rewritten equations."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text().splitlines(keepends=True)
header='''subroutine native_upper(nVertLevels,nCells,nEdges,eddy_visc_horz,tend_u,rho_edge,u, &
 config_len_disp,config_mpas_cam_coef,config_number_cam_damping_levels, &
 config_number_rayleigh_damp_u_levels,config_rayleigh_damp_u_timescale_days,edgeSolveEnd)
 use mpas_kind_types, only: RKIND
 implicit none
 integer,intent(in)::nVertLevels,nCells,nEdges,config_number_cam_damping_levels
 integer,intent(in)::config_number_rayleigh_damp_u_levels,edgeSolveEnd
 integer::iCell,iEdge,k,cellStart,cellEnd,edgeSolveStart
 real(kind=RKIND),intent(in)::config_len_disp,config_mpas_cam_coef,config_rayleigh_damp_u_timescale_days
 real(kind=RKIND),intent(inout)::eddy_visc_horz(nVertLevels,nCells+1),tend_u(nVertLevels,nEdges+1)
 real(kind=RKIND),intent(in)::rho_edge(nVertLevels,nEdges+1),u(nVertLevels,nEdges+1)
 real(kind=RKIND)::visc2cam,rayleigh_coef_inverse,rayleigh_damp_coef(nVertLevels)
 real(kind=RKIND),parameter::seconds_per_day=86400.0_RKIND
 cellStart=1;cellEnd=nCells;edgeSolveStart=1
'''
# Pinned mpas_atm_time_integration.F:6390-6408,6610-6629.
assert 'if (config_mpas_cam_coef > 0.0)' in source[6389]
assert 'rayleigh_coef_inverse = ' in source[6610]
Path(sys.argv[2]).write_text(header+''.join(source[6389:6408])+''.join(source[6609:6629])+'end subroutine\n')
