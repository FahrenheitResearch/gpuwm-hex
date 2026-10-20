"""Add unchanged LBC/mesh-scaling numerical slices to the direct oracle.

Framework time/stream operations are outside this component deck. The pool's
numeric stages are copied from v8.4.2 and receive explicit arrays and times.
"""
from pathlib import Path
import hashlib


def append_slices(source_root: Path, out: Path, pins: list):
    boundaries=(source_root/'src/core_atmosphere/dynamics/mpas_atm_boundaries.F').read_text()
    core=(source_root/'src/core_atmosphere/mpas_atm_core.F').read_text()
    integration=(source_root/'src/core_atmosphere/dynamics/mpas_atm_time_integration.F').read_text()
    def slice_text(source, start, end, file, name):
        first=source.index(start)
        last=source.index(end,first)
        text=source[first:last]
        pins.append({'file':file,'routine':name,'first_line':source[:first].count('\n')+1,
                     'last_line':source[:last].count('\n')+1,'sha256':hashlib.sha256(text.encode()).hexdigest()})
        return text
    derive=slice_text(boundaries,'        ! Compute lbc_rho_zz','        if (.not. firstCall) then\n            lbc_interval',
                      'mpas_atm_boundaries.F','mpas_atm_update_bdy_tend:derive')
    tendency=slice_text(boundaries,'            dt = 1.0_RKIND / dt','            !\n            ! Logging',
                        'mpas_atm_boundaries.F','mpas_atm_update_bdy_tend:tendency')
    state=slice_text(boundaries,'        !$acc parallel default(present)\n        !$acc loop gang vector collapse(3)\n        do i=1, horizDim+1',
                     '    end subroutine mpas_atm_get_bdy_state_3d',
                     'mpas_atm_boundaries.F','mpas_atm_get_bdy_state_3d:numeric')
    scaling=slice_text(core,'      meshScalingRegionalCell(:) = 1.0','   end subroutine atm_compute_mesh_scaling',
                       'mpas_atm_core.F','atm_compute_mesh_scaling:regional')
    velocity=slice_text(integration,'               do iEdge = 1, nEdgesSolve\n',
                        '               !$acc end parallel',
                        'mpas_atm_time_integration.F','atm_srk3:specified_velocity')
    clamp=slice_text(integration,'      where ( scalars_2(:,:,:) < 0.0)',
                     '      MPAS_ACC_TIMER_START(',
                     'mpas_atm_time_integration.F','atm_srk3:scalar_clamp')
    declarations='''
integer, intent(in) :: nCells,nEdges,nVertLevels,nScalars,index_qv
integer, intent(in) :: cellsOnEdge(2,nEdges+1)
real(RKIND), intent(inout) :: zz(nVertLevels,nCells+1)
real(RKIND), intent(inout) :: rho(nVertLevels,nCells+1),theta(nVertLevels,nCells+1)
real(RKIND), intent(inout) :: rho_zz(nVertLevels,nCells+1),rtheta_m(nVertLevels,nCells+1)
real(RKIND), intent(inout) :: rho_edge(nVertLevels,nEdges+1),ru(nVertLevels,nEdges+1),u(nVertLevels,nEdges+1)
real(RKIND), intent(inout) :: w(nVertLevels+1,nCells+1),scalars(nScalars,nVertLevels,nCells+1)
integer :: iCell,iEdge,cell1,cell2,k,j
'''
    args='nCells,nEdges,nVertLevels,nScalars,index_qv,cellsOnEdge,zz,rho,theta,rho_zz,rtheta_m,rho_edge,ru,u,w,scalars'
    code='module direct_lbc\nuse mpas_kind_types\nimplicit none\ninteger, parameter :: nRelaxZone=5\nreal(RKIND), parameter :: rvord=461.6_RKIND/287.0_RKIND\ncontains\n'
    code+=f'subroutine derive_lbc({args})\n{declarations}\n{derive}\nend subroutine\n'
    code+=f'subroutine tend_lbc({args},dt,lbc_tend_u,lbc_tend_ru,lbc_tend_rho_edge,lbc_tend_w,lbc_tend_theta,lbc_tend_rtheta_m,lbc_tend_rho_zz,lbc_tend_rho,lbc_tend_scalars)\n{declarations}\n'
    code+='''real(RKIND), intent(inout) :: dt
real(RKIND), intent(inout) :: lbc_tend_u(nVertLevels,nEdges+1),lbc_tend_ru(nVertLevels,nEdges+1),lbc_tend_rho_edge(nVertLevels,nEdges+1)
real(RKIND), intent(inout) :: lbc_tend_w(nVertLevels+1,nCells+1)
real(RKIND), intent(inout) :: lbc_tend_theta(nVertLevels,nCells+1),lbc_tend_rtheta_m(nVertLevels,nCells+1),lbc_tend_rho_zz(nVertLevels,nCells+1),lbc_tend_rho(nVertLevels,nCells+1)
real(RKIND), intent(inout) :: lbc_tend_scalars(nScalars,nVertLevels,nCells+1)
'''+tendency+'end subroutine\n'
    code+='''subroutine state_lbc(innerDim,vertDim,horizDim,state,tend,dt,return_state)
integer, intent(in) :: innerDim,vertDim,horizDim
real(RKIND), intent(in) :: state(innerDim,vertDim,horizDim+1),tend(innerDim,vertDim,horizDim+1),dt
real(RKIND), intent(out) :: return_state(innerDim,vertDim,horizDim+1)
integer :: i,j,k
'''+state+'end subroutine\n'
    code+='''subroutine scaling_lbc(nCells,nEdges,cellsOnEdge,meshDensity,config_h_ScaleWithMesh,meshScalingRegionalCell,meshScalingRegionalEdge)
integer, intent(in) :: nCells,nEdges,cellsOnEdge(2,nEdges+1)
real(RKIND), intent(in) :: meshDensity(nCells+1)
logical, intent(in) :: config_h_ScaleWithMesh
real(RKIND), intent(out) :: meshScalingRegionalCell(nCells+1),meshScalingRegionalEdge(nEdges+1)
integer :: iCell,iEdge,cell1,cell2
'''+scaling+'end subroutine\n'
    code+='''subroutine overwrite_native(nVertLevels,nEdges,nEdgesSolve,bdyMaskEdge,u,ru_driving_values)
integer, intent(in) :: nVertLevels,nEdges,nEdgesSolve,bdyMaskEdge(nEdges+1)
real(RKIND), intent(inout) :: u(nVertLevels,nEdges+1)
real(RKIND), intent(in) :: ru_driving_values(nVertLevels,nEdges+1)
integer :: iEdge,k
'''+velocity+'end subroutine\n'
    code+='''subroutine clamp_native(nt,nl,nc,scalars_2)
integer, intent(in) :: nt,nl,nc
real(RKIND), intent(inout) :: scalars_2(nt,nl,nc+1)
'''+clamp+'end subroutine\nend module\n'
    # Append after the full unchanged regional routines, before the fixture.
    with (out/'oracle.f90').open('a') as stream:
        stream.write(code)
