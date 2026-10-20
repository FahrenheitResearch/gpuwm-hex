subroutine gfl_spread_native(nCells,nCellsSolve,nVertLevels,maxEdges,maxEdges2, &
 config_gfl_sub3d,bdyMaskCell,nEdgesOnCell,cellsOnCell,nCellsOnCellsOnCell,cellsOnCellsOnCell, &
 sigma_deep,sigma_deep_far,sub3d_rthcuten,sub3d_rqvcuten,sub3d_rucuten,sub3d_rvcuten, &
 rthcuten,rqvcuten,rucuten,rvcuten) bind(C)
use iso_c_binding
implicit none
integer,parameter::RKIND=c_float
integer(c_int),value:: nCells,nCellsSolve,nVertLevels,maxEdges,maxEdges2,config_gfl_sub3d
integer(c_int)::bdyMaskCell(nCells),nEdgesOnCell(nCells),cellsOnCell(maxEdges,nCells)
integer(c_int)::nCellsOnCellsOnCell(nCells),cellsOnCellsOnCell(maxEdges2,nCells)
real(c_float)::sigma_deep(nCells),sigma_deep_far(nCells)
real(c_float)::sub3d_rthcuten(nVertLevels,nCells),sub3d_rqvcuten(nVertLevels,nCells)
real(c_float)::sub3d_rucuten(nVertLevels,nCells),sub3d_rvcuten(nVertLevels,nCells)
real(c_float)::rthcuten(nVertLevels,nCells),rqvcuten(nVertLevels,nCells)
real(c_float)::rucuten(nVertLevels,nCells),rvcuten(nVertLevels,nCells)
integer::iCell,k,i,j,ineighbor,ineighbor_far
if(config_gfl_sub3d>0)then
            !-- part 3: spread the subsidence tendencies to the 1st neighbors
            do iCell = 1, nCellsSolve
            ! do not do this at the boundary of the regional domain!
            !if(bdyMaskCell(icell) == 7 )cycle
            if(bdyMaskCell(iCell) > 4 )cycle
              do k = 1, nVertLevels
                 rthcuten(k,iCell) = rthcuten(k,iCell) + (1._RKIND - sigma_deep(iCell)) * sub3d_rthcuten(k,iCell)
                 rqvcuten(k,iCell) = rqvcuten(k,iCell) + (1._RKIND - sigma_deep(iCell)) * sub3d_rqvcuten(k,iCell)
                 rucuten(k,iCell)  = rucuten(k,iCell)  + (1._RKIND - sigma_deep(iCell)) * sub3d_rucuten(k,iCell)
                 rvcuten(k,iCell)  = rvcuten(k,iCell)  + (1._RKIND - sigma_deep(iCell)) * sub3d_rvcuten(k,iCell)
                 do i = 1, nEdgesOnCell(iCell)
                   ineighbor = cellsOnCell(i,iCell)
                   ! do not do this if there is no convection in that neighbor cell!
                   ! ineighbor here is the i-th cell from the first row, which should have CellsOnCells elements
                   if(sigma_deep(ineighbor) == 0.)cycle
! do this for first row cells
! 1-sigma_deep-sigma_deep_far what comes in from the first row neighbor,
! 1-sigma_deep_far            what comes in from second row neighbor

                   rthcuten(k,iCell) = rthcuten(k,iCell) + &
                                       (sigma_deep(ineighbor)- sigma_deep_far(ineighbor)) &
                                       *sub3d_rthcuten(k,ineighbor)/ &
                                       real(nEdgesOnCell(icell),RKIND)
                   rqvcuten(k,iCell) = rqvcuten(k,iCell) + &
                                       (sigma_deep(ineighbor)- sigma_deep_far(ineighbor)) &
                                       *sub3d_rqvcuten(k,ineighbor)/ &
                                       real(nEdgesOnCell(icell),RKIND)
                   rucuten(k,iCell) = rucuten(k,iCell) + &
                                       (sigma_deep(ineighbor) - sigma_deep_far(ineighbor)) &
                                      *sub3d_rucuten(k,ineighbor)/ &
                                       real(nEdgesOnCell(icell),RKIND)
                   rvcuten(k,iCell) = rvcuten(k,iCell) + &
                                       (sigma_deep(ineighbor)- sigma_deep_far(ineighbor))  &
                                       *sub3d_rvcuten(k,ineighbor)/ &
                                       real(nEdgesOnCell(icell),RKIND)
                   enddo ! i, first row
! now for second row for this cell 
!
                   do j=1,nCellsOnCellsOnCell(icell)
                   ! ineighbor_far here is the j-th cell from the second row, which should have nCellsOnCellsOnCell elements
                   ineighbor_far = cellsOnCellsOnCell(j,iCell)
                   rthcuten(k,iCell) = rthcuten(k,iCell) + &
                                       sigma_deep_far(ineighbor_far)*sub3d_rthcuten(k,ineighbor_far)/ &
                                       real(nCellsOnCellsOnCell(icell),RKIND)                               
                   rqvcuten(k,iCell) = rqvcuten(k,iCell) + &
                                       sigma_deep_far(ineighbor_far)*sub3d_rqvcuten(k,ineighbor_far)/ &
                                       real(nCellsOnCellsOnCell(icell),RKIND)
                   rucuten(k,iCell) = rucuten(k,iCell) + &
                                       sigma_deep_far(ineighbor_far)*sub3d_rucuten(k,ineighbor_far)/ &
                                       real(nCellsOnCellsOnCell(icell),RKIND)
                   rvcuten(k,iCell) = rvcuten(k,iCell) + &
                                       sigma_deep_far(ineighbor_far)*sub3d_rvcuten(k,ineighbor_far)/ &
                                       real(nCellsOnCellsOnCell(icell),RKIND)
                   enddo  ! j - second row
              enddo ! k
            enddo ! icell
       else !if (bdyMaskCell(icell) > 4  .or. config_gfl_sub3d == 0 ) then
            do i = 1, nCellsSolve
            do k = 1, nVertLevels
                 rucuten(k,i)  = rucuten(k,i)  + sub3d_rucuten(k,i)
                 rvcuten(k,i)  = rvcuten(k,i)  + sub3d_rvcuten(k,i)
                 rthcuten(k,i) = rthcuten(k,i) + sub3d_rthcuten(k,i)
                 rqvcuten(k,i) = rqvcuten(k,i) + sub3d_rqvcuten(k,i)
            enddo
            enddo

       endif
end subroutine

! RRFS-MPAS 1d6e370e mpas_atmphys_driver_convection.F:1025-1067.
subroutine gfl_prepare_native(n,nz,nc,len_disp,density,lw,sw,dx,rad) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n,nz,nc
real(c_float),value::len_disp
real(c_float)::density(nc),lw(nz,nc),sw(nz,nc),dx(n),rad(n,nz+1)
integer::i,k
do i=1,n
 dx(i)=len_disp/density(i)**0.25
 do k=1,nz
  rad(i,k)=lw(k,i)+sw(k,i)
 enddo
enddo
end subroutine
