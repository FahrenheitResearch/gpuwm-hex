! Standalone copied algorithm: RRFS-MPAS 1d6e370e mpas_atm_core.F:1481-1584
subroutine gfl_topology_native(nCells,nCellsSolve,maxEdges,maxEdges2, &
 nEdgesOnCell,cellsOnCell,nCellsOnCellsOnCell,cellsOnCellsOnCell) bind(C)
 use iso_c_binding
 implicit none
 integer(c_int),value::nCells,nCellsSolve,maxEdges,maxEdges2
 integer(c_int)::nEdgesOnCell(nCells),cellsOnCell(maxEdges,nCells)
 integer(c_int)::nCellsOnCellsOnCell(nCells),cellsOnCellsOnCell(maxEdges2,nCells)
 integer::iCell,jCell,candidate,i,j,k,cellGroup(50),nCellsGroup,nOuterRow,nInnerRow
 logical::isNeighbor
      do iCell = 1, nCellsSolve  ! for each owned cell

         cellGroup(:) = 0
         nCellsGroup = 1 + nEdgesOnCell(iCell)
         nInnerRow = nCellsGroup
         cellGroup(1) = iCell
         cellGroup(2:nCellsGroup) = cellsOnCell(1:nCellsGroup-1,iCell)

         ! here cellGroup is a list of cells including iCell and its neighbors.
         ! we'll add the neighbors of the neighbors next

         do i = 1, nEdgesOnCell(iCell)

            jCell = cellsOnCell(i,iCell)
            do j=1,nEdgesOnCell(jCell)

               candidate = cellsOnCell(j,jCell)
               isNeighbor = .true.

               do k=1,nCellsGroup
                if(cellGroup(k) .eq. candidate) isNeighbor = .false.
               end do
               if(isNeighbor) then
                  cellGroup(nCellsGroup+1) = candidate
                  nCellsGroup = nCellsGroup +1
               end if
            end do
         end do

         nOuterRow = nCellsGroup - nInnerRow

         if(nOuterRow > maxEdges2) stop 'native maxEdges2 < nOuterRow'

         do i=1,nOuterRow
            cellsOnCellsOnCell(i,iCell) = cellGroup(i+nInnerRow)
         end do

         nCellsOnCellsOnCell(iCell) = nOuterRow


      end do
end subroutine
