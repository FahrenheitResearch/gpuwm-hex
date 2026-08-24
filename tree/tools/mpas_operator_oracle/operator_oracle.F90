! Standalone, source-extracted M1 oracle for the frozen MPAS-A v8.2.3 loops.
!
! This executable intentionally has no dependency on the Python port.  It reads
! the published MPAS x1.2562 static file with NetCDF-Fortran, constructs fixed
! deterministic fields, and executes the scalar loop bodies used by the frozen
! atmosphere core.  The generated manifest labels this accurately as a
! source-extracted Fortran oracle, not a linked full-model oracle.
!
! Frozen authority:
!   src/core_atmosphere/mpas_atm_core.F:1137-1224       incidence signs
!   src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5561-5617
!   src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5710-5731
!   src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5787-5794

program mpas_operator_oracle
   use, intrinsic :: iso_fortran_env, only : real64
   use netcdf
   implicit none

   character(len=2048) :: mesh_path, output_dir
   integer :: ncid, n_cells, n_edges, n_vertices, max_edges, max_edges2
   integer :: vertex_degree, i, j, edge, cell, vertex, unit
   integer :: cell1, cell2, sign_value
   integer, allocatable :: cells_on_edge(:,:), edges_on_cell(:,:)
   integer, allocatable :: n_edges_on_cell(:), vertices_on_edge(:,:)
   integer, allocatable :: edges_on_vertex(:,:), cells_on_vertex(:,:)
   integer, allocatable :: edges_on_edge(:,:), n_edges_on_edge(:)
   real(real64), allocatable :: dc_edge(:), dv_edge(:), area_cell(:)
   real(real64), allocatable :: area_triangle(:), kite_areas_on_vertex(:,:)
   real(real64), allocatable :: weights_on_edge(:,:)
   real(real64), allocatable :: phi_cell(:), u_edge(:)
   real(real64), allocatable :: gradient(:), divergence(:), curl(:)
   real(real64), allocatable :: tangential(:), cell_edge(:), cell_vertex(:)

   call get_command_argument(1, mesh_path)
   call get_command_argument(2, output_dir)
   if (len_trim(mesh_path) == 0 .or. len_trim(output_dir) == 0) then
      error stop 'usage: operator_oracle STATIC_MESH_NC OUTPUT_DIRECTORY'
   end if

   call check(nf90_open(trim(mesh_path), nf90_nowrite, ncid), 'open static mesh')
   call dimension_length(ncid, 'nCells', n_cells)
   call dimension_length(ncid, 'nEdges', n_edges)
   call dimension_length(ncid, 'nVertices', n_vertices)
   call dimension_length(ncid, 'maxEdges', max_edges)
   call dimension_length(ncid, 'maxEdges2', max_edges2)
   call dimension_length(ncid, 'vertexDegree', vertex_degree)

   allocate(cells_on_edge(2,n_edges), edges_on_cell(max_edges,n_cells))
   allocate(n_edges_on_cell(n_cells), vertices_on_edge(2,n_edges))
   allocate(edges_on_vertex(vertex_degree,n_vertices))
   allocate(cells_on_vertex(vertex_degree,n_vertices))
   allocate(edges_on_edge(max_edges2,n_edges), n_edges_on_edge(n_edges))
   allocate(dc_edge(n_edges), dv_edge(n_edges), area_cell(n_cells))
   allocate(area_triangle(n_vertices), kite_areas_on_vertex(vertex_degree,n_vertices))
   allocate(weights_on_edge(max_edges2,n_edges))
   allocate(phi_cell(n_cells), u_edge(n_edges))
   allocate(gradient(n_edges), divergence(n_cells), curl(n_vertices))
   allocate(tangential(n_edges), cell_edge(n_edges), cell_vertex(n_vertices))

   call read_integer_2d(ncid, 'cellsOnEdge', cells_on_edge)
   call read_integer_2d(ncid, 'edgesOnCell', edges_on_cell)
   call read_integer_1d(ncid, 'nEdgesOnCell', n_edges_on_cell)
   call read_integer_2d(ncid, 'verticesOnEdge', vertices_on_edge)
   call read_integer_2d(ncid, 'edgesOnVertex', edges_on_vertex)
   call read_integer_2d(ncid, 'cellsOnVertex', cells_on_vertex)
   call read_integer_2d(ncid, 'edgesOnEdge', edges_on_edge)
   call read_integer_1d(ncid, 'nEdgesOnEdge', n_edges_on_edge)
   call read_real_1d(ncid, 'dcEdge', dc_edge)
   call read_real_1d(ncid, 'dvEdge', dv_edge)
   call read_real_1d(ncid, 'areaCell', area_cell)
   call read_real_1d(ncid, 'areaTriangle', area_triangle)
   call read_real_2d(ncid, 'kiteAreasOnVertex', kite_areas_on_vertex)
   call read_real_2d(ncid, 'weightsOnEdge', weights_on_edge)
   call check(nf90_close(ncid), 'close static mesh')

   do i = 1, n_cells
      phi_cell(i) = sin(0.017_real64 * real(i,real64)) &
                  + cos(0.013_real64 * real(i,real64))
   end do
   do i = 1, n_edges
      u_edge(i) = sin(0.019_real64 * real(i,real64)) &
                - 0.3_real64 * cos(0.011_real64 * real(i,real64))
   end do

   ! Normal scalar gradient and arithmetic cell-to-edge interpolation.
   do edge = 1, n_edges
      cell1 = cells_on_edge(1,edge)
      cell2 = cells_on_edge(2,edge)
      gradient(edge) = (phi_cell(cell2) - phi_cell(cell1)) / dc_edge(edge)
      cell_edge(edge) = 0.5_real64 * (phi_cell(cell1) + phi_cell(cell2))
   end do

   ! Finite-volume divergence, preserving the frozen edge-slot loop order.
   divergence = 0.0_real64
   do cell = 1, n_cells
      do j = 1, n_edges_on_cell(cell)
         edge = edges_on_cell(j,cell)
         if (cells_on_edge(1,edge) == cell) then
            sign_value = 1
         else if (cells_on_edge(2,edge) == cell) then
            sign_value = -1
         else
            error stop 'non-reciprocal cellsOnEdge/edgesOnCell'
         end if
         divergence(cell) = divergence(cell) &
                          + real(sign_value,real64) * dv_edge(edge) * u_edge(edge)
      end do
      divergence(cell) = divergence(cell) / area_cell(cell)
   end do

   ! Vertex circulation divided by dual-triangle area.
   curl = 0.0_real64
   do vertex = 1, n_vertices
      do j = 1, vertex_degree
         edge = edges_on_vertex(j,vertex)
         if (vertices_on_edge(2,edge) == vertex) then
            sign_value = 1
         else if (vertices_on_edge(1,edge) == vertex) then
            sign_value = -1
         else
            error stop 'non-reciprocal verticesOnEdge/edgesOnVertex'
         end if
         curl(vertex) = curl(vertex) &
                      + real(sign_value,real64) * dc_edge(edge) * u_edge(edge)
      end do
      curl(vertex) = curl(vertex) / area_triangle(vertex)
   end do

   ! Perpendicular edge velocity reconstruction.
   tangential = 0.0_real64
   do edge = 1, n_edges
      do j = 1, n_edges_on_edge(edge)
         tangential(edge) = tangential(edge) &
                          + weights_on_edge(j,edge) * u_edge(edges_on_edge(j,edge))
      end do
   end do

   ! Kite-area cell-to-vertex interpolation.
   cell_vertex = 0.0_real64
   do vertex = 1, n_vertices
      do j = 1, vertex_degree
         cell_vertex(vertex) = cell_vertex(vertex) &
                             + kite_areas_on_vertex(j,vertex) &
                             * phi_cell(cells_on_vertex(j,vertex))
      end do
      cell_vertex(vertex) = cell_vertex(vertex) / area_triangle(vertex)
   end do

   open(newunit=unit, file=trim(output_dir)//'/inputs.csv', &
        status='replace', action='write')
   write(unit,'(A)') 'field,index,value'
   do i = 1, n_cells
      write(unit,'(A,",",I0,",",ES26.17E3)') 'phi_cell', i, phi_cell(i)
   end do
   do i = 1, n_edges
      write(unit,'(A,",",I0,",",ES26.17E3)') 'u_edge', i, u_edge(i)
   end do
   close(unit)

   open(newunit=unit, file=trim(output_dir)//'/outputs.csv', &
        status='replace', action='write')
   write(unit,'(A)') 'operator,index,value'
   call write_vector(unit, 'edge_scalar_gradient', gradient)
   call write_vector(unit, 'edge_to_cell_divergence', divergence)
   call write_vector(unit, 'edge_to_vertex_curl', curl)
   call write_vector(unit, 'tangential_velocity', tangential)
   call write_vector(unit, 'cell_to_edge', cell_edge)
   call write_vector(unit, 'cell_to_vertex', cell_vertex)
   close(unit)

contains

   subroutine check(status, context)
      integer, intent(in) :: status
      character(len=*), intent(in) :: context
      if (status /= nf90_noerr) then
         write(*,'(A,": ",A)') trim(context), trim(nf90_strerror(status))
         error stop 2
      end if
   end subroutine check

   subroutine dimension_length(file_id, name, length_value)
      integer, intent(in) :: file_id
      character(len=*), intent(in) :: name
      integer, intent(out) :: length_value
      integer :: dimension_id
      call check(nf90_inq_dimid(file_id, name, dimension_id), 'find '//name)
      call check(nf90_inquire_dimension(file_id, dimension_id, len=length_value), &
                 'read dimension '//name)
   end subroutine dimension_length

   subroutine read_integer_1d(file_id, name, values)
      integer, intent(in) :: file_id
      character(len=*), intent(in) :: name
      integer, intent(out) :: values(:)
      integer :: variable_id
      call check(nf90_inq_varid(file_id, name, variable_id), 'find '//name)
      call check(nf90_get_var(file_id, variable_id, values), 'read '//name)
   end subroutine read_integer_1d

   subroutine read_integer_2d(file_id, name, values)
      integer, intent(in) :: file_id
      character(len=*), intent(in) :: name
      integer, intent(out) :: values(:,:)
      integer :: variable_id
      call check(nf90_inq_varid(file_id, name, variable_id), 'find '//name)
      call check(nf90_get_var(file_id, variable_id, values), 'read '//name)
   end subroutine read_integer_2d

   subroutine read_real_1d(file_id, name, values)
      integer, intent(in) :: file_id
      character(len=*), intent(in) :: name
      real(real64), intent(out) :: values(:)
      integer :: variable_id
      call check(nf90_inq_varid(file_id, name, variable_id), 'find '//name)
      call check(nf90_get_var(file_id, variable_id, values), 'read '//name)
   end subroutine read_real_1d

   subroutine read_real_2d(file_id, name, values)
      integer, intent(in) :: file_id
      character(len=*), intent(in) :: name
      real(real64), intent(out) :: values(:,:)
      integer :: variable_id
      call check(nf90_inq_varid(file_id, name, variable_id), 'find '//name)
      call check(nf90_get_var(file_id, variable_id, values), 'read '//name)
   end subroutine read_real_2d

   subroutine write_vector(file_unit, name, values)
      integer, intent(in) :: file_unit
      character(len=*), intent(in) :: name
      real(real64), intent(in) :: values(:)
      integer :: index_value
      do index_value = 1, size(values)
         write(file_unit,'(A,",",I0,",",ES26.17E3)') &
            trim(name), index_value, values(index_value)
      end do
   end subroutine write_vector

end program mpas_operator_oracle
