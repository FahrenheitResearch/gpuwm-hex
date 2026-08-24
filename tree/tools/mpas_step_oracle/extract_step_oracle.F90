program extract_step_oracle
   ! Lossless extractor for the frozen MPAS-A whole-step authority trajectory.
   !
   ! NetCDF-Fortran reverses the on-disk (Time, horizontal, vertical) dimensions.
   ! The arrays below are therefore (vertical, horizontal, Time).  A Fortran
   ! stream write emits vertical levels contiguously for each horizontal point,
   ! exactly matching a C-order array shaped (horizontal, vertical).
   use, intrinsic :: iso_fortran_env, only : real32, int64, error_unit
   use netcdf
   implicit none

   character(len=4096) :: history_t0, history_t1, output_dir
   integer :: argc

   argc = command_argument_count()
   if (argc /= 3) then
      write(error_unit, '(a)') &
         'usage: extract_step_oracle HISTORY_T0 HISTORY_T1 OUTPUT_DIRECTORY'
      error stop 2
   end if
   call get_command_argument(1, history_t0)
   call get_command_argument(2, history_t1)
   call get_command_argument(3, output_dir)

   call extract_history(trim(history_t0), trim(output_dir), 't0')
   call extract_history(trim(history_t1), trim(output_dir), 't1')

contains

   subroutine check(status, context)
      integer, intent(in) :: status
      character(len=*), intent(in) :: context

      if (status /= nf90_noerr) then
         write(error_unit, '(a,2a)') trim(context), ': ', trim(nf90_strerror(status))
         error stop 3
      end if
   end subroutine check

   subroutine dimension_length(ncid, name, length)
      integer, intent(in) :: ncid
      character(len=*), intent(in) :: name
      integer, intent(out) :: length
      integer :: dimid

      call check(nf90_inq_dimid(ncid, name, dimid), 'dimension '//trim(name))
      call check(nf90_inquire_dimension(ncid, dimid, len=length), &
         'dimension length '//trim(name))
   end subroutine dimension_length

   subroutine read_variable(ncid, name, values)
      integer, intent(in) :: ncid
      character(len=*), intent(in) :: name
      real(real32), intent(out) :: values(:, :, :)
      integer :: varid

      call check(nf90_inq_varid(ncid, name, varid), 'variable '//trim(name))
      call check(nf90_get_var(ncid, varid, values), 'read '//trim(name))
   end subroutine read_variable

   subroutine write_raw(path, values)
      character(len=*), intent(in) :: path
      real(real32), intent(in) :: values(:, :, :)
      integer :: unit, stat
      character(len=512) :: message

      open(newunit=unit, file=path, access='stream', form='unformatted', &
         action='write', status='replace', iostat=stat, iomsg=message)
      if (stat /= 0) then
         write(error_unit, '(3a)') 'open ', trim(path), ': '//trim(message)
         error stop 4
      end if
      write(unit, iostat=stat, iomsg=message) values
      if (stat /= 0) then
         write(error_unit, '(3a)') 'write ', trim(path), ': '//trim(message)
         error stop 5
      end if
      close(unit)
   end subroutine write_raw

   subroutine extract_one(ncid, output_dir, time_id, name, horizontal, vertical)
      integer, intent(in) :: ncid, horizontal, vertical
      character(len=*), intent(in) :: output_dir, time_id, name
      real(real32), allocatable :: values(:, :, :)
      character(len=8192) :: path
      integer(int64) :: count

      allocate(values(vertical, horizontal, 1))
      call read_variable(ncid, name, values)
      path = trim(output_dir)//'/'//trim(time_id)//'_'//trim(name)//'.f32le'
      call write_raw(trim(path), values)
      count = int(size(values), int64)
      write(*, '(a,1x,a,1x,i0,1x,a,1x,es24.16,1x,es24.16)') &
         'wrote', trim(path), count, 'values min/max', minval(values), maxval(values)
      deallocate(values)
   end subroutine extract_one

   subroutine extract_history(history, output_dir, time_id)
      character(len=*), intent(in) :: history, output_dir, time_id
      integer :: ncid, n_cells, n_edges, n_vertices, n_levels, n_levels_p1, n_time

      call check(nf90_open(history, nf90_nowrite, ncid), 'open '//trim(history))
      call dimension_length(ncid, 'nCells', n_cells)
      call dimension_length(ncid, 'nEdges', n_edges)
      call dimension_length(ncid, 'nVertices', n_vertices)
      call dimension_length(ncid, 'nVertLevels', n_levels)
      call dimension_length(ncid, 'nVertLevelsP1', n_levels_p1)
      call dimension_length(ncid, 'Time', n_time)

      if (n_time /= 1) then
         write(error_unit, '(a,i0)') 'expected exactly one Time record; found ', n_time
         error stop 6
      end if
      if (n_cells /= 2562 .or. n_edges /= 7680 .or. n_vertices /= 5120 .or. &
          n_levels /= 15 .or. &
          n_levels_p1 /= 16) then
         write(error_unit, '(a,5(1x,i0))') &
            'refusing unexpected dimensions:', n_cells, n_edges, n_vertices, &
            n_levels, n_levels_p1
         error stop 7
      end if

      write(*, '(a,1x,a,4(1x,i0))') &
         'extracting', trim(history), n_cells, n_edges, n_levels, n_levels_p1
      call extract_one(ncid, output_dir, time_id, 'rho', n_cells, n_levels)
      call extract_one(ncid, output_dir, time_id, 'theta', n_cells, n_levels)
      call extract_one(ncid, output_dir, time_id, 'u', n_edges, n_levels)
      call extract_one(ncid, output_dir, time_id, 'w', n_cells, n_levels_p1)
      call extract_one(ncid, output_dir, time_id, 'pressure', n_cells, n_levels)
      call extract_one(ncid, output_dir, time_id, 'qv', n_cells, n_levels)
      call extract_one(ncid, output_dir, time_id, 'divergence', n_cells, n_levels)
      call extract_one(ncid, output_dir, time_id, 'vorticity', n_vertices, n_levels)
      call extract_one(ncid, output_dir, time_id, 'ke', n_cells, n_levels)
      call check(nf90_close(ncid), 'close '//trim(history))
   end subroutine extract_history

end program extract_step_oracle
