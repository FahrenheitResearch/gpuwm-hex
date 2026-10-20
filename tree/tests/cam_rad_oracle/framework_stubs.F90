! Test-only single-rank MPAS framework adapters. No radiation math is replaced.
module mpas_atmphys_utilities
  implicit none
  character(len=512) :: mpas_err_message
contains
  subroutine physics_error_fatal(message)
    character(len=*), intent(in) :: message
    write(*,'(a)') trim(message)
    error stop 1
  end subroutine
  subroutine physics_message(message)
    character(len=*), intent(in) :: message
    write(*,'(a)') trim(message)
  end subroutine
  subroutine mpas_log_write(message, intArgs, realArgs)
    character(len=*), intent(in) :: message
    integer, optional, intent(in) :: intArgs(:)
    real, optional, intent(in) :: realArgs(:)
  end subroutine
end module

module mpas_dmpar
  use mpas_kind_types
  implicit none
  integer, parameter :: IO_NODE=0
  type dm_info
    integer :: my_proc_id=0
  end type
  interface mpas_dmpar_bcast_doubles
    module procedure bcast1, bcast2, bcast5
  end interface
contains
  subroutine mpas_dmpar_bcast_int(info, value)
    type(dm_info), intent(in) :: info
    integer, intent(inout) :: value
  end subroutine
  subroutine mpas_dmpar_max_int(info, value, result)
    type(dm_info), intent(in) :: info
    integer, intent(in) :: value
    integer, intent(out) :: result
    result=value
  end subroutine
  subroutine bcast1(info, n, a)
    type(dm_info), intent(in) :: info
    integer, intent(in) :: n
    real(R8KIND), intent(inout) :: a(:)
  end subroutine
  subroutine bcast2(info, n, a)
    type(dm_info), intent(in) :: info
    integer, intent(in) :: n
    real(R8KIND), intent(inout) :: a(:,:)
  end subroutine
  subroutine bcast5(info, n, a)
    type(dm_info), intent(in) :: info
    integer, intent(in) :: n
    real(R8KIND), intent(inout) :: a(:,:,:,:,:)
  end subroutine
end module

module mpas_pool_routines
  use mpas_kind_types
  implicit none
  type mpas_pool_type
    integer, pointer :: nc=>null(), nv=>null(), na=>null(), no=>null(), nm=>null()
    integer, pointer :: sul=>null(), salt=>null(), dust=>null(), ocpho=>null()
    integer, pointer :: bcpho=>null(), ocphi=>null(), bcphi=>null(), bg=>null(), volc=>null()
    real(RKIND), pointer :: lat(:)=>null(), lon(:)=>null(), pin(:)=>null(), ps(:)=>null()
    real(RKIND), pointer :: hybi(:,:)=>null(), pressure(:,:)=>null()
    real(RKIND), pointer :: ozone(:,:,:)=>null(), aerp(:,:,:)=>null(), aern(:,:,:)=>null()
  end type
  interface mpas_pool_get_array
    module procedure array1, array2, array3
  end interface
contains
  subroutine mpas_pool_get_dimension(pool, name, value)
    type(mpas_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    integer, pointer, intent(out) :: value
    select case(name)
    case('nCells','nCellsSolve'); value=>pool%nc
    case('nVertLevels'); value=>pool%nv
    case('nAerLevels'); value=>pool%na
    case('nOznLevels'); value=>pool%no
    case('nMonths'); value=>pool%nm
    case('index_sul'); value=>pool%sul
    case('index_sslt'); value=>pool%salt
    case('index_dust1'); value=>pool%dust
    case('index_ocpho'); value=>pool%ocpho
    case('index_bcpho'); value=>pool%bcpho
    case('index_ocphi'); value=>pool%ocphi
    case('index_bcphi'); value=>pool%bcphi
    case('index_bg'); value=>pool%bg
    case('index_volc'); value=>pool%volc
    case default; error stop 'oracle stub: unknown dimension'
    end select
    if (.not.associated(value)) error stop 'oracle stub: absent dimension'
  end subroutine
  subroutine array1(pool, name, value)
    type(mpas_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(RKIND), pointer, intent(out) :: value(:)
    select case(name)
    case('latCell'); value=>pool%lat
    case('lonCell'); value=>pool%lon
    case('pin'); value=>pool%pin
    case('m_ps'); value=>pool%ps
    case default; error stop 'oracle stub: unknown rank-1 field'
    end select
  end subroutine
  subroutine array2(pool, name, value)
    type(mpas_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(RKIND), pointer, intent(out) :: value(:,:)
    select case(name)
    case('m_hybi'); value=>pool%hybi
    case('pressure_base'); value=>pool%pressure
    case default; error stop 'oracle stub: unknown rank-2 field'
    end select
  end subroutine
  subroutine array3(pool, name, value, time)
    type(mpas_pool_type), intent(in) :: pool
    character(len=*), intent(in) :: name
    real(RKIND), pointer, intent(out) :: value(:,:,:)
    integer, optional, intent(in) :: time
    select case(name)
    case('ozmixm'); value=>pool%ozone
    case('aerosols')
      if (.not.present(time)) error stop 'oracle stub: aerosol time missing'
      select case(time)
      case(1); value=>pool%aerp
      case(2); value=>pool%aern
      case default; error stop 'oracle stub: invalid aerosol time'
      end select
    case default; error stop 'oracle stub: unknown rank-3 field'
    end select
  end subroutine
end module
