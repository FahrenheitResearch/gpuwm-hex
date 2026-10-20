! Framework-only pool borrows for the correct-mode (config_physics_mode) mpas_atm_iau.F module.
! ABI v3: adds config_physics_mode, state u, mesh cellsOnEdge and diag iau_window_elapsed.
module mpas_kind_types
  implicit none
  integer, parameter :: RKIND=kind(1.0), StrKIND=512
end module

module mpas_derived_types
  use mpas_kind_types
  implicit none
  type mpas_pool_type
    type(mpas_pool_type), pointer :: mesh=>null(), state=>null(), diag=>null(), tend=>null(), tend_iau=>null()
    character(len=StrKIND), pointer :: option=>null(), mode=>null()
    integer, pointer :: cellsOnEdge(:,:)=>null()
    real(RKIND), pointer :: elapsed=>null()
    real(RKIND), pointer :: window=>null()
    integer, pointer :: nc=>null(), ne=>null(), ncs=>null(), nes=>null(), nl=>null(), ms=>null(), me=>null(), qv=>null()
    real(RKIND), pointer :: zz(:,:)=>null(), theta_m(:,:)=>null(), rho_zz(:,:)=>null(), rho_edge(:,:)=>null()
    real(RKIND), pointer :: theta(:,:)=>null(), rho(:,:)=>null(), u(:,:)=>null()
    real(RKIND), pointer :: scalars(:,:,:)=>null(), scalars_tend(:,:,:)=>null()
  end type
end module

module mpas_pool_routines
  use mpas_derived_types
  implicit none
  interface mpas_pool_get_config
    module procedure get_character, get_real
  end interface
  interface mpas_pool_get_array
    module procedure get_array0, get_array2, get_array2i, get_array3
  end interface
contains
  subroutine get_character(pool,name,out)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    character(len=StrKIND), pointer :: out
    select case(name)
    case('config_IAU_option'); out=>pool%option
    case('config_physics_mode'); out=>pool%mode
    case default; stop 11
    end select
  end subroutine
  subroutine get_real(pool,name,out)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    real(RKIND), pointer :: out
    if(name/='config_IAU_window_length_s') stop 12
    out=>pool%window
  end subroutine
  subroutine mpas_pool_get_subpool(pool,name,out)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    type(mpas_pool_type), pointer :: out
    select case(name)
    case('mesh'); out=>pool%mesh
    case('state'); out=>pool%state
    case('diag'); out=>pool%diag
    case('tend'); out=>pool%tend
    case('tend_iau'); out=>pool%tend_iau
    case default; stop 13
    end select
  end subroutine
  subroutine mpas_pool_get_dimension(pool,name,out)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    integer, pointer :: out
    select case(name)
    case('nCells'); out=>pool%nc
    case('nEdges'); out=>pool%ne
    case('nCellsSolve'); out=>pool%ncs
    case('nEdgesSolve'); out=>pool%nes
    case('nVertLevels'); out=>pool%nl
    case('moist_start'); out=>pool%ms
    case('moist_end'); out=>pool%me
    case('index_qv'); out=>pool%qv
    case default; stop 14
    end select
  end subroutine
  subroutine get_array2(pool,name,out,time_level)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    real(RKIND), pointer :: out(:,:)
    integer, optional :: time_level
    select case(name)
    case('zz'); out=>pool%zz
    case('theta_m')
      if (.not.present(time_level)) stop 15
      if (time_level/=1) stop 16
      out=>pool%theta_m
    case('rho_zz')
      if (.not.present(time_level)) stop 17
      if (time_level/=2) stop 18
      out=>pool%rho_zz
    case('rho_edge'); out=>pool%rho_edge
    case('theta'); out=>pool%theta
    case('rho'); out=>pool%rho
    case('u'); out=>pool%u
    case default; stop 19
    end select
  end subroutine
  subroutine get_array0(pool,name,out,time_level)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    real(RKIND), pointer :: out
    integer, optional :: time_level
    if(name/='iau_window_elapsed') stop 22
    out=>pool%elapsed
  end subroutine
  subroutine get_array2i(pool,name,out,time_level)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    integer, pointer :: out(:,:)
    integer, optional :: time_level
    if(name/='cellsOnEdge') stop 23
    out=>pool%cellsOnEdge
  end subroutine
  subroutine get_array3(pool,name,out,time_level)
    type(mpas_pool_type), intent(in) :: pool
    character(*), intent(in) :: name
    real(RKIND), pointer :: out(:,:,:)
    integer, optional :: time_level
    select case(name)
    case('scalars')
      if (present(time_level)) then
        if(time_level/=1) stop 20
      end if
      out=>pool%scalars
    case('scalars_tend'); out=>pool%scalars_tend
    case default; stop 21
    end select
  end subroutine
end module

module mpas_dmpar
end module
module mpas_log
  use mpas_kind_types
contains
  subroutine mpas_log_write(message,intArgs,realArgs)
    character(*) :: message
    integer, optional :: intArgs(:)
    real(RKIND), optional :: realArgs(:)
  end subroutine
end module
module mpas_timer
contains
  subroutine mpas_timer_start(name)
    character(*) :: name
  end subroutine
  subroutine mpas_timer_stop(name)
    character(*) :: name
  end subroutine
end module
