! Intrinsic-only seam oracle for mpas_atm_dissipation_models.F:859-860.
! Complete scientific dissipation module is qualified by the separate deck.
program native_angle_reference
  use iso_fortran_env, only: int32
  implicit none
  character(len=1024) :: source, destination
  integer :: n, k, bytes
  real, allocatable :: angle(:), c(:), s(:)
  call get_command_argument(1,source)
  call get_command_argument(2,destination)
  if (trim(source) == '--synthetic') then
    allocate(angle(1036))
    angle(1:12) = [0.0,-0.0,transfer(1_int32,0.0),transfer(-2147483647_int32,0.0), &
      tiny(0.0),-tiny(0.0),1.0,-1.0,1.0e20,-1.0e20,huge(0.0),-huge(0.0)]
    do k=13,size(angle)
      angle(k)=real(k-525)*0.006134033203125
    end do
    open(11,file=trim(destination),access='stream',form='unformatted',convert='little_endian',status='new')
    write(11) angle
    close(11)
    stop
  end if
  inquire(file=trim(source),size=bytes)
  if (mod(bytes,4)/=0 .or. bytes<4) stop 2
  n=bytes/4
  allocate(angle(n),c(n),s(n))
  open(11,file=trim(source),access='stream',form='unformatted',convert='little_endian',status='old')
  read(11) angle
  close(11)
  do k=1,n
    c(k)=cos(angle(k))
    s(k)=sin(angle(k))
  end do
  open(12,file=trim(destination)//'/native-cos.bin',access='stream',form='unformatted',convert='little_endian',status='new')
  write(12) c
  close(12)
  open(13,file=trim(destination)//'/native-sin.bin',access='stream',form='unformatted',convert='little_endian',status='new')
  write(13) s
  close(13)
end program native_angle_reference
