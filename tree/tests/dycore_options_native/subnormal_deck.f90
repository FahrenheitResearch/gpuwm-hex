! Gradual-underflow sentinel plus unchanged native upper-driver blocks.
! MPAS-A v8.4.2 mpas_atm_time_integration.F:6390-6408,6610-6629.
program subnormal_deck
 implicit none
 real :: input(8), factor(8), product(8), cam(8,1), ray(8,1), rho(8,1), u(8,1)
 integer :: k
 do k=1,8
 input(k)=transfer(1048576+k,0.0)
 if(mod(k,2)==0)input(k)=-input(k)
 factor(k)=16.0
 end do
 product=input*factor
 cam=0.0;ray=0.0;rho(:,1)=factor;u(:,1)=input
 call native_upper(8,1,1,cam,ray,rho,u,12000.0,0.0,4,8,0.03125,1)
 open(20,file='subnormal-upper.bin',access='stream',form='unformatted',convert='little_endian')
 write(20)input,factor,product,cam,ray
 close(20)
end program
