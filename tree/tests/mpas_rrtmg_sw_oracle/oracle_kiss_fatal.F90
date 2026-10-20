! MPAS module_ra_rrtmg_sw.F:1742-1744 native pressure-order STOP.
program oracle_kiss_fatal
 use mcica_subcol_gen_sw,only:mcica_subcol_sw
 implicit none
 integer,parameter::nc=1,nl=4,ng=112
 integer::rng
 real::play(nc,nl),x(nc,nl),optics(14,nc,nl),output(8,ng,nc,nl),radius(3,nc,nl)
 play(1,:)=[100.,200.,300.,400.]
 x=1.;optics=.5;output=-987.25;radius=-987.25;rng=0
 call mcica_subcol_sw(1,nc,nl,1,7,rng,play,x,x,x,x,x,x,x,optics,optics,optics,optics, &
   x,0,279,-41.25,output(1,:,:,:),output(2,:,:,:),output(3,:,:,:),output(4,:,:,:), &
   radius(1,:,:),radius(2,:,:),radius(3,:,:),output(5,:,:,:),output(6,:,:,:),output(7,:,:,:),output(8,:,:,:))
 print *, 'UNEXPECTED RETURN FROM NATIVE KISS PRESSURE STOP'
end program oracle_kiss_fatal
