! Native initialization and exact MPAS driver-slice arithmetic.
! module_cu_ntiedtke.F:281-351; driver_convection.F:837-850.
program lifecycle
 use mpas_kind_types,only: RKIND
 use module_cu_ntiedtke,only: ntiedtkeinit
 use cu_ntiedtke,only: cu_ntiedtke_finalize
 use cu_ntiedtke_pre,only: cu_ntiedtke_pre_init,cu_ntiedtke_pre_finalize
 use cu_ntiedtke_post,only: cu_ntiedtke_post_init,cu_ntiedtke_post_finalize
 implicit none
 real(RKIND):: a(2,3,1,8), rt(7), rb(7), lw(7), sw(7), pi(7), qd(7), qb(7), density(7)
 real(RKIND):: th(7),qv(7),dx(7)
 integer:: mode,f,k,i,err
 character(len=256)::msg
 rt=[1.e10,1.e-8,-3.e-5,1.0,-1.0,1.e-20,0.0]
 rb=[-1.e10,2.e-8,3.e-5,-1.0,1.e-7,2.e-20,-0.0]
 lw=[1.0,3.e-8,1.e-12,1.e-6,1.0,-1.e-20,0.0]
 sw=[-1.0,-4.e-8,1.e-8,-1.e-6,-1.e-7,1.e-20,-0.0]
 pi=[0.75,1.01,0.92,0.3,0.6,0.7,1.0]
 qd=rb*0.1; qb=sw*0.1
 density=[0.01,0.1251,0.1,0.75,1.0,4.0,100.0]
 open(71,file='driver-slice.csv',status='replace')
 write(71,'(a)') 'i,rt,rb,lw,sw,pi,qd,qb,density,th,qv,dx'
 do i=1,7
   dx(i)=120000.0 / density(i)**0.25
   qv(i)=qd(i)+qb(i)
   th(i)=(rt(i)+rb(i)+lw(i)+sw(i))*pi(i)
   write(71,'(i0,11(",",z8.8))') i,transfer(rt(i),0),transfer(rb(i),0), &
     transfer(lw(i),0),transfer(sw(i),0),transfer(pi(i),0),transfer(qd(i),0), &
     transfer(qb(i),0),transfer(density(i),0),transfer(th(i),0),transfer(qv(i),0),transfer(dx(i),0)
 enddo
 close(71)
 open(72,file='lifecycle.csv',status='replace')
 write(72,'(a)') 'mode,field,k,i,word'
 do mode=0,3
   a=0.125
   call ntiedtkeinit(a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),a(:,:,:,5),a(:,:,:,6), &
     a(:,:,:,7),a(:,:,:,8),mode==1,merge(0,2,mode==2),merge(0,2,mode==3),1,.true., &
     1,3,1,2,1,4,1,2,1,1,1,3,1,2,1,1,1,3)
   do f=1,8
   do k=1,3
   do i=1,2
     write(72,'(4(i0,","),z8.8)')mode,f,k,i,transfer(a(i,k,1,f),0)
   enddo
   enddo
   enddo
 enddo
 close(72)
 call cu_ntiedtke_finalize(msg,err)
 if(err/=0.or.trim(msg)/='cu_ntiedtke_finalize OK') stop 1
 call cu_ntiedtke_pre_init(msg,err)
 if(err/=0.or.trim(msg)/='') stop 2
 call cu_ntiedtke_pre_finalize(msg,err)
 if(err/=0.or.trim(msg)/='') stop 3
 call cu_ntiedtke_post_init(msg,err)
 if(err/=0.or.trim(msg)/='') stop 4
 call cu_ntiedtke_post_finalize(msg,err)
 if(err/=0.or.trim(msg)/='') stop 5
end program
