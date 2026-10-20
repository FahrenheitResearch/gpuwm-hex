module cu6_oracle_boundary
  use iso_c_binding, only: c_int, c_float
  use module_cu_tiedtke, only: cu_tiedtke, tiedtkeinit, cuflx, cuadjtq, tlucua, tlucub, tlucuc, c2es, c5les, c5ies, vtmpc1, cevapcu1, cevapcu2, cprcon
  use mpas_atmphys_constants, only: r_d, r_v, cp, g=>gravity
  implicit none
contains
  ! Exact-word sentinel witness for module_cu_tiedtke.F:2329,2339-2341.
  ! The pristine routine is unchanged; a single contiguous argument slab
  ! makes its level-zero writes observable in the preceding argument.
  subroutine cu6_unsafe_flux_values(nc,values,flags) bind(C)
    integer(c_int), value :: nc
    real(c_float), intent(out) :: values(nc*120)
    integer(c_int), intent(out) :: flags(nc*3+1)
    real :: a(nc,5,24),sig(nc,4)
    integer :: bot(nc),top(nc),ddtop(nc),ktype(nc),ktopm2,i,k,v
    logical :: dd(nc),active(nc)
    do v=1,24
      do k=1,5
        do i=1,nc
          a(i,k,v)=real(100*v+10*k+i)
        enddo
      enddo
    enddo
    sig=0.5;bot=4;top=3;ddtop=4;ktype=0;dd=.false.;active=.false.
    call cuflx(nc,4,5,a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6), &
      bot,top,ddtop,ktype,dd,active,a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10), &
      a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15),a(:,:,16), &
      a(:,1,17),a(:,1,18),a(:,:,19),a(:,1,20),a(:,:,21),ktopm2,60.,sig)
    values=reshape(a,(/nc*120/))
    flags=(/merge(1,0,dd),merge(1,0,active),ktype,ktopm2/)
  end subroutine
  subroutine cu6_subnormal_values(n,input,output) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: input(n)
    real(c_float), intent(out) :: output(n,3)
    integer :: i
    do i=1,n
      output(i,1)=input(i)*2.
      output(i,2)=input(i)+input(i)
      output(i,3)=input(i)*0.5
    enddo
  end subroutine
  ! Exact equilibrium reaches module_cu_tiedtke.F:2864-2867 without
  ! manufacturing saturation in Python or reading expected outputs.
  subroutine cu6_equilibrium(n,p,t,q) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: p(n),t(n)
    real(c_float), intent(out) :: q(n)
    real :: sat,cor
    integer :: i
    do i=1,n
      sat=min(0.5,tlucua(t(i))*(1./p(i)))
      cor=1./(1.-vtmpc1*sat)
      q(i)=sat*cor
    enddo
  end subroutine
  ! MPAS module_cu_tiedtke.F:2749-2904; include the uncalled kcall=4 arm.
  subroutine cu6_adjust(n,kcall,p,t,q,flags) bind(C)
    integer(c_int), value :: n,kcall
    real(c_float), intent(in) :: p(n)
    real(c_float), intent(inout) :: t(n),q(n)
    integer(c_int), intent(in) :: flags(n)
    real :: pp(1),tt(1,1),qq(1,1)
    logical :: active(1)
    integer :: i
    do i=1,n
      pp(1)=p(i);tt(1,1)=t(i);qq(1,1)=q(i);active(1)=flags(i)/=0
      call cuadjtq(1,1,1,pp,tt,qq,active,kcall)
      t(i)=tt(1,1);q(i)=qq(1,1)
    enddo
  end subroutine
  subroutine cu6_unsafe_flux() bind(C)
    real :: a(1,5,24),sig(1,4)
    integer :: bot(1),top(1),ddtop(1),ktype(1),ktopm2
    logical :: dd(1),active(1)
    a=0.;sig=0.5;bot=4;top=3;ddtop=4;ktype=0;dd=.false.;active=.false.
    call cuflx(1,4,5,a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6), &
      bot,top,ddtop,ktype,dd,active,a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10), &
      a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15),a(:,:,16), &
      a(:,1,17),a(:,1,18),a(:,:,19),a(:,1,20),a(:,:,21),ktopm2,60.,sig)
  end subroutine
  ! Valid public routine probe for the downdraft-above-cloud rejection,
  ! module_cu_tiedtke.F:2310. The second ascent can change cloud top after
  ! cudlfs has chosen downdraft top; this arm is absent from current dumps.
  subroutine cu6_flux_probe(mode,values,flags) bind(C)
    integer(c_int), value :: mode
    real(c_float), intent(out) :: values(216)
    integer(c_int), intent(out) :: flags(4)
    real :: a(1,9,24),sig(1,8)
    integer :: bot(1),top(1),ddtop(1),ktype(1),ktopm2,k
    logical :: dd(1),active(1)
    a=0.;sig=0.5;bot=7;top=5;ddtop=4+2*mode;ktype=1;dd=.true.;active=.true.
    do k=1,9
      a(1,k,5)=10000.+10000.*real(k)
    enddo
    a(:,:,1)=0.005;a(:,:,2)=0.010;a(:,:,3)=280.;a(:,:,4)=0.005
    a(:,:,19)=280.
    call cuflx(1,8,9,a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6), &
      bot,top,ddtop,ktype,dd,active,a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10), &
      a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15),a(:,:,16), &
      a(:,1,17),a(:,1,18),a(:,:,19),a(:,1,20),a(:,:,21),ktopm2,60.,sig)
    values=reshape(a,(/216/))
    flags=(/merge(1,0,dd(1)),merge(1,0,active(1)),ktype(1),ktopm2/)
  end subroutine
  ! Empty j tile exercises compiler-generated optional-array descriptor
  ! presence edges at module_cu_tiedtke.F:475 without dereferencing absent
  ! tendency arguments. Nonempty native tiles require four tendencies.
  subroutine cu6_empty_tile(values,flag_value) bind(C)
    real(c_float), intent(inout) :: values(3)
    integer(c_int), intent(inout) :: flag_value
    real :: a(1,5,1),sig(1,5),rain(1,1),prate(1,1),flux(1,1),land(1,1)
    logical :: flag(1,1)
    a=0.;sig=0.5;land=1.
    rain=values(1);prate=values(2);flux=values(3);flag=flag_value/=0
    call cu_tiedtke(60.,0,1,rain,prate,flux,sig, &
      a,a,a,a,a,a,a,a,a,a,a,a,a,a,land,flag, &
      1,2,1,2,1,5,1,1,1,1,1,5,1,1,2,1,1,4, &
      f_qc=.false.,f_qi=.false.)
    values=(/rain(1,1),prate(1,1),flux(1,1)/)
    flag_value=merge(1,0,flag(1,1))
  end subroutine
  subroutine cu6_constants(values) bind(C)
    real(c_float), intent(out) :: values(7)
    values=(/c2es,c5les,c5ies,vtmpc1,cevapcu1,cevapcu2,cprcon/)
  end subroutine
  subroutine cu6_thermo(n,t,a,b,d) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: t(n)
    real(c_float), intent(out) :: a(n),b(n),d(n)
    integer :: k
    do k=1,n
      a(k)=tlucua(t(k)); b(k)=tlucub(t(k)); d(k)=tlucuc(t(k))
    enddo
  end subroutine
  subroutine cu6_tan(n,t,a) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: t(n)
    real(c_float), intent(out) :: a(n)
    integer :: k
    do k=1,n
      a(k)=tan(t(k))
    enddo
  end subroutine
  subroutine cu6_oracle(nc,nz,dt,itimestep,stepcu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu, &
      rt,rqv,rqc,rqi,ru,rv,rain,prate,active,fqc,fqi) bind(C)
    integer(c_int), value :: nc,nz,itimestep,stepcu,fqc,fqi
    real(c_float), value :: dt
    real(c_float), intent(in) :: u(nc,nz+1,1),v(nc,nz+1,1),w(nc,nz+1,1)
    real(c_float), intent(in) :: t(nc,nz+1,1),qv(nc,nz+1,1),qc(nc,nz+1,1),qi(nc,nz+1,1)
    real(c_float), intent(in) :: pi(nc,nz+1,1),rho(nc,nz+1,1)
    real(c_float), intent(in) :: qvf(nc,nz+1,1),qvbl(nc,nz+1,1),dz(nc,nz+1,1)
    real(c_float), intent(in) :: p(nc,nz+1,1),p8(nc,nz+1,1),xland(nc,1),qfx(nc,1),znu(nc,nz+1)
    real(c_float), intent(inout) :: rt(nc,nz+1,1),rqv(nc,nz+1,1),rqc(nc,nz+1,1)
    real(c_float), intent(inout) :: rqi(nc,nz+1,1),ru(nc,nz+1,1),rv(nc,nz+1,1)
    real(c_float), intent(inout) :: rain(nc,1),prate(nc,1)
    integer(c_int), intent(out) :: active(nc,1)
    logical :: flag(nc,1)
    flag=.false.
    if (kind(1.0) /= c_float) error stop 'cu6 requires default REAL32'
    if(fqc<0.and.fqi<0) then
      ! Whole-module optional absence, module_cu_tiedtke.F:450-472.
      ! MPAS's real driver always supplies these two optional outputs.
      call cu_tiedtke(dt,itimestep,stepcu,rain,prate,qfx,znu, &
        u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,flag, &
        1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz, &
        rthcuten=rt,rqvcuten=rqv,rucuten=ru,rvcuten=rv,f_qc=.false.,f_qi=.false.)
      active=merge(1_c_int,0_c_int,flag)
      return
    endif
    call cu_tiedtke(dt,itimestep,stepcu,rain,prate,qfx,znu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,flag, &
      1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz, &
      rt,rqv,rqc,rqi,ru,rv,fqc/=0,fqi/=0)
    active=merge(1_c_int,0_c_int,flag)
  end subroutine
  subroutine cu6_driver(nc,nz,dt,itimestep,stepcu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu, &
      rt,rqv,rqc,rqi,ru,rv,rain,prate,active,fqc,fqi,do_restart) bind(C)
    integer(c_int), value :: nc,nz,itimestep,stepcu,fqc,fqi,do_restart
    real(c_float), value :: dt
    real(c_float), intent(in) :: u(nc,nz+1,1),v(nc,nz+1,1),w(nc,nz+1,1)
    real(c_float), intent(in) :: t(nc,nz+1,1),qv(nc,nz+1,1),qc(nc,nz+1,1),qi(nc,nz+1,1)
    real(c_float), intent(in) :: pi(nc,nz+1,1),rho(nc,nz+1,1)
    real(c_float), intent(in) :: qvf(nc,nz+1,1),qvbl(nc,nz+1,1),dz(nc,nz+1,1)
    real(c_float), intent(in) :: p(nc,nz+1,1),p8(nc,nz+1,1),xland(nc,1),qfx(nc,1),znu(nc,nz+1)
    real(c_float), intent(inout) :: rt(nc,nz+1,1),rqv(nc,nz+1,1),rqc(nc,nz+1,1)
    real(c_float), intent(inout) :: rqi(nc,nz+1,1),ru(nc,nz+1,1),rv(nc,nz+1,1)
    real(c_float), intent(inout) :: rain(nc,1),prate(nc,1)
    integer(c_int), intent(out) :: active(nc,1)
    logical :: flag(nc,1)
    integer :: initflag
    flag=.false.
    if (kind(1.0) /= c_float) error stop 'cu6 requires default REAL32'
    ! mpas_atmphys_driver_convection.F:476-478,555-578.
    initflag=1
    if(do_restart/=0.or.itimestep>1) initflag=0
    call cu_tiedtke(dt,initflag,stepcu,rain,prate,qfx,znu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,flag, &
      1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz, &
      rt,rqv,rqc,rqi,ru,rv,fqc/=0,fqi/=0)
    active=merge(1_c_int,0_c_int,flag)
  end subroutine
  subroutine cu6_init(nc,nz,rt,rqv,rqc,rqi,ru,rv,restart,pqc,pqi,pfirst) bind(C)
    ! Whole module initializer; module_cu_tiedtke.F:478-540.
    integer(c_int), value :: nc,nz,restart,pqc,pqi,pfirst
    real(c_float), intent(inout) :: rt(nc,nz+1,1),rqv(nc,nz+1,1),rqc(nc,nz+1,1)
    real(c_float), intent(inout) :: rqi(nc,nz+1,1),ru(nc,nz+1,1),rv(nc,nz+1,1)
    call tiedtkeinit(rt,rqv,rqc,rqi,ru,rv,restart/=0,pqc,pqi,pfirst,.false., &
      1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz)
  end subroutine


  ! Synthetic inputs are made by Fortran and consumed unchanged by both
  ! implementations. These regimes are fixture coverage, not observations.
  subroutine cu6_stress_fixture(nc,nz,mode,u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu) bind(C)
    integer(c_int), value :: nc,nz,mode
    real(c_float), intent(out) :: u(nc,nz+1),v(nc,nz+1),w(nc,nz+1),t(nc,nz+1)
    real(c_float), intent(out) :: qv(nc,nz+1),qc(nc,nz+1),qi(nc,nz+1),pi(nc,nz+1),rho(nc,nz+1)
    real(c_float), intent(out) :: qvf(nc,nz+1),qvbl(nc,nz+1),dz(nc,nz+1),p(nc,nz+1),p8(nc,nz+1)
    real(c_float), intent(out) :: xland(nc),qfx(nc),znu(nc,nz+1)
    integer :: c,k,regimes(nc),code
    real :: h,dzz,ts,lapse,rh,sat,forcing,flux,omega,inversion
    call cu6_fixture(nc,nz,u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu,regimes,0)
    dzz=16000./real(nz)
    do c=1,nc
      code=mode*nc+c-1
      ts=286.+real(mod(code,5))*5.
      lapse=0.004+real(mod(code/5,5))*0.0015
      rh=0.35+real(mod(code/25,5))*0.15
      forcing=10.**real(mod(code/125,5)-9)
      flux=forcing*real(mod(code/625,4))*4000.
      omega=real(mod(code/2500,5))*0.5
      inversion=real(mod(code/12500,4))*6.
      xland(c)=1.+real(mod(code,2));qfx(c)=flux
      p8(c,1)=100000.
      do k=1,nz
        h=(real(k)-0.5)*dzz
        t(c,k)=max(200.,ts-lapse*min(h,10000.))
        if(h>10000.) t(c,k)=t(c,k)+(h-10000.)*0.004
        if(h>1500..and.h<8000.) t(c,k)=t(c,k)+inversion
        p(c,k)=p8(c,k)*exp(-g*dzz*0.5/(r_d*t(c,k)))
        sat=tlucua(t(c,k))/p(c,k)
        sat=sat/(1.-(r_v/r_d-1.)*sat)
        qv(c,k)=rh*sat/(1.-sat)
        if(h>9000.) qv(c,k)=qv(c,k)*0.1
        p8(c,k+1)=p8(c,k)*exp(-g*dzz/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k))))
        p(c,k)=0.5*(p8(c,k)+p8(c,k+1))
        pi(c,k)=(p(c,k)/100000.)**(r_d/cp)
        rho(c,k)=p(c,k)/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k)))
        w(c,k)=omega*sin(3.14159265*h/16000.)
        qvf(c,k)=forcing;qvbl(c,k)=0.2*forcing
        if(h>3000.) then
          qvf(c,k)=0.;qvbl(c,k)=0.
        endif
      enddo
    enddo
  end subroutine

  subroutine cu6_fixture(nc,nz,u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu,regime,unsafe_top) bind(C)
    integer(c_int), value :: nc,nz,unsafe_top
    real(c_float), intent(out) :: u(nc,nz+1),v(nc,nz+1),w(nc,nz+1),t(nc,nz+1)
    real(c_float), intent(out) :: qv(nc,nz+1),qc(nc,nz+1),qi(nc,nz+1),pi(nc,nz+1),rho(nc,nz+1)
    real(c_float), intent(out) :: qvf(nc,nz+1),qvbl(nc,nz+1),dz(nc,nz+1),p(nc,nz+1),p8(nc,nz+1)
    real(c_float), intent(out) :: xland(nc),qfx(nc),znu(nc,nz+1)
    integer(c_int), intent(out) :: regime(nc)
    integer :: c,k,r,variant
    real :: ts,lapse,ps,qs,h,zh,dzz,variation,saturation,rh
    u=0.;v=0.;w=0.;t=0.;qv=0.;qc=0.;qi=0.;pi=1.;rho=1.
    qvf=0.;qvbl=0.;dz=0.;p=0.;p8=0.;xland=1.;qfx=0.
    dzz=16000./real(nz)
    do k=1,nz+1
      do c=1,nc
        znu(c,k)=(1.-(real(k)-0.5)/real(nz))*(0.8+0.015*real(mod(c-1,12)))
      enddo
    end do
    do c=1,nc
      r=1+mod((c-1)/12,6)
      variant=mod(c-1,12)
      variation=0.7+0.06*real(variant)
      regime(c)=r
      ts=304.;lapse=0.007;ps=100000.;qs=0.020
      select case(r)
      case(2)
        ts=299.;lapse=0.007;qs=0.016
      case(3)
        ts=286.;lapse=0.003;qs=0.003
      case(4)
        ts=303.;lapse=0.0065;qs=0.023;xland(c)=2.
      case(5)
        ts=294.;lapse=0.0075;ps=68000.;qs=0.014
      case(6)
        ts=300.;lapse=0.0065;qs=0.018
      end select
      ts=ts+(real(variant)-5.)*0.25
      qfx(c)=0.00008*variation
      if(r==3) qfx(c)=0.
      do k=1,nz+1
        h=real(k-1)*dzz
        p8(c,k)=ps*exp(-h/7800.)
        w(c,k)=0.05*variation*sin(3.14159265*h/16000.)
      end do
      do k=1,nz
        zh=(real(k)-0.5)*dzz
        t(c,k)=max(205.,ts-lapse*min(zh,10000.))
        if(zh>10000.) t(c,k)=t(c,k)+(zh-10000.)*0.003
        if(unsafe_top/=0) t(c,k)=max(205.,ts-lapse*zh)
        if(r==2.and.zh>2000.) t(c,k)=t(c,k)+min(12.,(zh-2000.)*0.01)
        p(c,k)=0.5*(p8(c,k)+p8(c,k+1))
        rh=0.75+0.01*real(variant)
        if(zh>10000.) rh=0.20
        if(unsafe_top/=0) rh=0.75+0.01*real(variant)
        if(r==2.and.zh>2000.) rh=0.40
        if(r==3) rh=0.35
        saturation=tlucua(t(c,k))/p(c,k)
        saturation=saturation/(1.-(r_v/r_d-1.)*saturation)
        qv(c,k)=rh*saturation/(1.-saturation)
        p8(c,k+1)=p8(c,k)*exp(-g*dzz/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k))))
        p(c,k)=0.5*(p8(c,k)+p8(c,k+1))
        pi(c,k)=(p(c,k)/100000.)**(r_d/cp)
        rho(c,k)=p(c,k)/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k)))
        dz(c,k)=dzz
        u(c,k)=5.+0.001*zh
        v(c,k)=-2.+0.0003*zh
        if(zh<3000.) then
          qvf(c,k)=1.e-7*variation
          qvbl(c,k)=2.e-8*variation
        endif
        if(r==3) then
          qvf(c,k)=0.;qvbl(c,k)=0.;w(c,k)=0.
        endif
        if(r==6) then
          select case(variant)
          case(0)
            qv(c,k)=0.;qvf(c,k)=0.;qvbl(c,k)=0.;qfx(c)=0.
          case(1)
            w(c,k)=0.;qvf(c,k)=0.;qvbl(c,k)=0.;qfx(c)=0.
          case(2)
            qc(c,k)=1.e-4
          case(3)
            qi(c,k)=1.e-5
          case(4)
            qvf(c,k)=-abs(qvf(c,k))
          case(5)
            xland(c)=2.;qfx(c)=0.
          case(6)
            t(c,k)=273.16
          case(7)
            t(c,k)=233.15
          case(8)
            qv(c,k)=0.03*exp(-zh/3000.)
          case(9)
            w(c,k)=-abs(w(c,k))
          case(10)
            qfx(c)=-0.00001
          case(11)
            qvf(c,k)=qvf(c,k)*5.
          end select
        endif
      end do
    end do
  end subroutine
end module
