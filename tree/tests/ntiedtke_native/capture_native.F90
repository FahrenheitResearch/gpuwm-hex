! MPAS module_cu_ntiedtke.F:20-276; strict pristine full captured slab.
program capture_native
 use ccpp_kind_types, only: kind_phys
 use module_cu_ntiedtke, only: cu_ntiedtke_driver
 use cu_ntiedtke_common, only: alf,als,alv,c2es,c5ies,c5les,cpd,g,r5alscp,r5alvcp,ralfdcp,ralsdcp,ralvdcp,rcpd,rd,rv,vtmpc1,zrg
 implicit none
 integer :: nc,nz,unit,stepcu,itimestep,errflg
 logical :: f_qc,f_qi
 real(kind_phys) :: dt,cp_in,rd_in,rv_in,xlv_in,xls_in,xlf_in,grav_in
 character(512) :: errmsg
 character(4096) :: root,output
 real(kind_phys), allocatable :: pcps(:,:,:),p8w(:,:,:),t3d(:,:,:),dz8w(:,:,:),u3d(:,:,:),v3d(:,:,:),w(:,:,:),qv3d(:,:,:),qc3d(:,:,:),qi3d(:,:,:),pi3d(:,:,:),rho3d(:,:,:),qvften(:,:,:),thften(:,:,:),rthcuten(:,:,:),rqvcuten(:,:,:),rqccuten(:,:,:),rqicuten(:,:,:),rucuten(:,:,:),rvcuten(:,:,:)
 real(kind_phys), allocatable :: raincv(:,:),pratec(:,:),qfx(:,:),hfx(:,:),xland(:,:),dx(:,:)
 logical, allocatable :: cu_act_flag(:,:)
 call get_command_argument(1,root)
 call get_command_argument(2,output)
 open(newunit=unit,file=trim(root)//'/settings.txt',status='old')
 read(unit,*)nc,nz
 read(unit,*)dt,stepcu,itimestep,f_qc,f_qi
 read(unit,*)cp_in,rd_in,rv_in,xlv_in,xls_in,xlf_in,grav_in
 close(unit)
 ! MPAS module_cu_ntiedtke.F:158-173 only initializes on the first call.
 ! Restore the captured entry module state for subsequent/restarted calls.
 open(newunit=unit,file=trim(root)//'/state.raw',access='stream',form='unformatted',status='old')
 read(unit)alf,als,alv,c2es,c5ies,c5les,cpd,g,r5alscp,r5alvcp,ralfdcp,ralsdcp,ralvdcp,rcpd,rd,rv,vtmpc1,zrg
 close(unit)
 allocate(pcps(nc,nz+1,1))
 allocate(p8w(nc,nz+1,1))
 allocate(t3d(nc,nz+1,1))
 allocate(dz8w(nc,nz+1,1))
 allocate(u3d(nc,nz+1,1))
 allocate(v3d(nc,nz+1,1))
 allocate(w(nc,nz+1,1))
 allocate(qv3d(nc,nz+1,1))
 allocate(qc3d(nc,nz+1,1))
 allocate(qi3d(nc,nz+1,1))
 allocate(pi3d(nc,nz+1,1))
 allocate(rho3d(nc,nz+1,1))
 allocate(qvften(nc,nz+1,1))
 allocate(thften(nc,nz+1,1))
 allocate(rthcuten(nc,nz+1,1))
 allocate(rqvcuten(nc,nz+1,1))
 allocate(rqccuten(nc,nz+1,1))
 allocate(rqicuten(nc,nz+1,1))
 allocate(rucuten(nc,nz+1,1))
 allocate(rvcuten(nc,nz+1,1))
 allocate(raincv(nc,1))
 allocate(pratec(nc,1))
 allocate(qfx(nc,1))
 allocate(hfx(nc,1))
 allocate(xland(nc,1))
 allocate(dx(nc,1))
 allocate(cu_act_flag(nc,1))
 open(newunit=unit,file=trim(root)//'/pcps.raw',access='stream',form='unformatted',status='old')
 read(unit)pcps
 close(unit)
 open(newunit=unit,file=trim(root)//'/p8w.raw',access='stream',form='unformatted',status='old')
 read(unit)p8w
 close(unit)
 open(newunit=unit,file=trim(root)//'/t3d.raw',access='stream',form='unformatted',status='old')
 read(unit)t3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/dz8w.raw',access='stream',form='unformatted',status='old')
 read(unit)dz8w
 close(unit)
 open(newunit=unit,file=trim(root)//'/u3d.raw',access='stream',form='unformatted',status='old')
 read(unit)u3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/v3d.raw',access='stream',form='unformatted',status='old')
 read(unit)v3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/w.raw',access='stream',form='unformatted',status='old')
 read(unit)w
 close(unit)
 open(newunit=unit,file=trim(root)//'/qv3d.raw',access='stream',form='unformatted',status='old')
 read(unit)qv3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/qc3d.raw',access='stream',form='unformatted',status='old')
 read(unit)qc3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/qi3d.raw',access='stream',form='unformatted',status='old')
 read(unit)qi3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/pi3d.raw',access='stream',form='unformatted',status='old')
 read(unit)pi3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/rho3d.raw',access='stream',form='unformatted',status='old')
 read(unit)rho3d
 close(unit)
 open(newunit=unit,file=trim(root)//'/qvften.raw',access='stream',form='unformatted',status='old')
 read(unit)qvften
 close(unit)
 open(newunit=unit,file=trim(root)//'/thften.raw',access='stream',form='unformatted',status='old')
 read(unit)thften
 close(unit)
 open(newunit=unit,file=trim(root)//'/rthcuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rthcuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/rqvcuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rqvcuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/rqccuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rqccuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/rqicuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rqicuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/rucuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rucuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/rvcuten.raw',access='stream',form='unformatted',status='old')
 read(unit)rvcuten
 close(unit)
 open(newunit=unit,file=trim(root)//'/raincv.raw',access='stream',form='unformatted',status='old')
 read(unit)raincv
 close(unit)
 open(newunit=unit,file=trim(root)//'/pratec.raw',access='stream',form='unformatted',status='old')
 read(unit)pratec
 close(unit)
 open(newunit=unit,file=trim(root)//'/qfx.raw',access='stream',form='unformatted',status='old')
 read(unit)qfx
 close(unit)
 open(newunit=unit,file=trim(root)//'/hfx.raw',access='stream',form='unformatted',status='old')
 read(unit)hfx
 close(unit)
 open(newunit=unit,file=trim(root)//'/xland.raw',access='stream',form='unformatted',status='old')
 read(unit)xland
 close(unit)
 open(newunit=unit,file=trim(root)//'/dx.raw',access='stream',form='unformatted',status='old')
 read(unit)dx
 close(unit)
 open(newunit=unit,file=trim(root)//'/cu_act_flag.raw',access='stream',form='unformatted',status='old')
 read(unit)cu_act_flag
 close(unit)
 call cu_ntiedtke_driver(pcps=pcps,p8w=p8w,t3d=t3d,dz8w=dz8w,dt=dt,itimestep=itimestep,stepcu=stepcu, &
 raincv=raincv,pratec=pratec,qfx=qfx,hfx=hfx,xland=xland,dx=dx,u3d=u3d,v3d=v3d,w=w,qv3d=qv3d,qc3d=qc3d,qi3d=qi3d, &
 pi3d=pi3d,rho3d=rho3d,qvften=qvften,thften=thften,cu_act_flag=cu_act_flag,f_qc=f_qc,f_qi=f_qi, &
 rthcuten=rthcuten,rqvcuten=rqvcuten,rqccuten=rqccuten,rqicuten=rqicuten,rucuten=rucuten,rvcuten=rvcuten, &
 grav=grav_in,xlf=xlf_in,xls=xls_in,xlv=xlv_in,rd=rd_in,rv=rv_in,cp=cp_in,errmsg=errmsg,errflg=errflg, &
 ids=1,ide=nc+1,jds=1,jde=2,kds=1,kde=nz+1,ims=1,ime=nc,jms=1,jme=1,kms=1,kme=nz+1, &
 its=1,ite=nc,jts=1,jte=1,kts=1,kte=nz)
 open(newunit=unit,file=trim(output)//'/rthcuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rthcuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/rqvcuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rqvcuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/rqccuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rqccuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/rqicuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rqicuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/rucuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rucuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/rvcuten.raw',access='stream',form='unformatted',status='new')
 write(unit)rvcuten
 close(unit)
 open(newunit=unit,file=trim(output)//'/raincv.raw',access='stream',form='unformatted',status='new')
 write(unit)raincv
 close(unit)
 open(newunit=unit,file=trim(output)//'/pratec.raw',access='stream',form='unformatted',status='new')
 write(unit)pratec
 close(unit)
 open(newunit=unit,file=trim(output)//'/cu_act_flag.raw',access='stream',form='unformatted',status='new')
 write(unit)cu_act_flag
 close(unit)
 open(newunit=unit,file=trim(output)//'/errmsg.raw',access='stream',form='unformatted',status='new')
 write(unit)errmsg
 close(unit)
 open(newunit=unit,file=trim(output)//'/errflg.raw',access='stream',form='unformatted',status='new')
 write(unit)errflg
 close(unit)
 open(newunit=unit,file=trim(output)//'/state.raw',access='stream',form='unformatted',status='new')
 write(unit)alf,als,alv,c2es,c5ies,c5les,cpd,g,r5alscp,r5alvcp,ralfdcp,ralsdcp,ralvdcp,rcpd,rd,rv,vtmpc1,zrg
 close(unit)
end program
