extern "C" void wsm6_host(int nz,float dt,int hail,const float *c,float *v,float *s,float *r,float *dbz,float *prod) {
 int status;wsm6_columns(nz,1,1,dt,hail,c,v,s,r,dbz,prod,dbz,0,1,1,1,&status);
}
extern "C" void wsm6_wrapper_host(int nz,float dt,int hail,const float *c,float *v,float *s,float *r,float *dbz,float *prod,const float *pi,int radii_on) {
 int status;wsm6_columns(nz,1,1,dt,hail,c,v,s,r,dbz,prod,pi,1,radii_on,1,0,&status);
}
// MPAS module_mp_wsm6.F:18-239, mp_wsm6_effectRad.F90:190-191.
extern "C" void wsm6_wrapper_message_host(int nz,float dt,int hail,const float *c,float *v,float *s,float *r,float *dbz,float *prod,const float *pi,int radii_on,int *status,char *message) {
 wsm6_columns(nz,1,1,dt,hail,c,v,s,r,dbz,prod,pi,1,radii_on,0,0,status,message);
}
// MPAS mp_wsm6.F90:73-191; mp_radar.F90:72-190: production snapshot kernel ABI.
extern "C" void wsm6_initialized_state_host(const float *c,int hail,int member,unsigned char *out) {
 wsm6_initialized_state(c,hail,member,out);
}
extern "C" void wsm6_driver_host(int nz,int nc,int ns,int top,float dt,float dt_dyn,int hail,
 const float *c,float **fields,int *counter,int radii_on,int radar_on,float bucket,int acrain,
 float rvord,float rcv,float p0) {
 float *v=new float[10*nz*nc],*s=new float[7*nc],*r=new float[3*nz*nc],*dbz=new float[nz*nc],
 *prod=new float[2*nz*nc],*pi=new float[nz*nc];int *status=new int[nc];
 for(int i=0;i<ns;++i){blockIdx.x=i;wsm6_from_mpas(nz,nc,ns,c[4]/c[3],fields,v,s,r,prod,pi);}
 for(int i=0;i<ns;++i){blockIdx.x=i;wsm6_columns(nz,nc,ns,dt,hail,c,v,s,r,dbz,prod,pi,1,radii_on,radar_on,0,status);}
 if(radar_on)for(int i=0;i<ns;++i){blockIdx.x=i;wsm6_radar_to_mpas(nz,nc,ns,fields,dbz,status);}
 for(int i=0;i<ns;++i){blockIdx.x=i;wsm6_to_mpas(nz,nc,ns,top,dt_dyn,c[3],c[4]/c[3],rvord,rcv,p0,c[0],bucket,acrain,c,fields,counter,v,s,r,prod);}
 blockIdx.x=0;
 delete[] v;delete[] s;delete[] r;delete[] dbz;delete[] prod;delete[] pi;delete[] status;
}
extern "C" void radar_constants_host(double *out) {
 Wsm6State S{};state_parameters(S);int err=0;
 mp_wsm6_init(S,1.28f,1000.f,100.f,4190.f,1846.4f,0,err);
 out[0]=S.pi5;out[1]=S.lamda4;out[2]=S.k_w;out[3]=S.m_w_0.real();out[4]=S.m_w_0.imag();out[5]=S.m_i_0.real();out[6]=S.m_i_0.imag();
 for(int i=0;i<4;++i){out[7+i]=S.xcre[i];out[11+i]=S.xcrg[i];out[15+i]=S.xcse[i];out[19+i]=S.xcsg[i];out[23+i]=S.xcge[i];out[27+i]=S.xcgg[i];}out[31]=S.xam_s;
}
extern "C" void height_guard_host(int nz,int nc,int ns,const float *z,int *status) {
 for(int i=0;i<ns;++i){blockIdx.x=i;wsm6_height_guard(nz,nc,ns,z,status);}blockIdx.x=0;
}

// MPAS-A v8.4.2 mp_radar.F90:192-292: actual device routine, host witness.
extern "C" void radar_branch_host(const double *a,const int *opts,double *out) {
 const char *names[]={"maxwellgarnett","invalid","air","ice","water","spherical","spheroidal","none","icewater","airwater","airice","iceair","waterair"};
 Wsm6State S;state_parameters(S);radar_init(S);
 rayleigh_soak_wetgraupel(S,a[0],a[1],a[2],a[3],a[4],S.m_w_0,S.m_i_0,a[5],*out,
 names[opts[0]-1],names[opts[1]-1],names[opts[2]-1],names[opts[3]-1],names[opts[4]-1],names[opts[5]-1]);
}

// MPAS-A v8.4.2 mp_wsm6_effectRad.F90:22-193, mp_wsm6.F90:191-211.
extern "C" void radii_boundary_host(int nz,int enabled,const float *c,float *v,float *r,int *status) {
 Wsm6State S;state_parameters(S);int err;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],0,err);
 mp_wsm6_effectrad_init(S,err);status[0]=err;
 mp_wsm6_effectrad_run(S,enabled,v,v+2*nz,v+3*nz,v+5*nz,v+7*nz,c[11],c[5],c[20],c[21],c[22],c[23],c[24],c[25],r,r+nz,r+2*nz,1,1,1,nz,err);
 mp_wsm6_effectrad_finalize(S,err);status[1]=err;
 mp_wsm6_finalize(S,err);status[2]=err;
}

// MPAS-A v8.4.2 mp_wsm6.F90:213-1468; optional diagnostic argument coverage.
extern "C" void optional_surface_host(int nz,float dt,int hail,int mask,const float *c,float *v,float *s,float *prod) {
 Wsm6State S;state_parameters(S);int err;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],hail,err);
 mp_wsm6_run(S,v,v+nz,v+2*nz,v+3*nz,v+4*nz,v+5*nz,v+6*nz,v+7*nz,v+8*nz,v+9*nz,dt,
 c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],
 s,s+1,s+6,(mask&1)?s+2:nullptr,(mask&2)?s+3:nullptr,(mask&4)?s+4:nullptr,(mask&8)?s+5:nullptr,
 prod,prod+nz,1,1,1,nz,err);
}

// MPAS module_mp_wsm6.F:214-232: independently present effective-radius flags.
extern "C" void wsm6_wrapper_flags_host(int nz,float dt,int hail,const float *c,float *v,float *s,float *r,float *dbz,float *prod,const float *pi,int cloud,int ice,int snow) {
 wsm6_wrapper_host(nz,dt,hail,c,v,s,r,dbz,prod,pi,cloud==1 && ice==1 && snow==1);
}

// MPAS-A v8.4.2 mp_wsm6.F90:1471-1494,1751-1994,1997-2272.
extern "C" float gamma_host(float x) {
 Wsm6State S{};return rgmma(S,x);
}
extern "C" void sedimentation_host(int nz,int iterations,int category,float dt,const float *c,float *a,float *precip) {
 Wsm6State S{};state_parameters(S);int err;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],0,err);
 if(category==1)nislfv_rain_plm(S,1,nz,a+nz,a+2*nz,a+3*nz,a,a+4*nz,a+5*nz,precip,dt,1,iterations);
 else nislfv_rain_plm6(S,1,nz,a+nz,a+2*nz,a+3*nz,a,a+4*nz,a+5*nz,a+6*nz,precip,precip+1,dt,1,iterations);
}

// MPAS-A v8.4.2 mp_radar.F90:536-589; production numeric/error result view.
extern "C" void maxwell_host(const double *volumes,int inclusion,double *out,int *error) {
 Wsm6State S{};state_parameters(S);radar_init(S);
 const char *name=inclusion==0?"spherical":(inclusion==1?"spheroidal":"invalid");
 cx answer=m_complex_maxwellgarnett(S,volumes[0],volumes[1],volumes[2],S.m_w_0,S.m_i_0,cx(1.,0.),name,*error);
 out[0]=answer.real();out[1]=answer.imag();
}
