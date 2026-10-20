// Test-only adapters; no scheme arithmetic.
// MPAS module_cu_gf.mpas.F:3353-3473,4060-4121.
__device__ void gf_test_init(GFWorkspace &ws,int nz,int scheme,int restart_i,
 int qc,int qi,GFReal *a,GFReal *s){
 int one=1,two=2,three=3,kde=nz+1;
 bool restart=restart_i!=0,read=false;GFReal cp=1004.5f;
 #define A(n) (a+(n)*4*nz)
 #define S(n) (s+(n)*4)
 #define DIMS &one,&three,&one,&three,&one,&kde,&one,&two,&one,&two,&one,&nz,&one,&two,&one,&two,&one,&nz
 if(scheme==0)
 g3init(ws,A(0),A(1),A(2),A(3),S(0),&cp,&restart,&qc,&qi,&two,A(4),A(5),
  S(1),S(2),S(3),S(4),S(5),S(6),S(7),S(8),A(6),A(7),A(8),A(9),A(10),&read,DIMS);
 else
 gdinit(ws,A(0),A(1),A(2),A(3),S(0),&cp,&restart,&qc,&qi,&two,A(4),A(5),
  S(1),S(2),S(3),S(4),S(5),S(6),S(7),S(8),&read,DIMS);
 #undef A
 #undef S
 #undef DIMS
}
// MPAS module_cu_gf.mpas.F:3106-3173.
__device__ void gf_test_ktop(GFWorkspace &ws,int nz,int ilo,GFReal *dby,
 int *kbcon,int *ktop,int *ierr,GFText *text){
 int one=1,four=4;
 cup_ktop(ws,text,&ilo,dby,kbcon,ktop,ierr,&four,&one,&nz,
  &one,&four,&one,&one,&one,&nz,50);
}
// MPAS module_cu_gf.mpas.F:2265-2397.
__device__ void gf_test_env(GFWorkspace &ws,int nz,int itest,GFReal *a,GFReal *s,int *ierr){
 int one=1,four=4;GFReal crit=258.f,xl=2500000.f,cp=1004.5f;
 cup_env(ws,a,a+4*nz,a+8*nz,a+12*nz,a+16*nz,a+20*nz,a+24*nz,
 s+4,s,ierr,&crit,&itest,&xl,&cp,&four,&one,&nz,&one,&four,&one,&one,&one,&nz);
}
#ifdef GF_HOST_WITNESS
// MPAS module_cu_gf.mpas.F:1989-2124.
extern "C" void gf_host_edt(int nz,int aero,float *a,float *s,float *e,int *ints);
// MPAS module_cu_gf.mpas.F:2987-3103.
extern "C" void gf_host_kbcon(int nz,int iloop,int excess,float *a,float *s,int *ints,char *text);
extern "C" void gf_host_init(int nz,int scheme,int restart,int qc,int qi,float *a,float *s){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_init(ws,nz,scheme,restart,qc,qi,(GFReal*)a,(GFReal*)s);
}
extern "C" void gf_host_ktop(int nz,int ilo,float *dby,int *kbcon,int *ktop,int *ierr,char *text){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_ktop(ws,nz,ilo,(GFReal*)dby,kbcon,ktop,ierr,(GFText*)text);
}
extern "C" void gf_host_env(int nz,int itest,float *a,float *s,int *ierr){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_env(ws,nz,itest,(GFReal*)a,(GFReal*)s,ierr);
}
#else
extern "C" __global__ void gf_aux_init(int nz,int scheme,int restart,int qc,int qi,
 float *a,float *s,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_init(ws,nz,scheme,restart,qc,qi,(GFReal*)a,(GFReal*)s);
}
extern "C" __global__ void gf_aux_ktop(int nz,int ilo,float *dby,int *kbcon,int *ktop,
 int *ierr,char *text,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_ktop(ws,nz,ilo,(GFReal*)dby,kbcon,ktop,ierr,(GFText*)text);
}
extern "C" __global__ void gf_aux_env(int nz,int itest,float *a,float *s,int *ierr,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_env(ws,nz,itest,(GFReal*)a,(GFReal*)s,ierr);
}
#endif

// MPAS module_cu_gf.mpas.F:3788-4058.
__device__ void gf_test_up(GFWorkspace &ws,int nz,int shallow,int excess,int autoconv,GFReal *a,GFReal *s,int *ints){
 int one=1,four=4,zero=0;GFReal xl=2500000.f,clean=100.f;
 char deep[]="deep   ",shal[]="shallow";
 #define A(n) (a+(n)*4*nz)
 cup_up_moisture(ws,shallow?shal:deep,ints,A(4),A(0),A(1),A(2),s,&clean,A(5),
 ints+4,ints+8,A(6),A(7),A(3),A(8),A(9),A(10),A(11),A(12),ints+12,A(13),&xl,
 s+12,&excess,s+16,A(14),A(15),A(16),s+4,s+8,&autoconv,&one,&zero,
 &four,&one,&nz,&one,&zero,&zero,&one,&four,&one,&one,&one,&nz,7);
 #undef A
}
// MPAS module_cu_gf.mpas.F:2127-2263.
__device__ void gf_test_down(GFWorkspace &ws,int nz,int iloop,GFReal *a,GFReal *s,int *ints,GFText *text){
 int one=1,four=4;GFReal xl=2500000.f;
 #define A(n) (a+(n)*4*nz)
 cup_dd_moisture_new(ws,text,A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),
 ints,ints+4,A(10),s,s+4,A(11),A(12),A(13),A(14),&iloop,&xl,&four,&one,&nz,
 &one,&four,&one,&one,&one,&nz,50);
 #undef A
}
#ifdef GF_HOST_WITNESS
extern "C" void gf_host_up(int nz,int shallow,int excess,int autoconv,float *a,float *s,int *ints){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_up(ws,nz,shallow,excess,autoconv,(GFReal*)a,(GFReal*)s,ints);
}
extern "C" void gf_host_down(int nz,int iloop,float *a,float *s,int *ints,char *text){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_down(ws,nz,iloop,(GFReal*)a,(GFReal*)s,ints,(GFText*)text);
}
#else
extern "C" __global__ void gf_aux_up(int nz,int shallow,int excess,int autoconv,float *a,float *s,int *ints,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_up(ws,nz,shallow,excess,autoconv,(GFReal*)a,(GFReal*)s,ints);
}
extern "C" __global__ void gf_aux_down(int nz,int iloop,float *a,float *s,int *ints,char *text,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_down(ws,nz,iloop,(GFReal*)a,(GFReal*)s,ints,(GFText*)text);
}
#endif

// MPAS module_cu_gf.mpas.F:1989-2124.
__device__ void gf_test_edt(GFWorkspace &ws,int nz,int aero,GFReal *a,GFReal *s,GFReal *e,int *ints){
 int one=1,four=4,three=3,zero=0;GFReal emax=.95f,emin=.01f,clean=100.f;
 cup_dd_edt(ws,ints,a,a+4*nz,a+8*nz,ints+4,ints+8,s,a+12*nz,s+4,a+16*nz,
 s+12,s+8,&emax,&emin,&three,e,s+16,s+20,&clean,a+20*nz,&aero,
 &four,&one,&nz,&one,&zero,&zero,&one,&four,&one,&one,&one,&nz);
}
// MPAS module_cu_gf.mpas.F:2987-3103.
__device__ void gf_test_kbcon(GFWorkspace &ws,int nz,int iloop,int excess,GFReal *a,GFReal *s,int *ints,GFText *text){
 int one=1,four=4;GFReal xl=2500000.f,cp=1004.5f;
 cup_kbcon(ws,text,s+8,&iloop,ints,ints+4,a,a+4*nz,s,ints+12,ints+8,a+8*nz,
 s+4,&xl,&cp,s+16,s+12,&excess,&four,&one,&nz,&one,&four,&one,&one,&one,&nz,50);
}
#ifdef GF_HOST_WITNESS
extern "C" void gf_host_edt(int nz,int aero,float *a,float *s,float *e,int *ints){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_edt(ws,nz,aero,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
extern "C" void gf_host_kbcon(int nz,int iloop,int excess,float *a,float *s,int *ints,char *text){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_kbcon(ws,nz,iloop,excess,(GFReal*)a,(GFReal*)s,ints,(GFText*)text);
}
#else
extern "C" __global__ void gf_aux_edt(int nz,int aero,float *a,float *s,float *e,int *ints,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_edt(ws,nz,aero,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
extern "C" __global__ void gf_aux_kbcon(int nz,int iloop,int excess,float *a,float *s,int *ints,char *text,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_kbcon(ws,nz,iloop,excess,(GFReal*)a,(GFReal*)s,ints,(GFText*)text);
}
#endif
