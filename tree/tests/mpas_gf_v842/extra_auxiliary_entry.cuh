// Test-only adapters, no scheme arithmetic.
// MPAS module_cu_gf.mpas.F:2400-2499,2502-2984,3176-3228.
__device__ void gf_test_extra(GFWorkspace &ws,int nz,int mode,int ens4,int maxens,
 int choice,int deeps,GFReal *a,GFReal *s,GFReal *e,int *ints){
 int one=1,two=2,four=4,sixteen=16,ensdim=48;GFReal dt=120.f,xl=2500000.f,rv=461.6f,cp=1004.5f;
 GFReal mbdt[3]={120.f,120.f,120.f};char name_deeps[]="deeps",name_deep[]="deep ";
 #define A(n) (a+(n)*4*nz)
 #define S(n) (s+(n)*4)
 #define I(n) (ints+(n)*4)
 #define DIMS &four,&one,&nz,&one,&four,&one,&one,&one,&nz
 if(mode==0)
  cup_env_clev(ws,A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),A(10),
   A(11),A(12),A(13),A(14),S(0),I(0),S(1),&xl,&rv,&cp,DIMS);
 else if(mode==1)
  cup_maximi(ws,A(0),&two,I(1),I(2),I(0),DIMS);
 else
  cup_forcing_ens_3d(ws,S(0),S(1),S(2),S(3),S(5),mbdt,&dt,I(0),I(1),I(2),e,
   &one,deeps?name_deeps:name_deep,S(8),&maxens,&one,&one,&one,&sixteen,S(11),A(0),
   I(3),A(3),A(1),I(4),A(2),e+4*48,S(4),I(5),&ensdim,&choice,&one,&one,DIMS,&ens4,&two,5);
 #undef A
 #undef S
 #undef I
 #undef DIMS
}
#ifdef GF_HOST_WITNESS
extern "C" void gf_host_extra(int nz,int mode,int ens4,int maxens,int choice,int deeps,
 float *a,float *s,float *e,int *ints){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_extra(ws,nz,mode,ens4,maxens,choice,deeps,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
#else
extern "C" __global__ void gf_aux_extra(int nz,int mode,int ens4,int maxens,int choice,int deeps,
 float *a,float *s,float *e,int *ints,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};
 gf_test_extra(ws,nz,mode,ens4,maxens,choice,deeps,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
#endif
