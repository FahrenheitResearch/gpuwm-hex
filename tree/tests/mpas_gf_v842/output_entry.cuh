// Test-only argument adapter. MPAS module_cu_gf.mpas.F:3580-3786.
__device__ void gf_test_output(GFWorkspace &ws,int nz,GFReal *a,GFReal *s,GFReal *e,int *ints){
 int one=1,four=4,sixteen=16,zero=0;char name[]="deep";
 #define A(n) (a+(n)*4*nz)
 #define S(n) (s+(n)*4)
 #define I(n) (ints+(n)*4)
 cup_output_ens_3d(ws,e,I(0),A(0),A(1),A(2),A(3),A(4),A(5),A(6),A(7),A(8),A(9),
  A(10),A(11),S(0),A(12),S(1),I(1),&one,name,&one,&one,&one,I(2),I(3),e+64,
  &sixteen,&sixteen,S(2),S(3),S(4),S(5),S(6),S(7),S(8),S(9),S(10),S(11),S(12),
  S(13),S(14),S(15),S(16),S(17),&zero,&one,&one,&four,&one,&nz,&one,&four,&one,&one,&one,&nz,4);
 #undef A
 #undef S
 #undef I
}
#ifdef GF_HOST_WITNESS
extern "C" void gf_host_output(int nz,float *a,float *s,float *e,int *ints){
 char arena[65536];GFWorkspace ws{arena,sizeof(arena),0,0,0};
 gf_test_output(ws,nz,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
#else
extern "C" __global__ void gf_aux_output(int nz,float *a,float *s,float *e,int *ints,char *arena){
 GFWorkspace ws{arena,65536,0,0,0};gf_test_output(ws,nz,(GFReal*)a,(GFReal*)s,(GFReal*)e,ints);
}
#endif
