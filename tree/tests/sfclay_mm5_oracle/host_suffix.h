extern "C" void host_tables(float *t) {
for(threadIdx.x=0;threadIdx.x<=1000;threadIdx.x++) mm5_init_tables(t);
}
extern "C" void host_run(float *a, const float *c, const float *t, int n, int r, int iff, int ift, int iz, int scm, int shallow) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_columns(a,c,t,n,r,iff,ift,iz,scm,shallow);
}
extern "C" void host_driver_from(float *a, float *sea, const float *md, const float *sst, const float *ice, int n, float dx, float threshold, int frac) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_from_mpas(a,sea,md,sst,ice,n,dx,threshold,frac);
}
extern "C" void host_driver_to(float *a, const float *sea, const float *held, const float *ice, int n, float threshold) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_to_mpas(a,sea,ice,held,n,threshold);
}
extern "C" void host_diagnostics(float *a, const float *c, const float *ice, const float *q, const float *t, int n, int lsm, float threshold) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_sfc_diagnostics(a,ice,q,t,c,n,lsm,threshold);
}
extern "C" void host_wrf_diagnostics(float *a, const float *c, int n, int ua) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_wrf_sfcdiags(a,c,n,ua);
}
extern "C" void host_run_optional(float *a, const float *c, const float *t, int n, int r, int iff, int ift, int iz, int scm, int shallow, unsigned int mask) {
for(threadIdx.x=0;threadIdx.x<n;threadIdx.x++) mm5_columns(a,c,t,n,r,iff,ift,iz,scm,shallow,mask);
}
