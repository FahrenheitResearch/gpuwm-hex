// Philox4x32-10: Random123 BSD notice in licenses/LICENSE-Random123.txt.
__device__ __forceinline__ uint4 stoch_philox(uint4 c, unsigned long long seed) {
    unsigned int k0 = (unsigned int)seed;
    unsigned int k1 = (unsigned int)(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = 0xD2511F53ULL * c.x;
        unsigned long long p1 = 0xCD9E8D57ULL * c.z;
        c = make_uint4((unsigned int)(p1 >> 32) ^ c.y ^ k0,
                       (unsigned int)p1,
                       (unsigned int)(p0 >> 32) ^ c.w ^ k1,
                       (unsigned int)p0);
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
    return c;
}


extern "C" __global__ void harmonics(const double *lat, const double *lon,
                                     float *basis, int points, int degree) {
    long long i = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= (long long)points*(degree+1)) return;
    int point=i/(degree+1), m=i%(degree+1), modes=(degree+1)*(degree+1)-1;
    // A valid basis can exceed 2^31 float words (8 GiB). Widen before
    // multiplying, so high rows never wrap into negative addresses.
    long long row=(long long)point*modes;
    double x=sin(lat[point]), s=sqrt(fmax(0.0,1-x*x));
    double diagonal=0.28209479177387814347;
    for (int j=1; j<=m; ++j) diagonal *= -sqrt((2.0*j+1)/(2.0*j))*s;
    double a=0, b=diagonal, cs=cos(m*lon[point]), sn=sin(m*lon[point]);
    for (int l=m; l<=degree; ++l) {
        double p;
        if (l==m) p=diagonal;
        else if (l==m+1) p=sqrt(2.0*m+3)*x*b;
        else p=sqrt((4.0*l*l-1)/(1.0*l*l-m*m))*x*b
                  -sqrt(((2.0*l+1)*((l-1.0)*(l-1)-m*m))/((2.0*l-3)*(1.0*l*l-m*m)))*a;
        if (l>0) {
            int center=l*l+l-1;
            if (m==0) basis[row+center]=(float)p;
            else {
                basis[row+center+m]=(float)(1.4142135623730951*p*cs);
                basis[row+center-m]=(float)(1.4142135623730951*p*sn);
            }
        }
        a=b; b=p;
    }
}
extern "C" __global__ void ar1(float *coeff, const float *weights, int n,
                               float phi, float innovation,
                               unsigned long long seed, unsigned long long draw) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=n) return;
    uint4 words=stoch_philox(make_uint4(i/3,(unsigned int)draw,i%3,(unsigned int)(draw>>32)),seed);
    double u=((double)words.x+0.5)*0x1p-32;
    double v=((double)words.y+0.5)*0x1p-32;
    float noise=(float)(sqrt(-2*log(u))*cos(6.2831853071795864769*v));
    coeff[i]=phi*coeff[i]+innovation*weights[i/3]*noise;
}

__device__ __forceinline__ float sppt_factor(float pattern, float taper,
                                            float clip, int logit) {
    if (logit) {
        float factor=-tanhf(pattern*taper*0.5f);
        return isnan(factor) ? factor : fminf(clip, fmaxf(-clip, factor));
    }
    if (isnan(pattern)) return pattern;
    return fminf(clip, fmaxf(-clip, pattern))*taper;
}

// Tendencies may have padding or strided level/cell layouts. Strides are in
// float words, not bytes. Spectral and mesh workspaces are contiguous.
extern "C" __global__ void force_cells(
    float *theta, float *qv, const float *rho, const float *taper,
    const float *pattern, const float *thermal, const float *dissipation,
    long long ts0, long long ts1, long long qs0, long long qs1,
    long long rs0, long long rs1, int cells, int levels,
    int sppt, int skeb, int has_qv, int logit, float clip, float thermal_scale) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=cells*levels) return;
    int k=i/cells, c=i%cells;
    long long t=k*ts0+c*ts1;
    float value=theta[t];
    if (sppt) {
        float multiplier=1.0f+sppt_factor(pattern[c],taper[i],clip,logit);
        value*=multiplier;
        if (has_qv) qv[k*qs0+c*qs1]*=multiplier;
    }
    if (skeb) value+=rho[k*rs0+c*rs1]*taper[i]*thermal[c]
                    *sqrtf(thermal_scale*dissipation[k]);
    theta[t]=value;
}

extern "C" __global__ void force_edges(
    float *momentum, const float *rho, const float *taper,
    const float *vertex_taper, const float *pattern, const float *psi,
    const float *dissipation, const float *norm, const int *coe, const int *voe,
    const float *orientation, const float *dv,
    long long ms0, long long ms1, long long rs0, long long rs1,
    int cells, int vertices, int edges, int levels,
    int sppt, int skeb, int logit, float clip, float kinetic_scale) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=edges*levels) return;
    int k=i/edges, e=i%edges;
    long long m=k*ms0+e*ms1;
    float value=momentum[m];
    if (sppt) {
        int a=coe[2*e], b=coe[2*e+1];
        float fa=sppt_factor(pattern[a],taper[k*cells+a],clip,logit);
        float fb=sppt_factor(pattern[b],taper[k*cells+b],clip,logit);
        value*=1.0f+0.5f*(fa+fb);
    }
    if (skeb) {
        int a=voe[2*e], b=voe[2*e+1];
        float rotational=orientation[e]*(vertex_taper[k*vertices+b]*psi[b]
                         -vertex_taper[k*vertices+a]*psi[a])/dv[e];
        float denominator=norm[0];
        if (denominator<1e-30f) denominator=1e-30f;
        rotational/=denominator;
        value+=rho[k*rs0+e*rs1]*rotational*sqrtf(kinetic_scale*dissipation[k]);
    }
    momentum[m]=value;
}
