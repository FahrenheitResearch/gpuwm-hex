// CPU execution of the unchanged CUDA transcription for a pre-GPU check.
// This shim does not predict GPU compiler behavior or prove GPU parity.
#include <cmath>
#include <cstdint>
#include <cstring>
#define __device__
#define __global__
#define __forceinline__ inline
template<class A,class B> A bits(B b) { A a; static_assert(sizeof(A)==sizeof(B)); std::memcpy(&a,&b,sizeof(a)); return a; }
#define ROUND_OP(name,type,op) inline type name(type a,type b) { volatile type r=a op b; return r; }
ROUND_OP(__fadd_rn,float,+)
ROUND_OP(__fsub_rn,float,-)
ROUND_OP(__fmul_rn,float,*)
ROUND_OP(__fdiv_rn,float,/)
ROUND_OP(__dadd_rn,double,+)
ROUND_OP(__dsub_rn,double,-)
ROUND_OP(__dmul_rn,double,*)
ROUND_OP(__ddiv_rn,double,/)
inline float __double2float_rn(double a) { volatile float r=a; return r; }
inline long long __double2ll_rn(double a) { return std::llrint(a); }
inline float __fmaf_rn(float a,float b,float c) { return std::fma(a,b,c); }
inline double __fma_rn(double a,double b,double c) { return std::fma(a,b,c); }
inline float __fsqrt_rn(float a) { return std::sqrt(a); }
inline double __dsqrt_rn(double a) { return std::sqrt(a); }
inline unsigned __float_as_uint(float a) { return bits<unsigned>(a); }
inline float __uint_as_float(unsigned a) { return bits<float>(a); }
inline float __int_as_float(int a) { return bits<float>(a); }
inline long long __double_as_longlong(double a) { return bits<long long>(a); }
inline double __longlong_as_double(long long a) { return bits<double>(a); }
inline unsigned long long __umul64hi(unsigned long long a,unsigned long long b) { return ((unsigned __int128)a*b)>>64; }
using std::isnan;
using std::isinf;
using std::signbit;
#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "support.cuh"
#ifdef TK_NATIVE_TAN_CONTROL
// Diagnostic only. No GPU or forecast source uses this control.
#define glibc_tanf(x) ::tanf(x)
#define gfk_exp(x) ::expf(x)
#define gfk_log(x) ::logf(x)
#define gfk_pow(x,y) ::powf(x,y)
#endif
#include "routines.cuh"
struct HostIndex { int x; };
static HostIndex blockIdx{0},blockDim{1},threadIdx{0};
extern "C" void cu6_set_column(int c) { threadIdx.x=c; }
#include "driver.cu"
extern "C" void cu6_constants(float* values) {
    float v[]={tk_c2es,tk_c5les,tk_c5ies,tk_vtmpc1,tk_cevapcu1,tk_cevapcu2,tk_cprcon};
    std::memcpy(values,v,sizeof(v));
}
// MPAS module_cu_tiedtke.F:2749-2904, including the uncalled kcall=4 arm.
extern "C" void cu6_adjust(int n,int kcall,const float* p,float* t,float* q,const int* flags) {
    for(int c=0;c<n;++c) {
        float f[16]{};int i[1]{};bool b[1]{};int status=1,activity=0;
        TkArena arena{f,i,b,0,0,0,1,&status,&activity,false};
        auto pp=arena.take<float>(1),tt=arena.take<float>(1),qq=arena.take<float>(1);
        auto active=arena.take<bool>(1);
        pp(1)=p[c];tt(1)=t[c];qq(1)=q[c];active(1)=flags[c]!=0;
        tk_cuadjtq(1,1,1,pp,tt,qq,active,kcall,arena);
        t[c]=tt(1);q[c]=qq(1);
    }
}
extern "C" void cu6_thermo(int n,const float* t,float* a,float* b,float* d) {
    TkArena arena{};
    for(int k=0;k<n;++k) {
        a[k]=tk_tlucua(t[k],arena);
        b[k]=tk_tlucub(t[k],arena);
        d[k]=tk_tlucuc(t[k],arena);
    }
}
extern "C" void cu6_tan(int n,const float* t,float* a) {
    for(int k=0;k<n;++k) a[k]=glibc_tanf(t[k]);
}
extern "C" int cu6_unsafe_flux() {
    float f[4096]{}; int i[64]{}; bool b[64]{};
    int status=1,activity=0;
    TkArena arena{f,i,b,0,0,0,1,&status,&activity,false};
    TkArray<float> a[24];
    for(int k=0;k<24;++k) a[k]=arena.take<float>(5);
    // The actual Fortran level arrays have extent 4, even where their
    // containing storage slab has 5 words to accommodate interface fields.
    for(int k=0;k<24;++k) if(k!=4) a[k].count=4;
    auto bot=arena.take<int>(1),top=arena.take<int>(1),ddtop=arena.take<int>(1),type=arena.take<int>(1);
    bot(1)=4;top(1)=3;ddtop(1)=4;type(1)=0;
    auto dd=arena.take<bool>(1),active=arena.take<bool>(1);
    dd(1)=false;active(1)=false;
    int ktopm2=0;
    tk_cuflx(1,4,5,a[0],a[1],a[2],a[3],a[4],a[5],bot,top,ddtop,type,dd,active,
             a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],
             a[18],a[19],a[20],ktopm2,60.0f,a[21],arena);
    return status;
}

// MPAS module_cu_tiedtke.F:2244-2422. Match the native contiguous-slab
// sentinel witness, including its writes at level zero into preceding slabs.
extern "C" void cu6_unsafe_flux_values(int nc,float* values,int* flags) {
    float f[4096]{};int i[64]{};bool b[64]{};
    for(int c=0;c<nc;++c) {
        int status=1,activity=0;
        TkArena arena{f+c,i+c,b+c,0,0,0,nc,&status,&activity,false};
        TkArray<float> a[24];
        for(int v=0;v<24;++v) {
            a[v]=arena.take<float>(5);
            for(int k=1;k<=5;++k) a[v](k)=float(100*(v+1)+10*k+c+1);
            if(v!=4) a[v].count=4;
        }
        auto bot=arena.take<int>(1),top=arena.take<int>(1),ddtop=arena.take<int>(1),type=arena.take<int>(1);
        bot(1)=4;top(1)=3;ddtop(1)=4;type(1)=0;
        auto dd=arena.take<bool>(1),active=arena.take<bool>(1);
        dd(1)=false;active(1)=false;
        auto sig=arena.take<float>(4);
        for(int k=1;k<=4;++k) sig(k)=0.5f;
        int ktopm2=0;
        tk_cuflx(1,4,5,a[0],a[1],a[2],a[3],a[4],a[5],bot,top,ddtop,type,dd,active,
                 a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],
                 a[18],a[19],a[20],ktopm2,60.0f,sig,arena);
        flags[c]=dd(1);flags[nc+c]=active(1);flags[2*nc+c]=type(1);flags[3*nc]=ktopm2;
    }
    for(int k=0;k<120*nc;++k) values[k]=f[k];
}

// MPAS module_cu_tiedtke.F:2244-2422, valid downdraft-top rejection probe.
extern "C" void cu6_flux_probe(int mode,float* values,int* flags) {
    float f[4096]{};int i[64]{};bool b[64]{};
    int status=1,activity=0;
    TkArena arena{f,i,b,0,0,0,1,&status,&activity,false};
    TkArray<float> a[24];
    for(int k=0;k<24;++k) a[k]=arena.take<float>(9);
    for(int k=0;k<24;++k) if(k!=4) a[k].count=8;
    for(int k=1;k<=9;++k) a[4](k)=10000.0f+10000.0f*float(k);
    for(int k=1;k<=8;++k) {
        a[0](k)=0.005f;a[1](k)=0.010f;a[2](k)=280.0f;
        a[3](k)=0.005f;a[18](k)=280.0f;
    }
    // The ninth storage word is preserved although the argument's extent is 8.
    a[0].data[8]=0.005f;a[1].data[8]=0.010f;a[2].data[8]=280.0f;
    a[3].data[8]=0.005f;a[18].data[8]=280.0f;
    auto bot=arena.take<int>(1),top=arena.take<int>(1),ddtop=arena.take<int>(1),type=arena.take<int>(1);
    bot(1)=7;top(1)=5;ddtop(1)=4+2*mode;type(1)=1;
    auto dd=arena.take<bool>(1),active=arena.take<bool>(1);
    dd(1)=true;active(1)=true;
    auto sig=arena.take<float>(8);
    for(int k=1;k<=8;++k) sig(k)=0.5f;
    int ktopm2=0;
    tk_cuflx(1,8,9,a[0],a[1],a[2],a[3],a[4],a[5],bot,top,ddtop,type,dd,active,
             a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17],
             a[18],a[19],a[20],ktopm2,60.0f,sig,arena);
    for(int k=0;k<24;++k) std::memcpy(values+9*k,a[k].data,9*sizeof(float));
    flags[0]=dd(1);flags[1]=active(1);flags[2]=type(1);flags[3]=ktopm2;
}
