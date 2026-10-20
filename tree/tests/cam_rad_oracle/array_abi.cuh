// Fortran descriptor and arithmetic ABI for the whole CAM transcription.
// This host witness uses host glibc. CUDA transcendental bindings must come
// from the shared glibc 2.43 device header; they are deliberately separate.
#pragma once
#ifndef __CUDACC__
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#define CAM_HD
#define CAM_ROUTINE
inline const char* cam_context="entry";
#define CAM_TRACE(x) (cam_context=(x))
#define CAM_ASSERT(x) do {if(!(x)) {fprintf(stderr,"CAM_ABI_ASSERT %s: %s\n",cam_context,#x);abort();}} while(0)
#else
#define CAM_HD __host__ __device__
#define CAM_ROUTINE __device__ __noinline__
#define CAM_ASSERT(x) ((void)0)
#define CAM_TRACE(x) ((void)0)
#endif

template<class A,class B> struct CamSame {static constexpr bool value=false;};
template<class A> struct CamSame<A,A> {static constexpr bool value=true;};

template<class A,class B> CAM_HD constexpr auto cam_add(A a,B b)->decltype(a+b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a+b);
  if constexpr (CamSame<R,double>::value) return __dadd_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fadd_rn(a,b);
#endif
  return a+b;
}
template<class A,class B> CAM_HD constexpr auto cam_sub(A a,B b)->decltype(a-b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a-b);
  if constexpr (CamSame<R,double>::value) return __dsub_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fsub_rn(a,b);
#endif
  return a-b;
}
template<class A,class B> CAM_HD constexpr auto cam_mul(A a,B b)->decltype(a*b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a*b);
  if constexpr (CamSame<R,double>::value) return __dmul_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fmul_rn(a,b);
#endif
  return a*b;
}
template<class A,class B> CAM_HD constexpr auto cam_div(A a,B b)->decltype(a/b) {
#if defined(__CUDA_ARCH__)
  using R=decltype(a/b);
  if constexpr (CamSame<R,double>::value) return __ddiv_rn(a,b);
  if constexpr (CamSame<R,float>::value) return __fdiv_rn(a,b);
#endif
  return a/b;
}
template<class T> CAM_HD constexpr T cam_abs(T a) {return a<0?-a:a;}
template<class A,class B> CAM_HD constexpr auto cam_min(A a,B b)->decltype(a+b) {return a<b?a:b;}
template<class A,class B,class... X> CAM_HD constexpr auto cam_min(A a,B b,X... x) {return cam_min(cam_min(a,b),x...);}
template<class A,class B> CAM_HD constexpr auto cam_max(A a,B b)->decltype(a+b) {return a>b?a:b;}
template<class A,class B,class... X> CAM_HD constexpr auto cam_max(A a,B b,X... x) {return cam_max(cam_max(a,b),x...);}
template<class A,class B> CAM_HD constexpr auto cam_min0(A a,B b) {return cam_min(a,b);}
template<class T> CAM_HD int cam_nint(T a) {return int(a>=0?cam_add(a,T(.5)):cam_sub(a,T(.5)));}
CAM_HD constexpr float cam_float(int a) {return float(a);}
CAM_HD constexpr int cam_mod(int a,int b) {return a%b;}
template<class T> CAM_HD T cam_aint(T a,int kind=0) {return T((long long)a);}
// GNU integer powers use multiplication trees, independent of libm pow.
template<class T> CAM_HD T cam_pow(T a,int n) {
  const bool reciprocal=n<0;
  unsigned exponent=reciprocal ? 0u-unsigned(n) : unsigned(n);
  T result=T(1);
  bool first=true;
  while(exponent) {
    if(exponent&1) {result=first?a:cam_mul(result,a);first=false;}
    exponent>>=1;
    if(exponent) a=cam_mul(a,a);
  }
  return reciprocal ? cam_div(T(1),result) : result;
}
#ifndef __CUDACC__
inline float cam_exp(float a) {return ::expf(a);}
inline double cam_exp(double a) {return ::exp(a);}
inline float cam_log(float a) {return ::logf(a);}
inline double cam_log(double a) {return ::log(a);}
inline float cam_log10(float a) {return ::log10f(a);}
inline double cam_log10(double a) {return ::log10(a);}
inline float cam_sqrt(float a) {return ::sqrtf(a);}
inline double cam_sqrt(double a) {return ::sqrt(a);}
inline float cam_cos(float a) {return ::cosf(a);}
inline double cam_cos(double a) {return ::cos(a);}
inline float cam_sin(float a) {return ::sinf(a);}
inline double cam_sin(double a) {return ::sin(a);}
inline float cam_asin(float a) {return ::asinf(a);}
inline double cam_asin(double a) {return ::asin(a);}
inline float cam_atan(float a) {return ::atanf(a);}
inline double cam_atan(double a) {return ::atan(a);}
inline float cam_mod(float a,float b) {return ::fmodf(a,b);}
inline double cam_mod(double a,double b) {return ::fmod(a,b);}
inline float cam_pow(float a,float b) {return ::powf(a,b);}
inline double cam_pow(double a,double b) {return ::pow(a,b);}
inline double cam_pow(double a,float b) {return ::pow(a,double(b));}
inline double cam_pow(float a,double b) {return ::pow(double(a),b);}
#endif

struct CamSection {int low,high;};
CAM_HD inline CamSection cam_section(int a,int b) {return {a,b};}
template<class T> struct CamArray {
  T* data=nullptr;
  int rank=0,count=0,low[5]{},extent[5]{},stride[5]{};
  CAM_HD CamArray()=default;
  CAM_HD static CamArray from(T* p) {CamArray r;r.data=p;return r;}
  template<class... B> CAM_HD static CamArray contiguous(T* p,B... b) {
    int bounds[]={int(b)...};
    CamArray r;r.data=p;r.rank=sizeof...(b)/2;r.count=1;
    CAM_ASSERT(r.rank<=5);
    for(int d=0;d<r.rank;++d) {
      r.low[d]=bounds[2*d];r.extent[d]=bounds[2*d+1]-bounds[2*d]+1;
      r.stride[d]=r.count;r.count*=r.extent[d];
    }
    return r;
  }
  template<class... I> CAM_HD T& operator()(I... index) const {
    int a[]={int(index)...};int offset=0;
    CAM_ASSERT(sizeof...(index)==rank);
    for(int d=0;d<rank;++d) {
      CAM_ASSERT(a[d]>=low[d] && a[d]<low[d]+extent[d]);
      offset+=(a[d]-low[d])*stride[d];
    }
    return data[offset];
  }
  CAM_HD T& linear(int index) const {
    CAM_ASSERT(index>=0 && index<count);
    int offset=0;
    for(int d=0;d<rank;++d) {offset+=(index%extent[d])*stride[d];index/=extent[d];}
    return data[offset];
  }
  template<class... B> CAM_HD CamArray rebound(B... b) const {
    if(rank==0) return contiguous(data,b...);
    int bounds[]={int(b)...};CamArray r=*this;
    int cumulative=1;bool packed=true;
    for(int d=0;d<rank;++d) {packed=packed && stride[d]==cumulative;cumulative*=extent[d];}
    if(packed) {
      auto result=contiguous(data,b...);
      // Explicit-shape dummy extents do not bound sequence association.
      // MPAS radctl declares pdel(pcols,pverp), while camrad supplies pver;
      // the extra last plane is never referenced by the native routine.
      return result;
    }
    CAM_ASSERT(sizeof...(b)/2==rank);
    r.count=1;
    for(int d=0;d<rank;++d) {
      CAM_ASSERT(bounds[2*d+1]-bounds[2*d]+1<=extent[d]);
      r.low[d]=bounds[2*d];r.extent[d]=bounds[2*d+1]-bounds[2*d]+1;
      r.count*=r.extent[d];
    }
    return r;
  }
  CAM_HD void take(CamArray& r,int d,int a) const {
    CAM_ASSERT(a>=low[d] && a<low[d]+extent[d]);
    r.data+=(a-low[d])*stride[d];
  }
  CAM_HD void take(CamArray& r,int d,CamSection a) const {
    CAM_ASSERT(a.low>=low[d] && a.high<low[d]+extent[d]);
    r.data+=(a.low-low[d])*stride[d];
    int j=r.rank++;r.low[j]=1;r.extent[j]=a.high-a.low+1;
    r.stride[j]=stride[d];r.count*=r.extent[j];
  }
  template<class... S> CAM_HD CamArray section(S... s) const {
    CAM_ASSERT(sizeof...(s)==rank);
    CamArray r;r.data=data;r.count=1;int d=0;
    (take(r,d++,s),...);
    return r;
  }
  CAM_HD void init(const T* p,int n) const {
    CAM_ASSERT(n==count);
    for(int i=0;i<n;++i) linear(i)=p[i];
  }
};
struct CamArena {
  char* data;size_t used=0,capacity,peak=0;
  CAM_HD CamArena(char* p,size_t n):data(p),capacity(n) {}
  template<class T,class... B> CAM_HD CamArray<T> array(B... b) {
    used=(used+alignof(T)-1)&~size_t(alignof(T)-1);
    auto r=CamArray<T>::contiguous(reinterpret_cast<T*>(data+used),b...);
    used+=r.count*sizeof(T);peak=used>peak?used:peak;CAM_ASSERT(used<=capacity);return r;
  }
};
struct CamFrame {
  CamArena& arena;size_t mark;
  CAM_HD CamFrame(CamArena& a):arena(a),mark(a.used) {}
  CAM_HD ~CamFrame() {arena.used=mark;}
};
template<class T> CAM_HD int cam_size(CamArray<T> a) {return a.count;}
template<class T> CAM_HD bool cam_present(CamArray<T> a) {return a.data!=nullptr;}
CAM_HD inline int cam_count(CamArray<bool> a) {
  int n=0;for(int i=0;i<a.count;++i) n+=a.linear(i);return n;
}
template<class T> CAM_HD T cam_sum(CamArray<T> a) {
  T n=0;for(int i=0;i<a.count;++i) n=cam_add(n,a.linear(i));return n;
}
template<class T> CAM_HD T cam_product(CamArray<T> a) {
  T n=1;for(int i=0;i<a.count;++i) n=cam_mul(n,a.linear(i));return n;
}
template<class T> CAM_HD T cam_maxval(CamArray<T> a,int dim,int index) {
  CAM_ASSERT(a.rank==2 && dim==2);
  T m=a(index+1,1);
  for(int k=2;k<=a.extent[1];++k) m=cam_max(m,a(index+1,k));
  return m;
}
template<class T> CAM_HD T cam_maxval(CamArray<T> a) {
  CAM_ASSERT(a.count>0);
  T m=a.linear(0);
  for(int i=1;i<a.count;++i) m=cam_max(m,a.linear(i));
  return m;
}
