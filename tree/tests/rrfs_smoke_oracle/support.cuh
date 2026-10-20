#ifndef SDDEV
#define SDDEV __device__
#endif
// RRFS src/core_atmosphere/physics/physics_noaa/SMOKE/*.F90: array and RKIND operation lowering.
const int SD_END = 2147483647;
struct Slice {int lo,hi;SDDEV Slice():lo(SD_END),hi(SD_END) {} SDDEV Slice(int a,int b):lo(a),hi(b) {}};
template<class T> struct FArray {
 T *p=nullptr; int rank=0;int lo[4]={1,1,1,1},hi[4]={0,0,0,0},stride[4]={1,1,1,1};
 SDDEV int size() const {int n=1;for(int d=0;d<rank;d++)n*=hi[d]-lo[d]+1;return rank?n:0;}
 SDDEV T &linear(int n) const {int off=0;for(int d=0;d<rank;d++){int z=hi[d]-lo[d]+1;off+=(n%z)*stride[d];n/=z;}return p[off];}
 SDDEV T &operator()(int a) const {return p[(a-lo[0])*stride[0]];}
 SDDEV T &operator()(int a,int b) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]];}
 SDDEV T &operator()(int a,int b,int c) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]+(c-lo[2])*stride[2]];}
 SDDEV T &operator()(int a,int b,int c,int e) const {return p[(a-lo[0])*stride[0]+(b-lo[1])*stride[1]+(c-lo[2])*stride[2]+(e-lo[3])*stride[3]];}
 SDDEV void add_section(FArray &r,int &d,int val) const {r.p+=(val-lo[d])*stride[d];d++;}
 SDDEV void add_section(FArray &r,int &d,Slice val) const {int l=val.lo==SD_END?lo[d]:val.lo,h=val.hi==SD_END?hi[d]:val.hi;r.p+=(l-lo[d])*stride[d];r.lo[r.rank]=1;r.hi[r.rank]=h-l+1;r.stride[r.rank++]=stride[d++];}
 template<class... A> SDDEV FArray section(A... a) const {FArray r;r.p=p;int d=0;(add_section(r,d,a),...);return r;}
 template<class... A> SDDEV void bind(A... a) {int v[]={int(a)...};int r=sizeof...(a)/2;int base=stride[0];bool reshape=rank!=r;for(int d=0;d<r && !reshape;d++)reshape=(hi[d]-lo[d]+1 != v[2*d+1]-v[2*d]+1);if(reshape){int st=base;for(int d=0;d<r;d++){stride[d]=st;st*=v[2*d+1]-v[2*d]+1;}}rank=r;for(int d=0;d<r;d++){lo[d]=v[2*d];hi[d]=v[2*d+1];}}
};
template<class T,class... I> SDDEV FArray<T> sd_sequence(FArray<T> a,I... i){FArray<T> r;r.p=&a(i...);r.bind(1,1);return r;}
// Port arena defect: writing beyond the allocated device workspace corrupts
// subsequent state. Launcher reserves from the complete tile descriptors;
// this assertion diagnoses a bad reservation, never a native undefined read.
SDDEV void sd_capacity_failure() {
#ifdef __CUDA_ARCH__
 asm("trap;");
#else
 __builtin_trap();
#endif
}
struct Arena {float *p;int used=0;int capacity;template<class T,class... A> SDDEV FArray<T> alloc(A... a) {FArray<T> r;r.p=(T*)(p+used);r.bind(a...);int words=r.size();if(words<0 || used>capacity-words)sd_capacity_failure();used+=words;return r;}};
struct ArenaScope {Arena &arena;int mark;SDDEV ArenaScope(Arena &a):arena(a),mark(a.used){} SDDEV ~ArenaScope(){arena.used=mark;}};
struct FStr {
 char *p;int len;const char *lit;
 SDDEV FStr():p(nullptr),len(0),lit(nullptr){}
 SDDEV FStr(const char *s):p(nullptr),len(0),lit(s){while(s[len])len++;}
 SDDEV FStr(char *s,int n):p(s),len(n),lit(nullptr){}
 SDDEV char at(int i) const {return i<len?(p?p[i]:lit[i]):' ';}
 SDDEV FStr &operator=(const FStr &s){if(p){for(int i=0;i<len;i++)p[i]=s.at(i);}else{lit=s.lit;len=s.len;p=s.p;}return *this;}
};
SDDEV bool operator==(FStr a,FStr b){int n=a.len>b.len?a.len:b.len;for(int i=0;i<n;i++)if(a.at(i)!=b.at(i))return false;return true;}
SDDEV bool operator!=(FStr a,FStr b){return !(a==b);}
// Native Fortran unary REAL negation is a sign-bit XOR, including sNaNs.
template<class T> SDDEV T sd_neg(T a){return -a;}
SDDEV float sd_neg(float a){return __uint_as_float(__float_as_uint(a)^0x80000000u);}
template<class T> SDDEV T sd_add(T a,T b){return a+b;}
template<class T> SDDEV T sd_sub(T a,T b){return a-b;}
template<class T> SDDEV T sd_mul(T a,T b){return a*b;}
template<class T> SDDEV T sd_div(T a,T b){return a/b;}
// Strict GNU scalar SSE operand semantics: first operand NaN wins, signaling
// payloads are quieted, and invalid operations yield the negative indefinite.
// Finite results still come from the explicit round-to-nearest CUDA operations.
// This is arithmetic at the operation, not a comparison-time NaN substitution.
SDDEV bool sd_float_nan_word(unsigned int u){return (u&0x7fffffffu)>0x7f800000u;}
SDDEV float sd_sse_binary(float a,float b,float result){
 unsigned int x=__float_as_uint(a),y=__float_as_uint(b);
 if(sd_float_nan_word(x))return __uint_as_float(x|0x00400000u);
 if(sd_float_nan_word(y))return __uint_as_float(y|0x00400000u);
 return sd_float_nan_word(__float_as_uint(result))?__uint_as_float(0xffc00000u):result;
}
SDDEV float sd_add(float a,float b){return sd_sse_binary(a,b,__fadd_rn(a,b));} SDDEV double sd_add(double a,double b){return __dadd_rn(a,b);}
SDDEV float sd_sub(float a,float b){return sd_sse_binary(a,b,__fsub_rn(a,b));} SDDEV double sd_sub(double a,double b){return __dsub_rn(a,b);}
SDDEV float sd_mul(float a,float b){return sd_sse_binary(a,b,__fmul_rn(a,b));} SDDEV double sd_mul(double a,double b){return __dmul_rn(a,b);}
SDDEV float sd_div(float a,float b){return sd_sse_binary(a,b,__fdiv_rn(a,b));} SDDEV double sd_div(double a,double b){return __ddiv_rn(a,b);}
#define SD_MIXED(name) template<class A,class B> SDDEV auto name(A a,B b)->decltype(a+b){using T=decltype(a+b);return name(T(a),T(b));}
SD_MIXED(sd_add) SD_MIXED(sd_sub) SD_MIXED(sd_mul) SD_MIXED(sd_div)
template<class T> SDDEV T sd_min(T a,T b){return a<b?a:b;} template<class T> SDDEV T sd_max(T a,T b){return a>b?a:b;}
SD_MIXED(sd_min) SD_MIXED(sd_max)
SDDEV float sd_sqrt(float a){
 unsigned int x=__float_as_uint(a);
 if(sd_float_nan_word(x))return __uint_as_float(x|0x00400000u);
 float result=__fsqrt_rn(a);
 return sd_float_nan_word(__float_as_uint(result))?__uint_as_float(0xffc00000u):result;
}SDDEV double sd_sqrt(double a){return __dsqrt_rn(a);}
SDDEV float sd_exp(float a){return gfk_exp_fma(a);}SDDEV float sd_log(float a){return gfk_log(a);}SDDEV float sd_pow(float a,float b){return gfk_pow(a,b);}
SDDEV double sd_exp(double a){return glibc_exp(a);}SDDEV double sd_log(double a){return glibc_log(a);}SDDEV double sd_pow(double a,double b){return glibc_pow(a,b);}
SD_MIXED(sd_pow)
SDDEV float sd_log10(float a){return thompson_aa_log10f(a);}SDDEV double sd_log10(double a){return thompson_aa_log10(a);}
SDDEV float sd_erf(float a){return smoke_shared_erf(a);}
SDDEV float sd_cos(float a){return glibc_cosf(a);}SDDEV float sd_sin(float a){return glibc_sinf(a);}
SDDEV float sd_abs(float a){return __uint_as_float(__float_as_uint(a)&0x7fffffffu);} SDDEV double sd_abs(double a){return fabs(a);}SDDEV int sd_abs(int a){return a<0?-a:a;}
SDDEV float sd_aint(float a){unsigned int u=__float_as_uint(a);return sd_float_nan_word(u)?__uint_as_float(u):truncf(a);}SDDEV int sd_floor(float a){return int(floorf(a));}SDDEV int sd_nint(float a){return a>=0?int(a+0.5f):int(a-0.5f);}
SDDEV double sd_dble(float a){return double(a);}SDDEV float sd_mod(float a,float b){return fmodf(a,b);}SDDEV int sd_mod(int a,int b){return a%b;}
SDDEV float sd_sign(float a,float b){return copysignf(a,b);}SDDEV bool sd_isnan(float a){return isnan(a);}
template<class T> SDDEV T sd_ipow(T a,int b){if(b==0)return T(1);if(b==1)return a;T x=sd_ipow(a,b/2);T y=sd_mul(x,x);return b%2?sd_mul(y,a):y;}
template<class T> struct ScalarExpr {T v; SDDEV T linear(int) const{return v;}};
template<class T> SDDEV ScalarExpr<T> sd_expr(T v){return {v};}template<class T> SDDEV FArray<T> sd_expr(FArray<T> v){return v;}
struct OpAdd {template<class A,class B> SDDEV static auto op(A a,B b){return sd_add(a,b);}};
struct OpSub {template<class A,class B> SDDEV static auto op(A a,B b){return sd_sub(a,b);}};
struct OpMul {template<class A,class B> SDDEV static auto op(A a,B b){return sd_mul(a,b);}};
struct OpDiv {template<class A,class B> SDDEV static auto op(A a,B b){return sd_div(a,b);}};
template<class A,class B,class Op> struct BinaryExpr {A a;B b;SDDEV auto linear(int i) const {return Op::op(a.linear(i),b.linear(i));}};
template<class A,class B,class O> SDDEV BinaryExpr<A,B,O> sd_expr(BinaryExpr<A,B,O> e){return e;}
#define SD_ARRAY_OP(name,op) \
 template<class T,class B> SDDEV auto name(FArray<T> a,B b){return BinaryExpr<FArray<T>,decltype(sd_expr(b)),op>{a,sd_expr(b)};} \
 template<class A,class T> SDDEV auto name(A a,FArray<T> b){return BinaryExpr<decltype(sd_expr(a)),FArray<T>,op>{sd_expr(a),b};} \
 template<class T,class U> SDDEV auto name(FArray<T> a,FArray<U> b){return BinaryExpr<FArray<T>,FArray<U>,op>{a,b};} \
 template<class A,class B,class O,class C> SDDEV auto name(BinaryExpr<A,B,O> a,C b){return BinaryExpr<BinaryExpr<A,B,O>,decltype(sd_expr(b)),op>{a,sd_expr(b)};}
SD_ARRAY_OP(sd_add,OpAdd) SD_ARRAY_OP(sd_sub,OpSub) SD_ARRAY_OP(sd_mul,OpMul) SD_ARRAY_OP(sd_div,OpDiv)
template<class T,class E> SDDEV void sd_assign(FArray<T> a,E rhs,Arena &arena){ArenaScope scope(arena);auto b=sd_expr(rhs);auto t=arena.alloc<T>(1,a.size());for(int i=0;i<a.size();i++)t.linear(i)=b.linear(i);for(int i=0;i<a.size();i++)a.linear(i)=t.linear(i);}
template<class T> SDDEV T sd_sum(FArray<T> a){T v=0;for(int i=0;i<a.size();i++)v=sd_add(v,a.linear(i));return v;}
template<class T> SDDEV T sd_minval(FArray<T> a){T v=a.linear(0);for(int i=1;i<a.size();i++)v=sd_min(v,a.linear(i));return v;}
template<class T> SDDEV T sd_maxval(FArray<T> a){T v=a.linear(0);for(int i=1;i<a.size();i++)v=sd_max(v,a.linear(i));return v;}
