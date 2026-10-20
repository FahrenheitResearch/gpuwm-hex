// Host glibc/standard complex is exclusively a test witness, not CUDA math.
#define MATH(NAME) inline float mp_##NAME(float a) { return NAME##f(a); } inline double mp_##NAME(double a) {return NAME(a);}
MATH(exp)
MATH(log)
MATH(sqrt)
MATH(log10)
MATH(sin)
MATH(cos)
MATH(atan)
inline double mp_abs(cx a) {return std::abs(a);}
inline cx mp_log(cx a) {return std::log(a);}
inline cx mp_sqrt(cx a) {return std::sqrt(a);}
inline cx4 mp_sqrt(cx4 a) {return std::sqrt(a);}
inline float mp_pow(float a,float b) {return powf(a,b);}
inline double mp_pow(double a,double b) {return pow(a,b);}
inline cx mp_complex_div(cx a,cx b) {return a/b;}
inline cx4 mp_complex_div(cx4 a,cx4 b) {return a/b;}
