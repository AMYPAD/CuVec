%include "std_vector.i"

%{
#include "cuvec.cuh"    // CuAlloc
#include "cuda_fp16.h"  // __half

template<class _Tp> size_t data(std::vector<_Tp, CuAlloc<_Tp>> &vec){
  return (size_t) vec.data();
};
%}

template<class _Tp> size_t data(std::vector<_Tp, CuAlloc<_Tp>> &vec);

// `%define X Y` rather than `using X = Y;`
// due to https://github.com/swig/swig/issues/1058
%define CuVec(Type)
std::vector<Type, CuAlloc<Type>>
%enddef

%define MKCUVEC(Type, typechar)
%template(Vector_ ## typechar) CuVec(Type);
%template(Vector_ ## typechar ## _data) data<Type>;
%enddef

MKCUVEC(signed char, b)
MKCUVEC(unsigned char, B)
MKCUVEC(char, c)
MKCUVEC(short, h)
MKCUVEC(unsigned short, H)
MKCUVEC(int, i)
MKCUVEC(unsigned int, I)
MKCUVEC(long long, q)
MKCUVEC(unsigned long long, Q)
MKCUVEC(__half, e)
MKCUVEC(float, f)
MKCUVEC(double, d)
