/**
 * SWIG template header wrapping `CuVec<T>`. Provides:
 *     CuVec(T)
 * for external use via `%include "cuvec.i"`.
 * Note that `CuVec(T)` is `%define`d to be `CuVec<T>`, which in turn is
 * defined in "cuvec.cuh"
 */
%include "std_vector.i"

%{
#include "cuvec.cuh"    // CuAlloc
#include "cuda_fp16.h"  // __half

template<class T> size_t data(CuVec<T> &vec) {
  return (size_t) vec.data();
};
%}

/// `%define X Y` rather than `using X = Y;`
/// due to https://github.com/swig/swig/issues/1058
%define CuVec(T) std::vector<T, CuAlloc<T>> %enddef

template<class T> size_t data(CuVec(T) &vec);

%define MKCUVEC(T, typechar)
%template(CuVec_ ## typechar) CuVec(T);
%template(CuVec_ ## typechar ## _data) data<T>;
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
