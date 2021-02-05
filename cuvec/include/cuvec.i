/**
 * SWIG template header wrapping `SwigCuVec<T>` as defined in "cuvec.cuh"
 * for external use via `%include "cuvec.i"`.
 */
%include "std_vector.i"

%{
#include "cuvec.cuh"   // SwigCuVec<T>
#include "cuda_fp16.h" // __half
%}

template <class T> struct SwigCuVec {
  CuVec<T> vec;
  std::vector<size_t> shape;
};
template <class T> SwigCuVec<T> *SwigCuVec_new(std::vector<size_t> shape);
template <class T> void SwigCuVec_del(SwigCuVec<T> *self);
template <class T> T *SwigCuVec_data(SwigCuVec<T> *self);
template <class T> size_t SwigCuVec_address(SwigCuVec<T> *self);
template <class T> std::vector<size_t> SwigCuVec_shape(SwigCuVec<T> *self);

%template(SwigCuVec_Shape) std::vector<size_t>;
%define MKCUVEC(T, typechar)
%template(SwigCuVec_ ## typechar) SwigCuVec<T>;
%template(SwigCuVec_ ## typechar ## _new) SwigCuVec_new<T>;
%template(SwigCuVec_ ## typechar ## _del) SwigCuVec_del<T>;
%template(SwigCuVec_ ## typechar ## _data) SwigCuVec_data<T>;
%template(SwigCuVec_ ## typechar ## _address) SwigCuVec_address<T>;
%template(SwigCuVec_ ## typechar ## _shape) SwigCuVec_shape<T>;
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
