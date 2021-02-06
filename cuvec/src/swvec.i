%module swvec

%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%{
#include "cuda_fp16.h" // __half
%}

%include "cuvec.i" // SwigCuVec<T>

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
