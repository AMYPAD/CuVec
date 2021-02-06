/**
 * SWIG template header wrapping `SwigCuVec<T>` as defined in "cuvec.cuh"
 * for external use via `%include "cuvec.i"`.
 */
%include "std_vector.i"

%{
#include "cuvec.cuh"   // SwigCuVec<T>
%}

template <class T> struct SwigCuVec {
  CuVec<T> vec;
  std::vector<size_t> shape;
};
