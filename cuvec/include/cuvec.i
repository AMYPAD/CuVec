/**
 * SWIG template header wrapping `NDCuVec<T>` as defined in "cuvec.cuh"
 * for external use via `%include "cuvec.i"`.
 */
%include "std_vector.i"
%{
#include "cuvec.cuh"   // NDCuVec<T>
%}
/// expose definitions
template <class T> struct NDCuVec {
  CuVec<T> vec;
  std::vector<size_t> shape;
};
