%module example_swig

%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "cuvec.i" // %{ CuVec<T> %}, CuVec(T)
%{
/// signatures from "example_swig.cu"
float increment_inplace_f(CuVec<float> &src);
CuVec<float> *increment_f(CuVec<float> &src);
%}
/// expose definitions
float increment_inplace_f(CuVec(float) &src);
CuVec(float) *increment_f(CuVec(float) &src);
