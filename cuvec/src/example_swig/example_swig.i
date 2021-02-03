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
CuVec<float> *increment_f(CuVec<float> &src, CuVec<float> *output = NULL, bool timing = false);
%}
/// expose definitions
CuVec(float) *increment_f(CuVec(float) &src, CuVec(float) *output = NULL, bool timing = false);
