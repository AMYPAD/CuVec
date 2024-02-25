%module example_swig

%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "cuvec.i" // NDCuVec<T>
%{
/// signatures from "example_swig.cu"
NDCuVec<float> *increment2d_f(NDCuVec<float> &src, NDCuVec<float> *output = NULL, bool timing = false);
%}
/// expose definitions
NDCuVec<float> *increment2d_f(NDCuVec<float> &src, NDCuVec<float> *output = NULL, bool timing = false);
