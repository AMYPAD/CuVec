%module example_swig

%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "cuvec.i" // SwigCuVec<T>
%{
/// signatures from "example_swig.cu"
SwigCuVec<float> *increment2d_f(SwigCuVec<float> *src, bool timing = false, SwigCuVec<float> *output = NULL);
%}
/// expose definitions
SwigCuVec<float> *increment2d_f(SwigCuVec<float> *src, bool timing = false, SwigCuVec<float> *output = NULL);
