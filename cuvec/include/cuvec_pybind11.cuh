/**
 * Python template header wrapping `CuVec<T>`. Provides:
 *     PYBIND11_BIND_CUVEC(T, typechar);
 */
#ifndef _CUVEC_PYBIND11_H_
#define _CUVEC_PYBIND11_H_

#include "cuvec.cuh"           // CuVec
#include <pybind11/stl_bind.h> // pybind11::bind_vector

PYBIND11_MAKE_OPAQUE(CuVec<signed char>);
PYBIND11_MAKE_OPAQUE(CuVec<unsigned char>);
PYBIND11_MAKE_OPAQUE(CuVec<char>);
PYBIND11_MAKE_OPAQUE(CuVec<short>);
PYBIND11_MAKE_OPAQUE(CuVec<unsigned short>);
PYBIND11_MAKE_OPAQUE(CuVec<int>);
PYBIND11_MAKE_OPAQUE(CuVec<unsigned int>);
PYBIND11_MAKE_OPAQUE(CuVec<long long>);
PYBIND11_MAKE_OPAQUE(CuVec<unsigned long long>);
#ifdef _CUVEC_HALF
PYBIND11_MAKE_OPAQUE(CuVec<_CUVEC_HALF>);
#endif
PYBIND11_MAKE_OPAQUE(CuVec<float>);
PYBIND11_MAKE_OPAQUE(CuVec<double>);

#define PYBIND11_BIND_CUVEC(T, typechar)                                                          \
  pybind11::bind_vector<CuVec<T>>(m, PYBIND11_TOSTRING(CuVec_##typechar))                         \
      .def("resize", [](CuVec<T> &v, size_t n) { v.resize(n); })

#endif // _CUVEC_PYBIND11_H_
