/**
 * Unifying Python/C++/CUDA memory.
 *
 * Python buffered array -> C++11 `std::vector` -> CUDA managed memory.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "cuvec.cuh" // CuVec
#include <Python.h>
#include <pybind11/pybind11.h> // pybind11
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

#define _PYBIND11_CUVEC_EXPOSE(T, typechar)                                                       \
  pybind11::bind_vector<CuVec<T>>(m, PYBIND11_TOSTRING(CuVec_##typechar))                         \
      .def("resize", [](CuVec<T> &v, size_t n) { v.resize(n); })

PYBIND11_MODULE(pybind11, m) {
  m.doc() = "PyBind11 external module.";
  _PYBIND11_CUVEC_EXPOSE(signed char, b);
  _PYBIND11_CUVEC_EXPOSE(unsigned char, B);
  _PYBIND11_CUVEC_EXPOSE(char, c);
  _PYBIND11_CUVEC_EXPOSE(short, h);
  _PYBIND11_CUVEC_EXPOSE(unsigned short, H);
  _PYBIND11_CUVEC_EXPOSE(int, i);
  _PYBIND11_CUVEC_EXPOSE(unsigned int, I);
  _PYBIND11_CUVEC_EXPOSE(long long, q);
  _PYBIND11_CUVEC_EXPOSE(unsigned long long, Q);
#ifdef _CUVEC_HALF
  _PYBIND11_CUVEC_EXPOSE(_CUVEC_HALF, e);
#endif
  _PYBIND11_CUVEC_EXPOSE(float, f);
  _PYBIND11_CUVEC_EXPOSE(double, d);
}
