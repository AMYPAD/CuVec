/**
 * Python template header wrapping `NDCuVec<T>`. Provides:
 *     PYBIND11_BIND_NDCUVEC(T, typechar);
 */
#ifndef _CUVEC_PYBIND11_H_
#define _CUVEC_PYBIND11_H_

#include "cuvec.cuh"           // NDCuVec
#include <pybind11/pybind11.h> // pybind11, PYBIND11_MAKE_OPAQUE
#include <pybind11/stl.h>      // std::vector

PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
PYBIND11_MAKE_OPAQUE(NDCuVec<signed char>);
PYBIND11_MAKE_OPAQUE(NDCuVec<unsigned char>);
PYBIND11_MAKE_OPAQUE(NDCuVec<char>);
PYBIND11_MAKE_OPAQUE(NDCuVec<short>);
PYBIND11_MAKE_OPAQUE(NDCuVec<unsigned short>);
PYBIND11_MAKE_OPAQUE(NDCuVec<int>);
PYBIND11_MAKE_OPAQUE(NDCuVec<unsigned int>);
PYBIND11_MAKE_OPAQUE(NDCuVec<long long>);
PYBIND11_MAKE_OPAQUE(NDCuVec<unsigned long long>);
#ifdef _CUVEC_HALF
PYBIND11_MAKE_OPAQUE(NDCuVec<_CUVEC_HALF>);
#endif
PYBIND11_MAKE_OPAQUE(NDCuVec<float>);
PYBIND11_MAKE_OPAQUE(NDCuVec<double>);

#define PYBIND11_BIND_NDCUVEC(T, typechar)                                                        \
  pybind11::class_<NDCuVec<T>>(m, PYBIND11_TOSTRING(NDCuVec_##typechar))                          \
      .def(pybind11::init<>())                                                                    \
      .def(pybind11::init<std::vector<size_t>>())                                                 \
      .def("reshape", &NDCuVec<T>::reshape)                                                       \
      .def("shape", [](const NDCuVec<T> &v) { return v.shape; })                                  \
      .def("address", [](NDCuVec<T> &v) { return (size_t)v.vec.data(); })

#endif // _CUVEC_PYBIND11_H_
