/**
 * Python template header wrapping `NDCuVec<T>`. Provides:
 *     PYBIND11_BIND_NDCUVEC(T, typechar);
 */
#ifndef _CUVEC_PYBIND11_H_
#define _CUVEC_PYBIND11_H_

#include "cuvec.cuh"           // NDCuVec
#include <pybind11/pybind11.h> // pybind11, PYBIND11_MAKE_OPAQUE
#include <pybind11/stl.h>      // std::vector

#ifndef CUVEC_DISABLE_CUDA // ensure CPU-only alternative exists
#define NDxVEC_MAKE_OPAQUE(T)                                                                     \
  PYBIND11_MAKE_OPAQUE(NDCuVec<T>);                                                               \
  PYBIND11_MAKE_OPAQUE(NDCVec<T>);
#else
#define NDxVEC_MAKE_OPAQUE(T) PYBIND11_MAKE_OPAQUE(NDCuVec<T>);
#endif // CUVEC_DISABLE_CUDA

PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
NDxVEC_MAKE_OPAQUE(signed char);
NDxVEC_MAKE_OPAQUE(unsigned char);
NDxVEC_MAKE_OPAQUE(char);
NDxVEC_MAKE_OPAQUE(short);
NDxVEC_MAKE_OPAQUE(unsigned short);
NDxVEC_MAKE_OPAQUE(int);
NDxVEC_MAKE_OPAQUE(unsigned int);
NDxVEC_MAKE_OPAQUE(long long);
NDxVEC_MAKE_OPAQUE(unsigned long long);
#ifdef _CUVEC_HALF
NDxVEC_MAKE_OPAQUE(_CUVEC_HALF);
template <> struct pybind11::format_descriptor<_CUVEC_HALF> : pybind11::format_descriptor<float> {
  static std::string format() { return "e"; }
};
#endif
NDxVEC_MAKE_OPAQUE(float);
NDxVEC_MAKE_OPAQUE(double);

#define PYBIND11_BIND_NDVEC(Vec, T, typechar)                                                     \
  pybind11::class_<Vec<T>>(m, PYBIND11_TOSTRING(Vec##_##typechar), pybind11::buffer_protocol())   \
      .def_buffer([](Vec<T> &v) -> pybind11::buffer_info {                                        \
        return pybind11::buffer_info(v.vec.data(), sizeof(T),                                     \
                                     pybind11::format_descriptor<T>::format(), v.shape.size(),    \
                                     v.shape, v.strides());                                       \
      })                                                                                          \
      .def(pybind11::init<>())                                                                    \
      .def(pybind11::init<std::vector<size_t>>())                                                 \
      .def_property(                                                                              \
          "shape", [](const Vec<T> &v) { return &v.shape; }, &Vec<T>::reshape)                    \
      .def_property_readonly("address", [](const Vec<T> &v) { return (size_t)v.vec.data(); })
#define PYBIND11_BIND_NDCUVEC(T, typechar) PYBIND11_BIND_NDVEC(NDCuVec, T, typechar)
#ifndef CUVEC_DISABLE_CUDA // ensure CPU-only alternative exists
#define PYBIND11_BIND_NDCVEC(T, typechar) PYBIND11_BIND_NDVEC(NDCVec, T, typechar)
#else
#define PYBIND11_BIND_NDCVEC(T, typechar)
#endif // CUVEC_DISABLE_CUDA
#define PYBIND11_BIND_NDxVEC(T, typechar)                                                         \
  PYBIND11_BIND_NDCVEC(T, typechar);                                                              \
  PYBIND11_BIND_NDCUVEC(T, typechar)

#endif // _CUVEC_PYBIND11_H_
