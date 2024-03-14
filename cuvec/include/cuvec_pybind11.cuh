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
template <> struct pybind11::format_descriptor<_CUVEC_HALF> : pybind11::format_descriptor<float> {
  static std::string format() { return "e"; }
};
#endif
PYBIND11_MAKE_OPAQUE(NDCuVec<float>);
PYBIND11_MAKE_OPAQUE(NDCuVec<double>);

#define PYBIND11_BIND_NDCUVEC(T, typechar)                                                        \
  pybind11::class_<NDCuVec<T>>(m, PYBIND11_TOSTRING(NDCuVec_##typechar),                          \
                               pybind11::buffer_protocol())                                       \
      .def_buffer([](NDCuVec<T> &v) -> pybind11::buffer_info {                                    \
        return pybind11::buffer_info(v.vec.data(), sizeof(T),                                     \
                                     pybind11::format_descriptor<T>::format(), v.shape.size(),    \
                                     v.shape, v.strides());                                       \
      })                                                                                          \
      .def(pybind11::init<>())                                                                    \
      .def(pybind11::init<std::vector<size_t>>())                                                 \
      .def_property(                                                                              \
          "shape", [](const NDCuVec<T> &v) { return &v.shape; }, &NDCuVec<T>::reshape)            \
      .def_property_readonly("address", [](const NDCuVec<T> &v) { return (size_t)v.vec.data(); }) \
      .def("__str__", [](const NDCuVec<T> &v) {                                                   \
        std::stringstream s;                                                                      \
        s << "cuvec.cuvec_pybind11." << PYBIND11_TOSTRING(NDCuVec_##typechar) << "((";            \
        if (v.shape.size() > 0) s << v.shape[0];                                                  \
        for (size_t i = 1; i < v.shape.size(); i++) s << ", " << v.shape[i];                      \
        s << "))";                                                                                \
        std::string c = s.str();                                                                  \
        return c;                                                                                 \
      })

#endif // _CUVEC_PYBIND11_H_
