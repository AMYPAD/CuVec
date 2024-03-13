/**
 * Unifying Python/C++/CUDA memory.
 *
 * pybind11 opaque vector -> C++11 `std::vector` -> CUDA managed memory.
 */
#include "cuvec_pybind11.cuh"  // PYBIND11_BIND_NDCUVEC
#include <pybind11/pybind11.h> // PYBIND11_MODULE
#include <pybind11/stl_bind.h> // pybind11::bind_vector

PYBIND11_MODULE(cuvec_pybind11, m) {
  m.doc() = "PyBind11 external module.";
  pybind11::bind_vector<std::vector<size_t>>(m, "Shape");
  pybind11::implicitly_convertible<pybind11::tuple, std::vector<size_t>>();
  PYBIND11_BIND_NDxVEC(signed char, b);
  PYBIND11_BIND_NDxVEC(unsigned char, B);
  PYBIND11_BIND_NDxVEC(char, c);
  PYBIND11_BIND_NDxVEC(short, h);
  PYBIND11_BIND_NDxVEC(unsigned short, H);
  PYBIND11_BIND_NDxVEC(int, i);
  PYBIND11_BIND_NDxVEC(unsigned int, I);
  PYBIND11_BIND_NDxVEC(long long, q);
  PYBIND11_BIND_NDxVEC(unsigned long long, Q);
#ifdef _CUVEC_HALF
  PYBIND11_BIND_NDxVEC(_CUVEC_HALF, e);
#endif
  PYBIND11_BIND_NDxVEC(float, f);
  PYBIND11_BIND_NDxVEC(double, d);
  m.attr("__author__") = "Casper da Costa-Luis (https://github.com/casperdcl)";
  m.attr("__date__") = "2024";
  m.attr("__version__") = "2.0.0";
}
