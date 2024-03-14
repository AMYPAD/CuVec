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
  pybind11::bind_vector<std::vector<size_t>>(m, "Shape", pybind11::module_local(true));
  pybind11::implicitly_convertible<pybind11::tuple, std::vector<size_t>>();
  PYBIND11_BIND_NDCUVEC(signed char, b);
  PYBIND11_BIND_NDCUVEC(unsigned char, B);
  PYBIND11_BIND_NDCUVEC(char, c);
  PYBIND11_BIND_NDCUVEC(short, h);
  PYBIND11_BIND_NDCUVEC(unsigned short, H);
  PYBIND11_BIND_NDCUVEC(int, i);
  PYBIND11_BIND_NDCUVEC(unsigned int, I);
  PYBIND11_BIND_NDCUVEC(long long, q);
  PYBIND11_BIND_NDCUVEC(unsigned long long, Q);
#ifdef _CUVEC_HALF
  PYBIND11_BIND_NDCUVEC(_CUVEC_HALF, e);
#endif
  PYBIND11_BIND_NDCUVEC(float, f);
  PYBIND11_BIND_NDCUVEC(double, d);
  m.attr("__author__") = "Casper da Costa-Luis (https://github.com/casperdcl)";
  m.attr("__date__") = "2024";
  m.attr("__version__") = "2.0.0";
}
