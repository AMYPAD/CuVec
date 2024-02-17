/**
 * Unifying Python/C++/CUDA memory.
 *
 * pybind11 opaque vector -> C++11 `std::vector` -> CUDA managed memory.
 *
 * Copyright (2024) Casper da Costa-Luis
 */
#include "cuvec_pybind11.cuh"  // PYBIND11_MODULE, PYBIND11_BIND_CUVEC
#include <pybind11/pybind11.h> // PYBIND11_MODULE

PYBIND11_MODULE(cuvec_pybind11, m) {
  m.doc() = "PyBind11 external module.";
  PYBIND11_BIND_CUVEC(signed char, b);
  PYBIND11_BIND_CUVEC(unsigned char, B);
  PYBIND11_BIND_CUVEC(char, c);
  PYBIND11_BIND_CUVEC(short, h);
  PYBIND11_BIND_CUVEC(unsigned short, H);
  PYBIND11_BIND_CUVEC(int, i);
  PYBIND11_BIND_CUVEC(unsigned int, I);
  PYBIND11_BIND_CUVEC(long long, q);
  PYBIND11_BIND_CUVEC(unsigned long long, Q);
#ifdef _CUVEC_HALF
  PYBIND11_BIND_CUVEC(_CUVEC_HALF, e);
#endif
  PYBIND11_BIND_CUVEC(float, f);
  PYBIND11_BIND_CUVEC(double, d);
}
