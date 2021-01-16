#include "Python.h"
#include "cuhelpers.h"
#include <cstdio>  // printf
#include <sstream> // std::stringstream

void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

bool PyHandleError(cudaError_t err, const char *file, int line) {
  std::stringstream ss;
  ss << file << ':' << line << ": " << cudaGetErrorString(err);
  std::string s = ss.str();
  if (err != cudaSuccess) {
    PyErr_SetString(PyExc_ValueError, s.c_str());
    return false;
  }
  return true;
}
