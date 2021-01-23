/**
 * Example external extension module using CuVec.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "Python.h"
#include "pycuvec.cuh" // PyCuVec
/** functions */
/// dst = src + 1
__global__ void _d_incr(float *dst, float *src, int X, int Y) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  if (x >= X) return;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (y >= Y) return;
  dst[y * X + x] = src[y * X + x] + 1;
}
static PyObject *increment_f(PyObject *self, PyObject *args) {
  PyCuVec<float> *src;
  if (!PyArg_ParseTuple(args, "O", (PyObject **)&src)) return NULL;
  std::vector<Py_ssize_t> &N = src->shape;

  cudaEvent_t eStart, eAlloc, eKern;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eAlloc);
  cudaEventCreate(&eKern);
  cudaEventRecord(eStart);
  PyCuVec<float> *dst = PyCuVec_zeros_like(src);
  cudaEventRecord(eAlloc);
  dim3 thrds((N[1] + 31) / 32, (N[0] + 31) / 32);
  dim3 blcks(32, 32);
  _d_incr<<<thrds, blcks>>>(dst->vec.data(), src->vec.data(), N[1], N[0]);
  // cudaDeviceSynchronize();
  cudaEventRecord(eKern);
  cudaEventSynchronize(eKern);
  float alloc_ms, kernel_ms;
  cudaEventElapsedTime(&alloc_ms, eStart, eAlloc);
  cudaEventElapsedTime(&kernel_ms, eAlloc, eKern);
  // fprintf(stderr, "%.3f ms, %.3f ms\n", alloc_ms, kernel_ms);
  return Py_BuildValue("ddO", double(alloc_ms), double(kernel_ms), (PyObject *)dst);
}
static PyMethodDef example_methods[] = {
    {"increment_f", increment_f, METH_VARARGS, "Returns (alloc_ms, kernel_ms, input + 1)."},
    {NULL, NULL, 0, NULL} // Sentinel
};

/** module */
static struct PyModuleDef example_mod = {PyModuleDef_HEAD_INIT,
                                         "example_mod", // module
                                         "Example external module.",
                                         -1, // module keeps state in global variables
                                         example_methods};
PyMODINIT_FUNC PyInit_example_mod(void) {
  Py_Initialize();
  return PyModule_Create(&example_mod);
}
