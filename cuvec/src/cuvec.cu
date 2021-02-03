/**
 * Unifying Python/C++/CUDA memory.
 *
 * Python buffered array -> C++11 `std::vector` -> CUDA managed memory.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "Python.h"
#include "pycuvec.cuh" // PyCuVec, PyCuVec_tp
/** functions */
/// required before accessing on host
static PyObject *dev_sync(PyObject *self, PyObject *args) {
  cudaDeviceSynchronize();

  Py_INCREF(Py_None);
  return Py_None;
}
static PyMethodDef cuvec_methods[] = {
    {"dev_sync", dev_sync, METH_NOARGS, "Required before accessing cuvec on host."},
    {NULL, NULL, 0, NULL} // Sentinel
};

/** module */
static struct PyModuleDef cuvec_module = {
    PyModuleDef_HEAD_INIT,
    "cuvec", // module
    "CUDA unified memory with python array buffer and C++ std::vector interfaces.",
    -1, // module keeps state in global variables
    cuvec_methods};
PyMODINIT_FUNC PyInit_cuvec(void) {
  Py_Initialize();
  // import_array();  // load NumPy functionality

  PyObject *m = PyModule_Create(&cuvec_module);
  if (m == NULL) return NULL;

#define _PYCUVEC_EXPOSE(T, typechar)                                                              \
  static PyCuVec_tp<T> PyCuVec_##typechar;                                                        \
  if (PyType_Ready(&PyCuVec_##typechar.tp_obj) < 0) return NULL;                                  \
  Py_INCREF(&PyCuVec_##typechar.tp_obj);                                                          \
  PyModule_AddObject(m, PyCuVec_##typechar.tp_obj.tp_name, (PyObject *)&PyCuVec_##typechar.tp_obj)

  _PYCUVEC_EXPOSE(signed char, b);
  _PYCUVEC_EXPOSE(unsigned char, B);
  _PYCUVEC_EXPOSE(char, c);
  // #ifdef _Bool
  // #endif
  _PYCUVEC_EXPOSE(short, h);
  _PYCUVEC_EXPOSE(unsigned short, H);
  _PYCUVEC_EXPOSE(int, i);
  _PYCUVEC_EXPOSE(unsigned int, I);
  _PYCUVEC_EXPOSE(long long, q);
  _PYCUVEC_EXPOSE(unsigned long long, Q);
  _PYCUVEC_EXPOSE(__half, e);
  _PYCUVEC_EXPOSE(float, f);
  _PYCUVEC_EXPOSE(double, d);

  /* aliases: inconsistent between `numpy.dtype` and `array.typecodes`
  Py_INCREF(&PyCuVec_q.tp_obj);
  PyModule_AddObject(m, "PyCuVec_l", (PyObject *)&PyCuVec_q.tp_obj);

  Py_INCREF(&PyCuVec_Q.tp_obj);
  PyModule_AddObject(m, "PyCuVec_L", (PyObject *)&PyCuVec_Q.tp_obj);
  */

  PyObject *author = Py_BuildValue("s", "Casper da Costa-Luis (https://github.com/casperdcl)");
  if (author == NULL) return NULL;
  PyModule_AddObject(m, "__author__", author);

  PyObject *date = Py_BuildValue("s", "2021");
  if (date == NULL) return NULL;
  PyModule_AddObject(m, "__date__", date);

  PyObject *version = Py_BuildValue("s", "0.4.0");
  if (version == NULL) return NULL;
  PyModule_AddObject(m, "__version__", version);

  return m;
}
