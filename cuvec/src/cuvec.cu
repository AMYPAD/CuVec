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

/** classes */
static PyCuVec_tp<char> Vector_c;
static PyCuVec_tp<signed char> Vector_b;
static PyCuVec_tp<unsigned char> Vector_B;
// #ifdef _Bool
// #endif
static PyCuVec_tp<short> Vector_h;
static PyCuVec_tp<unsigned short> Vector_H;
static PyCuVec_tp<int> Vector_i;
static PyCuVec_tp<unsigned int> Vector_I;
static PyCuVec_tp<long long> Vector_q;          // _l
static PyCuVec_tp<unsigned long long> Vector_Q; // _L
static PyCuVec_tp<__half> Vector_e;
static PyCuVec_tp<float> Vector_f;
static PyCuVec_tp<double> Vector_d;

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

  if (PyType_Ready(&Vector_c.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_c.tp_obj);
  PyModule_AddObject(m, Vector_c.name.c_str(), (PyObject *)&Vector_c.tp_obj);

  if (PyType_Ready(&Vector_b.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_b.tp_obj);
  PyModule_AddObject(m, Vector_b.name.c_str(), (PyObject *)&Vector_b.tp_obj);

  if (PyType_Ready(&Vector_B.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_B.tp_obj);
  PyModule_AddObject(m, Vector_B.name.c_str(), (PyObject *)&Vector_B.tp_obj);

  // #ifdef _Bool
  // #endif

  if (PyType_Ready(&Vector_h.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_h.tp_obj);
  PyModule_AddObject(m, Vector_h.name.c_str(), (PyObject *)&Vector_h.tp_obj);

  if (PyType_Ready(&Vector_H.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_H.tp_obj);
  PyModule_AddObject(m, Vector_H.name.c_str(), (PyObject *)&Vector_H.tp_obj);

  if (PyType_Ready(&Vector_i.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_i.tp_obj);
  PyModule_AddObject(m, Vector_i.name.c_str(), (PyObject *)&Vector_i.tp_obj);

  if (PyType_Ready(&Vector_I.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_I.tp_obj);
  PyModule_AddObject(m, Vector_I.name.c_str(), (PyObject *)&Vector_I.tp_obj);

  if (PyType_Ready(&Vector_q.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_q.tp_obj);
  PyModule_AddObject(m, Vector_q.name.c_str(), (PyObject *)&Vector_q.tp_obj);
  Py_INCREF(&Vector_q.tp_obj);
  PyModule_AddObject(m, "Vector_l", (PyObject *)&Vector_q.tp_obj);

  if (PyType_Ready(&Vector_Q.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_Q.tp_obj);
  PyModule_AddObject(m, Vector_Q.name.c_str(), (PyObject *)&Vector_Q.tp_obj);
  Py_INCREF(&Vector_Q.tp_obj);
  PyModule_AddObject(m, "Vector_L", (PyObject *)&Vector_Q.tp_obj);

  if (PyType_Ready(&Vector_e.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_e.tp_obj);
  PyModule_AddObject(m, Vector_e.name.c_str(), (PyObject *)&Vector_e.tp_obj);

  if (PyType_Ready(&Vector_f.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_f.tp_obj);
  PyModule_AddObject(m, Vector_f.name.c_str(), (PyObject *)&Vector_f.tp_obj);

  if (PyType_Ready(&Vector_d.tp_obj) < 0) return NULL;
  Py_INCREF(&Vector_d.tp_obj);
  PyModule_AddObject(m, Vector_d.name.c_str(), (PyObject *)&Vector_d.tp_obj);

  PyObject *author = Py_BuildValue("s", "Casper da Costa-Luis (https://github.com/casperdcl)");
  if (author == NULL) return NULL;
  PyModule_AddObject(m, "__author__", author);

  PyObject *date = Py_BuildValue("s", "2021");
  if (date == NULL) return NULL;
  PyModule_AddObject(m, "__date__", date);

  PyObject *version = Py_BuildValue("s", "0.3.0");
  if (version == NULL) return NULL;
  PyModule_AddObject(m, "__version__", version);

  return m;
}
