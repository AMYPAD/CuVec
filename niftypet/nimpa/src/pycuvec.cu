/**
 * CUDA C++11 extension vector for Python
 * - Casper da Costa-Luis (https://github.com/casperdcl) 2021
 */
#include "Python.h"
#include "pycuvec.cuh"

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
    "CUDA managed memory with python array buffer and C++ std::vector interfaces.",
    -1, // module keeps state in global variables
    cuvec_methods};
PyMODINIT_FUNC PyInit_cuvec(void) {
  Py_Initialize();
  // import_array();  // load NumPy functionality

  PyObject *m = PyModule_Create(&cuvec_module);
  if (m == NULL)
    return NULL;

  static PyCuVec_t<char> Vector_c;
  if (PyType_Ready(&Vector_c.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_c.type_obj);
  PyModule_AddObject(m, "Vector_c", (PyObject *)&Vector_c.type_obj);

  static PyCuVec_t<signed char> Vector_b;
  if (PyType_Ready(&Vector_b.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_b.type_obj);
  PyModule_AddObject(m, "Vector_b", (PyObject *)&Vector_b.type_obj);

  static PyCuVec_t<unsigned char> Vector_B;
  if (PyType_Ready(&Vector_B.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_B.type_obj);
  PyModule_AddObject(m, "Vector_B", (PyObject *)&Vector_B.type_obj);

  // #ifdef _Bool
  // #endif

  static PyCuVec_t<short> Vector_h;
  if (PyType_Ready(&Vector_h.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_h.type_obj);
  PyModule_AddObject(m, "Vector_h", (PyObject *)&Vector_h.type_obj);

  static PyCuVec_t<unsigned short> Vector_H;
  if (PyType_Ready(&Vector_H.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_H.type_obj);
  PyModule_AddObject(m, "Vector_H", (PyObject *)&Vector_H.type_obj);

  static PyCuVec_t<int> Vector_i;
  if (PyType_Ready(&Vector_i.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_i.type_obj);
  PyModule_AddObject(m, "Vector_i", (PyObject *)&Vector_i.type_obj);

  static PyCuVec_t<unsigned int> Vector_I;
  if (PyType_Ready(&Vector_I.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_I.type_obj);
  PyModule_AddObject(m, "Vector_I", (PyObject *)&Vector_I.type_obj);

  static PyCuVec_t<long long> Vector_q;
  if (PyType_Ready(&Vector_q.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_q.type_obj);
  PyModule_AddObject(m, "Vector_q", (PyObject *)&Vector_q.type_obj);

  static PyCuVec_t<unsigned long long> Vector_Q;
  if (PyType_Ready(&Vector_Q.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_Q.type_obj);
  PyModule_AddObject(m, "Vector_Q", (PyObject *)&Vector_Q.type_obj);

  static PyCuVec_t<float> Vector_f;
  if (PyType_Ready(&Vector_f.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_f.type_obj);
  PyModule_AddObject(m, "Vector_f", (PyObject *)&Vector_f.type_obj);

  static PyCuVec_t<double> Vector_d;
  if (PyType_Ready(&Vector_d.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_d.type_obj);
  PyModule_AddObject(m, "Vector_d", (PyObject *)&Vector_d.type_obj);

  return m;
}
