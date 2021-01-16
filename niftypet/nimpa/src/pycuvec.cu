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

/** classes */
static PyCuVec_t<char> Vector_c;
static PyCuVec_t<signed char> Vector_b;
static PyCuVec_t<unsigned char> Vector_B;
// #ifdef _Bool
// #endif
static PyCuVec_t<short> Vector_h;
static PyCuVec_t<unsigned short> Vector_H;
static PyCuVec_t<int> Vector_i;
static PyCuVec_t<unsigned int> Vector_I;
static PyCuVec_t<long long> Vector_q;
static PyCuVec_t<unsigned long long> Vector_Q;
static PyCuVec_t<float> Vector_f;
static PyCuVec_t<double> Vector_d;

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

  if (PyType_Ready(&Vector_c.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_c.type_obj);
  PyModule_AddObject(m, Vector_c.name.c_str(), (PyObject *)&Vector_c.type_obj);

  if (PyType_Ready(&Vector_b.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_b.type_obj);
  PyModule_AddObject(m, Vector_b.name.c_str(), (PyObject *)&Vector_b.type_obj);

  if (PyType_Ready(&Vector_B.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_B.type_obj);
  PyModule_AddObject(m, Vector_B.name.c_str(), (PyObject *)&Vector_B.type_obj);

  // #ifdef _Bool
  // #endif

  if (PyType_Ready(&Vector_h.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_h.type_obj);
  PyModule_AddObject(m, Vector_h.name.c_str(), (PyObject *)&Vector_h.type_obj);

  if (PyType_Ready(&Vector_H.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_H.type_obj);
  PyModule_AddObject(m, Vector_H.name.c_str(), (PyObject *)&Vector_H.type_obj);

  if (PyType_Ready(&Vector_i.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_i.type_obj);
  PyModule_AddObject(m, Vector_i.name.c_str(), (PyObject *)&Vector_i.type_obj);

  if (PyType_Ready(&Vector_I.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_I.type_obj);
  PyModule_AddObject(m, Vector_I.name.c_str(), (PyObject *)&Vector_I.type_obj);

  if (PyType_Ready(&Vector_q.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_q.type_obj);
  PyModule_AddObject(m, Vector_q.name.c_str(), (PyObject *)&Vector_q.type_obj);

  if (PyType_Ready(&Vector_Q.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_Q.type_obj);
  PyModule_AddObject(m, Vector_Q.name.c_str(), (PyObject *)&Vector_Q.type_obj);

  if (PyType_Ready(&Vector_f.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_f.type_obj);
  PyModule_AddObject(m, Vector_f.name.c_str(), (PyObject *)&Vector_f.type_obj);

  if (PyType_Ready(&Vector_d.type_obj) < 0)
    return NULL;
  Py_INCREF(&Vector_d.type_obj);
  PyModule_AddObject(m, Vector_d.name.c_str(), (PyObject *)&Vector_d.type_obj);

  PyObject *all = Py_BuildValue(
      "[s, s, s, s, s, s, s, s, s, s, s, s]", "dev_sync", Vector_c.name.c_str(),
      Vector_b.name.c_str(), Vector_B.name.c_str(), Vector_h.name.c_str(), Vector_H.name.c_str(),
      Vector_i.name.c_str(), Vector_I.name.c_str(), Vector_q.name.c_str(), Vector_Q.name.c_str(),
      Vector_f.name.c_str(), Vector_d.name.c_str());
  if (all == NULL)
    return NULL;
  PyModule_AddObject(m, "__all__", all);

  return m;
}
