/**
 * CUDA C++11 extension vector for Python
 * - Casper da Costa-Luis (https://github.com/casperdcl) 2021
 */
// #include "cuhelpers.h"
#include "cuvec.cuh"
#include <Python.h>
#include <sstream>  // std::stringstream
#include <typeinfo> // typeid

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
/// class PyCuVec<T> and PyCuVec_f = PyCuVec<float>
template <class T> struct PyCuVec { PyObject_HEAD CuVec<T> vec; };
/// __init__
template <class T> static int PyCuVec_init(PyCuVec<T> *self, PyObject *args, PyObject *kwds) {
  int length = 0;
  static char *kwlist[2] = {(char *)"length", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &length))
    return -1;
  if (length < 0)
    length = 0;
  self->vec.resize(length);
  return 0;
}
/// __del__
template <class T> static void PyCuVec_dealloc(PyCuVec<T> *self) {
  self->vec.clear();
  Py_TYPE(self)->tp_free((PyObject *)self);
}
/// __str__
template <class T> static PyObject *PyCuVec_str(PyCuVec<T> *self) {
  std::stringstream s;
  s << "cuvec.Vector<" << typeid(T).name() << ">[" << self->vec.size() << "]";
  std::string c = s.str();
  PyObject *ret = PyUnicode_FromString(c.c_str());
  return ret;
}
/// class
static PyTypeObject PyCuVec_f = {
    PyVarObject_HEAD_INIT(NULL, 0) "cuvec.Vector_f", /* tp_name */
    sizeof(PyCuVec<float>),                          /* tp_basicsize */
    0,                                               /* tp_itemsize */
    (destructor)PyCuVec_dealloc<float>,              /* tp_dealloc */
    0,                                               /* tp_print */
    0,                                               /* tp_getattr */
    0,                                               /* tp_setattr */
    0,                                               /* tp_reserved */
    0,                                               /* tp_repr */
    0,                                               /* tp_as_number */
    0,                                               /* tp_as_sequence */
    0,                                               /* tp_as_mapping */
    0,                                               /* tp_hash  */
    0,                                               /* tp_call */
    (reprfunc)PyCuVec_str<float>,                    /* tp_str */
    0,                                               /* tp_getattro */
    0,                                               /* tp_setattro */
    0,                                               /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                              /* tp_flags */
    "cuvec.Vector<f> object",                        /* tp_doc */
    0,                                               /* tp_traverse */
    0,                                               /* tp_clear */
    0,                                               /* tp_richcompare */
    0,                                               /* tp_weaklistoffset */
    0,                                               /* tp_iter */
    0,                                               /* tp_iternext */
    0,                                               /* tp_methods */
    0,                                               /* tp_members */
    0,                                               /* tp_getset */
    0,                                               /* tp_base */
    0,                                               /* tp_dict */
    0,                                               /* tp_descr_get */
    0,                                               /* tp_descr_set */
    0,                                               /* tp_dictoffset */
    (initproc)PyCuVec_init<float>,                   /* tp_init */
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

  // class PyCuVec_f
  PyCuVec_f.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyCuVec_f) < 0)
    return NULL;
  Py_INCREF(&PyCuVec_f);
  PyModule_AddObject(m, "Vector_f", (PyObject *)&PyCuVec_f);

  return m;
}
