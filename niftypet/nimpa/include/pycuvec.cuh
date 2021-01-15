#ifndef _PYCUVEC_H_
#define _PYCUVEC_H_

#include "Python.h"
#include "cuvec.cuh" // CuVec
#include <cstdlib>   // malloc, free
#include <sstream>   // std::stringstream
#include <typeinfo>  // typeid

template <typename T> struct PyType {
  static const char *format() { return typeid(T).name(); }
};
template <> struct PyType<char> {
  static const char *format() { return "c"; }
};
template <> struct PyType<signed char> {
  static const char *format() { return "b"; }
};
template <> struct PyType<unsigned char> {
  static const char *format() { return "B"; }
};
#ifdef _Bool
template <> struct PyType<_Bool> {
  static const char *format() { return "?"; }
};
#endif
template <> struct PyType<short> {
  static const char *format() { return "h"; }
};
template <> struct PyType<unsigned short> {
  static const char *format() { return "H"; }
};
template <> struct PyType<int> {
  static const char *format() { return "i"; }
};
template <> struct PyType<unsigned int> {
  static const char *format() { return "I"; }
};
template <> struct PyType<long long> {
  static const char *format() { return "q"; }
};
template <> struct PyType<unsigned long long> {
  static const char *format() { return "Q"; }
};
template <> struct PyType<float> {
  static const char *format() { return "f"; }
};
template <> struct PyType<double> {
  static const char *format() { return "d"; }
};

/** classes */
/// class PyCuVec<T>
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
  s << "cuvec.Vector<" << PyType<T>::format() << ">[" << self->vec.size() << "]";
  std::string c = s.str();
  PyObject *ret = PyUnicode_FromString(c.c_str());
  return ret;
}
/// buffer interface
template <class T> static int PyCuVec_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
  if (view == NULL) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }

  PyCuVec<T> *self = (PyCuVec<T> *)obj;
  Py_ssize_t *shape = (Py_ssize_t *)malloc(sizeof(Py_ssize_t));
  shape[0] = self->vec.size();
  view->buf = (void *)self->vec.data();
  view->obj = (PyObject *)self;
  view->len = self->vec.size() * sizeof(T);
  view->readonly = 0;
  view->itemsize = sizeof(T);
  view->format = (char *)PyType<T>::format();
  view->ndim = 1;
  view->shape = shape;
  view->strides = &view->itemsize;
  view->suboffsets = NULL;
  view->internal = NULL;

  Py_INCREF(self);
  return 0;
}
template <class T> static void PyCuVec_release(PyObject *obj, Py_buffer *view) {
  if (view == NULL) {
    PyErr_SetString(PyExc_ValueError, "NULL view in release");
    return;
  }
  free(view->shape);

  PyCuVec<T> *self = (PyCuVec<T> *)obj;
  Py_DECREF(self);
}
/// class
template <class T> struct PyCuVec_t {
  PyBufferProcs as_buffer;
  PyTypeObject type_obj;
  PyCuVec_t()
      : as_buffer({
            (getbufferproc)PyCuVec_getbuffer<T>,
            (releasebufferproc)PyCuVec_release<T>,
        }),
        type_obj({
            PyVarObject_HEAD_INIT(NULL, 0) "cuvec.Vector", /* tp_name */
            sizeof(PyCuVec<T>),                            /* tp_basicsize */
            0,                                             /* tp_itemsize */
            (destructor)PyCuVec_dealloc<T>,                /* tp_dealloc */
            0,                                             /* tp_print */
            0,                                             /* tp_getattr */
            0,                                             /* tp_setattr */
            0,                                             /* tp_reserved */
            0,                                             /* tp_repr */
            0,                                             /* tp_as_number */
            0,                                             /* tp_as_sequence */
            0,                                             /* tp_as_mapping */
            0,                                             /* tp_hash  */
            0,                                             /* tp_call */
            (reprfunc)PyCuVec_str<T>,                      /* tp_str */
            0,                                             /* tp_getattro */
            0,                                             /* tp_setattro */
            &as_buffer,                                    /* tp_as_buffer */
            Py_TPFLAGS_DEFAULT,                            /* tp_flags */
            "cuvec.Vector object",                         /* tp_doc */
            0,                                             /* tp_traverse */
            0,                                             /* tp_clear */
            0,                                             /* tp_richcompare */
            0,                                             /* tp_weaklistoffset */
            0,                                             /* tp_iter */
            0,                                             /* tp_iternext */
            0,                                             /* tp_methods */
            0,                                             /* tp_members */
            0,                                             /* tp_getset */
            0,                                             /* tp_base */
            0,                                             /* tp_dict */
            0,                                             /* tp_descr_get */
            0,                                             /* tp_descr_set */
            0,                                             /* tp_dictoffset */
            (initproc)PyCuVec_init<T>,                     /* tp_init */
        }) {
    type_obj.tp_new = PyType_GenericNew;
  }
};

#endif // _PYCUVEC_H_
