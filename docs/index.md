# CuVec

Unifying Python/C++/CUDA memory: Python buffered array ↔ C++11 `std::vector` ↔ CUDA managed memory.

[![Version](https://img.shields.io/pypi/v/cuvec.svg?logo=python&logoColor=white)](https://github.com/AMYPAD/CuVec/releases)
[![Downloads](https://img.shields.io/pypi/dm/cuvec.svg?logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/cuvec)
[![Py-Versions](https://img.shields.io/pypi/pyversions/cuvec.svg?logo=python&logoColor=white)](https://pypi.org/project/cuvec)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4446211.svg)](https://doi.org/10.5281/zenodo.4446211)
[![Licence](https://img.shields.io/pypi/l/cuvec.svg?label=licence)](https://github.com/AMYPAD/CuVec/blob/master/LICENCE)
[![Tests](https://img.shields.io/github/workflow/status/AMYPAD/CuVec/Test?logo=GitHub)](https://github.com/AMYPAD/CuVec/actions)
[![Coverage](https://codecov.io/gh/AMYPAD/CuVec/branch/master/graph/badge.svg)](https://codecov.io/gh/AMYPAD/CuVec)

## Why

Data should be manipulated using the existing functionality and design paradigms of each programming language. Python code should be Pythonic. CUDA code should be... CUDActic? C code should be... er, Clean.

However, in practice converting between data formats across languages can be a pain.

Other libraries which expose functionality to convert/pass data formats between these different language spaces tend to be bloated, unnecessarily complex, and relatively unmaintainable. By comparison, `cuvec` uses the latest functionality of Python, C/C++11, and CUDA to keep its code (and yours) as succinct as possible. "Native" containers are exposed so your code follows the conventions of your language. Want something which works like a `numpy.ndarray`? Not a problem. Want to convert it to a `std::vector`? Or perhaps a raw `float *` to use in a CUDA kernel? Trivial.

- Less boilerplate code (fewer bugs, easier debugging, and faster prototyping)
- Fewer memory copies (faster execution)
- Lower memory usage (do more with less hardware)

### Non objectives

Anything to do with mathematical functionality. The aim is to expose functionality, not create it.

Even something as simple as setting element values is left to the user and/or pre-existing features - for example:

- Python: `arr[:] = value`
- NumPy: `arr.fill(value)`
- CuPy: `cupy.asarray(arr).fill(value)`
- C++: `std::fill(vec.begin(), vec.end(), value)`
- C & CUDA: `memset(vec.data(), value, sizeof(T) * vec.size())`

## Install

```sh
pip install cuvec
```

Requirements:

- Python 3.6 or greater (e.g. via [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda))
- (optional) [CUDA SDK/Toolkit](https://developer.nvidia.com/cuda-downloads) (including drivers for an NVIDIA GPU)
  + note that if the CUDA SDK/Toolkit is installed *after* CuVec, then CuVec must be re-installed to enable CUDA support

## Usage

### Creating

=== "Python"
    ```py
    import cuvec
    # from cuvec import swigcuvec as cuvec   # SWIG alternative
    arr = cuvec.zeros((1337, 42), "float32") # like `numpy.ndarray`
    # print(sum(arr))
    # some_numpy_func(arr)
    # some_cpython_api_func(arr.cuvec)
    # import cupy; cupy_arr = cupy.asarray(arr)
    ```

=== "CPython API"
    ```cpp
    #include "Python.h"
    #include "pycuvec.cuh"
    PyObject *obj = (PyObject *)PyCuVec_zeros<float>({1337, 42});
    // don't forget to Py_DECREF(obj) if not returning it.

    /// N.B.: convenience functions provided by "pycuvec.cuh":
    // PyCuVec<T> *PyCuVec_zeros(std::vector<Py_ssize_t> shape);
    // PyCuVec<T> *PyCuVec_zeros_like(PyCuVec<T> *other);
    // PyCuVec<T> *PyCuVec_deepcopy(PyCuVec<T> *other);
    ```

=== "C++/SWIG API"
    ```cpp
    #include "cuvec.cuh"
    SwigCuVec<float> *swv = SwigCuVec_new<float>({1337, 42});

    /// N.B.: convenience functions provided by "cuvec.cuh":
    // SwigCuVec<T> *SwigCuVec_new(std::vector<size_t> shape);
    // void SwigCuVec_del(SwigCuVec<T> *swv);
    // T *SwigCuVec_data(SwigCuVec<T> *swv);
    // size_t SwigCuVec_address(SwigCuVec<T> *swv);
    // std::vector<size_t> SwigCuVec_shape(SwigCuVec<T> *swv);
    ```

=== "C++/CUDA"
    ```cpp
    #include "cuvec.cuh"
    CuVec<float> vec(1337 * 42); // like std::vector<float>
    ```

### Converting

The following involve no memory copies.

=== "**Python** to **CPython API**"
    ```py
    # import cuvec, my_custom_lib
    # arr = cuvec.zeros((1337, 42), "float32")
    my_custom_lib.some_cpython_api_func(arr.cuvec)
    ```

=== "**CPython API** to **Python**"
    ```py
    import cuvec, my_custom_lib
    arr = cuvec.asarray(my_custom_lib.some_cpython_api_func())
    ```

=== "**CPython API** to **C++**"
    ```cpp
    /// input: `PyObject *obj` (obtained from e.g.: `PyArg_ParseTuple()`, etc)
    /// output: `CuVec<type> vec`
    CuVec<float> &vec = ((PyCuVec<float> *)obj)->vec; // like std::vector<float>
    std::vector<Py_ssize_t> &shape = ((PyCuVec<float> *)obj)->shape;
    ```

=== "**C++** to **C & CUDA**"
    ```cpp
    /// input: `CuVec<type> vec`
    /// output: `type *arr`
    float *arr = vec.data(); // pointer to `cudaMallocManaged()` data
    ```

=== "**Python** to **SWIG API**"
    ```py
    # import cuvec, my_custom_lib
    # arr = cuvec.swigcuvec.zeros((1337, 42), "float32")
    my_custom_lib.some_swig_api_func(arr.cuvec)
    ```

=== "**SWIG API** to **Python**"
    ```py
    import cuvec, my_custom_lib
    arr = cuvec.swigcuvec.asarray(my_custom_lib.some_swig_api_func())
    ```

=== "**SWIG API** to **C++**"
    ```cpp
    /// input: `SwigCuVec<type> *swv`
    /// output: `CuVec<type> vec`, `std::vector<size_t> shape`
    CuVec<float> &vec = swv->vec; // like std::vector<float>
    std::vector<size_t> &shape = swv->shape;
    ```

### Examples

Here's a before and after comparison of a Python ↔ CUDA interface.

Python:

=== "Before: pure NumPy"
    ```{.py linenums="1"}
    import numpy, mymod
    arr = numpy.zeros((1337, 42, 7), "float32")
    assert all(numpy.mean(arr, axis=(0, 1)) == 0)
    print(mymod.myfunc(arr).sum())
    ```

=== "After: with CuVec"
    ```{.py linenums="1"}
    import cuvec, numpy, mymod
    arr = cuvec.zeros((1337, 42, 7), "float32")
    assert all(numpy.mean(arr, axis=(0, 1)) == 0)
    print(mymod.myfunc(arr.cuvec).sum())
    ```

C++:

=== "Before: pure NumPy"
    ```{.cpp linenums="1"}
    #include <numpy/arrayobject.h>
    #include "mycudafunction.h"

    static PyObject *myfunc(PyObject *self, PyObject *args, PyObject *kwargs) {
      PyObject *o_src = NULL;
      PyObject *o_dst = NULL;
      static const char *kwds[] = {"src", "output", NULL};
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char **)kwds,
                                       &o_src, &o_dst))
        return NULL;
      PyArrayObject *p_src = (PyArrayObject *)PyArray_FROM_OTF(
        o_src, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
      if (p_src == NULL) return NULL;
      float *src = (float *)PyArray_DATA(p_src);

      // hardcode upsampling factor 2
      npy_intp *src_shape = PyArray_SHAPE(p_src);
      Py_ssize_t dst_shape[3];
      dst_shape[2] = src_shape[2] * 2;
      dst_shape[1] = src_shape[1] * 2;
      dst_shape[0] = src_shape[0] * 2;

      PyArrayObject *p_dst = NULL;
      if (o_dst == NULL)
        p_dst = (PyArrayObject *)PyArray_ZEROS(3, dst_shape, NPY_FLOAT32, 0);
      else
        p_dst = (PyArrayObject *)PyArray_FROM_OTF(
          o_dst, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY);
      if (p_dst == NULL) return NULL;
      float *dst = (float *)PyArray_DATA(p_dst);

      mycudafunction(dst, src, dst_shape);

      Py_DECREF(p_src);
      return PyArray_Return(p_dst);
    }
    ...
    PyMODINIT_FUNC PyInit_mymod(void) {
      ...
      import_array();
      ...
    }
    ```

=== "After: with CuVec"
    ```{.cpp linenums="1"}
    #include "pycuvec.cuh"
    #include "mycudafunction.h"

    static PyObject *myfunc(PyObject *self, PyObject *args, PyObject *kwargs) {
      PyCuVec<float> *src = NULL;
      PyCuVec<float> *dst = NULL;
      static const char *kwds[] = {"src", "output", NULL};
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char **)kwds,
                                       (PyObject **)&src, (PyObject **)&dst))
        return NULL;


      if (!src) return NULL;


      // hardcode upsampling factor 2

      std::vector<Py_ssize_t> dst_shape = src->shape();
      dst_shape[2] *= 2;
      dst_shape[1] *= 2;
      dst_shape[0] *= 2;


      if (!dst)
        dst = PyCuVec_zeros<float>(dst_shape);



      if (!dst) return NULL;


      mycudafunction(dst->vec.data(), src->vec.data(), dst_shape.data());


      return dst;
    }
    ...
    PyMODINIT_FUNC PyInit_mymod(void) {
      ...

      ...
    }
    ```

CUDA:

=== "Before: pure NumPy"
    ```{.cpp linenums="1"}
    void mycudafunction(float *dst, float *src, Py_ssize_t *shape) {
      float *d_src;
      int src_size = shape[0]/2 * shape[1]/2 * shape[2]/2 * sizeof(float);
      cudaMalloc(&d_src, src_size);
      cudaMemcpy(d_src, src, src_size, cudaMemcpyHostToDevice);
      float *d_dst;
      cudaMalloc(&d_dst, shape[0] * shape[1] * shape[2] * sizeof(float));
      mykernel<<<...>>>(d_dst, d_src, shape[0], shape[1], shape[2]);
      cudaMemcpy(dst, d_dst, cudaMemcpyDeviceToHost);
      cudaFree(d_dst);
      cudaFree(d_src);
    }
    ```

=== "After: with CuVec"
    ```{.cpp linenums="1"}
    void mycudafunction(float *dst, float *src, Py_ssize_t *shape) {






      mykernel<<<...>>>(dst, src, shape[0], shape[1], shape[2]);
      cudaDeviceSynchronize();


    }
    ```

For a full reference, see `cuvec.example_mod`'s source code: [example_mod.cu](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_mod/example_mod.cu).

## External Projects

=== "Python"
    Python objects (`arr`, returned by `cuvec.zeros()`, `cuvec.asarray()`, or `cuvec.copy()`) contain all the attributes of a `numpy.ndarray`. Additionally, `arr.cuvec` implements the [buffer protocol](https://docs.python.org/3/c-api/buffer.html), while `arr.__cuda_array_interface__` provides [compatibility with other libraries](https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html) such as Numba, CuPy, PyTorch, PyArrow, and RAPIDS.

    When using the SWIG alternative module, `arr.cuvec` is a wrapper around `SwigCuVec<type> *`.

=== "C++ & CUDA"
    `cuvec` is a header-only library so simply `#include "pycuvec.cuh"` (or `#include "cuvec.cuh"`). You can find the location of the headers using:

    ```py
    python -c "import cuvec; print(cuvec.include_path)"
    ```

    For reference, see `cuvec.example_mod`'s source code: [example_mod.cu](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_mod/example_mod.cu).

=== "SWIG"
    `cuvec` is a header-only library so simply `%include "cuvec.i"` in a SWIG interface file. You can find the location of the headers using:

    ```py
    python -c "import cuvec; print(cuvec.include_path)"
    ```

    For reference, see `cuvec.example_swig`'s source code: [example_swig.i](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_swig/example_swig.i) and [example_swig.cu](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_swig/example_swig.cu).

=== "CMake"
    This is likely unnecessary (see the "C++ & CUDA" tab above for simpler `#include` instructions).

    The raw C++/CUDA libraries may be included in external projects using `cmake`. Simply build the project and use `find_package(AMYPADcuvec)`.

    ```sh
    # print installation directory (after `pip install cuvec`)...
    python -c "import cuvec; print(cuvec.cmake_prefix)"

    # ... or build & install directly with cmake
    mkdir build && cd build
    cmake ../cuvec && cmake --build . && cmake --install . --prefix /my/install/dir
    ```

    At this point any external project may include `cuvec` as follows (Once setting `-DCMAKE_PREFIX_DIR=<installation prefix from above>`):

    ```{.cmake linenums="1" hl_lines="3 6"}
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
    project(myproj)
    find_package(AMYPADcuvec COMPONENTS cuvec REQUIRED)
    add_executable(myexe ...)
    set_target_properties(myexe PROPERTIES CXX_STANDARD 11)
    target_link_libraries(myexe PRIVATE AMYPAD::cuvec)
    ```
