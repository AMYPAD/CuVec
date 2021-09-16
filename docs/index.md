# CuVec

Unifying Python/C++/CUDA memory: Python buffered array <-> C++11 `std::vector` <-> CUDA managed memory.

[![Version](https://img.shields.io/pypi/v/cuvec.svg?logo=python&logoColor=white)](https://github.com/AMYPAD/CuVec/releases)
[![Downloads](https://img.shields.io/pypi/dm/cuvec.svg?logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/cuvec)
[![Py-Versions](https://img.shields.io/pypi/pyversions/cuvec.svg?logo=python&logoColor=white)](https://pypi.org/project/cuvec)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4446211.svg)](https://doi.org/10.5281/zenodo.4446211)
[![Licence](https://img.shields.io/pypi/l/cuvec.svg?label=licence)](https://github.com/AMYPAD/CuVec/blob/master/LICENCE)
[![Tests](https://img.shields.io/github/workflow/status/AMYPAD/CuVec/Test?logo=GitHub)](https://github.com/AMYPAD/CuVec/actions)
[![Coverage](https://codecov.io/gh/AMYPAD/CuVec/branch/master/graph/badge.svg)](https://codecov.io/gh/AMYPAD/CuVec)

## Why

Data should be manipulated using the existing functionality and design paradigms
of each programming language. Python code should be Pythonic. CUDA code should
be... CUDActic? C code should be... er, Clean.

However, in practice converting between data formats across languages can be a
pain.

Other libraries which expose functionality to convert/pass data formats between
these different language spaces tend to be bloated, unnecessarily complex, and
relatively unmaintainable. By comparison, `cuvec` uses the latest functionality
of Python, C/C++11, and CUDA to keep its code (and yours) as succinct as
possible. "Native" containers are exposed so your code follows the conventions
of your language. Want something which works like a `numpy.ndarray`? Not a
problem. Want to convert it to a `std::vector`? Or perhaps a raw `float *` to
use in a CUDA kernel? Trivial.

### Non objectives

Anything to do with mathematical functionality. The aim is to expose
functionality, not create it.

Even something as simple as setting element values is left to the user and/or
pre-existing features - for example:

- Python: `arr[:] = value`
- NumPy: `arr.fill(value)`
- CuPy: `cupy.asarray(arr).fill(value)`
- C++: `std::fill(vec.begin(), vec.end(), value)`
- C & CUDA: `memset(vec.data(), value, sizeof(T) * vec.size())`

## Install

Requirements:

- Python 3.6 or greater (e.g. via
  [Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda))
- [CUDA SDK/Toolkit](https://developer.nvidia.com/cuda-downloads) (including drivers for an NVIDIA GPU)

```sh
pip install cuvec
```

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

## External Projects

=== "Python"

    Python objects (`arr`, returned by `cuvec.zeros()`, `cuvec.asarray()`,
    or `cuvec.copy()`) contain all the attributes of a `numpy.ndarray`.
    Additionally, `arr.cuvec` implements the [buffer
    protocol](https://docs.python.org/3/c-api/buffer.html), while
    `arr.__cuda_array_interface__` provides [compatibility with other
    libraries](https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html)
    such as Numba, CuPy, PyTorch, PyArrow, and RAPIDS.

    When using the SWIG alternative module, `arr.cuvec` is a wrapper around
    `SwigCuVec<type> *`.

=== "C++ & CUDA"

    `cuvec` is a header-only library so simply `#include "pycuvec.cuh"` (or
    `#include "cuvec.cuh"`). You can find the location of the headers using:

    ```py
    python -c "import cuvec; print(cuvec.include_path)"
    ```

    For reference, see `cuvec.example_mod`\'s source code:
    [example_mod.cu](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_mod/example_mod.cu).

=== "SWIG"

    Using the include path from above, simply `%include "cuvec.i"` in a SWIG
    interface file.

    For reference, see `cuvec.example_swig`\'s source code:
    [example_swig.i](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_swig/example_swig.i)
    and
    [example_swig.cu](https://github.com/AMYPAD/CuVec/blob/master/cuvec/src/example_swig/example_swig.cu).

=== "CMake"

    This is likely unnecessary (see above for simpler `#include`
    instructions).

    The raw C++/CUDA libraries may be included in external projects using
    `cmake`. Simply build the project and use `find_package(AMYPADcuvec)`.

    ```sh
    # print installation directory (after `pip install cuvec`)...
    python -c "import cuvec; print(cuvec.cmake_prefix)"

    # ... or build & install directly with cmake
    mkdir build && cd build
    cmake ../cuvec && cmake --build . && cmake --install . --prefix /my/install/dir
    ```

    At this point any external project may include `cuvec` as follows (Once
    setting `-DCMAKE_PREFIX_DIR=<installation prefix from above>`):

    ```{.cmake linenums="1" hl_lines="3 5"}
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
    project(myproj)
    find_package(AMYPADcuvec COMPONENTS cuvec REQUIRED)
    add_executable(myexe ...)
    target_link_libraries(myexe PRIVATE AMYPAD::cuvec)
    ```
