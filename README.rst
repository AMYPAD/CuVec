CuVec
=====

Unifying Python/C++/CUDA memory: Python buffered array ↔ C++11 ``std::vector`` ↔ CUDA managed memory.

|Version| |Downloads| |Py-Versions| |DOI| |Licence| |Tests| |Coverage|

.. contents:: Table of contents
   :backlinks: top
   :local:

Why
~~~

Data should be manipulated using the existing functionality and design paradigms of each programming language. Python code should be Pythonic. CUDA code should be... CUDActic? C code should be... er, Clean.

However, in practice converting between data formats across languages can be a pain.

Other libraries which expose functionality to convert/pass data formats between these different language spaces tend to be bloated, unnecessarily complex, and relatively unmaintainable. By comparison, ``cuvec`` uses the latest functionality of Python, C/C++11, and CUDA to keep its code (and yours) as succinct as possible. "Native" containers are exposed so your code follows the conventions of your language. Want something which works like a ``numpy.ndarray``? Not a problem. Want to convert it to a ``std::vector``? Or perhaps a raw ``float *`` to use in a CUDA kernel? Trivial.

- Less boilerplate code (fewer bugs, easier debugging, and faster prototyping)
- Fewer memory copies (faster execution)
- Lower memory usage (do more with less hardware)

Non objectives
--------------

Anything to do with mathematical functionality. The aim is to expose functionality, not create it.

Even something as simple as setting element values is left to the user and/or pre-existing features - for example:

- Python: ``arr[:] = value``
- NumPy: ``arr.fill(value)``
- CuPy: ``cupy.asarray(arr).fill(value)``
- C++: ``std::fill(vec.begin(), vec.end(), value)``
- C & CUDA: ``memset(vec.data(), value, sizeof(T) * vec.size())``

Install
~~~~~~~

Requirements:

- Python 3.6 or greater (e.g. via `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_)
- (optional) `CUDA SDK/Toolkit <https://developer.nvidia.com/cuda-downloads>`_ (including drivers for an NVIDIA GPU)

  * note that if the CUDA SDK/Toolkit is installed *after* CuVec, then CuVec must be re-installed to enable CUDA support

.. code:: sh

    pip install cuvec

Usage
~~~~~

See `the usage documentation <https://amypad.github.io/CuVec/#usage>`_ and `quick examples <https://amypad.github.io/CuVec/#examples>`_ of how to upgrade a Python ↔ C++ ↔ CUDA interface.

External Projects
~~~~~~~~~~~~~~~~~

For integration into Python, C++, CUDA, CMake, and general SWIG projects, see `the external project documentation <https://amypad.github.io/CuVec/#external-projects>`_.
Full and explicit example modules using the `CPython API <https://github.com/AMYPAD/CuVec/tree/master/cuvec/src/example_mod>`_ and `SWIG <https://github.com/AMYPAD/CuVec/tree/master/cuvec/src/example_swig>`_ are also provided.

Contributing
~~~~~~~~~~~~

See `CONTRIBUTING.md <https://github.com/AMYPAD/CuVec/blob/master/CONTRIBUTING.md>`_.

Licence
~~~~~~~

|Licence| |DOI|

Copyright 2021

- `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ University College London/King's College London
- `Contributors <https://github.com/AMYPAD/cuvec/graphs/contributors>`__

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4446211.svg
   :target: https://doi.org/10.5281/zenodo.4446211
.. |Licence| image:: https://img.shields.io/pypi/l/cuvec.svg?label=licence
   :target: https://github.com/AMYPAD/CuVec/blob/master/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/AMYPAD/CuVec/Test?logo=GitHub
   :target: https://github.com/AMYPAD/CuVec/actions
.. |Downloads| image:: https://img.shields.io/pypi/dm/cuvec.svg?logo=pypi&logoColor=white&label=PyPI%20downloads
   :target: https://pypi.org/project/cuvec
.. |Coverage| image:: https://codecov.io/gh/AMYPAD/CuVec/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AMYPAD/CuVec
.. |Version| image:: https://img.shields.io/pypi/v/cuvec.svg?logo=python&logoColor=white
   :target: https://github.com/AMYPAD/CuVec/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/cuvec.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/cuvec
