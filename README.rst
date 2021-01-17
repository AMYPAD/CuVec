cuvec
=====

Unifying Python/C++/CUDA memory: Python buffered array -> C++11 ``std::vector`` -> CUDA managed memory.

|Version| |Downloads| |Py-Versions| |Licence| |Tests| |Coverage|

Install
~~~~~~~

Requirements:

- Python 3.6 or greater (e.g. via `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_)
- `CUDA SDK/Toolkit <https://developer.nvidia.com/cuda-downloads>`_ (including drivers for an NVIDIA GPU)

.. code:: sh

    pip install cuvec

External CMake Projects
~~~~~~~~~~~~~~~~~~~~~~~

The raw C++/CUDA libraries may be included in external projects using ``cmake``.
Simply build the project and use ``find_package(AMYPADcuvec)``.

.. code:: sh

    # print installation directory (after `pip install cuvec`)...
    python -c "import cuvec; print(cuvec.cmake_prefix)"

    # ... or build & install directly with cmake
    mkdir build && cd build
    cmake ../cuvec && cmake --build . && cmake --install . --prefix /my/install/dir

At this point any external project may include ``cuvec`` as follows
(Once setting ``-DCMAKE_PREFIX_DIR=<installation prefix from above>``):

.. code:: cmake

    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
    project(myproj)
    find_package(AMYPADcuvec COMPONENTS cuvec REQUIRED)
    add_executable(myexe ...)
    target_link_libraries(myexe PRIVATE AMYPAD::cuvec)

Licence
~~~~~~~

|Licence|

Copyright 2021

- `Casper O. da Costa-Luis <https://github.com/casperdcl>`__ @ University College London/King's College London
- `Contributors <https://github.com/AMYPAD/cuvec/graphs/contributors>`__

.. |Licence| image:: https://img.shields.io/pypi/l/cuvec.svg?label=licence
   :target: https://github.com/AMYPAD/cuvec/blob/master/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/AMYPAD/cuvec/Test?logo=GitHub
   :target: https://github.com/AMYPAD/cuvec/actions
.. |Downloads| image:: https://img.shields.io/pypi/dm/cuvec.svg?logo=pypi&logoColor=white&label=PyPI%20downloads
   :target: https://pypi.org/project/cuvec
.. |Coverage| image:: https://codecov.io/gh/AMYPAD/cuvec/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AMYPAD/cuvec
.. |Version| image:: https://img.shields.io/pypi/v/cuvec.svg?logo=python&logoColor=white
   :target: https://github.com/AMYPAD/cuvec/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/cuvec.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/cuvec
