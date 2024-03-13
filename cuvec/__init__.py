"""
Unifying Python/C++/CUDA memory.

Python buffered array -> C++11 `std::vector` -> CUDA managed memory.
"""
__author__ = "Casper da Costa-Luis (https://github.com/casperdcl)"
__date__ = "2021-2024"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError: # pragma: nocover
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
# config
__all__ = ['cmake_prefix', 'include_path']

try:          # py<3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources  # type: ignore # yapf: disable

try:
    from .cuvec_cpython import dev_set, dev_sync
except ImportError as err: # pragma: no cover
    from warnings import warn
    warn(str(err), UserWarning)
else:                      # backwards compatibility: import from .cpython
    from .cpython import CuVec, asarray, copy, ones, ones_like, typecodes, zeros, zeros_like
    __all__ += [
        # classes
        'CuVec',
        # functions
        'dev_set', 'dev_sync', 'copy', 'asarray',
        'zeros', 'ones', 'zeros_like', 'ones_like',
        # data
        'typecodes'] # yapf: disable

p = resources.files('cuvec').resolve()
# for C++/CUDA/SWIG includes
include_path = p / 'include'
# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = p / 'cmake'
