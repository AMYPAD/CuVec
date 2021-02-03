"""
Unifying Python/C++/CUDA memory.

Python buffered array -> C++11 `std::vector` -> CUDA managed memory.
"""
__author__ = "Casper O. da Costa-Luis"
__date__ = "2021"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError: # pragma: nocover
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"
__all__ = [
    # config
    'cmake_prefix', 'include_path',
    # classes
    'CuVec',
    # functions
    'dev_sync', 'copy', 'zeros', 'asarray', 'cu_copy', 'cu_zeros',
    # data
    'typecodes', 'vec_types'] # yapf: disable

from pathlib import Path

from pkg_resources import resource_filename

try:
    from .cuvec import dev_sync
except ImportError as err: # pragma: no cover
    from warnings import warn
    warn(str(err), UserWarning)
else:
    from .pycuvec import CuVec, asarray, copy, cu_copy, cu_zeros, typecodes, vec_types, zeros

# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = Path(resource_filename(__name__, "cmake")).resolve()
include_path = Path(resource_filename(__name__, "include")).resolve()
