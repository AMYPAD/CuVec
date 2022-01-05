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
    'dev_set', 'dev_sync', 'cu_copy', 'cu_zeros',
    'copy', 'asarray',
    'zeros', 'ones', 'zeros_like', 'ones_like',
    # data
    'typecodes', 'vec_types'] # yapf: disable

try:          # py<3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources

try:
    from .cuvec import dev_set, dev_sync
except ImportError as err: # pragma: no cover
    from warnings import warn
    warn(str(err), UserWarning)
else:
    from .pycuvec import (
        CuVec,
        asarray,
        copy,
        cu_copy,
        cu_zeros,
        ones,
        ones_like,
        typecodes,
        vec_types,
        zeros,
        zeros_like,
    )

p = resources.files('cuvec').resolve()
# for C++/CUDA/SWIG includes
include_path = p / 'include'
# for use in `cmake -DCMAKE_PREFIX_PATH=...`
cmake_prefix = p / 'cmake'
