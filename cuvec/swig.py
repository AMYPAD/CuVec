"""
Thin wrappers around `cuvec_swig` C++/CUDA module

A SWIG-driven equivalent of the CPython Extension API-driven `cpython.py`
"""
import logging
import re
from collections.abc import Sequence
from functools import partial
from textwrap import dedent
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import cuvec_swig as sw  # type: ignore [attr-defined] # yapf: disable
from ._utils import CVector, Shape, _generate_helpers, typecodes

__all__ = [
    'CuVec', 'zeros', 'ones', 'zeros_like', 'ones_like', 'copy', 'asarray', 'retarray', 'Shape',
    'typecodes']
__author__, __date__, __version__ = sw.__author__, sw.__date__, sw.__version__

log = logging.getLogger(__name__)
if hasattr(sw, 'NDCuVec_e_new'):
    typecodes += 'e'


class SWIGVector(CVector):
    RE_CUVEC_TYPE = re.compile("<.*(?:ND|Swig)CuVec_(.); proxy of <Swig Object of type"
                               r" '(?:ND|Swig)CuVec<\s*(\w+)\s*>\s*\*' at 0x\w+>")

    def __init__(self, typechar: str, shape: Shape, cuvec=None):
        """
        Args:
          typechar(char)
          shape(tuple(int))
          cuvec(SwigPyObject<CuVec<Type>>): if given,
            `typechar` and `shape` are ignored
        """
        if cuvec is None:
            shape = shape if isinstance(shape, Sequence) else (shape,)
            cuvec = getattr(sw, f'NDCuVec_{typechar}_new')(shape)
        else:
            typechar = self.is_raw_cuvec(cuvec).group(1)
        self.cuvec = cuvec
        super().__init__(typechar)

    def __del__(self):
        getattr(sw, f'NDCuVec_{self.typechar}_del')(self.cuvec)

    @property
    def shape(self) -> tuple:
        return getattr(sw, f'NDCuVec_{self.typechar}_shape')(self.cuvec)

    @shape.setter
    def shape(self, shape: Shape):
        shape = shape if isinstance(shape, Sequence) else (shape,)
        getattr(sw, f'NDCuVec_{self.typechar}_reshape')(self.cuvec, shape)

    @property
    def address(self) -> int:
        return getattr(sw, f'NDCuVec_{self.typechar}_address')(self.cuvec)


SWIGVector.vec_types = {np.dtype(c): partial(SWIGVector, c) for c in typecodes}


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `SWIGVector` object (for use in CPython API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.swig.CuVec`, raw `SWIGVector`, or `numpy.ndarray`"""
        if SWIGVector.is_instance(arr):
            log.debug("wrap swraw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj._vec = arr
            obj.cuvec = arr.cuvec
            return obj
        if isinstance(arr, CuVec) and hasattr(arr, '_vec'):
            log.debug("new view")
            obj = np.asarray(arr).view(cls)
            obj._vec = arr._vec
            obj.cuvec = arr._vec.cuvec
            return obj
        if isinstance(arr, np.ndarray):
            log.debug("copy")
            return copy(arr)
        raise NotImplementedError(
            dedent("""\
            Not intended for explicit construction
            (do not do `cuvec.swig.CuVec((42, 1337))`;
            instead use `cuvec.swig.zeros((42, 137))`"""))

    __array_interface__: Dict[str,
                              Any] # <https://numpy.org/doc/stable/reference/arrays.interface.html>

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>"""
        if not hasattr(self, 'cuvec'):
            raise AttributeError(
                dedent("""\
                `numpy.ndarray` object has no attribute `cuvec`:
                try using `cuvec.asarray()` first."""))
        return self._vec.__cuda_array_interface__

    def resize(self, new_shape: Shape):
        """Change shape (but not size) of array in-place."""
        self._vec.shape = new_shape
        super().resize(new_shape, refcheck=False)

    @property
    def shape(self) -> Tuple[int, ...]:
        return super().shape

    @shape.setter
    def shape(self, new_shape: Shape):
        self.resize(new_shape)


def zeros(shape: Shape, dtype="float32") -> CuVec:
    """
    Returns a `cuvec.swig.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(SWIGVector.zeros(shape, dtype))


ones, zeros_like, ones_like = _generate_helpers(zeros, CuVec)


def copy(arr) -> CuVec:
    """
    Returns a `cuvec.swig.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(SWIGVector.copy(arr))


def asarray(arr, dtype=None, order=None, ownership: str = 'warning') -> CuVec:
    """
    Returns a `cuvec.swig.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).

    Args:
      ownership: logging level if `is_raw_cuvec(arr)`.
        WARNING: `asarray()` should not be used on an existing reference, e.g.:
        >>> res = asarray(some_swig_api_func(..., output=getattr(out, 'cuvec', None)))
        `res.cuvec` and `out.cuvec` are now the same
        yet garbage collected separately (dangling ptr).
        Instead, use the `retarray` helper:
        >>> raw = some_swig_api_func(..., output=getattr(out, 'cuvec', None))
        >>> res = retarray(raw, out)
        NB: `asarray()`/`retarray()` are safe if the raw cuvec was created in C++/SWIG, e.g.:
        >>> res = retarray(some_swig_api_func(..., output=None))
    """
    if SWIGVector.is_raw_cuvec(arr):
        ownership = ownership.lower()
        if ownership in {'critical', 'fatal', 'error'}:
            raise IOError("Can't take ownership of existing cuvec (would create dangling ptr)")
        getattr(log, ownership)("taking ownership")
        arr = SWIGVector('', (), arr)
    if not isinstance(arr, np.ndarray) and SWIGVector.is_instance(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))


def retarray(raw, out: Optional[CuVec] = None):
    """
    Returns `out if hasattr(out, 'cuvec') else asarray(raw, ownership='debug')`.
    See `asarray` for explanation.
    Args:
      raw: a raw CuVec (returned by C++/SWIG function).
      out: preallocated output array.
    """
    return out if hasattr(out, 'cuvec') else asarray(raw, ownership='debug')
