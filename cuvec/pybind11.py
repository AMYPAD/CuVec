"""
Thin wrappers around `cuvec_pybind11` C++/CUDA module

A pybind11-driven equivalent of the CPython Extension API-driven `cpython.py`
"""
import logging
import re
from collections.abc import Sequence
from functools import partial
from textwrap import dedent
from typing import Any, Dict, Optional

import numpy as np

from . import cuvec_pybind11 as cu  # type: ignore [attr-defined] # yapf: disable
from ._utils import CVector, Shape, _generate_helpers, typecodes

__all__ = [
    'CuVec', 'zeros', 'ones', 'zeros_like', 'ones_like', 'copy', 'asarray', 'retarray', 'Shape',
    'typecodes']

log = logging.getLogger(__name__)
if hasattr(cu, 'NDCuVec_e'):
    typecodes += 'e'


class Pybind11Vector(CVector):
    RE_CUVEC_TYPE = re.compile(r"<.*NDCuVec_(.) object at 0x\w+>")

    def __init__(self, typechar: str, shape: Shape, cuvec=None):
        """
        Args:
          typechar(char)
          shape(tuple(int))
          cuvec(NDCuVec<Type>): if given, `typechar` and `shape` are ignored
        """
        if cuvec is None:
            shape = cu.Shape(shape if isinstance(shape, Sequence) else (shape,))
            cuvec = getattr(cu, f'NDCuVec_{typechar}')(shape)
        else:
            typechar = self.is_raw_cuvec(cuvec).group(1)
        self.cuvec = cuvec
        super().__init__(typechar)

    @property
    def shape(self) -> tuple:
        return tuple(self.cuvec.shape)

    @shape.setter
    def shape(self, shape: Shape):
        shape = cu.Shape(shape if isinstance(shape, Sequence) else (shape,))
        self.cuvec.shape = shape

    @property
    def address(self) -> int:
        return self.cuvec.address


Pybind11Vector.vec_types = {np.dtype(c): partial(Pybind11Vector, c) for c in typecodes}


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `Pybind11Vector` object (for use in pybind11 API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.pybind11.CuVec`, raw `Pybind11Vector`, or `numpy.ndarray`"""
        if Pybind11Vector.is_instance(arr):
            log.debug("wrap pyraw %s", type(arr))
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
            (do not do `cuvec.pybind11.CuVec((42, 1337))`;
            instead use `cuvec.pybind11.zeros((42, 137))`"""))

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
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


def zeros(shape: Shape, dtype="float32") -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(Pybind11Vector.zeros(shape, dtype))


ones, zeros_like, ones_like = _generate_helpers(zeros, CuVec)


def copy(arr) -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(Pybind11Vector.copy(arr))


def asarray(arr, dtype=None, order=None, ownership: str = 'warning') -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).

    Args:
      ownership: logging level if `is_raw_cuvec(arr)`.
        WARNING: `asarray()` should not be used on an existing reference, e.g.:
        >>> res = asarray(some_pybind11_api_func(..., output=getattr(out, 'cuvec', None)))
        `res.cuvec` and `out.cuvec` are now the same
        yet garbage collected separately (dangling ptr).
        Instead, use the `retarray` helper:
        >>> raw = some_pybind11_api_func(..., output=getattr(out, 'cuvec', None))
        >>> res = retarray(raw, out)
        NB: `asarray()`/`retarray()` are safe if the raw cuvec was created in C++, e.g.:
        >>> res = retarray(some_pybind11_api_func(..., output=None))
    """
    if Pybind11Vector.is_raw_cuvec(arr):
        ownership = ownership.lower()
        if ownership in {'critical', 'fatal', 'error'}:
            raise IOError("Can't take ownership of existing cuvec (would create dangling ptr)")
        getattr(log, ownership)("taking ownership")
        arr = Pybind11Vector('', (), arr)
    if not isinstance(arr, np.ndarray) and Pybind11Vector.is_instance(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))


def retarray(raw, out: Optional[CuVec] = None):
    """
    Returns `out if hasattr(out, 'cuvec') else asarray(raw, ownership='debug')`.
    See `asarray` for explanation.
    Args:
      raw: a raw CuVec (returned by C++/pybind11 function).
      out: preallocated output array.
    """
    return out if hasattr(out, 'cuvec') else asarray(raw, ownership='debug')
