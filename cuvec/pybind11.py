"""Thin wrappers around `cuvec_pybind11` C++/CUDA module"""
import logging
import re
from collections.abc import Sequence
from functools import partial
from textwrap import dedent
from typing import Any, Dict, Optional

import numpy as np

from . import cuvec_pybind11 as cu  # type: ignore [attr-defined] # yapf: disable
from ._common import Shape, _generate_helpers, typecodes

log = logging.getLogger(__name__)
RE_NDCUVEC_TYPE = r"<.*NDCuVec_(.) object at 0x\w+>"
NDCUVEC_TYPES = {
    "signed char": 'b',
    "unsigned char": 'B',
    "char": 'c',
    "short": 'h',
    "unsigned short": 'H',
    "int": 'i',
    "unsigned int": 'I',
    "long long": 'q',
    "unsigned long long": 'Q',
    "float": 'f',
    "double": 'd'} # yapf: disable
if hasattr(cu, 'NDCuVec_e'):
    typecodes += 'e'
    NDCUVEC_TYPES["__half"] = 'e'


class Pybind11Vector:
    def __init__(self, typechar: str, shape: Shape, cuvec=None):
        """
        Thin wrapper around `NDCuVec<Type>`. Always takes ownership.
        Args:
          typechar(str)
          shape(tuple(int))
          cuvec(NDCuVec<Type>): if given, `typechar` and `shape` are ignored
        """
        if cuvec is not None:
            assert is_raw_cuvec(cuvec)
            self.typechar = re.match(RE_NDCUVEC_TYPE, str(cuvec)).group(1) # type: ignore
            self.cuvec = cuvec
            return

        self.typechar = typechar
        shape = cu.Shape(shape if isinstance(shape, Sequence) else (shape,))
        self.cuvec = getattr(cu, f'NDCuVec_{typechar}')(shape)

    @property
    def shape(self) -> tuple:
        return tuple(self.cuvec.shape())

    @property
    def address(self) -> int:
        return self.cuvec.address()

    @property
    def __array_interface__(self) -> Dict[str, Any]:
        return {
            'shape': self.shape, 'typestr': np.dtype(self.typechar).str,
            'data': (self.address, False), 'version': 3}

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        return self.__array_interface__

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.typechar}', {self.shape})"

    def __str__(self) -> str:
        return f"{np.dtype(self.typechar)}{self.shape} at 0x{self.address:x}"


vec_types = {np.dtype(c): partial(Pybind11Vector, c) for c in typecodes}


def cu_zeros(shape: Shape, dtype="float32"):
    """
    Returns a new `Pybind11Vector` of the specified shape and data type.
    """
    return vec_types[np.dtype(dtype)](shape)


def cu_copy(arr):
    """
    Returns a new `Pybind11Vector` with data copied from the specified `arr`.
    """
    res = cu_zeros(arr.shape, arr.dtype)
    np.asarray(res).flat = arr.flat
    return res


def is_raw_cuvec(arr):
    return re.match(RE_NDCUVEC_TYPE, str(arr))


def is_raw_pyvec(arr):
    return isinstance(arr, Pybind11Vector) or type(arr).__name__ == "Pybind11Vector"


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `Pybind11Vector` object (for use in pybind11 API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.pybind11.CuVec`, raw `Pybind11Vector`, or `numpy.ndarray`"""
        if is_raw_pyvec(arr):
            log.debug("wrap pyraw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj.pyvec = arr
            obj.cuvec = arr.cuvec
            return obj
        if isinstance(arr, CuVec) and hasattr(arr, 'pyvec'):
            log.debug("new view")
            obj = np.asarray(arr).view(cls)
            obj.pyvec = arr.pyvec
            obj.cuvec = arr.pyvec.cuvec
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
        res = self.__array_interface__
        return {
            'shape': res['shape'], 'typestr': res['typestr'], 'data': res['data'], 'version': 3}


def zeros(shape: Shape, dtype="float32") -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(cu_zeros(shape, dtype))


ones, zeros_like, ones_like = _generate_helpers(zeros, CuVec)


def copy(arr) -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(cu_copy(arr))


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
    if is_raw_cuvec(arr):
        ownership = ownership.lower()
        if ownership in {'critical', 'fatal', 'error'}:
            raise IOError("Can't take ownership of existing cuvec (would create dangling ptr)")
        getattr(log, ownership)("taking ownership")
        arr = Pybind11Vector('', (), arr)
    if not isinstance(arr, np.ndarray) and is_raw_pyvec(arr):
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
