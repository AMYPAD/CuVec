"""
Thin wrappers around `swvec` C++/CUDA module

A SWIG-driven equivalent of the CPython Extension API-driven `cpython.py`
"""
import logging
import re
from collections.abc import Sequence
from functools import partial
from textwrap import dedent
from typing import Any, Dict, Optional

import numpy as np

from . import swvec as sw  # type: ignore [attr-defined] # yapf: disable
from ._common import Shape, _generate_helpers, typecodes

log = logging.getLogger(__name__)
RE_SWIG_TYPE = ("<.*SwigCuVec_(.); proxy of <Swig Object of type"
                r" 'SwigCuVec<\s*(\w+)\s*>\s*\*' at 0x\w+>")
SWIG_TYPES = {
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
if hasattr(sw, 'SwigCuVec_e_new'):
    typecodes += 'e'
    SWIG_TYPES["__half"] = 'e'


class SWIGVector:
    def __init__(self, typechar: str, shape: Shape, cuvec=None):
        """
        Thin wrapper around `SwigPyObject<CuVec<Type>>`. Always takes ownership.
        Args:
          typechar(char)
          shape(tuple(int))
          cuvec(SwigPyObject<CuVec<Type>>): if given,
            `typechar` and `shape` are ignored
        """
        if cuvec is not None:
            assert is_raw_cuvec(cuvec)
            self.typechar = re.match(RE_SWIG_TYPE, str(cuvec)).group(1) # type: ignore
            self.cuvec = cuvec
            return

        self.typechar = typechar # type: ignore
        self.cuvec = getattr(
            sw, f'SwigCuVec_{typechar}_new')(shape if isinstance(shape, Sequence) else (shape,))

    def __del__(self):
        getattr(sw, f'SwigCuVec_{self.typechar}_del')(self.cuvec)

    @property
    def shape(self) -> tuple:
        return getattr(sw, f'SwigCuVec_{self.typechar}_shape')(self.cuvec)

    @property
    def address(self) -> int:
        return getattr(sw, f'SwigCuVec_{self.typechar}_address')(self.cuvec)

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


vec_types = {np.dtype(c): partial(SWIGVector, c) for c in typecodes}


def cu_zeros(shape: Shape, dtype="float32"):
    """
    Returns a new `SWIGVector` of the specified shape and data type.
    """
    return vec_types[np.dtype(dtype)](shape)


def cu_copy(arr):
    """
    Returns a new `SWIGVector` with data copied from the specified `arr`.
    """
    res = cu_zeros(arr.shape, arr.dtype)
    np.asarray(res).flat = arr.flat
    return res


def is_raw_cuvec(arr):
    return re.match(RE_SWIG_TYPE, str(arr))


def is_raw_swvec(arr):
    return isinstance(arr, SWIGVector) or type(arr).__name__ == "SWIGVector"


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `SWIGVector` object (for use in CPython API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.swig.CuVec`, raw `SWIGVector`, or `numpy.ndarray`"""
        if is_raw_swvec(arr):
            log.debug("wrap swraw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj.swvec = arr
            obj.cuvec = arr.cuvec
            return obj
        if isinstance(arr, CuVec) and hasattr(arr, 'swvec'):
            log.debug("new view")
            obj = np.asarray(arr).view(cls)
            obj.swvec = arr.swvec
            obj.cuvec = arr.swvec.cuvec
            return obj
        if isinstance(arr, np.ndarray):
            log.debug("copy")
            return copy(arr)
        raise NotImplementedError(
            dedent("""\
            Not intended for explicit construction
            (do not do `cuvec.swig.CuVec((42, 1337))`;
            instead use `cuvec.swig.zeros((42, 137))`"""))

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
    Returns a `cuvec.swig.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(cu_zeros(shape, dtype))


ones, zeros_like, ones_like = _generate_helpers(zeros, CuVec)


def copy(arr) -> CuVec:
    """
    Returns a `cuvec.swig.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(cu_copy(arr))


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
    if is_raw_cuvec(arr):
        ownership = ownership.lower()
        if ownership in {'critical', 'fatal', 'error'}:
            raise IOError("Can't take ownership of existing cuvec (would create dangling ptr)")
        getattr(log, ownership)("taking ownership")
        arr = SWIGVector('', (), arr)
    if not isinstance(arr, np.ndarray) and is_raw_swvec(arr):
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