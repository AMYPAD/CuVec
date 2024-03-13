"""
Thin wrappers around `cuvec_pybind11` C++/CUDA module

A pybind11-driven equivalent of the CPython Extension API-driven `cpython.py`
"""
import logging
from collections.abc import Sequence
from textwrap import dedent
from typing import Any, Dict, Tuple

import numpy as np

from . import cuvec_pybind11 as cu  # type: ignore [attr-defined] # yapf: disable
from ._utils import Shape, _generate_helpers, typecodes

__all__ = [
    'CuVec', 'zeros', 'ones', 'zeros_like', 'ones_like', 'copy', 'asarray', 'Shape', 'typecodes']
__author__, __date__, __version__ = cu.__author__, cu.__date__, cu.__version__

log = logging.getLogger(__name__)
vec_types = {
    np.dtype('int8'): cu.NDCuVec_b,
    np.dtype('uint8'): cu.NDCuVec_B,
    np.dtype('S1'): cu.NDCuVec_c,
    np.dtype('int16'): cu.NDCuVec_h,
    np.dtype('uint16'): cu.NDCuVec_H,
    np.dtype('int32'): cu.NDCuVec_i,
    np.dtype('uint32'): cu.NDCuVec_I,
    np.dtype('int64'): cu.NDCuVec_q,
    np.dtype('uint64'): cu.NDCuVec_Q,
    np.dtype('float32'): cu.NDCuVec_f,
    np.dtype('float64'): cu.NDCuVec_d}
if hasattr(cu, 'NDCuVec_e'):
    typecodes += 'e'
    vec_types[np.dtype('float16')] = cu.NDCuVec_e


def cu_zeros(shape: Shape, dtype="float32"):
    """
    Returns a new `<cuvec.cuvec_pybind11.NDCuVec_*>` of the specified shape and data type.
    """
    return vec_types[np.dtype(dtype)](shape if isinstance(shape, Sequence) else (shape,))


def cu_copy(arr):
    """
    Returns a new `<cuvec.cuvec_pybind11.NDCuVec_*>` with data copied from the specified `arr`.
    """
    res = cu_zeros(arr.shape, arr.dtype)
    np.asarray(res).flat = arr.flat
    return res


_NDCuVec_types = tuple(vec_types.values())
_NDCuVec_types_s = tuple(map(str, vec_types.values()))


def is_raw_cuvec(cuvec):
    """
    Returns `True` when given the output of
    pybind11 API functions returning `NDCuVec<T> *` PyObjects.

    This is needed since conversely `isinstance(cuvec, CuVec)` may be `False`
    due to external libraries
    `#include "cuvec_pybind11.cuh"` making a distinct type object.
    """
    return isinstance(cuvec, _NDCuVec_types) or str(type(cuvec)) in _NDCuVec_types_s


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `cuvec.cuvec_pybind11.NDCuVec_*` object (for use in pybind11 API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.pybind11.CuVec`, raw `cuvec.cuvec_pybind11.NDCuVec_*`, or `numpy.ndarray`"""
        if is_raw_cuvec(arr):
            log.debug("wrap raw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj.cuvec = arr
            return obj
        if isinstance(arr, CuVec) and hasattr(arr, 'cuvec'):
            log.debug("new view")
            obj = np.asarray(arr).view(cls)
            obj.cuvec = arr.cuvec
            return obj
        if isinstance(arr, np.ndarray):
            log.debug("copy")
            return copy(arr)
        raise NotImplementedError(
            dedent("""\
            Not intended for explicit construction
            (do not do `cuvec.pybind11.CuVec((42, 1337))`;
            instead use `cuvec.pybind11.zeros((42, 137))`"""))

    __array_interface__: Dict[str,
                              Any] # <https://numpy.org/doc/stable/reference/arrays.interface.html>

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        """<https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>"""
        if not hasattr(self, 'cuvec'):
            raise AttributeError(
                dedent("""\
                `numpy.ndarray` object has no attribute `cuvec`:
                try using `cuvec.pybind11.asarray()` first."""))
        res = self.__array_interface__
        return {
            'shape': res['shape'], 'typestr': res['typestr'], 'data': res['data'], 'version': 3}

    def resize(self, new_shape: Shape):
        """Change shape (but not size) of array in-place."""
        self.cuvec.shape = new_shape if isinstance(new_shape, Sequence) else (new_shape,)
        super().resize(new_shape, refcheck=False)

    @property
    def shape(self) -> Tuple[int, ...]:
        return super().shape

    @shape.setter
    def shape(self, new_shape: Shape):
        self.resize(new_shape)


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


def asarray(arr, dtype=None, order=None) -> CuVec:
    """
    Returns a `cuvec.pybind11.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).
    """
    if not isinstance(arr, np.ndarray) and is_raw_cuvec(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))
