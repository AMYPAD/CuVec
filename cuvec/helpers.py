"""Useful helper functions."""
import logging
from textwrap import dedent

import numpy as np

from .pycuvec import cu_copy, cu_zeros, vec_types

log = logging.getLogger(__name__)
_Vector_types = tuple(vec_types.values())
_Vector_types_s = tuple(map(str, vec_types.values()))


def is_raw_cuvec(cuvec):
    """
    Returns `True` when given the output of
    CPython API functions returning `PyCuVec<T> *` PyObjects.

    This is needed since conversely `isinstance(cuvec, CuVec)` may be `False`
    due to external libraries
    `#include "pycuvec.cuh"` making a distinct type object.
    """
    return isinstance(cuvec, _Vector_types) or str(type(cuvec)) in _Vector_types_s


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `cuvec.Vector_*` object (for use in CPython API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.CuVec`, raw `cuvec.Vector_*`, or `numpy.ndarray`"""
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
            (do not do `cuvec.CuVec((42, 1337))`;
            instead use `cuvec.zeros((42, 137))`"""))

    @property
    def __cuda_array_interface__(self):
        if not hasattr(self, 'cuvec'):
            raise AttributeError(
                dedent("""\
                `numpy.ndarray` object has no attribute `cuvec`:
                try using `cuvec.asarray()` first."""))
        res = self.__array_interface__
        return {
            'shape': res['shape'], 'typestr': res['typestr'], 'data': res['data'], 'version': 3}


def zeros(shape, dtype="float32"):
    """
    Returns a `cuvec.CuVec` view of a new `numpy.ndarray`
    of the specified shape and data type (`cuvec` equivalent of `numpy.zeros`).
    """
    return CuVec(cu_zeros(shape, dtype))


def copy(arr):
    """
    Returns a `cuvec.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    return CuVec(cu_copy(arr))


def asarray(arr, dtype=None, order=None):
    """
    Returns a `cuvec.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).
    """
    if not isinstance(arr, np.ndarray) and is_raw_cuvec(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))
