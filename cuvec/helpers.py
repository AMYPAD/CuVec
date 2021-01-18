"""Useful helper functions."""
import logging
from textwrap import dedent

import numpy as np

from .pycuvec import cu_copy, cu_zeros, vec_types

log = logging.getLogger(__name__)


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `cuvec.Vector_*` object (for use in CPython API function calls).
    """
    _Vector_types = tuple(vec_types.values())

    def __new__(cls, arr, cuvec=None):
        """arr: `cuvec.CuVec`, raw `cuvec.Vector_*`, or `numpy.ndarray`"""
        if isinstance(arr, CuVec._Vector_types):
            log.debug("wrap raw %s", type(arr))
            obj = np.asarray(arr).view(cls)
            obj.cuvec = arr
            return obj
        if isinstance(arr, CuVec):
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
