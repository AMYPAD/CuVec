"""Useful helper functions."""
from collections.abc import Sequence

import numpy as np

from .pycuvec import vec_types


def zeros(shape, dtype="float32"):
    """
    Returns a new `Vector_*` of the specified shape and data type
    (`cuvec` equivalent of `numpy.zeros`).
    """
    return vec_types[np.dtype(dtype)](shape if isinstance(shape, Sequence) else (shape,))


def from_numpy(arr):
    """
    Returns a new `Vector_*` of the specified shape and data type
    (`cuvec` equivalent of `numpy.copy`).
    """
    res = zeros(arr.shape, arr.dtype)
    np.asarray(res)[:] = arr[:]
    return res
