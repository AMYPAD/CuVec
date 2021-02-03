"""
Thin wrappers around `example_swig` C++/CUDA module

A SWIG-driven equivalent of the CPython Extension API-driven `pycuvec.py`
"""
import array
import logging
import re
from functools import partial
from textwrap import dedent

import numpy as np

from . import example_swig as sw

log = logging.getLogger(__name__)
# u: non-standard np.dype('S2'); l/L: inconsistent between `array` and `numpy`
typecodes = ''.join(i for i in array.typecodes if i not in "ulL") + "e"
RE_SWIG_TYPE = ("<Swig Object of type"
                r" 'std::vector<\s*(\w+)\s*,\s*CuAlloc<\s*\1\s*>\s*>\s*\*' at 0x\w+>")
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
    "__half": 'e',
    "float": 'f',
    "double": 'd'} # yapf: disable


class SWIGVector:
    def __init__(self, typechar, size, cuvec=None):
        """
        Thin wrapper around `SwigPyObject<CuVec<Type>>`. Always takes ownership.
        Args:
          typechar(char)
          size(tuple(int))
          cuvec(SwigPyObject<CuVec<Type>>): if given,
            `typechar` and `size` are ignored
        """
        if cuvec is not None:
            assert is_raw_cuvec(cuvec)
            self.typechar = SWIG_TYPES[re.match(RE_SWIG_TYPE, str(cuvec)).group(1)]
            self.cuvec = cuvec
            return
        self.typechar = typechar
        self.cuvec = getattr(sw, f'new_CuVec_{typechar}')(size)

    def __del__(self):
        getattr(sw, f'delete_CuVec_{self.typechar}')(self.cuvec)

    def __len__(self):
        return getattr(sw, f'CuVec_{self.typechar}___len__')(self.cuvec)

    def data(self):
        return getattr(sw, f'CuVec_{self.typechar}_data')(self.cuvec)

    @property
    def __array_interface__(self):
        return {
            'shape': (len(self),), 'typestr': np.dtype(self.typechar).str,
            'data': (self.data(), False), 'version': 3}

    @property
    def __cuda_array_interface__(self):
        return self.__array_interface__

    def __repr__(self):
        return f"{type(self).__name__}('{self.typechar}', {len(self)})"

    def __str__(self):
        return f"{np.dtype(self.typechar)}[{len(self)}] at 0x{self.data():x}"


vec_types = {np.dtype(c): partial(SWIGVector, c) for c in typecodes}


def cu_zeros(size, dtype="float32"):
    """
    Returns a new `SWIGVector` of the specified size and data type.
    """
    return vec_types[np.dtype(dtype)](size)


def cu_copy(arr):
    """
    Returns a new `SWIGVector` with data copied from the specified `arr`.
    """
    res = cu_zeros(arr.size, arr.dtype)
    np.asarray(res).flat = arr.flat
    return res


def is_raw_cuvec(arr):
    return type(arr).__name__ == "SwigPyObject" and re.match(RE_SWIG_TYPE, str(arr))


def is_raw_swvec(arr):
    return isinstance(arr, SWIGVector) or type(arr).__name__ == "SWIGVector"


class CuVec(np.ndarray):
    """
    A `numpy.ndarray` compatible view with a `cuvec` member containing the
    underlying `SWIGVector` object (for use in CPython API function calls).
    """
    def __new__(cls, arr):
        """arr: `cuvec.CuVec`, raw `SWIGVector`, or `numpy.ndarray`"""
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
    res = CuVec(cu_zeros(int(np.prod(shape)), dtype))
    res.resize(shape)
    return res


def copy(arr):
    """
    Returns a `cuvec.CuVec` view of a new `numpy.ndarray`
    with data copied from the specified `arr`
    (`cuvec` equivalent of `numpy.copy`).
    """
    res = CuVec(cu_copy(arr))
    res.resize(arr.shape)
    return res


def asarray(arr, dtype=None, order=None):
    """
    Returns a `cuvec.CuVec` view of `arr`, avoiding memory copies if possible.
    (`cuvec` equivalent of `numpy.asarray`).
    """
    if is_raw_cuvec(arr):
        log.debug("taking ownership")
        arr = SWIGVector(None, None, arr)
    if not isinstance(arr, np.ndarray) and is_raw_swvec(arr):
        res = CuVec(arr)
        if dtype is None or res.dtype == np.dtype(dtype):
            return CuVec(np.asanyarray(res, order=order))
    return CuVec(np.asanyarray(arr, dtype=dtype, order=order))
