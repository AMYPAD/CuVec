"""Common helpers for cuvec.{cpython,pybind11,swig} modules."""
import array
import re
from abc import ABC, abstractmethod
from typing import Any, Dict
from typing import Sequence as Seq
from typing import Union

import numpy as np

Shape = Union[Seq[int], int]
# u: non-standard np.dype('S2'); l/L: inconsistent between `array` and `numpy`
typecodes = ''.join(i for i in array.typecodes if i not in "ulL")


class CVector(ABC):
    """Thin wrapper around `CuVec<Type>`. Always takes ownership."""
    vec_types: Dict[np.dtype, Any]
    RE_CUVEC_TYPE: re.Pattern

    def __init__(self, typechar: str):
        self.typechar = typechar

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def address(self) -> int:
        pass  # pragma: no cover

    @property
    def __array_interface__(self) -> Dict[str, Any]:
        return {
            'shape': self.shape, 'typestr': np.dtype(self.typechar).str,
            'data': (self.address, False), 'version': 3}

    __cuda_array_interface__ = __array_interface__

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.typechar}', {self.shape})"

    def __str__(self) -> str:
        return f"{np.dtype(self.typechar)}{self.shape} at 0x{self.address:x}"

    @classmethod
    def zeros(cls, shape: Shape, dtype="float32"):
        """Returns a new Vector of the specified shape and data type."""
        return cls.vec_types[np.dtype(dtype)](shape)

    @classmethod
    def copy(cls, arr):
        """Returns a new Vector with data copied from the specified `arr`."""
        res = cls.zeros(arr.shape, arr.dtype)
        np.asarray(res).flat = arr.flat
        return res

    @classmethod
    def is_instance(cls, arr):
        return isinstance(arr, cls) or type(arr).__name__ == cls.__name__

    @classmethod
    def is_raw_cuvec(cls, arr):
        return cls.RE_CUVEC_TYPE.match(str(arr))


def _generate_helpers(zeros, CuVec):
    def ones(shape: Shape, dtype="float32") -> CuVec:
        """
        Returns a `CuVec` view of a new `numpy.ndarray`
        of the specified shape and data type (equivalent of `numpy.ones`).
        """
        res = zeros(shape, dtype)
        res[:] = 1
        return res

    def zeros_like(arr) -> CuVec:
        """
        Returns `zeros(arr.shape, arr.dtype)`.
        """
        return zeros(arr.shape, arr.dtype)

    def ones_like(arr) -> CuVec:
        """
        Returns `ones(arr.shape, arr.dtype)`.
        """
        return ones(arr.shape, arr.dtype)

    return ones, zeros_like, ones_like
