"""Common helpers for pycuvec & swigcuvec modules."""
import array
from typing import Sequence as Seq
from typing import Union

Shape = Union[Seq[int], int]
# u: non-standard np.dype('S2'); l/L: inconsistent between `array` and `numpy`
typecodes = ''.join(i for i in array.typecodes if i not in "ulL")


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
