import numpy as np
from pytest import mark

import cuvec


@mark.parametrize("vtype", list(cuvec.typecodes))
def test_Vector_asarray(vtype):
    """vtype(char): any of bBhHiIqQfd"""
    v = getattr(cuvec.cuvec, f"Vector_{vtype}")((1, 2, 3))
    assert str(v) == f"Vector_{vtype}((1, 2, 3))"
    a = np.asarray(v)
    assert not a.any()
    a[0, 0] = 42
    b = np.asarray(v)
    assert (b[0, 0] == 42).all()
    assert not b[1:, 1:].any()
    assert a.dtype.char == vtype
    del a, b


def test_Vector_strides():
    shape = 127, 344, 344
    v = cuvec.cuvec.Vector_f(shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)


def test_vector():
    shape = 127, 344, 344
    a = np.asarray(cuvec.vector(shape, "i"))
    assert a.dtype == np.int32

    a = np.asarray(cuvec.vector(shape, "d"))
    assert a.dtype == np.float64
