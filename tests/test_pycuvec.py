import numpy as np
from pytest import mark

import cuvec as cu


@mark.parametrize("tp", list(cu.typecodes))
def test_Vector_asarray(tp):
    """tp(char): any of bBhHiIqQfd"""
    v = getattr(cu.cuvec, f"Vector_{tp}")((1, 2, 3))
    assert str(v) == f"Vector_{tp}((1, 2, 3))"
    a = np.asarray(v)
    assert not a.any()
    a[0, 0] = 42
    b = np.asarray(v)
    assert (b[0, 0] == 42).all()
    assert not b[1:, 1:].any()
    assert a.dtype.char == tp
    del a, b, v


def test_Vector_strides():
    shape = 127, 344, 344
    v = cu.cuvec.Vector_f(shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)
