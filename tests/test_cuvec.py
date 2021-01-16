import numpy as np
from pytest import importorskip, mark

cuvec = importorskip("niftypet.nimpa.cuvec")
improc = importorskip("niftypet.nimpa.prc.improc")


@mark.parametrize("vtype", list("bBhHiIqQfd"))
def test_Vector_asarray(vtype):
    v = getattr(cuvec, f"Vector_{vtype}")((1, 2, 3))
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
    v = cuvec.Vector_f(shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)
