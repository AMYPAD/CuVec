import numpy as np
from pytest import importorskip, mark

cu = importorskip("cuvec.swig")
shape = 127, 344, 344


@mark.parametrize("tp", list(cu.typecodes))
def test_SWIGVector_asarray(tp):
    v = cu.SWIGVector(tp, (1, 2, 3))
    assert repr(v) == f"SWIGVector('{tp}', (1, 2, 3))"
    a = np.asarray(v)
    assert not a.any()
    a[0, 0] = 42
    b = np.asarray(v)
    assert (b[0, 0] == 42).all()
    assert not b[1:, 1:].any()
    assert a.dtype == np.dtype(tp)
    del a, b, v
