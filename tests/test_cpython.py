import numpy as np
from pytest import mark, raises

import cuvec.cpython as cu
from cuvec import cuvec_cpython
from cuvec import example_cpython as ex  # type: ignore # yapf: disable


@mark.parametrize("tp", list(cu.typecodes))
def test_PyCuVec_asarray(tp):
    v = getattr(cuvec_cpython, f"PyCuVec_{tp}")((1, 2, 3))
    assert str(v) == f"PyCuVec_{tp}((1, 2, 3))"
    a = np.asarray(v)
    assert not a.any()
    a[0, 0] = 42
    b = np.asarray(v)
    assert (b[0, 0] == 42).all()
    assert not b[1:, 1:].any()
    assert a.dtype.char == tp
    del a, b, v


def test_np_types():
    f = cu.zeros((1337, 42), 'f')
    d = cu.zeros((1337, 42), 'd')
    cu.asarray(ex.increment2d_f(f))
    cu.asarray(ex.increment2d_f(f, f))
    with raises(TypeError):
        cu.asarray(ex.increment2d_f(d))
    with raises(SystemError):
        # the TypeError is suppressed since a new output is generated
        cu.asarray(ex.increment2d_f(f, d))
