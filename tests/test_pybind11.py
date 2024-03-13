import re

import numpy as np
from pytest import importorskip, mark, raises

from . import shape

cu = importorskip("cuvec.pybind11")
cuvec_pybind11 = importorskip("cuvec.cuvec_pybind11")
ex = importorskip("cuvec.example_pybind11")


@mark.parametrize("tp", list(cu.typecodes))
def test_NDCuVec_asarray(tp):
    v = getattr(cuvec_pybind11, f"NDCuVec_{tp}")((1, 2, 3))
    assert re.match(f"<cuvec.cuvec_pybind11.NDCuVec_{tp} object at 0x[0-9a-f]+>", str(v))
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
    cu.asarray(ex.increment2d_f(f.cuvec))
    cu.asarray(ex.increment2d_f(f.cuvec, f.cuvec))
    with raises(TypeError):
        cu.asarray(ex.increment2d_f(d.cuvec))
    with raises(TypeError):
        cu.asarray(ex.increment2d_f(f.cuvec, d.cuvec))


def test_resize():
    v = cu.asarray(np.random.random(shape))
    v.resize(shape[::-1])
    assert v.shape == shape[::-1]
    assert v.cuvec.shape == v.shape
    v.resize(v.size)
    assert v.shape == (v.size,)
    assert v.cuvec.shape == v.shape
    v.shape = shape
    assert v.shape == shape
    assert v.cuvec.shape == v.shape
