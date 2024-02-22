import numpy as np
from pytest import mark, raises

import cuvec.cpython as cu
from cuvec import cuvec_cpython

shape = 127, 344, 344


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


def test_CVector_strides():
    v = cuvec_cpython.PyCuVec_f(shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)


@mark.timeout(20)
def test_asarray():
    v = cu.asarray(np.random.random(shape))
    w = cu.CuVec(v)
    assert w.cuvec == v.cuvec
    assert (w == v).all()
    assert np.asarray(w.cuvec).data == np.asarray(v.cuvec).data
    x = cu.asarray(w.cuvec)
    assert x.cuvec == v.cuvec
    assert (x == v).all()
    assert np.asarray(x.cuvec).data == np.asarray(v.cuvec).data
    y = cu.asarray(x.tolist())
    assert y.cuvec != v.cuvec
    assert (y == v).all()
    assert np.asarray(y.cuvec).data == np.asarray(v.cuvec).data
    z = cu.asarray(v[:])
    assert z.cuvec != v.cuvec
    assert (z == v[:]).all()
    assert np.asarray(z.cuvec).data == np.asarray(v.cuvec).data
    s = cu.asarray(v[1:])
    assert s.cuvec != v.cuvec
    assert (s == v[1:]).all()
    assert np.asarray(s.cuvec).data != np.asarray(v.cuvec).data


def test_increment():
    # `example_cpython` is defined in ../cuvec/src/example_cpython/
    from cuvec.example_cpython import increment2d_f
    a = cu.zeros((1337, 42), 'f')
    assert (a == 0).all()
    res = cu.asarray(increment2d_f(a.cuvec, a.cuvec))
    assert (a == 1).all()
    assert (res == 1).all()

    a[:] = 0
    assert (a == 0).all()
    assert (res == 0).all()

    res = cu.asarray(increment2d_f(a))
    assert (res == 1).all()


def test_increment_return():
    from cuvec.example_cpython import increment2d_f
    a = cu.zeros((1337, 42), 'f')
    assert (a == 0).all()
    res = cu.asarray(increment2d_f(a, a))
    assert (a == 1).all()
    del a
    assert (res == 1).all()


def test_np_types():
    from cuvec.example_cpython import increment2d_f
    f = cu.zeros((1337, 42), 'f')
    d = cu.zeros((1337, 42), 'd')
    cu.asarray(increment2d_f(f))
    cu.asarray(increment2d_f(f, f))
    with raises(TypeError):
        cu.asarray(increment2d_f(d))
    with raises(SystemError):
        # the TypeError is suppressed since a new output is generated
        cu.asarray(increment2d_f(f, d))
