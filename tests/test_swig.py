import numpy as np
from pytest import importorskip, mark, raises

from . import shape

cu = importorskip("cuvec.swig")
# `example_swig` is defined in ../cuvec/src/example_swig/
ex = importorskip("cuvec.example_swig")


def test_SWIGVector_strides():
    v = cu.SWIGVector('f', shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (512, 32, 4)


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


@mark.timeout(20)
def test_asarray():
    v = cu.asarray(np.random.random(shape))
    w = cu.CuVec(v)
    assert w.cuvec == v.cuvec
    assert (w == v).all()
    assert str(w._vec) == str(v._vec)
    assert np.asarray(w._vec).data == np.asarray(v._vec).data
    x = cu.asarray(w._vec)
    assert x.cuvec == v.cuvec
    assert (x == v).all()
    assert str(x._vec) == str(v._vec)
    assert np.asarray(x._vec).data == np.asarray(v._vec).data
    y = cu.asarray(x.tolist())
    assert y.cuvec != v.cuvec
    assert (y == v).all()
    assert str(y._vec) != str(v._vec)
    assert np.asarray(y._vec).data == np.asarray(v._vec).data
    z = cu.asarray(v[:])
    assert z.cuvec != v.cuvec
    assert (z == v[:]).all()
    assert str(z._vec) != str(v._vec)
    assert np.asarray(z._vec).data == np.asarray(v._vec).data
    s = cu.asarray(v[1:])
    assert s.cuvec != v.cuvec
    assert (s == v[1:]).all()
    assert str(s._vec) != str(v._vec)
    assert np.asarray(s._vec).data != np.asarray(v._vec).data
    with raises(IOError):
        cu.asarray(s._vec.cuvec, ownership='error')


def test_resize():
    v = cu.asarray(np.random.random(shape))
    v.resize(shape[::-1])
    assert v.shape == shape[::-1]
    assert v._vec.shape == v.shape
    v.resize(v.size)
    assert v.shape == (v.size,)
    assert v._vec.shape == v.shape
    v.shape = shape
    assert v.shape == shape
    assert v._vec.shape == v.shape


def test_increment():
    a = cu.zeros((1337, 42), 'f')
    assert (a == 0).all()
    ex.increment2d_f(a.cuvec, a.cuvec)
    assert (a == 1).all()

    a[:] = 0
    assert (a == 0).all()

    b = cu.retarray(ex.increment2d_f(a.cuvec))
    assert (b == 1).all()

    c = cu.retarray(ex.increment2d_f(b.cuvec, a.cuvec), a)
    assert (a == 2).all()
    assert c.cuvec == a.cuvec
    assert (c == a).all()
    assert str(c._vec) == str(a._vec)
    assert np.asarray(c._vec).data == np.asarray(a._vec).data
