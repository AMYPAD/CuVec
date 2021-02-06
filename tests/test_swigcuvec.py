import logging

import numpy as np
from pytest import importorskip, mark, raises

from cuvec import dev_sync

cu = importorskip("cuvec.swigcuvec")
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


def test_PyCuVec_strides():
    v = cu.SWIGVector('f', shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)


@mark.parametrize("spec,result", [("i", np.int32), ("d", np.float64)])
def test_zeros(spec, result):
    a = np.asarray(cu.zeros(shape, spec))
    assert a.dtype == result
    assert a.shape == shape
    assert not a.any()


def test_copy():
    a = np.random.random(shape)
    b = np.asarray(cu.copy(a))
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert (a == b).all()


def test_CuVec_creation(caplog):
    with raises(TypeError):
        cu.CuVec()

    with raises(NotImplementedError):
        cu.CuVec(shape)

    caplog.set_level(logging.DEBUG)
    caplog.clear()
    v = cu.CuVec(np.ones(shape, dtype='h'))
    assert [i[1:] for i in caplog.record_tuples] == [
        (10, 'copy'), (10, "wrap swraw <class 'cuvec.swigcuvec.SWIGVector'>")]
    assert v.shape == shape
    assert v.dtype.char == 'h'
    assert (v == 1).all()

    caplog.clear()
    v = cu.zeros(shape, 'd')
    assert [i[1:] for i in caplog.record_tuples] == [
        (10, "wrap swraw <class 'cuvec.swigcuvec.SWIGVector'>")]

    caplog.clear()
    v[0, 0, 0] = 1
    assert not caplog.record_tuples
    w = cu.CuVec(v)
    assert [i[1:] for i in caplog.record_tuples] == [(10, "new view")]

    caplog.clear()
    assert w[0, 0, 0] == 1
    v[0, 0, 0] = 9
    assert w[0, 0, 0] == 9
    assert v.cuvec is w.cuvec
    assert v.data == w.data
    assert not caplog.record_tuples


def test_asarray():
    v = cu.asarray(np.random.random(shape))
    w = cu.CuVec(v)
    assert w.cuvec == v.cuvec
    assert (w == v).all()
    assert str(w.swvec) == str(v.swvec)
    assert np.asarray(w.swvec).data == np.asarray(v.swvec).data
    x = cu.asarray(w.swvec)
    assert x.cuvec == v.cuvec
    assert (x == v).all()
    assert str(x.swvec) == str(v.swvec)
    assert np.asarray(x.swvec).data == np.asarray(v.swvec).data
    y = cu.asarray(x.tolist())
    assert y.cuvec != v.cuvec
    assert (y == v).all()
    assert str(y.swvec) != str(v.swvec)
    assert np.asarray(y.swvec).data == np.asarray(v.swvec).data
    z = cu.asarray(v[:])
    assert z.cuvec != v.cuvec
    assert (z == v[:]).all()
    assert str(z.swvec) != str(v.swvec)
    assert np.asarray(z.swvec).data == np.asarray(v.swvec).data
    s = cu.asarray(v[1:])
    assert s.cuvec != v.cuvec
    assert (s == v[1:]).all()
    assert str(s.swvec) != str(v.swvec)
    assert np.asarray(s.swvec).data != np.asarray(v.swvec).data


def test_cuda_array_interface():
    cupy = importorskip("cupy")
    v = cu.asarray(np.random.random(shape))
    assert hasattr(v, '__cuda_array_interface__')

    c = cupy.asarray(v)
    assert (c == v).all()
    c[0, 0, 0] = 1
    dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]
    c[0, 0, 0] = 0
    dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]

    d = cupy.asarray(v.swvec)
    d[0, 0, 0] = 1
    dev_sync()
    assert d[0, 0, 0] == v[0, 0, 0]
    d[0, 0, 0] = 0
    dev_sync()
    assert d[0, 0, 0] == v[0, 0, 0]

    ndarr = v + 1
    assert ndarr.shape == v.shape
    assert ndarr.dtype == v.dtype
    with raises(AttributeError):
        ndarr.__cuda_array_interface__


def test_increment():
    # `example_swig` is defined in ../cuvec/src/example_swig/
    from cuvec.example_swig import increment2d_f
    a = cu.zeros((1337, 42), 'f')
    assert (a == 0).all()
    increment2d_f(a.cuvec, a.cuvec)
    assert (a == 1).all()

    a[:] = 0
    assert (a == 0).all()

    res = cu.asarray(increment2d_f(a.cuvec))
    assert (res == 1).all()
