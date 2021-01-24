import logging

import numpy as np
from pytest import mark, raises

import cuvec as cu

shape = 127, 344, 344


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
    assert [i[1:] for i in caplog.record_tuples] == [(10, 'copy'),
                                                     (10, "wrap raw <class 'Vector_h'>")]
    assert v.shape == shape
    assert v.dtype.char == 'h'
    assert (v == 1).all()

    caplog.clear()
    v = cu.zeros(shape, 'd')
    assert [i[1:] for i in caplog.record_tuples] == [(10, "wrap raw <class 'Vector_d'>")]

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
