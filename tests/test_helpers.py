import logging

import numpy as np
from pytest import mark, raises

import cuvec

shape = 127, 344, 344


@mark.parametrize("spec,result", [("i", np.int32), ("d", np.float64)])
def test_zeros(spec, result):
    a = np.asarray(cuvec.zeros(shape, spec))
    assert a.dtype == result
    assert a.shape == shape
    assert not a.any()


def test_copy():
    a = np.random.random(shape)
    b = np.asarray(cuvec.copy(a))
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert (a == b).all()


def test_CuVec_creation(caplog):
    with raises(TypeError):
        cuvec.CuVec()

    with raises(NotImplementedError):
        cuvec.CuVec(shape)

    caplog.set_level(logging.DEBUG)
    caplog.clear()
    v = cuvec.CuVec(np.ones(shape, dtype='h'))
    assert [i[1:] for i in caplog.record_tuples] == [(10, 'copy'),
                                                     (10, "wrap raw <class 'Vector_h'>")]
    assert v.shape == shape
    assert v.dtype.char == 'h'
    assert (v == 1).all()

    caplog.clear()
    v = cuvec.zeros(shape, 'd')
    assert [i[1:] for i in caplog.record_tuples] == [(10, "wrap raw <class 'Vector_d'>")]

    caplog.clear()
    v[0, 0, 0] = 1
    assert not caplog.record_tuples
    w = cuvec.CuVec(v)
    assert [i[1:] for i in caplog.record_tuples] == [(10, "new view")]

    caplog.clear()
    assert w[0, 0, 0] == 1
    v[0, 0, 0] = 9
    assert w[0, 0, 0] == 9
    assert v.cuvec is w.cuvec
    assert v.data == w.data
    assert not caplog.record_tuples
