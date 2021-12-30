import logging

import numpy as np
from pytest import importorskip, mark, raises

import cuvec as cu

shape = 127, 344, 344


@mark.parametrize("tp", list(cu.typecodes))
def test_PyCuVec_asarray(tp):
    v = getattr(cu.cuvec, f"PyCuVec_{tp}")((1, 2, 3))
    assert str(v) == f"PyCuVec_{tp}((1, 2, 3))"
    a = np.asarray(v)
    assert not a.any()
    a[0, 0] = 42
    b = np.asarray(v)
    assert (b[0, 0] == 42).all()
    assert not b[1:, 1:].any()
    assert a.dtype.char == tp
    del a, b, v


def test_PyCuVec_strides():
    v = cu.cuvec.PyCuVec_f(shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (473344, 1376, 4)


@mark.parametrize("spec,result", [("i", np.int32), ("d", np.float64)])
@mark.parametrize("init", ["zeros", "ones"])
def test_create(init, spec, result):
    a = np.asarray(getattr(cu, init)(shape, spec))
    assert a.dtype == result
    assert a.shape == shape
    assert (a == (0 if init == 'zeros' else 1)).all()

    b = getattr(cu, f'{init}_like')(a)
    assert b.shape == a.shape
    assert b.dtype == a.dtype


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
                                                     (10, "wrap raw <class 'PyCuVec_h'>")]
    assert v.shape == shape
    assert v.dtype.char == 'h'
    assert (v == 1).all()

    caplog.clear()
    v = cu.zeros(shape, 'd')
    assert [i[1:] for i in caplog.record_tuples] == [(10, "wrap raw <class 'PyCuVec_d'>")]

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


def test_cuda_array_interface():
    cupy = importorskip("cupy")
    v = cu.asarray(np.random.random(shape))
    assert hasattr(v, '__cuda_array_interface__')

    c = cupy.asarray(v)
    assert (c == v).all()
    c[0, 0, 0] = 1
    cu.dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]
    c[0, 0, 0] = 0
    cu.dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]

    ndarr = v + 1
    assert ndarr.shape == v.shape
    assert ndarr.dtype == v.dtype
    with raises(AttributeError):
        ndarr.__cuda_array_interface__


def test_increment():
    # `example_mod` is defined in ../cuvec/src/example_mod/
    from cuvec.example_mod import increment2d_f
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
    from cuvec.example_mod import increment2d_f
    a = cu.zeros((1337, 42), 'f')
    assert (a == 0).all()
    res = cu.asarray(increment2d_f(a, a))
    assert (a == 1).all()
    del a
    assert (res == 1).all()


def test_np_types():
    from cuvec.example_mod import increment2d_f
    f = cu.zeros((1337, 42), 'f')
    d = cu.zeros((1337, 42), 'd')
    cu.asarray(increment2d_f(f))
    cu.asarray(increment2d_f(f, f))
    with raises(TypeError):
        cu.asarray(increment2d_f(d))
    with raises(SystemError):
        # the TypeError is suppressed since a new output is generated
        cu.asarray(increment2d_f(f, d))
