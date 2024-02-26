"""Common (parameterised) tests for cuvec.{cpython,pybind11,swig}"""
import logging

import numpy as np
from pytest import importorskip, mark, raises, skip

import cuvec as cu
import cuvec.cpython as cp

from . import shape

try:
    # `cuvec.pybind11` alternative to `cuvec.cpython`
    # `example_pybind11` is defined in ../cuvec/src/example_pybind11/
    from cuvec import example_pybind11  # type: ignore # yapf: disable
    from cuvec import pybind11 as py
except ImportError:
    py, example_pybind11 = None, None  # type: ignore # yapf: disable

try:
    # `cuvec.swig` alternative to `cuvec.cpython`
    # `example_swig` is defined in ../cuvec/src/example_swig/
    from cuvec import example_swig  # type: ignore # yapf: disable
    from cuvec import swig as sw
except ImportError:
    sw, example_swig = None, None  # type: ignore # yapf: disable


def test_includes():
    assert cu.include_path.is_dir()
    assert {i.name
            for i in cu.include_path.iterdir()} == {
                'cuvec.cuh', 'cuvec_cpython.cuh', 'cuvec_pybind11.cuh', 'cuvec.i', 'pycuvec.cuh'}


def test_cmake_prefix():
    assert cu.cmake_prefix.is_dir()
    assert {i.name
            for i in cu.cmake_prefix.iterdir()} == {
                f'AMYPADcuvec{i}.cmake'
                for i in ('Config', 'ConfigVersion', 'Targets', 'Targets-relwithdebinfo')}


@mark.parametrize("cu,CVector", [(py, 'Pybind11Vector'), (sw, 'SWIGVector')])
def test_CVector_strides(cu, CVector):
    if cu is None:
        skip("cuvec.pybind11 or cuvec.swig not available")
    v = getattr(cu, CVector)('f', shape)
    a = np.asarray(v)
    assert a.shape == shape
    assert a.strides == (512, 32, 4)


@mark.parametrize("spec,result", [("i", np.int32), ("d", np.float64)])
@mark.parametrize("init", ["zeros", "ones"])
@mark.parametrize("cu", filter(None, [cp, py, sw]))
def test_create(cu, init, spec, result):
    a = np.asarray(getattr(cu, init)(shape, spec))
    assert a.dtype == result
    assert a.shape == shape
    assert (a == (0 if init == 'zeros' else 1)).all()

    b = getattr(cu, f'{init}_like')(a)
    assert b.shape == a.shape
    assert b.dtype == a.dtype


@mark.parametrize("cu", filter(None, [cp, py, sw]))
def test_copy(cu):
    a = np.random.random(shape)
    b = np.asarray(cu.copy(a))
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert (a == b).all()


@mark.parametrize("cu,classname", [(cp, "raw <class 'PyCuVec_{typechar}'>"),
                                   (py, "pyraw <class 'cuvec.pybind11.Pybind11Vector'>"),
                                   (sw, "swraw <class 'cuvec.swig.SWIGVector'>")])
def test_CuVec_creation(cu, classname, caplog):
    if cu is None:
        skip("cuvec.pybind11 or cuvec.swig not available")
    with raises(TypeError):
        cu.CuVec()

    with raises(NotImplementedError):
        cu.CuVec(shape)

    caplog.set_level(logging.DEBUG)
    caplog.clear()
    v = cu.CuVec(np.ones(shape, dtype='h'))
    assert [i[1:] for i in caplog.record_tuples] == [
        (10, 'copy'), (10, f"wrap {classname}".format(typechar='h'))]
    assert v.shape == shape
    assert v.dtype.char == 'h'
    assert (v == 1).all()

    caplog.clear()
    v = cu.zeros(shape, 'd')
    assert [i[1:] for i in caplog.record_tuples] == [
        (10, f"wrap {classname}".format(typechar='d'))]

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


@mark.parametrize("cu", filter(None, [py, sw]))
@mark.timeout(20)
def test_asarray(cu):
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


@mark.parametrize("cu", filter(None, [py, sw]))
def test_resize(cu):
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


@mark.timeout(60)
@mark.parametrize("cu", filter(None, [cp, py, sw]))
def test_cuda_array_interface(cu):
    cupy = importorskip("cupy")
    from cuvec import dev_sync

    v = cu.asarray(np.random.random(shape))
    assert set(v.__cuda_array_interface__) == {'shape', 'typestr', 'data', 'version'}

    c = cupy.asarray(v)
    assert (c == v).all()
    c[0, 0, 0] = 1
    dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]
    c[0, 0, 0] = 0
    dev_sync()
    assert c[0, 0, 0] == v[0, 0, 0]

    if hasattr(v, '_vec'):
        d = cupy.asarray(v._vec)
        d[0, 0, 0] = 1
        dev_sync()
        assert d[0, 0, 0] == c[0, 0, 0] == v[0, 0, 0]
        d[0, 0, 0] = 0
        dev_sync()
        assert d[0, 0, 0] == c[0, 0, 0] == v[0, 0, 0]

    ndarr = v + 1
    assert ndarr.shape == v.shape
    assert ndarr.dtype == v.dtype
    with raises(AttributeError):
        ndarr.__cuda_array_interface__


@mark.parametrize("cu,ex", [(py, example_pybind11), (sw, example_swig)])
def test_increment(cu, ex):
    if cu is None:
        skip("cuvec.pybind11 or cuvec.swig not available")
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
