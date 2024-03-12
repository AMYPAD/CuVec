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
                                   (py, "raw <class 'cuvec.cuvec_pybind11.NDCuVec_{typechar}'>"),
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
