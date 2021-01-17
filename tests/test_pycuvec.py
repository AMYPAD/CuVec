import numpy as np
from pytest import mark

import cuvec


@mark.parametrize("spec,result", [("i", np.int32), ("d", np.float64)])
def test_zeros(spec, result):
    shape = 127, 344, 344
    a = np.asarray(cuvec.zeros(shape, spec))
    assert a.dtype == result
    assert a.shape == shape
    assert not a.any()


def test_from_numpy():
    shape = 127, 344, 344
    a = np.random.random(shape)
    b = np.asarray(cuvec.from_numpy(a))
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert (a == b).all()
