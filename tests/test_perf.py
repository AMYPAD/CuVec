from time import time

import numpy as np

import cuvec as cu


def _time_overhead():
    tic = time()
    pass
    res = time() - tic
    return res


def timer(func):
    def inner(*args, **kwargs):
        overhead = np.mean([_time_overhead() for _ in range(100)])
        tic = time()
        res = func(*args, **kwargs)
        return (time() - tic - overhead) * 1000, res

    return inner


def test_perf():
    overhead = np.mean([_time_overhead() for _ in range(100)])
    t = {}
    t['init'], _ = timer(cu.dev_sync)()
    t['create'], src = timer(cu.zeros)((1337, 42), "float32")

    tic = time()
    src.flat = range(1337, 42)
    t['assign'] = (time() - tic - overhead) * 1000

    # `_increment_f` is defined in ../cuvec/src/pycuvec.cu
    t['call'], (t['kernel'], res) = timer(cu.cuvec._increment_f)(src.cuvec)
    t['view'], dst = timer(cu.asarray)(res)

    assert (src + 1 == dst).all()
    print(t)
    # even a fast kernel takes longer than API overhead
    assert t['kernel'] / t['call'] >= 0.5
