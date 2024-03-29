from functools import wraps
from time import time

import numpy as np
from pytest import mark, skip

import cuvec.cpython as cu

# `example_cpython` is defined in ../cuvec/src/example_cpython/
from cuvec import example_cpython  # type: ignore # yapf: disable

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


def _time_overhead():
    tic = time()
    pass
    res = time() - tic
    return res


def timer(func):
    @wraps(func)
    def inner(*args, **kwargs):
        overhead = np.mean([_time_overhead() for _ in range(100)])
        tic = time()
        res = func(*args, **kwargs)
        return (time() - tic - overhead) * 1000, res

    return inner


def retry_on_except(n=3):
    """decroator for retrying `n` times before raising Exceptions"""
    def wrapper(func):
        @wraps(func)
        def test_inner(*args, **kwargs):
            for i in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i >= n:
                        raise

        return test_inner

    return wrapper


@mark.parametrize("cu,ex", [(cu, example_cpython), (py, example_pybind11), (sw, example_swig)])
@retry_on_except()
def test_perf(cu, ex, shape=(1337, 42), quiet=False, return_time=False):
    if cu is None:
        skip("cuvec.pybind11 or cuvec.swig not available")
    retarray = getattr(cu, 'retarray', cu.asarray)
    overhead = np.mean([_time_overhead() for _ in range(100)])
    t = {}
    t['create src'], src = timer(cu.zeros)(shape, "float32")

    rnd = np.random.random(shape)
    tic = time()
    src[:] = rnd
    t['assign'] = (time() - tic - overhead) * 1000

    if not quiet:
        t['warmup'], res = timer(ex.increment2d_f)(src.cuvec, None, True)
        t['> create dst'], t['> kernel'] = retarray(res)[0, :2]
    t['call ext'], res = timer(ex.increment2d_f)(src.cuvec, None, True)
    t['- create dst'], t['- kernel'] = None, None
    t['view'], dst = timer(retarray)(res)
    t['- create dst'], t['- kernel'] = dst[0, :2]

    if not quiet:
        print("\n".join(f"{k.ljust(14)} | {v:.3f}" for k, v in t.items()))
    assert (src + 1 == dst)[1:].all()
    assert (src + 1 == dst)[0, 2:].all()
    # even a fast kernel takes longer than API overhead
    assert t['- kernel'] / (t['call ext'] - t['- create dst']) > 0.3
    # API call should be <0.1 ms... but set a higher threshold of 5 ms
    assert t['call ext'] - t['- create dst'] - t['- kernel'] < 5
    if return_time:
        return t


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    try:
        from tqdm import trange
    except ImportError:
        trange = range
    nruns = 500
    N = 1024

    for args in [(cu, example_cpython), (py, example_pybind11), (sw, example_swig)]:
        print(f"# One run ({args[1].__name__}):")
        test_perf(*args, shape=(N, N))

        print(f"# Average over {nruns} runs:")
        res_runs = [
            test_perf(*args, shape=(N, N), quiet=True, return_time=True) for _ in trange(nruns)]
        pretty = {
            'create src': 'Create input', 'assign': 'Assign', 'call ext': 'Call extension',
            '- create dst': '-- Create output', '- kernel': '-- Launch kernel', 'view': 'View'}
        runs = {pretty[k]: [i[k] for i in res_runs] for k in res_runs[0]}
        print("\n".join(
            f"{k.ljust(16)} | {np.mean(v):.3f} ± {np.std(v, ddof=1)/np.sqrt(len(v)):.3f}"
            for k, v in runs.items()))
