from functools import wraps
from time import time

import numpy as np
from pytest import mark, skip

# `example_mod` is defined in ../cuvec/src/example_mod/
from cuvec import example_mod
from cuvec import pycuvec as cu

try:
    # alternative to `cu`
    # `example_swig` is defined in ../cuvec/src/example_swig/
    from cuvec import example_swig
    from cuvec import swigcuvec as sw
except ImportError:
    sw, example_swig = None, None


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


@mark.parametrize("cu,ex", [(cu, example_mod), (sw, example_swig)])
@retry_on_except()
def test_perf(cu, ex, shape=(1337, 42), quiet=False):
    if cu is None:
        skip("SWIG not available")
    overhead = np.mean([_time_overhead() for _ in range(100)])
    t = {}
    t['create src'], src = timer(cu.zeros)(shape, "float32")

    rnd = np.random.random(shape)
    tic = time()
    src[:] = rnd
    t['assign'] = (time() - tic - overhead) * 1000

    if not quiet:
        if cu is sw:
            t['warmup'], res = timer(ex.increment2d_f)(src.cuvec, None, True)
            t['> create dst'], t['> kernel'] = cu.asarray(res)[0, :2]
        else:
            t['warmup'], (t['> create dst'], t['> kernel'], _) = timer(ex.increment2d_f)(src.cuvec)
    if cu is sw:
        t['call ext'], res = timer(ex.increment2d_f)(src.cuvec, None, True)
        t['- create dst'], t['- kernel'] = None, None
        t['view'], dst = timer(cu.asarray)(res)
        t['- create dst'], t['- kernel'] = dst[0, :2]
    else:
        t['call ext'], (t['- create dst'], t['- kernel'], res) = timer(ex.increment2d_f)(src.cuvec)
        t['view'], dst = timer(cu.asarray)(res)

    if not quiet:
        print("\n".join(f"{k.ljust(14)} | {v:.3f}" for k, v in t.items()))
    assert (src + 1 == dst)[1:].all()
    assert (src + 1 == dst)[0, 2 if cu is sw else 0:].all()
    # even a fast kernel takes longer than API overhead
    assert t['- kernel'] / (t['call ext'] - t['- create dst']) > 0.5
    # API call should be <0.1 ms... but set a higher threshold of 2 ms
    assert t['call ext'] - t['- create dst'] - t['- kernel'] < 2
    return t


if __name__ == "__main__":
    try:
        from tqdm import trange
    except ImportError:
        trange = range
    nruns = 1000

    for args in [(cu, example_mod), (sw, example_swig)]:
        print(f"# One run ({args[1].__name__}):")
        test_perf(*args, shape=(1000, 1000))

        print(f"# Average over {nruns} runs:")
        runs = [test_perf(*args, shape=(1000, 1000), quiet=True) for _ in trange(nruns)]
        pretty = {
            'create src': 'Create input', 'assign': 'Assign', 'call ext': 'Call extension',
            '- create dst': '-- Create output', '- kernel': '-- Launch kernel', 'view': 'View'}
        runs = {pretty[k]: [i[k] for i in runs] for k in runs[0]}
        print("\n".join(
            f"{k.ljust(16)} | {np.mean(v):.3f} ± {np.std(v, ddof=1)/np.sqrt(len(v)):.3f}"
            for k, v in runs.items()))
