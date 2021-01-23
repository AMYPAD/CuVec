from functools import wraps
from time import time

import numpy as np

import cuvec as cu


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


def test_perf(shape=(1337, 42), quiet=False):
    # `example_mod` is defined in ../cuvec/src/example_mod/
    from cuvec.example_mod import increment_f

    overhead = np.mean([_time_overhead() for _ in range(100)])
    t = {}
    t['create src'], src = timer(cu.zeros)(shape, "float32")

    rnd = np.random.random(shape)
    tic = time()
    src[:] = rnd
    t['assign'] = (time() - tic - overhead) * 1000

    if not quiet:
        t['warmup'], (t['> create dst'], t['> kernel'], _) = timer(increment_f)(src.cuvec)
    t['call ext'], (t['- create dst'], t['- kernel'], res) = timer(increment_f)(src.cuvec)
    t['view'], dst = timer(cu.asarray)(res)

    if not quiet:
        print("\n".join(f"{k.ljust(14)} | {v:.3f}" for k, v in t.items()))
    assert (src + 1 == dst).all()
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

    print("# One run:")
    test_perf((1000, 1000))

    print("Repeating & averaging performance test metrics over {nruns} runs.")
    runs = [test_perf((1000, 1000), True) for _ in trange(nruns)]
    pretty = {
        'create src': 'Create input', 'assign': 'Assign', 'call ext': 'Call extension',
        '- create dst': '-- Create output', '- kernel': '-- Launch kernel', 'view': 'View'}
    runs = {pretty[k]: [i[k] for i in runs] for k in runs[0]}
    print("\n".join(f"{k.ljust(16)} | {np.mean(v):.3f} ± {np.std(v, ddof=1)/np.sqrt(len(v)):.3f}"
                    for k, v in runs.items()))
