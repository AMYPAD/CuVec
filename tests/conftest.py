from pytest import fixture, skip


@fixture
def dev_sync():
    try:
        from cuvec import dev_sync as res
    except ImportError:
        skip("CUDA not available")
    return res
