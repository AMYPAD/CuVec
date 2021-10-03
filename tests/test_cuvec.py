import cuvec as cu


def test_includes():
    assert cu.include_path.is_dir()
    assert {i.name for i in cu.include_path.iterdir()} == {'cuvec.cuh', 'pycuvec.cuh', 'cuvec.i'}


def test_cmake_prefix():
    assert cu.cmake_prefix.is_dir()
    assert {i.name
            for i in cu.cmake_prefix.iterdir()} == {
                f'AMYPADcuvec{i}.cmake'
                for i in ('Config', 'ConfigVersion', 'Targets', 'Targets-release')}
