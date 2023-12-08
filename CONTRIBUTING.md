# Contributing

Install in "development/editable" mode including dev/test dependencies:

```sh
git clone https://github.com/AMYPAD/CuVec && cd CuVec

# `pip install -e .[dev]` won't work due to https://github.com/scikit-build/scikit-build-core/issues/114
# work-around:
# 1. install dependencies (one-off)
pip install toml
python -c 'import toml; c=toml.load("pyproject.toml")
print("\0".join(c["build-system"]["requires"] + c["project"]["dependencies"] + c["project"]["optional-dependencies"]["dev"]), end="")' \
| xargs -0 pip install -U ninja cmake
# 2. delete build artefacts, (re)build & install in-place with debug info
git clean -Xdf
pip install --no-build-isolation --no-deps -t . -U -v . \
  -Ccmake.define.CUVEC_DEBUG=1
  -Ccmake.define.CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers -Wno-unused-parameter -Wno-cast-function-type"
git restore cuvec/src # undo deletion of sources
```

Once installed in development/editable mode, tests may be run using:

```sh
pytest
```

To run performance tests, build with debugging disabled (`CUVEC_DEBUG=0`), then run:

```sh
python tests/test_perf.py
```
