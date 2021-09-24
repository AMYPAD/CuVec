# Contributing

Install in "development/editable" mode including dev/test dependencies:

```sh
git clone https://github.com/AMYPAD/CuVec && cd CuVec
pip install -e .[dev]
```

Alternatively, if `cmake` and a generator (such as `make` or `ninja`) are
available, then `setup.py build` and `develop` can be explicitly called;
optionally with extra `cmake` and generator arguments:

```sh
python setup.py build develop easy_install cuvec[dev] -- -DCUVEC_DEBUG:BOOL=ON -- -j8
```

Once installed in development/editable mode, tests may be run using:

```sh
pytest
```
