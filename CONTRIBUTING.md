# Contributing

Install in "development/editable" mode including dev/test dependencies:

```sh
# clone & install dependencies (one-off)
git clone https://github.com/AMYPAD/CuVec
cd CuVec
make deps-build deps-run

# delete build artefacts, (re)build & install in-place with debug info
make CUVEC_DEBUG=1 build-editable
```

Once installed in development/editable mode, tests may be run using:

```sh
pytest -k "not perf"
```

To run performance tests, build with debugging disabled (`CUVEC_DEBUG=0`), then run:

```sh
pytest -k "perf" -n=0
python tests/test_perf.py
```
