# NOTE: cannot `pip install -U -e .[dev]` (install in-place for pytest)
# since `-e .` doesn't work yet (https://github.com/scikit-build/scikit-build-core/issues/114).
# Instead, do `make deps-build build-editable deps-run`
CUVEC_DEBUG=0
CXX_FLAGS=-Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers -Wno-unused-parameter -Wno-cast-function-type
CUDA_ARCHITECTURES=native
BUILD_CMAKE_FLAGS=-Ccmake.define.CUVEC_DEBUG=$(CUVEC_DEBUG) -Ccmake.define.CMAKE_CXX_FLAGS="$(CXX_FLAGS)" -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHITECTURES)
CCACHE=
ifneq ($(CCACHE),)
	BUILD_CMAKE_FLAGS+= -Ccmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache
endif
.PHONY: build-editable clean deps-build deps-run build-wheel
build-editable:
	git diff --exit-code --quiet '*/src/**' || (echo "Uncommitted changes in */src"; exit 1)
	@$(MAKE) clean
	pip install --no-build-isolation --check-build-dependencies --no-deps -t . -U -v . $(BUILD_CMAKE_FLAGS)
	git restore '*/src/**'
test:
	pytest -k "not perf" -n=3
	pytest -k "perf" -n=0 --cov-append
clean:
	git clean -Xdf
deps-build:
	pip install toml
	python -c 'import toml; c=toml.load("pyproject.toml"); print("\0".join(c["build-system"]["requires"] + ["cmake>=" + c["tool"]["scikit-build"]["cmake"]["minimum-version"]]), end="")' | xargs -0 pip install ninja
deps-run:
	pip install toml
	python -c 'import toml; c=toml.load("pyproject.toml"); print("\0".join(c["project"]["dependencies"] + c["project"]["optional-dependencies"]["dev"]), end="")' | xargs -0 pip install
build-wheel:
	@$(MAKE) clean
	pip install build
	python -m build -n -w $(BUILD_CMAKE_FLAGS)
