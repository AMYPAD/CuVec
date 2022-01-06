#!/usr/bin/env python3
import logging
import re
import sys
from pathlib import Path

from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cuvec.setup")

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
cmake_args = [f"-DCUVEC_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}"]
try:
    from miutil import cuinfo
    nvcc_arch_raw = map(cuinfo.compute_capability, range(cuinfo.num_devices()))
    nvcc_arches = {"%d%d" % i for i in nvcc_arch_raw if i >= (3, 5)}
    if nvcc_arches:
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(sorted(nvcc_arches)))
except Exception as exc:
    if "sdist" not in sys.argv or any(i in sys.argv for i in ["build", "bdist", "wheel"]):
        log.warning("Import or CUDA device detection error:\n%s", exc)
for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
    i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
setup(use_scm_version=True, packages=find_packages(exclude=["docs", "tests"]),
      cmake_source_dir="cuvec", cmake_languages=("C", "CXX"),
      cmake_minimum_required_version="3.18", cmake_args=cmake_args)
