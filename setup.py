#!/usr/bin/env python3
import logging
import re
import sys
from pathlib import Path

from setuptools import find_packages
from setuptools_scm import get_version
from skbuild import setup as sksetup

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cuvec.setup")

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
setup_kwargs = {"use_scm_version": True, "packages": find_packages(exclude=["tests"])}
cmake_args = [f"-DCUVEC_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}"]

try:
    from miutil import cuinfo
    nvcc_arches = {"%d%d" % cuinfo.compute_capability(i) for i in range(cuinfo.num_devices())}
except Exception as exc:
    log.warning("could not detect CUDA architectures:\n%s", exc)
else:
    cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + " ".join(sorted(nvcc_arches)))

for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
    i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
sksetup(cmake_source_dir="cuvec", cmake_languages=("C", "CXX", "CUDA"),
        cmake_minimum_required_version="3.18", cmake_args=cmake_args, **setup_kwargs)
