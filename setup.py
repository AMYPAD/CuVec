#!/usr/bin/env python3
import logging
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools_scm import get_version

__version__ = get_version(root=".", relative_to=__file__)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cuvec.setup")

build_ver = ".".join(__version__.split('.')[:3]).split(".dev")[0]
setup_kwargs = {"use_scm_version": True, "packages": find_packages(exclude=["tests"])}
cmake_args = [f"-DCUVEC_BUILD_VERSION={build_ver}", f"-DPython3_ROOT_DIR={sys.prefix}"]

try:
    from miutil import cuinfo
    nvcc_arches = map(cuinfo.compute_capability, range(cuinfo.num_devices()))
    nvcc_arches = {"%d%d" % i for i in nvcc_arches if i >= (3, 5)}
    if nvcc_arches:
        cmake_args.append("-DCMAKE_CUDA_ARCHITECTURES=" + " ".join(sorted(nvcc_arches)))
except Exception as exc:
    log.warning("Import or CUDA device detection error:\n%s", exc)

try:
    from skbuild import setup as sksetup
except ImportError:
    log.warning("`skbuild.setup` not found: Using `setuptools.setup`")
    setup(**setup_kwargs)
else:
    for i in (Path(__file__).resolve().parent / "_skbuild").rglob("CMakeCache.txt"):
        i.write_text(re.sub("^//.*$\n^[^#].*pip-build-env.*$", "", i.read_text(), flags=re.M))
    sksetup(cmake_source_dir="cuvec", cmake_minimum_required_version="3.18", cmake_args=cmake_args,
            **setup_kwargs)
