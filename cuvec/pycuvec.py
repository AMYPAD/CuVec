"""Thin wrappers around `cuvec` C++/CUDA module"""
from collections.abc import Sequence

import numpy as np

from .cuvec import (
    Vector_b,
    Vector_B,
    Vector_c,
    Vector_d,
    Vector_f,
    Vector_h,
    Vector_H,
    Vector_i,
    Vector_I,
    Vector_q,
    Vector_Q,
)

vec_types = {
    np.dtype('int8'): Vector_b,
    np.dtype('uint8'): Vector_B,
    np.dtype('S1'): Vector_c,
    np.dtype('int16'): Vector_h,
    np.dtype('uint16'): Vector_H,
    np.dtype('int32'): Vector_i,
    np.dtype('uint32'): Vector_I,
    np.dtype('int64'): Vector_q,
    np.dtype('uint64'): Vector_Q,
    np.dtype('float32'): Vector_f,
    np.dtype('float64'): Vector_d}


def vector(shape, dtype=np.float32):
    return vec_types[np.dtype(dtype)](shape if isinstance(shape, Sequence) else (shape,))
