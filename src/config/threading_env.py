"""Limit BLAS/OpenMP threads before mixing OpenCV and scikit-learn.

OpenCV and HistGradientBoostingRegressor both use OpenMP; loading both in one
process without thread limits can segfault on macOS/Anaconda builds.
"""

from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_cpu_threads(max_threads: int = 1) -> None:
    value = str(max_threads)
    for name in _THREAD_VARS:
        os.environ.setdefault(name, value)
