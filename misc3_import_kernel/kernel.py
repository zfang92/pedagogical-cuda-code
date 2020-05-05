# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Here defines the kernel to be fetched by Fetch in __init__.py
"""


import math

from numba import cuda
import numpy as np


@cuda.jit
def k__hypot_in_kernel(a, b, c):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if start >= a.shape[0]:
        return
    
    for idx in range(start, a.shape[0], stride):
        c[idx] = math.hypot(a[idx], b[idx])

