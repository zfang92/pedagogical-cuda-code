# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from numba import cuda
import numpy as np


@cuda.jit
def k__linearop1d(Q, r, S, T):
    start_b = cuda.grid(1)
    stride_b = cuda.gridsize(1)

    if start_b >= len(T):
        return

    for b in range(start_b, len(T), stride_b):
        T[b] = Q[b] + r * S[b]


S = np.array([1.2, np.inf, -np.inf])
r = 3.4
Q = np.array([5.6, 7.8, np.inf])

d_S = cuda.to_device(S)
d_Q = cuda.to_device(Q)
d_T = cuda.device_array((3,))

k__linearop1d[40, 256](d_Q, r, d_S, d_T)

# we should expect [9.68, inf, nan]
T = d_T.copy_to_host()

