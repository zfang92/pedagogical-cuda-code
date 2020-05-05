# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Key points:
1) device functions,
2) ufuncs calling device functions.

Refer to 'development note' for more notes.
"""


import math

from numba import cuda, vectorize
import numpy as np


# this is a CUDA device function and is not directly callable from the CPU
@cuda.jit(device=True)  # N.B., device=False means this is a custom CUDA kernel
def polar_to_cartesian(rho, theta):
    """
    Note that a device function is just a function spawned on the GPU. It is 
    NOT to be parallelized as is the case for ufuncs and custom kernels.
    """
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return x, y  # a device function can have return value(s)

# this is a ufunc that calls the device function above
@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# now supply the data
n = int(1e6)

rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

# the routine--copy, allocate, perform, transfer
d_rho1 = cuda.to_device(rho1)
d_rho2 = cuda.to_device(rho2)
d_theta1 = cuda.to_device(theta1)
d_theta2 = cuda.to_device(theta2)

d_distance = cuda.device_array_like(rho1)

def timer():  # define a convenience function
    polar_distance(d_rho1, d_theta1, d_rho2, d_theta2, out=d_distance)
    cuda.synchronize()

distance = d_distance.copy_to_host()


#======================================================================
for _ in range(5):
    timer()
"""
%timeit -r 50 -n 10 timer()
"""

