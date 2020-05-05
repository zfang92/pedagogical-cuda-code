# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Key points:
1) ufuncs,
2) memory transfer and general workflow.

Refer to 'development note' for more notes.
"""


import math

from numba import cuda, vectorize
import numpy as np


@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32, float32)'], target='cuda')
def weigh(values, weights):
    return values * weights

@vectorize(['float32(float32)'], target='cuda')
def activate(values):
    return (math.exp(values) - math.exp(-values)) \
           / (math.exp(values) + math.exp(-values))


# now supply the data
n = int(1e6)

greyscales = np.floor(np.random.uniform(0, 255, n).astype(np.float32))
weights = np.random.normal(0.5, 0.1, n).astype(np.float32)

# step 1: copy data to device
d_greyscales = cuda.to_device(greyscales)
d_weights = cuda.to_device(weights)

# step 2: allocate space on device for all the GPU variables
d_normalized = cuda.device_array(shape=(n,), dtype=np.float32)
d_weighted = cuda.device_array(shape=(n,), dtype=np.float32)
d_activated = cuda.device_array(shape=(n,), dtype=np.float32)

# step 3: perform (a series of) calculations on device
def timer():  # define a convenience function
    normalize(d_greyscales, out=d_normalized)
    weigh(d_normalized, d_weights, out=d_weighted)
    activate(d_weighted, out=d_activated)
    cuda.synchronize()

# step 4: copy final result back to host
activated = d_activated.copy_to_host()


#======================================================================
for _ in range(5):
    timer()
"""
%timeit -r 50 -n 10 timer()
"""

