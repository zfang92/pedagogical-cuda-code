# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Benchmarking results (40 blocks with 256 threads per block) - 
1e6, SP:  (346 ± 11) µs;
1e6, DP:  (409 ± 21) µs;
1e7, SP:  (903 ± 29) µs;
1e7, DP: (1510 ± 32) µs.
==>  (62 ± 3) µs per 1e6 SP calculations, and the overhead was (284 ± 11) µs;
    (122 ± 4) µs per 1e6 DP calculations, and the overhead was (287 ± 21) µs.

One can conclude that for the calculation time, t_f64 = 2 * t_f32, and for the
overhead time, t_f64 = t_f32.

Key points:
1) 1d grids,
2) strides.

Refer to 'development note' for more notes.
"""


import math

from numba import cuda, jit
import numpy as np


# try 1e6/1e7 and f32/f64 for the following varibles for benchmarking
n = int(1e6)
TYPE = 'float32'

# let's write a custom CUDA kernel that calculates the hypotenuse
@cuda.jit
def k__hypot(a, b, c):
    """
    Below is a snapshot of tasks presented to one thread; 'start' has a 
    different value for each thread, and each thread takes on 
    len(range(start, a.shape[0], stride)) individual tasks.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # though unnecessary in the present case, below is always a good habit
    if start >= a.shape[0]:
        return
    
    for idx in range(start, a.shape[0], stride):
        c[idx] = math.hypot(a[idx], b[idx])


# now supply the data and the specs for our kernel
a = np.random.uniform(-12, 12, n).astype(TYPE)
b = np.random.uniform(-12, 12, n).astype(TYPE)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(d_a)


# define some convenience functions
def timer1():  # CPU no jit
    np.hypot(a, b)

@jit
def timer2():  # CPU with jit
    np.hypot(a, b)

def timer3():  # GPU single thread
    k__hypot[1, 1](d_a, d_b, d_c)
    cuda.synchronize()  # synchronization is important here

def timer4():  # GPU multiple threads
    k__hypot[40, 256](d_a, d_b, d_c)
    cuda.synchronize()  # synchronization is important here


# make sure everything is correct
timer4()
c = d_c.copy_to_host()
np.testing.assert_almost_equal(np.hypot(a, b), c, decimal=5)


#======================================================================
for _ in range(5):
    timer1(); timer2(); timer3(); timer4()
"""
%timeit timer1()
%timeit timer2()
%timeit -r 50 -n 10 timer3()
%timeit -r 50 -n 10 timer4()
"""

