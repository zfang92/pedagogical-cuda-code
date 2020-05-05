# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Here we test if CUDA kernels can be casted as class attributes, and if so, 
whether an imported kernels is slower than a normal one. The answers are: yes, 
no.

Workflow: fetch the kernel from kernel.py into __init__.Fetch.k__ via here
      --> from __init__.Fetch.k__, Timer fetches the kernel as its attribute
      --> timer1() executes the kernel
"""


import math

from numba import cuda
import numpy as np

from misc3_import_kernel.__init__ import Fetch
from misc3_import_kernel import kernel
from misc3_import_kernel.exe import Timer


# let's import the hypot kernel from the 'misc3_import_kernel' module
Fetch.k__ = getattr(kernel, 'k__hypot_in_kernel')

# now the good'ol hypot kernel
@cuda.jit
def k__hypot(a, b, c):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    if start >= a.shape[0]:
        return
    
    for idx in range(start, a.shape[0], stride):
        c[idx] = math.hypot(a[idx], b[idx])


# now supply the data and the specs for our kernels
n = int(1e6)
TYPE = 'float32'

a = np.random.uniform(-12, 12, n).astype(TYPE)
b = np.random.uniform(-12, 12, n).astype(TYPE)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(d_a)


# define convenience function
def timer2():
    k__hypot[40, 256](d_a, d_b, d_c)
    cuda.synchronize()


# make sure everything is correct
timer = Timer(d_a, d_b, d_c); timer.timer1()
c = d_c.copy_to_host()
np.testing.assert_almost_equal(np.hypot(a, b), c, decimal=5)


#======================================================================
for _ in range(5):
    timer.timer1(); timer2()
"""
%timeit -r 50 -n 10 timer.timer1()
%timeit -r 50 -n 10 timer2()
"""

