# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from numba import cuda
import numpy as np


@cuda.jit
def k__abc1(a, b, c, d):
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    D, M = a.shape

    for x in range(start_x, D, stride_x):
        for y in range(start_y, M, stride_y):
            d[x, y] = a[x, y] + b[0] * c[x, y]

@cuda.jit
def k__abc2(a, b, c, d):
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    D, M = a.shape

    for x in range(start_x, D, stride_x):
        for y in range(start_y, M, stride_y):
            d[x, y] = a[x, y] + b * c[x, y]


D, M = 1000, 1000
griddim2, blockdim2 = (8, 64), (4, 64)

a = np.ones((D,M), dtype='float64')
b = np.array([2.0])
c = 3.0 * np.ones((D,M), dtype='float64')

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)


def timer1():
    k__abc1[griddim2, blockdim2](d_a, d_b, d_c, d_a)
    cuda.synchronize()

def timer2():
    k__abc2[griddim2, blockdim2](d_a, b[0], d_c, d_a)
    cuda.synchronize()


#======================================================================
for _ in range(5):
    timer1(); timer2()
"""
%timeit -r 50 -n 10 timer1()
%timeit -r 50 -n 10 timer2()
"""

