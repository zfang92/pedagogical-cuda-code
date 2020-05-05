# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from numba import cuda, vectorize
import numpy as np


@vectorize(['float64(float64, float64, float64, float64)'], target='cuda')
def abcd(a, b, c, d):
    return a + b * c / d

@cuda.jit
def k__abcd(a, b, c, d, e):
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    D, M = a.shape

    for x in range(start_x, D, stride_x):
        for y in range(start_y, M, stride_y):
            e[x, y] = a[x, y] + b[x, y] * c[x, y] / d[x, y]


D, M = 2000, 2000
griddim2, blockdim2 = (8, 64), (4, 64)

a = np.random.uniform(0, 1, (D,M)).astype('float64')
b = np.random.uniform(0, 1, (D,M)).astype('float64')
c = np.random.uniform(0, 1, (D,M)).astype('float64')
d = np.random.uniform(0, 1, (D,M)).astype('float64')


d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)
d_d = cuda.to_device(d)

d_e = cuda.device_array(shape=(D,M), dtype='float64')


def timer1():
    abcd(d_a, d_b, d_c, d_d, out=d_e)
    cuda.synchronize()

def timer2():
    k__abcd[griddim2, blockdim2](d_a, d_b, d_c, d_d, d_e)
    cuda.synchronize()


#======================================================================
for _ in range(5):
    timer1(); timer2()
"""
%timeit -r 50 -n 10 timer1()
%timeit -r 50 -n 10 timer2()
"""

