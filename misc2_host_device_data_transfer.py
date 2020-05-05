# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from numba import cuda
import numpy as np


n = int(2e4)  # this is not to exceed 10^7

# supply data
data = np.random.normal(size=n, loc=0, scale=1).astype('float64')


# define convenience function
def timer():
    d_data = cuda.to_device(data)
    d_data.copy_to_host()


#======================================================================
for _ in range(5):
    timer()
"""
%timeit -r 50 -n 10 timer()
"""

