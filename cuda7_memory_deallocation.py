# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Key points:
1) as described in the first choice,
2) as described in the second choice.

Refer to 'development note' for more notes.
"""


import time

from numba import cuda
import numpy as np


"""
Choices:
1 - Show that once created, device arrays will stay there in the device memory 
    the same way as regular variables in the host memory. This phenomenon is 
    not to be confused with the 'deallocation behavior' mentioned in section 
    3.3.7 of Numba's documentation, which merely describes a mechanism also 
    existing in the host memory.
2 - Show that both host and device arrays created inside a host function will 
    be automatically deallocated, or 'freed up', once the function is finished. 
    Open your task manager and be ready.
"""
choice = 2


if choice == 1:
    n = int(1e6)

    base = np.ones(n, dtype='float64')

    d_array01 = cuda.to_device(base); d_array02 = cuda.to_device(base)
    d_array03 = cuda.to_device(base); d_array04 = cuda.to_device(base)
    d_array05 = cuda.to_device(base); d_array06 = cuda.to_device(base)
    d_array07 = cuda.to_device(base); d_array08 = cuda.to_device(base)
    d_array09 = cuda.to_device(base); d_array10 = cuda.to_device(base)
    d_array11 = cuda.to_device(base); d_array12 = cuda.to_device(base)
    d_array13 = cuda.to_device(base); d_array14 = cuda.to_device(base)
    d_array15 = cuda.to_device(base); d_array16 = cuda.to_device(base)
    d_array17 = cuda.to_device(base); d_array18 = cuda.to_device(base)
    d_array19 = cuda.to_device(base); d_array20 = cuda.to_device(base)
    d_array21 = cuda.to_device(base); d_array22 = cuda.to_device(base)
    d_array23 = cuda.to_device(base); d_array24 = cuda.to_device(base)
    d_array25 = cuda.to_device(base); d_array26 = cuda.to_device(base)
    d_array27 = cuda.to_device(base); d_array28 = cuda.to_device(base)
    d_array29 = cuda.to_device(base); d_array30 = cuda.to_device(base)


if choice == 2:
    n = int(1e8)  # 10^8 float64 numbers take up roughly 800 MB of space

    base = np.ones(n, dtype='float64')
    
    def test_deallocation():
        d_array1 = cuda.to_device(1*base)
        d_array2 = cuda.to_device(2*base)
        d_array3 = cuda.to_device(3*base)
        d_array4 = cuda.to_device(4*base)
        d_array5 = cuda.to_device(5*base)
        d_array6 = cuda.to_device(6*base)

        print('Done.')
        time.sleep(10)

        return


    test_deallocation()

