# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


import numpy as np


"""
TYPE can be any data type--float32, float64, int32, int64, etc. However, 
special attention has to be paid when using float32, because failing to convert
everyting into float32 will induce serious numerical errors, as outlined below.
"""
TYPE = 'float32'

def test_matmul(A, B):
    # this ensures A, B and result have the same data type
    result = np.zeros((A.shape[0],B.shape[1]), dtype=TYPE)

    for row in range(A.shape[0]):
        for col in range(B.shape[1]):

            tmp = 0.0  # this will induce problems when TYPE='float32'
            # tmp = np.array(0, dtype=TYPE)  # this is correct
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]

            result[row, col] = tmp

    return result


M, N = 128, 32
A = np.arange(M*N).reshape(M, N).astype(TYPE)
B = np.arange(M*N).reshape(N, M).astype(TYPE)

print(np.sum(np.matmul(A, B)-test_matmul(A, B)))

