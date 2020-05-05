# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

This particular example shows no speed up from CPU implementation, as expected.
The reason is that the input size is not large enough to compensate for the 
extra time introduce by the computational overheads.

To be able to handle input data with size larger than the total number of 
threads, one needs more advanced approaches to deal with strides. There are 
standard funcitons in the CUDA library for that.

Another thing to be noted is that one should be careful with data types. When 
using float32, all three functions below fail because they do not have 
everything in f32 (so there's problem when the f64->f32 conversion happens).

Key points:
1) 2d grids,
2) shared memory usage (caching).

Refer to 'development note' for more notes.
"""


from numba import cuda, jit, types
import numpy as np


# naive cpu implementation
@jit(nopython=True)
def matmul(A, B):
    result = np.zeros_like(C)

    for row in range(A.shape[0]):
        for col in range(B.shape[1]):

            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]

            result[row, col] = tmp

    return result

# let's first write a kernel that do not take advantage of the shared memory
@cuda.jit
def k__matmul(A, B, C):
    # no strides
    row, col = cuda.grid(2)

    # a necessary check
    if row >= A.shape[0] or col >= B.shape[1]:
        return

    tmp = 0.0
    for k in range(A.shape[1]):
        tmp += A[row, k] * B[k, col]

    C[row, col] = tmp

# let's now write a kernel that takes advantage of the shared memory
# a better example (with strides!) is presented in the next program
@cuda.jit
def k__matmul_sharedmem(A, B, C):
    # no strides
    row, col = cuda.grid(2)

    # a necessary check
    if row >= A.shape[0] or col >= B.shape[1]:
        return

    # shared memory allocation
    tid_x, tid_y = cuda.threadIdx.x, cuda.threadIdx.y
    a_cache = cuda.shared.array(blockdim, types.float64)
    b_cache = cuda.shared.array(blockdim, types.float64)

    a_cache[tid_x, tid_y] = A[row, tid_y]
    b_cache[tid_x, tid_y] = B[tid_x, col]

    # synchronization among threads is very important when using shared memory
    cuda.syncthreads()

    # real calculations
    tmp = 0.0
    for k in range(A.shape[1]):
        tmp += a_cache[tid_x, k] * b_cache[k, tid_y]

    C[row, col] = tmp


# set the dimensions, note that we cannot have strides in the kernels above
M, N = 128, 32
blockdim = (N, N)
griddim = (int(M/N), int(M/N))

# now supply data
A = np.arange(M*N).reshape(M, N).astype(np.float64)
B = np.arange(M*N).reshape(N, M).astype(np.float64)
C = np.zeros((M,M)).astype(np.float64)

d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.to_device(C)


# define some convenience functions
def timer1():  # CPU with jit
    return matmul(A, B)

def timer2():  # GPU, no shared memory involved
    k__matmul[griddim, blockdim](d_A, d_B, d_C)
    cuda.synchronize()

def timer3():  # GPU with shared memory tehnique
    k__matmul_sharedmem[griddim, blockdim](d_A, d_B, d_C)
    cuda.synchronize()


# make sure the results are correct
np.testing.assert_array_equal(A@B, timer1())

timer2()
C_timer2 = d_C.copy_to_host()
np.testing.assert_array_equal(A@B, C_timer2)

timer3()
C_timer3 = d_C.copy_to_host()
np.testing.assert_array_equal(A@B, C_timer3)


#======================================================================
for _ in range(5):
    timer1(); timer2(); timer3();
"""
%timeit timer1()
%timeit -r 50 -n 10 timer2()
%timeit -r 50 -n 10 timer3()
"""

