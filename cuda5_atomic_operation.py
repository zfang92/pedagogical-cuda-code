# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

Key points:
1) atomic operations,
2) shared memory usage (buffering) when strides are present, and when block 
   size is not equal to the size of the array in shared memory.

Refer to 'development note' for more notes.
"""


from numba import cuda, jit, types
import numpy as np


@jit(nopython=True)
def histogram(x, xmin, xmax, histogram):
    '''
    Increment bin counts in histogram_out, given histogram range [xmin, xmax).
    '''
    bin_width = (xmax - xmin) / nbins
    
    for element in x:
        bin_number = np.int32((element-xmin)/bin_width)

        if 0 <= bin_number < nbins:
            # only increment if in range
            histogram[bin_number] += 1

@cuda.jit
def k__histogram(x, xmin, xmax, histogram):
    '''
    Increment bin counts in histogram_out, given histogram range [xmin, xmax).
    '''
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # though unnecessary in the present case, below is always a good habit
    if start >= x.shape[0]:
        return
    
    # real calculations
    bin_width = (xmax - xmin) / nbins
    
    for idx in range(start, x.shape[0], stride):
        bin_number = np.int32((x[idx]-xmin)/bin_width)

        if 0 <= bin_number < nbins:
            # watch out!
            cuda.atomic.add(histogram, bin_number, 1)

@cuda.jit
def k__histogram_shmem(x, xmin, xmax, histogram):
    """
    Recall the common usage of shared memory, i.e., caching and buffering, this
    exmample demonstrates usage #2. Here we are benefited by putting the write-
    intensive operations into the shared memory and then collect the tally once
    a block has finished its calculation.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # though unnecessary in the present case, below is always a good habit
    if start >= x.shape[0]:
        return

    # allocate space in the shared memory whose size must be a constant
    tid = cuda.threadIdx.x
    hist_buffer = cuda.shared.array(nbins, types.int32)

    for i in range(nbins):
        hist_buffer[i] = 0

    cuda.syncthreads()  # this is important

    # real calculations
    bin_width = (xmax - xmin) / nbins

    for idx in range(start, x.shape[0], stride):
        bin_number = np.int32((x[idx]-xmin)/bin_width)

        if 0 <= bin_number < nbins:
            # writing in the shared memory
            cuda.atomic.add(hist_buffer, bin_number, 1)

    cuda.syncthreads()  # this is important

    # move the tallied result back to the output array
    if tid < nbins:  # assuming griddim >= nbins
        cuda.atomic.add(histogram, tid, hist_buffer[tid])


# now supply the data and the specs for our kernel
x = np.random.normal(size=int(1e7), loc=0, scale=1).astype(np.float32)
xmin = np.float32(-4.0)
xmax = np.float32(4.0)

d_x = cuda.to_device(x)

nbins = 10

blockdim = 40
griddim = 256


# define some convenience functions
def timer1():  # CPU with jit
    histogram(x, xmin, xmax, hist)

def timer2():  # GPU
    k__histogram[blockdim, griddim](d_x, xmin, xmax, d_hist)
    cuda.synchronize()

def timer3():  # GPU with shared memory
    k__histogram_shmem[blockdim, griddim](d_x, xmin, xmax, d_hist)
    cuda.synchronize()


# make sure everything is correct
hist = np.zeros(shape=nbins, dtype=np.int32); timer1();
d_hist = cuda.to_device(np.zeros(shape=nbins, dtype=np.int32)); timer2()
hist_gpu = d_hist.copy_to_host()
np.testing.assert_equal(hist, hist_gpu)

hist = np.zeros(shape=nbins, dtype=np.int32); timer1()
d_hist = cuda.to_device(np.zeros(shape=nbins, dtype=np.int32)); timer3()
hist_gpu = d_hist.copy_to_host()
np.testing.assert_equal(hist, hist_gpu)


#======================================================================
for _ in range(5):
    hist = np.zeros(shape=nbins, dtype=np.int32); timer1()
    d_hist = cuda.to_device(np.zeros(shape=nbins, dtype=np.int32)); timer2()
    d_hist = cuda.to_device(np.zeros(shape=nbins, dtype=np.int32)); timer3()
"""
%timeit timer1()
%timeit -r 50 -n 10 timer2()
%timeit -r 50 -n 10 timer3()
"""

