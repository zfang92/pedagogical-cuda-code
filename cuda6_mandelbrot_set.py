# -*- coding: utf-8 -*-
"""
@author: Zheng Fang

It's interesting to see some usage of 'scope' here--a variable/function defined 
outside a function/class could be used and modified by that function/class, but 
a variable/function newly defined within a function/class (and is not equated
with an argument) will never leak to the outside.

Additionally, since Python passes references (a.k.a. pointers) into functions, 
modifying an argument within a function would result in its counterpart being 
modified outside that function, as it is the case for 'd_image' in 'timer2()'.

Key points:
1) 2d grids calling device functions,
2) calling kernels from wrapper methods within classes.

Refer to 'development note' for more notes.
"""

from matplotlib.pyplot import imshow
from numba import cuda, jit
import numpy as np


# let's first create a Mandelbrot set using the CPU
@jit(nopython=True)
def mandelbrot(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number, determine if it's a
    candidate for membership in the Mandelbrot set.
    """
    c = complex(x, y)
    z = 0.0j

    for i in range(max_iters):
        z = z * z + c

        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters

@jit
def fractal(x_min, x_max, y_min, y_max, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixlsize_x = (x_max - x_min) / height
    pixlsize_y = (y_max - y_min) / width

    for x in range(height):
        real = x_min + x * pixlsize_x

        for y in range(width):
            imag = y_min + y * pixlsize_y
            image[x, y] = mandelbrot(real, imag, iters)

# now write a GPU version
mandelbrot_gpu = cuda.jit(device=True)(mandelbrot)  # an interesting syntax

@cuda.jit
def k__fractal(x_min, x_max, y_min, y_max, image, iters):
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    height = image.shape[0]
    width = image.shape[1]

    # a necessary check
    if start_x >= height or start_y >= width:
        return

    pixlsize_x = (x_max - x_min) / height
    pixlsize_y = (y_max - y_min) / width

    for x in range(start_x, height, stride_x):
        real = x_min + x * pixlsize_x

        for y in range(start_y, width, stride_y):
            imag = y_min + y * pixlsize_y
            image[x, y] = mandelbrot_gpu(real, imag, iters)


# now supply the data and the specs for our kernel
image = np.zeros((1536, 1024), dtype=np.int32)
x_min, x_max, y_min, y_max = -2.0, 1.0, -1.0, 1.0
iters = 20

d_image = cuda.device_array_like(image)

blockdim = (16, 16)
griddim = (8, 8)


# define some convenience functions, this time making use of Class
def timer1():  # CPU with jit
    fractal(x_min, x_max, y_min, y_max, image, iters)

def timer2():  # GPU
    k__fractal[griddim, blockdim](x_min, x_max, y_min, y_max, d_image, iters)
    cuda.synchronize()

class Timer3:  # GPU kernel wrapped by a Class
    def __init__(self, x_min, x_max, y_min, y_max, image, iters):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.d_image = cuda.device_array_like(image)
        self.iters = iters

    def do(self):
        k__fractal[griddim, blockdim](self.x_min, self.x_max, self.y_min, 
                                      self.y_max, self.d_image, self.iters)
        cuda.synchronize()

    def check(self):
        # for benchmarking purpose we are splitting 'do' and 'check'
        return (self.d_image).copy_to_host()


# make sure everything is correct
timer1()
timer3 = Timer3(x_min, x_max, y_min, y_max, image, iters)
timer3.do()
image_gpu = timer3.check()
np.testing.assert_equal(image, image_gpu)

imshow(image_gpu.T)


#======================================================================
for _ in range(5):
    timer1(); timer2(); timer3.do()
"""
%timeit timer1()
%timeit -r 50 -n 10 timer2()
%timeit -r 50 -n 10 timer3.do()
"""

