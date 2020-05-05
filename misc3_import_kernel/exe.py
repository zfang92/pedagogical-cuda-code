# -*- coding: utf-8 -*-
"""
@author: Zheng Fang
"""


from numba import cuda

from misc3_import_kernel.__init__ import Fetch


class Timer:
    def __init__(self, d_a, d_b, d_c):
        self.d_a = d_a
        self.d_b = d_b
        self.d_c = d_c
        self.k__hypot_in_exe = Fetch.k__

    def timer1(self):
        self.k__hypot_in_exe[40, 256](self.d_a, self.d_b, self.d_c)
        cuda.synchronize()

