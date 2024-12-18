import numpy as np

from dp_mechanisms.delta_base cimport delta_base

cdef class DeltaWeighted (delta_base):

    cdef delta_base delta1
    cdef delta_base delta2
    cdef double w1
    cdef double w2

    def __cinit__(self, delta_base delta1, delta_base delta2, double w1, double w2):
        self.delta1 = delta1
        self.delta2 = delta2
        self.w1 = w1
        self.w2 = w2

    cdef double compute_next(self) noexcept nogil:
        return self.w1*self.delta1.compute_next() + self.w2*self.delta1.compute_next()
    