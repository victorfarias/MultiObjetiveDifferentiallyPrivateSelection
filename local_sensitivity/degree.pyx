# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

from dp_mechanisms.delta_base cimport delta_base
# from local_sensitivity.metric cimport Metric

cimport numpy as np
import numpy as np


# cdef class Degree(Metric):
    
#     def __init__(self, g):
#         self.name = "Degree"
#         self.abbr = "degree"
#         self.gs = 1
#         self.u = g.degrees().astype(np.float)
    
#     cdef delta_base get_delta(self, int i):
#         return delta_degree(i)


cdef class delta_degree(delta_base):

    def __cinit__(self, long i):
        pass

    cdef double compute_next(self) noexcept nogil:
        return 1.