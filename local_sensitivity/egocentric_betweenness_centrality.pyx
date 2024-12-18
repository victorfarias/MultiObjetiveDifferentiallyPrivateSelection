# cython: infer_types=True, language_level=3, cdivision=True, boundscheck=False, wraparound=False

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport numpy as np
import numpy as np

from local_dampening.delta_base cimport delta_base
from local_sensitivity.metric cimport Metric

cpdef double gs_ebc(long[:] degrees):
    cdef unsigned long md = np.amax(degrees)
    return max((md)*(md-1)/4, md)

cpdef double gs_ebc_md(long max_degree):
    # cdef unsigned long md = np.amax(degrees)
    return max((max_degree)*(max_degree-1)/4, max_degree)

cdef class EBC(Metric):
    
    cdef long[:] degrees
    cdef long max_degree
    
    def __init__(self, g):
        self.name = "Egocentric Betweeness Centrality"
        self.abbr = "ebc"
        self.degrees = g.degrees().astype(np.int)
        self.gs = gs_ebc(self.degrees)
        self.u = g.ebc()
        self.max_degree = g.max_degree()
    
    cdef delta_base get_delta(self, int i):
        return delta_ebc(self.degrees, self.max_degree, i)


cdef class delta_ebc(delta_base):
    cdef double degree, t

    def __cinit__(self, long[:] degrees, long max_degree, long i):

        self.degree = degrees[i]
        self.t = 0        
        self.gs = gs_ebc_md(max_degree)


    cdef double compute_next(self) noexcept nogil:
        cdef double ds = self.degree + self.t
        cdef double ls = max(((ds)*(ds-1))/4., ds)
        self.t += 11
        return min(ls, self.gs)