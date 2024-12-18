# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

from dp_mechanisms.delta_base cimport delta_base
# from local_sensitivity.metric cimport Metric

import numpy as np

cdef class delta_ed_gs(delta_base):    

    cdef double compute_next(self) noexcept nogil:
        return 1.

cdef class delta_ed(delta_base):
    cdef double d, T, N, t

    def __cinit__(self, double[:] T, double[:] degrees, long i):        
        with nogil:
            self.t = 0        
            self.T = T[i]   
            self.N = degrees[i]
            if(self.N == 1):
                self.d = 0.
            else:
                self.d = self.T/((self.N)*(self.N-1))
            self.gs = 1.
    

    cdef double compute_next(self) noexcept nogil:
        cdef double N_ = self.N - self.t
        self.t = self.t + 1.

        if(N_ <= 2):
            return 1. 
        return 2/(N_-1)


# cdef class ED(Metric):    
#     cdef double[:] T, degrees
    
#     def __init__(self, g):
#         self.name = "Egocentric Density"
#         self.abbr = "ed"
#         self.gs = 1
#         self.u = g.ed()
#         self.T = g.T().astype(np.float)
#         self.degrees = g.degrees().astype(np.float)
    
#     cdef delta_base get_delta(self, int i): 
#         return delta_ed(self.T, self.degrees, i)