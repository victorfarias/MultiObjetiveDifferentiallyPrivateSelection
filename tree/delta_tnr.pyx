# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from dp_mechanisms.delta_base cimport delta_base
from local_sensitivity.metric cimport Metric


cdef double f1(float tn, float n) noexcept nogil:
    return fabs((tn/n)-((tn+1)/(n+1)))

cdef double f2(float tn, float n) noexcept nogil:
    return fabs((tn/n)-((tn)/(n+1)))

cdef double f3(float tn, float n) noexcept nogil:
    if tn>1 and n>1:
        return fabs((tn/n)-((tn-1)/(n-1)))
    else :
        return 0

cdef double f4(float tn, float n) noexcept nogil:
    if n>1 and tn<n:
        return fabs((tn/n)-((tn)/(n-1)))
    else :
        return 0

cdef double f5(float tn, float n) noexcept nogil:
    if n>0 and tn>0:
        return fabs((tn/n)-((tn-1)/(n)))
    else :
        return 0

cdef double f6(float tn, float n) noexcept nogil:
    if n>0 and tn<n:
        return fabs((tn/n)-((tn+1)/(n)))
    else:
        return 0

cdef double f(float tn, float n) noexcept nogil:
    return max(f1(tn, n), max(f2(tn, n), max(f3(tn, n), max(f4(tn, n), max(f5(tn, n), f6(tn, n))))))

cdef double ls(float tn, float n, float t) noexcept nogil:
    cdef double tn_ = max(tn - max(t-(n-tn),0),1) 
    cdef double n_ = max(n-t,1)
    return f(tn_, n_)

cdef class Tnr(Metric):    
    cdef double[:] tns
    cdef double[:] ns
    
    def __init__(self, double[:] tns, double[:] ns):
        self.name = "True Positive Rate"
        self.abbr = "tnr"
        self.gs = 1
        self.tns = tns
        self.ns = ns
    
    cdef delta_base get_delta(self, int i): 
        return DeltaTnr(self.tns[i], self.ns[i])

cdef class DeltaTnr(delta_base):

    cdef double tn
    cdef double n
    cdef int t
    
    def __cinit__(self, double tn, double n):
        self.tn = tn
        self.n = n
        self.t = 0

    cdef double compute_next(self) noexcept nogil:
        self.t += 1
        return ls(self.tn, self.n, self.t-1)
        
