import numpy as np

from libc.math cimport fabs
from dp_mechanisms.delta_base cimport delta_base
from local_sensitivity.metric cimport Metric


cdef double f1(float tp, float p) noexcept nogil:
    return fabs((tp/p)-((tp+1)/(p+1)))

cdef double f2(float tp, float p) noexcept nogil:
    return fabs((tp/p)-((tp)/(p+1)))

cdef double f3(float tp, float p) noexcept nogil:
    if tp>1 and p>1:
        return fabs((tp/p)-((tp-1)/(p-1)))
    else :
        return 0

cdef double f4(float tp, float p) noexcept nogil:
    if p>1 and tp<p:
        return fabs((tp/p)-((tp)/(p-1)))
    else :
        return 0

cdef double f5(float tp, float p) noexcept nogil:
    if p>0 and tp>0:
        return fabs((tp/p)-((tp-1)/(p)))
    else :
        return 0

cdef double f6(float tp, float p) noexcept nogil:
    if p>0 and tp<p:
        return fabs((tp/p)-((tp+1)/(p)))
    else :
        return 0

cdef double f(float tp, float p) noexcept nogil:
    return max(f1(tp, p), max(f2(tp, p), max(f3(tp, p), max(f4(tp, p), max(f5(tp, p), f6(tp, p))))))

cdef double ls(float tp, float p, float t) noexcept nogil:
    cdef double tp_ = max(tp - max(t-(p-tp),0),1) 
    cdef double p_ = max(p-t,1)
    return f(tp_, p_)

cdef class Tpr(Metric):    
    cdef double[:] tps
    cdef double[:] ps
    
    def __init__(self, double[:] tps, double[:] ps):        
        self.gs = 1.
        self.tps = tps
        self.ps = ps
    
    cdef delta_base get_delta(self, int i): 
        return DeltaTpr(self.tps[i], self.ps[i])


cdef class DeltaTpr(delta_base):

    cdef double tp
    cdef double p
    cdef int t
    
    def __init__(self, double tp, double p):
        self.tp = tp
        self.p = p
        self.t = 0

    cdef double compute_next(self) noexcept nogil:
        self.t += 1
        return ls(self.tp, self.p, self.t-1)
