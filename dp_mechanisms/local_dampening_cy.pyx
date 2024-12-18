# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport cython

cimport numpy as np
import numpy as np

from dp_mechanisms.delta_base cimport delta_base
# from libc.math cimport fabs

cpdef double D_(double u_, delta_base delta) noexcept nogil:    
    cdef int sign =  1
    if u_<0:
        sign = -1
    u_ = u_*sign

    cdef double gs = delta.gs
    cdef double new_delta
    cdef double delta_cum = 0.
    cdef double new_delta_cum = 0.

    cdef int t = 0    
    while True:        
        new_delta = delta.compute_next()         
        new_delta_cum = delta_cum + new_delta  
        # with gil:
            # print(u_, new_delta, new_delta_cum, t, delta.gs)   
        # if(new_delta >= gs - 1e-7):
        #     with gil:
        #         print("1", u_, new_delta, new_delta_cum, t, delta.gs)   
        #     return (t + (u_ - delta_cum)/gs)*sign        
        if(u_>=delta_cum and u_<new_delta_cum):
            return ((u_ - delta_cum)/new_delta + t)*sign
        delta_cum = new_delta_cum        
        t += 1


# cpdef double[:] D_cy(double[:] u, Delta):
#     cdef int nu = u.shape[0]
#     cdef double[:] du = np.empty(nu, dtype=np.float64)

#     cdef int i    
#     for i in range(nu):        
#         print(i, nu)
#         delta = Delta(i)
#         du[i] = D_(u[i], delta)
#     return du
