# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport numpy as np
import numpy as np

cpdef double[:] pareto_set_score2d(double[:] u1, double[:] u2) noexcept:
    cdef int n = len(u1)
    cdef double[:] ps = np.zeros_like(u1, dtype=np.double)
    cdef int i,j 
    for i in range(n): 
        for j in range(n):
            if(u1[i] < u1[j] and u2[i] < u2[j]):
                ps[i] -= 1
    return np.array(ps)

cpdef pareto_set_score(double[:,:] data) noexcept:
    cdef int n = data.shape[0]
    cdef int m = data.shape[1]
    cdef double[:] score = np.zeros(n, dtype=np.float64)
    
    cdef int i, j, k, flag
    for i in range(n):
        for k in range(n):
            if(i!=k):
                flag = 1
                for j in range(m):
                    if(data[i,j] > data[k,j]):
                        flag = 0
                        break
                score[i] -= flag
    return np.array(score)

cpdef pareto_domination_score(double[:,:] data) noexcept:
    cdef int n = data.shape[0]
    cdef int m = data.shape[1]
    cdef double[:] score = np.zeros(n, dtype=np.float64)
    
    cdef int i,j,k, flag
    for i in range(n):
        for k in range(n):
            if(i!=k):
                flag = 1
                for j in range(m):
                    if(data[i,j] <= data[k,j]):
                        flag = 0
                        break;
                score[i] += flag
    return np.array(score)