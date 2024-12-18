# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport cython
from dp_mechanisms.delta_base cimport delta_base

cdef class Metric:
    cdef double[:] u
    cdef double gs
    cdef str name, abbr

    cdef delta_base get_delta(self, int i)
    cpdef double[:] get_u(self)
    cpdef double get_gs(self)
    cpdef str get_name(self)
    cpdef str get_abbr(self)