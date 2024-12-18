# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++


cimport cython

cdef class Metric:
    cdef delta_base get_delta(self, int i):
        pass

    cpdef double[:] get_u(self):
        return self.u

    cpdef double get_gs(self):
        return self.gs

    cpdef str get_name(self):
        return self.name

    cpdef str get_abbr(self):
        return self.abbr