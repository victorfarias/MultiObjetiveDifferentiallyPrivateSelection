cimport cython


cdef class delta_base:
    cdef double gs
    cdef double compute_next(self) noexcept nogil
    cdef void deallocate(self) noexcept nogil