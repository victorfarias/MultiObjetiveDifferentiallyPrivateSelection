cimport cython

cdef class delta_base:
    cdef double compute_next(self) noexcept nogil:
        pass
    cdef void deallocate(self) noexcept nogil:
        pass