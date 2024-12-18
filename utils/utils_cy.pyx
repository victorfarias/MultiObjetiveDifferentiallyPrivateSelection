# # cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# # distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# # distutils: language = c++

# from dp_mechanisms.delta_base cimport delta_base

# cpdef compute_next(delta_base delta) noexcept nogil:
#     return delta.compute_next()