# cython: infer_types=False, language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport cython
cimport numpy as np
import numpy as np

from dp_mechanisms.delta_base cimport delta_base
from local_sensitivity.metric cimport Metric

from cpython cimport PyObject, Py_INCREF, Py_DECREF

from libc.stdlib cimport malloc, free

# ctypedef PyObject *PyObject_p

cpdef gs_pareto(n):
    return n-1

@cython.final
cdef class delta_pareto(delta_base):
    cdef double [:,:] data
    cdef double sensitivity1, sensitivity2
    cdef double x_left_strip1, x_right_strip1, y_upper_strip1, y_lower_strip1
    cdef double x_left_strip2, x_right_strip2, y_upper_strip2, y_lower_strip2
    
    def __cinit__(self, data, int index_element, double sensitivity1, double sensitivity2):        
        self.gs = data.shape[0]
        self.sensitivity1 = sensitivity1
        self.sensitivity2 = sensitivity2
        # np.copyto(data[:,0], np.divide(data[:,0], sensitivity1))
        # np.copyto(data[:,1], np.divide(data[:,1], sensitivity2)) 
        
        x_element = data[index_element, 0]
        y_element = data[index_element, 1]

        data = np.delete(data, index_element, axis=0)        

        self.data = data

        self.x_left_strip1 = x_element - 2*sensitivity1
        self.x_right_strip1 = x_element
        self.y_upper_strip1 = y_element
        self.y_lower_strip1 = y_element - 2*sensitivity2        

        self.x_left_strip2 = x_element
        self.x_right_strip2 = x_element + 2*sensitivity1
        self.y_upper_strip2 = y_element + 2*sensitivity2
        self.y_lower_strip2 = y_element 
        
    cpdef bint is_inside_strip(self, double x, double y, double x_left_strip, double x_right_strip, double y_upper_strip, double y_lower_strip) noexcept nogil:
        if(x >= x_left_strip and y >= y_lower_strip and (x <= x_right_strip or y <= y_upper_strip)):
            return True
        else:
            return False
    
    cpdef bint is_inside_first_strip(self, double x, double y) noexcept nogil:        
        return self.is_inside_strip(x, y, self.x_left_strip1, self.x_right_strip1, self.y_upper_strip1, self.y_lower_strip1)

    cpdef bint is_inside_second_strip(self, double x, double y) noexcept nogil:
        return self.is_inside_strip(x, y, self.x_left_strip2, self.x_right_strip2, self.y_upper_strip2, self.y_lower_strip2)

    cpdef double sensitivity_first_strip(self) noexcept nogil:
        cdef int count = 0,i
        cdef double x,y
        for i in range(self.data.shape[0]):
            x = self.data[i, 0]
            y = self.data[i, 1]            
            if(self.is_inside_first_strip(x,y) == True):                
                count += 1
        self.x_left_strip1 = self.x_left_strip1 - 2*self.sensitivity1
        self.x_right_strip1 = self.x_right_strip1 + 2*self.sensitivity1
        self.y_upper_strip1 = self.y_upper_strip1 + 2*self.sensitivity2
        self.y_lower_strip1 = self.y_lower_strip1 - 2*self.sensitivity2        
        return count 

    cpdef double sensitivity_second_strip(self) noexcept nogil:
        cdef int count = 0,i
        cdef double x,y
        for i in range(self.data.shape[0]):
            x = self.data[i, 0]
            y = self.data[i, 1]            
            if(self.is_inside_second_strip(x,y) == True):                
                count += 1
        
        self.x_left_strip2  = self.x_left_strip2  - 2*self.sensitivity1
        self.x_right_strip2 = self.x_right_strip2 + 2*self.sensitivity1
        self.y_upper_strip2 = self.y_upper_strip2 + 2*self.sensitivity2
        self.y_lower_strip2 = self.y_lower_strip2 - 2*self.sensitivity2        
        return count 

    cpdef double compute_next(self) noexcept nogil:
        return max(self.sensitivity_first_strip(), self.sensitivity_second_strip())



@cython.final
cdef class delta_pareto_set_ls(delta_base):

    cdef double[:] xs, ys, sensitivities_x, sensitivities_y
    cdef double x, y
    # cdef double sensitivity_x, sensitivity_y, x, y
    cdef PyObject **deltas_x
    cdef PyObject **deltas_y
    cdef int n,i
                  

    def __cinit__(self, int index_element, double[:] u1, double[:] u2, Delta1, Delta2):
        
        self.xs = u1
        self.ys = u2

        self.n = self.xs.shape[0]
        self.i = index_element
        self.x = self.xs[index_element]
        self.y = self.ys[index_element]

        
        self.gs = self.n-1
        
        self.sensitivities_x = np.zeros(self.n)
        self.sensitivities_y = np.zeros(self.n)

        self.deltas_x = <PyObject **> malloc(self.n * sizeof(PyObject *))
        self.deltas_y = <PyObject **> malloc(self.n * sizeof(PyObject *))
            
        cdef int i
        cdef delta_base tmp1, tmp2

        # start_time = time.time()

        for i in range(self.n):
            
            tmp1 = Delta1(i)
            Py_INCREF(tmp1)            
            self.deltas_x[i] = <PyObject *>tmp1

            tmp2 = Delta2(i)
            Py_INCREF(tmp2)
            self.deltas_y[i] = <PyObject *>tmp2                     
        
    cdef bint has_crossed(self, int j) noexcept nogil:
        cdef double x_border1 = self.x + self.sensitivities_x[self.i]
        cdef double y_border1 = self.y + self.sensitivities_y[self.i]
        cdef double x_border2 = self.x - self.sensitivities_x[self.i]
        cdef double y_border2 = self.y - self.sensitivities_y[self.i]

        cdef double x_j = self.xs[j]
        cdef double y_j = self.ys[j]        
        cdef double sensitivity_x_j = self.sensitivities_x[j]
        cdef double sensitivity_y_j = self.sensitivities_y[j]

        
        if(j!=self.i and x_j > self.x  and y_j > self.y and ((x_j - sensitivity_x_j < x_border1) or (y_j - sensitivity_y_j < y_border1))):
            return True
        elif(j!=self.i and ( x_j <= self.x or y_j <= self.y) and (x_j + sensitivity_x_j > x_border2) and  (y_j + sensitivity_y_j > y_border2)):
            return True
        return False
 
    cdef void update(self, int j) noexcept nogil:            
        self.sensitivities_x[j] += (<delta_base>self.deltas_x[j]).compute_next()
        self.sensitivities_y[j] += (<delta_base>self.deltas_y[j]).compute_next()            
        
    cdef double compute_next(self) noexcept nogil:
        cdef int count = 0, j  
        # with gil:
        #     start_time = time.time()
        for j in range(self.n):
            # with gil:
            self.update(j)        
            if(self.has_crossed(j)):
                count += 1               
        # with gil:
        #     print("2.  %s seconds ---" % (time.time() - start_time))
        
        return count

    def __dealloc__(self):
        cdef int i
        for i in range(self.n):
            Py_DECREF( <object>self.deltas_x[ i ] )
            Py_DECREF( <object>self.deltas_y[ i ] )
        free(self.deltas_x)
        free(self.deltas_y)


    # def __cinit__(self, int index_element, Metric m1, Metric m2):
        
    #     self.xs = m1.u
    #     self.ys = m2.u

    #     self.n = self.xs.shape[0]
    #     self.i = index_element
    #     self.x = self.xs[index_element]
    #     self.y = self.ys[index_element]

        
    #     self.gs = self.n-1
        
    #     self.sensitivities_x = np.zeros(self.n)
    #     self.sensitivities_y = np.zeros(self.n)

    #     self.deltas_x = <PyObject **> malloc(self.n * sizeof(PyObject *))
    #     self.deltas_y = <PyObject **> malloc(self.n * sizeof(PyObject *))
            
    #     cdef int i
    #     cdef delta_base tmp1, tmp2

    #     # start_time = time.time()

    #     for i in range(self.n):
            
    #         tmp1 = m1.ge/t_delta(i)
    #         Py_INCREF(tmp1)            
    #         self.deltas_x[i] = <PyObject *>tmp1

    #         tmp2 = m2.get_delta(i)
    #         Py_INCREF(tmp2)
    #         self.deltas_y[i] = <PyObject *>tmp2       