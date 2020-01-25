#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: nonecheck=False

# Imports
import numpy as np
from cython.parallel import prange
import scipy.sparse

# C Imports 
cimport numpy as np
cimport sparse_tools
cimport cd_fast
from cython cimport floating
from base_class cimport RecModel

cdef class NaiveBaseline(RecModel):
    cdef public unsigned int num_items
    
    def __init__(self, int num_items):
        self.num_items = num_items

    cpdef np.ndarray[np.int32_t, ndim=1] rank(self, np.ndarray[np.int32_t, ndim=1] items, long user, unsigned int topn, int cores):           
        cdef: 
            int[:] predicted_items = np.zeros(topn,  dtype=np.int32)
            int num_items = len(items)

        if topn > num_items:
            print(f"Sampling {topn} elements from a vector of length {num_items} without replacement does not make sense!")
        else:
            return np.random.choice(items, size=topn, replace=False)

    cpdef void train(self):
        print("This naive Baseline does not need to be trained.")

    cpdef np.ndarray[np.double_t, ndim=1] predict(self, int[:] users, int[:] items, unsigned int cores):
        print("The naive Baseline cannot predict, only topn can be used.")
