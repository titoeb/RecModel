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

cdef class Baseline(RecModel):
    cdef:
        public unsigned int num_items
        public long[:] top_items
    
    def __init__(self, int num_items):
        self.num_items = num_items

    cpdef void train(self, object X):
        self.top_items = np.argsort(X.sum(axis = 0).A1.astype(np.float32))[::-1]
        
    cpdef np.ndarray[np.int32_t, ndim=1] rank(self, np.ndarray[np.int32_t, ndim=1] items, long user, int topn, int cores):           
        return np.array(self.top_items[:topn])

    cpdef np.ndarray[np.double_t, ndim=1] predict(self, int[:] users, int[:] items, unsigned int cores):
        print("The Baseline cannot predict, only topn can be used.")
