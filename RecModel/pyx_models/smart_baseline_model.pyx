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

cdef class SmartBaseline(RecModel):
    cdef:
        public unsigned int num_items
        public float[:] item_counts
    
    def __init__(self, int num_items):
        self.num_items = num_items

    cpdef void train(self, object X):
        self.item_counts = X.sum(axis = 0).A1.astype(np.float32)
        
    cpdef np.ndarray[np.int32_t, ndim=1] rank(self, np.ndarray[np.int32_t, ndim=1] items, long user, int topn, int cores):           
        cdef: 
            int num_items = len(items), i
            float[:] relevant_item_scores = np.empty(num_items, dtype=np.float32)
            

        for i in range(num_items):
            relevant_item_scores[i] = self.item_counts[items[i]]

        return items[np.argsort(relevant_item_scores)[::-1]]

    cpdef np.ndarray[np.double_t, ndim=1] predict(self, int[:] users, int[:] items, unsigned int cores):
        print("The naive Baseline cannot predict, only topn can be used.")
