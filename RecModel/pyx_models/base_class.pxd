#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: nonecheck=False

# C Imports 
cimport numpy as np

cdef class RecModel:
    cdef unsigned int compute_hit(self, int user, int[:] items_selected,  unsigned int rand_sampled, int[:] topn, int[:] hits, int max_topn, unsigned int cores)
    cpdef eval_prec(self, mat, metric, cores)
    cpdef np.ndarray[np.int32_t, ndim=1] compute_hit_par(self, int thread, int[:] X_idx, int[:] X_indptr, unsigned int rand_sampled, int[:] topn, int max_topn, int n_threads)
