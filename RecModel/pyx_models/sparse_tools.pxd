# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport numpy as np

# Dynamic_Sparse_Mat Structure
cdef struct sp_vec_arr:
    long dimension # True dimensions of the sparse vector
    long elems # The current number of non-zero elements in the vector.
    long total_elems # Current allocated length of underlying arrays
    long *idx # Indicees of the non-zero entries
    double *data # Values of the non-zero entries

# Multiplying two sparse matrices.
cdef void sparse_mult_vecs(int elem, double[:] result, int[:] A_idx, double[:] A_data, int[:] W_idx, double[:] W_data) nogil

# Fast prediction for the ease model
cpdef np.ndarray[np.float32_t, ndim=1] _predict_ease(int[:] X_indptr, int[:] X_idx, float[:] X_data, float[:, :] W, int[:] users, int[:] items)

# Methods for the Sparse matrices.
# Manipulate elements in the sparse vec.
cdef double get_elem_arr(long pos, sp_vec_arr *vec) nogil
cdef void set_elem_arr(long insert_pos, double val, sp_vec_arr *vec_arr) nogil

# Allocate, realloc, free sp_vec
cdef sp_vec_arr* alloc_sp_vec_arr(long dimension, long init_size) nogil
cdef void free_sp_vec_arr(sp_vec_arr *vec) nogil
cdef void realloc_sp_vec_arr(sp_vec_arr *vec) nogil

# Print
cdef void print_sp_vec_arr(sp_vec_arr *vec, int full)

# Compute the L1, L2 norm of a sparse vec
cdef double l1_norm_arr(sp_vec_arr *vec) nogil 
cdef double l2_norm_arr(sp_vec_arr *vec) nogil

# Helper funtions
cdef long bin_search(long *arr, long start, long end, long val) nogil
