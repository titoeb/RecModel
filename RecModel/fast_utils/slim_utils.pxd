from cython cimport floating
cimport sparse_tools
cimport numpy as np

cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous

cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'

ctypedef np.npy_uint32 UINT32_t
cdef inline UINT32_t DEFAULT_SEED = 1

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0): seed[0] = DEFAULT_SEED

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Note: we must be careful with the final line cast to np.uint32 so that
    # the function behaves consistently across platforms.
    #
    # The following cast might yield different results on different platforms:
    # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    #
    # We can use:
    # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # or:
    # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)
# ----------------------------------------------------------------------------- #

# Internal functions of cd_fast, shamelessly copied from sklearn
cdef floating _dot(int n, floating *x, int incx, floating *y, int incy) nogil
cdef UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil
cdef floating fmax(floating x, floating y) nogil
cdef floating fsign(floating f) nogil
cdef floating abs_max(int n, floating* a) nogil
cdef floating max(int n, floating* a) nogil

# Train the slim model
cpdef train_Slim(object X, floating alpha, floating l1_ratio, int max_iter, floating tol, int cores, int verbose)

cpdef void _predict_slim(int[:] users, int[:] items, double[:] output, int[:] A_indptr, int[:] W_indptr, int[:] A_idx, int[:] W_idx, double[:] A_data, double[:] W_data)

# cd_fast method itself, that fits Elastic Net with coordinate descent and soft thresholding
cdef void sparse_co_descent_arr_intern(int pos, sparse_tools.sp_vec_arr **w, floating alpha, floating beta,
                            floating[:] X_data,
                            int[:] X_indices,
                            int[:] X_indptr,
                            floating[:] y_data,
                            int[:] y_idx,
                            floating[:] x_squared_row_sum, int max_iter,
                            floating[:]  X_T_R, floating[:]  XtA, floating[:] R,
                            floating tol, unsigned int n_samples, unsigned int n_features, int pos_empty) nogil

cdef void sparse_co_descent_arr_par(int start_item, int end_item, sparse_tools.sp_vec_arr **w, floating alpha, floating beta,
        floating[:] X_data,
        int[:] X_indices,
        int[:] X_indptr,
        floating[:] y_data,
        int[:] y_idx,
        floating[:] x_squared_row_sum, int max_iter,
        floating tol, unsigned int n_samples, unsigned int n_features) nogil

cdef void sparse_co_descent_par(int pos, sparse_tools.sp_vec_arr **w, floating alpha, floating beta,
                            floating[:] X_data,
                            int[:] X_indices,
                            int[:] X_indptr,
                            floating[:] y_data,
                            int[:] y_idx,
                            floating[:] x_squared_row_sum, int max_iter,
                            floating[:]  X_T_R, floating[:]  XtA, floating[:] R,
                            floating tol, unsigned int n_samples, unsigned int n_features, int pos_empty) nogil