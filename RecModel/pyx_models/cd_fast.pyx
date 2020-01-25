#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# Imports
import numpy as np
import numbers
import scipy.sparse
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport sasum, dasum
from cython.parallel import prange

# cimports
from libc.math cimport fabs
cimport sparse_tools
cimport cython
cimport numpy as np
from cython cimport floating
from libc.stdlib cimport malloc, free

# Debug-imports
from libc.stdio cimport printf
np.import_array()

# Definitions 
cdef DEFAULT_SEED = 1

cdef floating _dot(int n, floating *x, int incx,
                   floating *y, int incy) nogil:
    """x.T.y"""
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)

cdef floating _asum(int n, floating *x, int incx) nogil:
    """sum(|x_i|)"""
    if floating is float:
        return sasum(&n, x, &incx)
    else:
        return dasum(&n, x, &incx)


# import from _random
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def _our_rand_r_py(seed):
    """Python utils to test the our_rand_r function"""
    cdef UINT32_t my_seed = seed
    return our_rand_r(&my_seed)

# --------------------------------------------------------------------------------------------------------------------------------- #
# The following two functions are shamelessly copied from scikit-learn

cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y

cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m

cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m

# Main funtion

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Using dynamic array

cdef train_Slim(object X, floating alpha, floating l1_ratio, int max_iter, floating tol, int cores, int verbose):
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64
    
    cdef:
        sparse_tools.sp_vec_arr **w 
        int i, j, item, start_col, end_col
        floating[:] X_T_R,  XtA, R, 
        floating[:] X_data, X_squared_row_sum
        int[:] X_idx, X_indptr
        unsigned int n_samples, n_features, pos

    # Get the  Array X
    n_samples, n_features = X.shape
    X_data = X.data.astype(np.float64)
    X_idx = X.indices.astype(np.int32)
    X_indptr = X.indptr.astype(np.int32)

    # Allocate internal arrays for the coordinate descent algorithm
    X_T_R = np.zeros(n_features, dtype=dtype)
    XtA = np.zeros(n_features, dtype=dtype)
    R = np.zeros(n_samples, dtype=dtype) #dimension of the sparse matrix

    # For the coordinate descent algorithm allocate the sum of the squared columns
    X_squared_row_sum = X.multiply(X).sum(axis=0).A1
    
    # Allocate array of sp_vec_arr pointers to store the weight matrices.
    w = <sparse_tools.sp_vec_arr **> malloc(sizeof(sparse_tools.sp_vec_arr *) * n_features) 

    # Fit the individual models
    #for user in prange(n_features, nogil=True, num_threads=cores):
    if verbose > 0:
        print(f"{n_features} models need to be fitted")
    for item in range(n_features):
        # Create
        start_col = X_indptr[item]
        end_col = X_indptr[item + 1]

        if start_col != end_col:
        
        # Fit the model
            if verbose > 0:
                if item % 1000 == 0:
                    printf("Current model is %lu\n", item)
         
            sparse_co_descent_arr_intern(pos=item, w=w, alpha=alpha, beta=l1_ratio, X_data=X_data, X_indices=X_idx, X_indptr=X_indptr, y_data=X_data[start_col:end_col], y_idx=X_idx[start_col:end_col],
                            x_squared_row_sum=X_squared_row_sum, max_iter=max_iter, X_T_R=X_T_R, XtA=XtA, R=R, tol=tol, n_samples=n_samples, n_features=n_features, pos_empty=1)
     

            # Clean the internal arrays.          
            for i in range(n_features):
                X_T_R[i] = 0
                XtA[i] = 0

            for i in range(n_samples):
                R[i] = 0
        else:
            w[item] = sparse_tools.alloc_sp_vec_arr(n_features, 1)
       
    indptr = np.empty(n_features + 1, dtype=np.int32)
    indptr[0] = 0

    for i in range(1, n_features + 1):
        indptr[i] = indptr[i - 1] + w[i - 1].elems
    
    
   
    # We know now that the number of non-zero elements will be idptr[-1].
    data = np.empty(indptr[n_features], dtype=np.float64)
    idx = np.empty(indptr[n_features], dtype=np.int32)

    # Get the acutal values from the c_sp_arr structs.
    pos = 0
    for i in range(n_features):
        for j in range(w[i].elems):
            data[pos] = w[i].data[j]
            idx[pos] = w[i].idx[j]
            pos += 1   
    
    # Free w 
    for i in range(n_features):   
        sparse_tools.free_sp_vec_arr(w[i])
    free(w)

    return indptr, idx, data
    
cdef sparse_tools.sp_vec_arr *sparse_co_descent_arr_intern(int pos, sparse_tools.sp_vec_arr **w, floating alpha, floating beta,
                            floating[:] X_data,
                            int[:] X_indices,
                            int[:] X_indptr,
                            floating[:] y_data,
                            int[:] y_idx,
                            floating[:] x_squared_row_sum, int max_iter,
                            floating[:]  X_T_R, floating[:]  XtA, floating[:] R,
                            floating tol, unsigned int n_samples, unsigned int n_features, int pos_empty) nogil:
    """Cython version of the coordinate descent algorithm for Elastic-Net
    We minimize:
        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) * norm(w, 2)^2
    """
    cdef:
        long ii,  endptr, jj
        long startptr = X_indptr[0]
        floating tmp, w_ii, d_w_max, w_max, d_w_ii,  R_norm2, w_norm2, A_norm2, l1_norm, normalize_sum, d_w_tol = tol, dual_norm_XtA, prod
        sparse_tools.sp_vec_arr *this_w
        floating  gap = tol + 1.0, w_ii_new = 0.0, R_sum = 0.0, l2_norm_tol = 0.0
        unsigned int f_iter, convergence_error
        unsigned int n_iter = 0

    w[pos] = sparse_tools.alloc_sp_vec_arr(n_features, 200)
    this_w = w[pos]

    # Initiallize residum with y
        # Sparse np.dot(Y.T, Y) plus initiallize R
    for ii in range(y_data.shape[0]):
        l2_norm_tol += y_data[ii] ** 2
        R[y_idx[ii]] = y_data[ii]

    tol *= l2_norm_tol

    # ---------------------------------
    # Main Loop of coordinate descent 
    # ---------------------------------
    for n_iter in range(max_iter):

        w_max = 0.0
        d_w_max = 0.0

        for ii in range(n_features):  # Loop over coordinates
            if x_squared_row_sum[ii] == 0.0:
                continue
            if pos_empty == 1 and ii == pos:
                continue

            startptr = X_indptr[ii]
            endptr = X_indptr[ii + 1]
            # ------------------------- www -------------------------------- #
            #w_ii = w[ii]  # Store previous value
            w_ii = sparse_tools.get_elem_arr(ii, this_w)
            # ------------------------- www -------------------------------- #

            if w_ii != 0.0:
                # R += w_ii * X[:,ii]
                for jj in range(startptr, endptr):
                    R[X_indices[jj]] += X_data[jj] * w_ii

            # tmp = (X[:,ii] * R).sum()
            tmp = 0.0
            for jj in range(startptr, endptr):
                tmp += R[X_indices[jj]] * X_data[jj]

            if tmp <= 0.0:
                if w_ii > 0: # This could also change the implementation!
                    # ------------------------- www -------------------------------- #
                    sparse_tools.set_elem_arr(ii, 0.0, this_w) 
                    #printf("set_elem(%ld, 0.0, w) \n", ii)
                    # ------------------------- www -------------------------------- #
                w_ii_new = 0.0
    
            else:
                w_ii_new = fsign(tmp) * fmax(tmp - alpha, 0) / (x_squared_row_sum[ii] + beta)
                if w_ii_new > 0 or (w_ii_new == 0  and w_ii > 0):
                    # ------------------------- www -------------------------------- #
                    # w[ii] = w_ii_new 
                    sparse_tools.set_elem_arr(ii, w_ii_new, this_w)   
                    #printf("set_elem(%ld, %lf, w) \n", ii, w_ii_new)                 
                    # ------------------------- www -------------------------------- #

            if w_ii_new != 0.0:
                # R -=  w[ii] * X[:,ii] # Update residual
                for jj in range(startptr, endptr):
                    R[X_indices[jj]] -= X_data[jj] * w_ii_new

            # update the maximum absolute coefficient update
            
            d_w_ii = fabs(w_ii_new - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii
            
            if fabs(w_ii_new) > w_max:                   
                w_max = fabs(w_ii_new)
        
        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller than
            # the tolerance: check the duality gap as ultimate stopping
            # criterion

            # sparse X.T / dense R dot product
            for ii in range(n_features):
                X_T_R[ii] = 0.0
                for jj in range(X_indptr[ii], X_indptr[ii + 1]):
                    X_T_R[ii] += X_data[jj] * R[X_indices[jj]]
                
                # ------------------------- www -------------------------------- #
                # This is not simply a lookup. This is some mathematical operation. optimize?
                # Maybe with sparse structrue on do
                # XtA[ii] = X_T_R[ii]
                # And then later iterate over w_sparse  and subtract - beta  * w[ii]
                #XtA[ii] = X_T_R[ii] - beta * w[ii]
                XtA[ii] = X_T_R[ii] - beta * sparse_tools.get_elem_arr(ii, this_w)
                # ------------------------- www -------------------------------- #

            dual_norm_XtA = max(n_features, &XtA[0])

            # R_norm2 = np.dot(R, R)
            R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)

            # w_norm2 = np.dot(w, w)
            # ------------------------- www -------------------------------- #
            # Compute the l2_norm of w
            w_norm2 = sparse_tools.l2_norm_arr(this_w)
            
            # ------------------------- www -------------------------------- #
            if (dual_norm_XtA > alpha):
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * const**2
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2
            
            # ------------------------- www -------------------------------- #
            # Compute the l1_norm of w
            l1_norm = sparse_tools.l1_norm_arr(this_w)
            # ------------------------- www -------------------------------- #

            prod = 0.0
            # Compute np.dot(R, y)
            for ii in range(y_data.shape[0]):
                prod += R[y_idx[ii]] * y_data[ii] 
            
            gap += (alpha * l1_norm - const * prod + 0.5 * beta * (1 + const ** 2) * w_norm2)
            #printf("l2_norm(w): %lf l1_norm(w): %lf, gap: %lf , tol: %lf at iteration %i\n", w_norm2, l1_norm, gap,tol, n_iter)
            if gap < tol:
                # return if we reached desired tolerance
                break


