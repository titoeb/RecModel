# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

import scipy.sparse
import numpy as np
cimport numpy as np

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Fast prediction for the ease model.
cpdef np.ndarray[np.float32_t, ndim=1] _predict_ease(int[:] X_indptr, int[:] X_idx, float[:] X_data, float[:, :] W, int[:] users, int[:] items):
    cdef:
        int i, j, user, item
    if len(users) == 0 or len(items) == 0:
        return np.full(1, 0.0, dtype=np.float32)
    else:
        #Pre-allocate output array.
        output = np.zeros(len(items))
        for i in range(len(items)):
            user = users[i]
            item = items[i]
            for j in range(X_indptr[user], X_indptr[user + 1]):
                output[i] += X_data[j] *  W[X_idx[j], item]
        return output
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Multiplication of two sparse vectors.
cdef void sparse_mult_vecs(int elem, double[:] result, int[:] A_idx, double[:] A_data, int[:] W_idx, double[:] W_data) nogil:
    cdef:
        double accumulator = 0.0
        unsigned int A_iter = 0
        unsigned int W_iter = 0
        unsigned int match = 0
        unsigned int len_W = W_idx.shape[0]
        unsigned int len_A = A_idx.shape[0]

    if not(A_idx.shape[0] == 0 or W_idx.shape[0] == 0):
        while(A_iter < len_A and W_iter < len_W):
            if A_idx[A_iter] > W_idx[W_iter]:
                W_iter += 1
            elif A_idx[A_iter] == W_idx[W_iter]:
                accumulator += A_data[A_iter] * W_data[W_iter]
                W_iter += 1
                A_iter += 1
            else:
                A_iter += 1
    result[elem] = accumulator

# ----------------------------------------------------------------------------------------------------------------------------------------
# The following functions belong to the implementation of sp_vec_arr, a sparse vector implementation using dynamically reallocating arrays.
cdef long bin_search(long *arr, long start, long end, long val) nogil:
    cdef:
        long midpos
    if end - start < 1:    
        return start
    else:
        midpos = <long> ((end - start) / 2) 
        midpos += start
        if arr[midpos] < val:
            return bin_search(arr, midpos + 1, end, val)
        elif arr[midpos] > val:
            return bin_search(arr, start, midpos, val)
        else:
            return midpos

# Manipulate elements in the sparse vec.
cdef double get_elem_arr(long pos, sp_vec_arr *vec) nogil:
    cdef:
        long cur_pos = 0
        long poss_pos

    if pos >= vec.dimension:
        raise ValueError("Value does not exist")
    
    # If the queried position is behind the last one return 0.0
    if pos > vec.idx[vec.elems - 1]:
        return 0.0
 
    if vec.elems == 0:
        return 0.0
    
    poss_pos = bin_search(vec.idx, 0, vec.elems, pos)
    if poss_pos < vec.elems and vec.idx[poss_pos] == pos:
        return vec.data[poss_pos]
    else:
        return 0.0

cdef void set_elem_arr(long insert_pos, double val, sp_vec_arr *vec) nogil:
    cdef:
        int found = 0, shift = 1, i = 0
        long cur_pos = 0, tmp_idx, extr_idx
        double tmp_val, extr_val

    if insert_pos >= vec.dimension:
        raise ValueError("Value does not exist")
       
    # Check whether enough storage available and if not double the size of the vectors.
    if vec.elems >= vec.total_elems:
        realloc_sp_vec_arr(vec)
    
    # Check whether this is the first item to insert
    if vec.elems == 0:
        vec.data[0] = val
        vec.idx[0] = insert_pos
        vec.elems += 1
        return

    if val == 0.0:
        # If the value is 0.0, then either, when the value does exists, it has to be remove, otherwise not added.
        if get_elem_arr(insert_pos, vec) != 0.0:

            # Find the position of the entry 
            cur_pos = bin_search(vec.idx, 0, vec.elems, insert_pos)

            #printf("cur_pos: %ld, vec.elems: %ld \n", cur_pos, vec.elems)
            # Just move ever y entry to the right of the entry one to the left (and with it remove the value itself)
            for i in range(cur_pos, vec.elems - 1):
                vec.data[i] = vec.data[i+1]
                vec.idx[i] = vec.idx[i+1]
                
            # Finally save that we removed one value from the sparse vector
            vec.elems -= 1

    else:
        # The value is nonzero and either has to be set or added to the vector.
    
        # The value does not have to be inserted at the first position.
        # Iterate through non-zero values and put it the right position.
        
        # Find either the position of the value or where it needs to be inserted.
        cur_pos = bin_search(vec.idx, 0, vec.elems, insert_pos)
        
        # This is not safe.
        if cur_pos < vec.elems and vec.idx[cur_pos] == insert_pos:
            # The value was already in the sparse list and just has to be changed
            vec.data[cur_pos] = val
        else:
            # The value has to be inserted.
            if cur_pos >= vec.elems:
                # There is no index smaller than insert_pos, insert val and insert_pos at the last position
                vec.data[cur_pos] = val
                vec.idx[cur_pos] = insert_pos
               
            else:
                 # First insert val, insert_pos at current position and save old elements
                extr_val = vec.data[cur_pos]
                extr_idx = vec.idx[cur_pos]
                vec.data[cur_pos] = val 
                vec.idx[cur_pos] = insert_pos
                
                cur_pos += 1
                # Iterate till vec.elems + 1 and shift elements
                while (cur_pos < vec.elems + 1):
                    tmp_val = extr_val
                    tmp_idx = extr_idx
                    extr_val = vec.data[cur_pos]
                    extr_idx = vec.idx[cur_pos]
                    vec.data[cur_pos] = tmp_val 
                    vec.idx[cur_pos] = tmp_idx
                    cur_pos += 1
            vec.elems += 1

# Compute L1, L2 norm
cdef double l1_norm_arr(sp_vec_arr *vec) nogil:   
    cdef:
        double accumulator = 0.0
        long i
    for i in range(vec.elems):
        accumulator += vec.data[i]
    return accumulator

cdef double l2_norm_arr(sp_vec_arr *vec) nogil:   
    cdef:
        double accumulator = 0.0
        long i
    for i in range(vec.elems):
        accumulator += vec.data[i] ** 2
    return accumulator

# Allocate, realloc, free sp_vec
cdef sp_vec_arr* alloc_sp_vec_arr(long dimension, long init_size) nogil:
    cdef sp_vec_arr* vec = <sp_vec_arr *> malloc(sizeof(sp_vec_arr))
    vec.elems = 0
    vec.dimension = dimension
    vec.total_elems = init_size
    vec.idx = <long *> calloc(sizeof(long), init_size)
    vec.data = <double *> calloc(sizeof(double), init_size)
    return vec
    
cdef void free_sp_vec_arr(sp_vec_arr *vec) nogil:
    free(vec.data)
    free(vec.idx)
    free(vec)

cdef void realloc_sp_vec_arr(sp_vec_arr *vec) nogil:
    # Make the arrays double the size and copy the elements into the new array.
    # Free old arrays.
    cdef:
        long *new_idx
        double * new_data
        long i, n_new
    n_new = 2 * vec.total_elems
    # Create new arrays
    new_idx = <long *> calloc(sizeof(long), n_new)
    new_data = <double *> calloc(sizeof(double), n_new)
    vec.total_elems = n_new
    
    # Copy data
    for i in range(vec.elems):
        new_idx[i] = vec.idx[i]
        new_data[i] = vec.data[i]

    # Free old array
    free(vec.data)
    free(vec.idx)

    # Insert new arrays
    vec.data = new_data
    vec.idx = new_idx
    
# Print sparse array!
cdef void print_sp_vec_arr(sp_vec_arr *vec, int full):
    # If full == 0, print only data and idx, otherwise print whole array.
    cdef:
        long i = 0
        long pos_array = 0
    if full == 0:
        print("Idx: ", end='')
        for i in range(vec.elems):
            print(vec.idx[i], end=' ')
        print('\n')
        print("Data: ", end=' ')
        for i in range(vec.elems):
            print(vec.data[i], end=' ')
    else:
        pos_array = 0
        for i in range(vec.dimension):
            if pos_array < vec.elems and vec.idx[pos_array] == i:
                print(vec.data[pos_array], end=' ')
                pos_array += 1
            else:
                print('0', end=' ')
    print('\n')
# ----------------------------------------------
cpdef void test():
    cdef sp_vec_arr *test = alloc_sp_vec_arr(5, 1)
    print_sp_vec_arr(test, 0)

    
    set_elem_arr(2, 2.0, test)
    
    set_elem_arr(4, 4.0, test)
    set_elem_arr(1, 1.0, test)
    set_elem_arr(0, 0.0, test)
    set_elem_arr(3, 3.0, test)  
    print_sp_vec_arr(test, 0)


    for i in range(5):
        print(get_elem_arr(i, test))

    print(f"L1-norm: {l1_norm_arr(test)}")
    print(f"L2-norm: {l2_norm_arr(test)}")

    print("Remove 4.0 at position 4.")
    print("before:")
    print_sp_vec_arr(test, 0)
    set_elem_arr(4, 0.0, test)
    print("after:")
    print_sp_vec_arr(test, 0)

    print("put 4.0 back in")
    set_elem_arr(4, 4.0, test)
    print_sp_vec_arr(test, 0)


    print("Remove 3.0 at position 3.")
    print("before:")
    print_sp_vec_arr(test, 0)
    set_elem_arr(3, 0.0, test)

    print("After:")
    print_sp_vec_arr(test, 0)

    free_sp_vec_arr(test)

cpdef void test_alloc():
    cdef:
        int n_features = 3, int = 0
        sp_vec_arr ** w
        
    w =  <sp_vec_arr **> malloc(sizeof(sp_vec_arr *) * n_features) 

    
    for i in range(n_features):
        w[i] = alloc_sp_vec_arr(n_features, 2)
    
    set_elem_arr(2, 4.0, w[0])
    set_elem_arr(1, 3.0, w[1])
   

    indptr = np.empty(n_features + 1, dtype=np.int_)
    indptr[0] = 0
    for i in range(1, n_features + 1):
        indptr[i] = indptr[i - 1] + w[i - 1].elems
    
   
    # We know now that the number of non-zero elements will be idptr[-1].
    data = np.empty(indptr[n_features], dtype=np.float64)
    idx = np.empty(indptr[n_features], dtype=np.int_)

    
    # Get the acutal values from the c_sp_arr structs.
    pos = 0
    for i in range(n_features):
        for j in range(w[i].elems):
            data[pos] = w[i].data[j]
            idx[pos] = w[i].idx[j]
            pos += 1   
    #print(f"data: {data}\n idx: {idx}\n indptr: {indptr}")

    #print(f"idx.shape: {idx.shape}, data: {data}\n idx: {idx}")
    out_mat = scipy.sparse.csc_matrix((data, idx, indptr), shape=(n_features, n_features), dtype=np.float64)
    
    # Free w 
    for i in range(n_features):   
        free_sp_vec_arr(w[i])
    free(w)

    print(out_mat.todense())



