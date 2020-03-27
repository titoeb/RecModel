#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

# Imports
import numpy as np
from cython.parallel import prange, threadid
import scipy.sparse

# C Imports 
cimport numpy as np
from cython cimport floating
from libc.stdlib cimport malloc, calloc, free
from libc.stdio cimport printf

# Import FLT_EPSILON from float.h
# This is used to avoid "division by zero"-error
cdef extern from 'float.h':
    float FLT_EPSILON

# Helper functions
# Get the qsort function for the internal argsort!
# This is used to find the k nearest neighbors out of all similar neigboors.
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compare)(const_void *, const_void *)) nogil

# This struct is used to track the elements in argsort.
# This is going too be sorted by the value and we will extract the indices.
cdef struct IndexedElement:
    int index
    float value

# This function will be given to the qsort function to compare the elements and sort them.
cdef int _compare(const_void *a, const_void *b) nogil:
    cdef float v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v >= 0: return -1
    if v < 0: return 1

# This function returns an integer array of length len(data) that contains the order of the data objects from largest to smallest.
# The pointer that is returnitemed needs to be freed.
cdef int* _argsort(float* data, int n, int result_n) nogil:
    cdef:
        int i
        int* order
        IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement)) # Allocate index tracking array.

    # I think this could be of size result_n
    order = <int*> malloc(sizeof(int) * n)
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(result_n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)
    return order

# Fast inline absolute value.
cdef inline float abs(float input) nogil:
    if input < 0:
        return -input
    else:
        return input
    
# Dot product between rel_sim and rel_data scaled by the sum of rel_data
cdef float _create_pred(float* rel_sim, float[:] rel_data, int* idx, int idx_len) nogil:
    cdef:
        float scaling_factor = 0.0, dot_product = 0.0
        int i = 0
    
    for i in range(idx_len):
        scaling_factor += abs(rel_sim[idx[i]])
        dot_product += rel_sim[idx[i]] * rel_data[idx[i]]
    return dot_product / (scaling_factor + FLT_EPSILON)


cdef float sum(float* rel_sim, int* idx, int length) nogil:
    cdef:
        float sum_rel_sim = 0.0
    
    for i in range(length):
        sum_rel_sim += abs(rel_sim[idx[i]])
        #printf("sum_rel_sim: %f, abs(rel_sim[idx[i]]): %f idx[i]: %i \n",sum_rel_sim,  abs(rel_sim[idx[i]]), idx[i])

    return sum_rel_sim

# Copy the relevant similarities from row item and colums in rel_item from sim to rel_sim
cdef void _cpy(int item, int[:] rel_items, float* rel_sim, float[:, :] sim) nogil:
    cdef:
        int i
    for i in range(len(rel_items)):
        rel_sim[i] = sim[item, rel_items[i]]

# ---------------------- #
# PREDICTION
# ---------------------- #
# The following methods will be used to compute the prediction of the neighborhood based methods given the pre-computed similarity matrix.

# Prediction function that is used within the neighborhood models.
cpdef void _predict_neighborhood_iterative(float[:, :] sim, int user, int[:] items, int nb_size, float[:] data, int[:] idx, int[:] indptr, float[:] predictions, int offset) nogil:
    """ Fast implementation of the predict method. 
    """
    cdef:
        int start_user, end_user, i, actual_nh_size, thread, n_items
        int[:] rel_items
        int* sorted_idx_internal
        float* rel_sim
        float[:] rel_data

    # Extract the row that belongs to the user from the sparse matrix.
    start_user = indptr[user]
    end_user = indptr[user + 1]
    n_items = len(items)

    if start_user == end_user:
        for i in range(n_items):
            predictions[offset + i] = 0.0
        return

    # BE AWARE! these are the original arrays. Use rel_items[:] = ... if you want to copy!
    rel_items = idx[start_user:end_user]
    rel_data = data[start_user:end_user]

    # Alloate rel_sim where the relevant similarity will be saved!
    rel_sim = <float*> calloc(len(rel_items), sizeof(float))

    # The actual neighborhood size is the minimum of the desired nh size and how many items the user consumed.
    actual_nh_size = min(nb_size, end_user - start_user)

    for i in range(len(items)):
        #rel_sim = sim[item, rel_items]
        _cpy(items[i], rel_items, rel_sim, sim)
        
        # Sorted_idx_internal is int* pointing to the ordered ranks of the indices sorted by rel_sim.
        # This is not efficient as all entries of rel_sim are sorted but only the first
        # actual_nh_size will be used. 
        sorted_idx_internal = _argsort(rel_sim, len(rel_items), result_n=actual_nh_size)
    
        # Put it in a typed memoryview to allow indexing with it
        # Only take the top actual_nh_size values.
        # Would be more efficient only to order the actual_nh_size top items ...       
        # With the top rated items create the prediction with a weighted sum.
        predictions[offset + i] = _create_pred(rel_sim, rel_data, idx=sorted_idx_internal, idx_len=actual_nh_size)

        # Don't forget to free the sorted_idx.
        free(sorted_idx_internal)

    # Don't forget to free rel_sim!
    free(rel_sim)

cpdef void _predict_neighborhood_iterative_weights_only(float[:, :] sim, int user, int[:] items, int nb_size, int[:] idx, int[:] indptr, float[:] predictions, int offset) nogil:
    """ Fast implementation of the predict method. 
    """
    cdef:
        int start_user, end_user, i, actual_nb_size, thread, n_items
        int[:] rel_items
        int* sorted_idx_internal
        float* rel_sim
        int test # tmp!!

    # Extract the row that belongs to the user from the sparse matrix.
    start_user = indptr[user]
    end_user = indptr[user + 1]
    n_items = len(items)

    if start_user == end_user:
        for i in range(n_items):
            predictions[offset + i] = 0.0
        return

    # BE AWARE! these are the original arrays. Use rel_items[:] = ... if you want to copy!
    rel_items = idx[start_user:end_user]

    # Alloate rel_sim where the relevant similarity will be saved!
    rel_sim = <float*> calloc(len(rel_items), sizeof(float))


    tmp = len(rel_items)
    # The actual neighborhood size is the minimum of the desired nh size and how many items the user consumed.
    actual_nb_size = min(nb_size, end_user - start_user)
    #printf("nb_size: %i, actual_nb_size: %i, end_user - start_user: %i, len(rel_items): %i", nb_size, actual_nb_size, end_user - start_user, tmp)

    for i in range(len(items)):
        #rel_sim = sim[item, rel_items]
        _cpy(items[i], rel_items, rel_sim, sim)
        
        # Sorted_idx_internal is int* pointing to the ordered ranks of the indices sorted by rel_sim.
        # This is not efficient as all entries of rel_sim are sorted but only the first
        # actual_nb_size will be used. 
        sorted_idx_internal = _argsort(rel_sim, len(rel_items), result_n=actual_nb_size)
    
        # Put it in a typed memoryview to allow indexing with it
        # Only take the top actual_nb_size values.
        # Would be more efficient only to order the actual_nb_size top items ...       
        # With the top rated items create the prediction with a weighted sum.
        predictions[offset + i] = sum(rel_sim, idx=sorted_idx_internal, length=actual_nb_size)

        # Don't forget to free the sorted_idx.
        free(sorted_idx_internal)

    # Don't forget to free rel_sim!
    free(rel_sim)

# Predict function for neighborhood.          
cpdef void _predict_neighborhood(float[:, :] sim, int[:] users, int[:] items, int nb_size, float[:] data, int[:] idx, int[:] indptr, float[:] predictions, int weights_only) nogil:
    cdef:
        int i

    if weights_only == 0:
        for i in range(len(users)):
            _predict_neighborhood_iterative(sim, users[i], items[i:(i+1)],  nb_size, data, idx, indptr, predictions, i)
    
    elif weights_only == 1:
        for i in range(len(users)):
            _predict_neighborhood_iterative_weights_only(sim, users[i], items[i:(i+1)],  nb_size, idx, indptr, predictions, i)
        
# ---------------------- #
# Similarity.
# ---------------------- #
# Correlation 
cdef float correlation(int[:] mat_idx, float[:] mat_data, int A_start, int A_end, int B_start, int B_end, int len) nogil:
    cdef:
        float product_a_b = 0.0, sum_a = 0.0, sum_b = 0.0, sum_a_squared = 0.0, sum_b_squared = 0.0, denominator = 0.0, numerator = 0.0
        int A_iter = 0, B_iter = 0

    # Save the current position we are in.
    A_iter = A_start
    B_iter = B_start

    # If there is no data for either of the items, return 0.
    if A_start == A_end or B_start == B_end:
        return 0.0

    # Now iterate over the two vectors and compute the neccesary parts.
    while((A_iter < A_end) and (B_iter < B_end)): 
        if mat_idx[A_iter] < mat_idx[B_iter]:
            # Collect for the individual vector
            sum_a += mat_data[A_iter]
            sum_a_squared += (mat_data[A_iter]) ** 2

            A_iter += 1
        elif mat_idx[A_iter] == mat_idx[B_iter]:

            # Collect for the individual vectors
            sum_b += mat_data[B_iter]
            sum_b_squared += (mat_data[B_iter]) ** 2            
            sum_a += mat_data[A_iter]
            sum_a_squared += (mat_data[A_iter]) ** 2

            # Collect for the vector product
            product_a_b += mat_data[A_iter] * mat_data[B_iter]
        
            # Proceed in the vector
            A_iter += 1
            B_iter += 1

        elif mat_idx[A_iter] > mat_idx[B_iter]:
            # Collect for the individual vector
            sum_b += mat_data[B_iter]
            sum_b_squared += (mat_data[B_iter]) ** 2

            B_iter += 1

    while(A_iter < A_end):
        sum_a += mat_data[A_iter]
        sum_a_squared += (mat_data[A_iter]) ** 2

        A_iter += 1

    while(B_iter < B_end):
        sum_b += mat_data[B_iter]
        sum_b_squared += (mat_data[B_iter]) ** 2

        B_iter += 1

    # All needed variables are collected.
    # Let's piece together the correlation.
    numerator = len * product_a_b - sum_a * sum_b
    denominator = ((len * sum_a_squared - sum_a ** 2) ** 0.5) * ((len * sum_b_squared - sum_b ** 2) ** 0.5)

    return numerator / (denominator + FLT_EPSILON)

cpdef void _compute_correlation_iter(float[:, :] sim_mat, int[:] X_indptr, int[:] X_idx, float[:] X_data, int num_items,int num_users,  int cores):
    cdef:
        int start_item_A, start_item_B, end_item_A, end_item_B, item_a, item_b
                    
    for item_a in prange(num_items, nogil=True, num_threads=cores, schedule='dynamic'):
        for item_b in range(item_a + 1, num_items):

            # Extract the vectors of the columns that represent item_a, item_b
            start_item_A = X_indptr[item_a]
            end_item_A = X_indptr[item_a + 1]
            start_item_B = X_indptr[item_b] 
            end_item_B = X_indptr[item_b + 1]

            # Compute correlation between itemA, itemB based on the similarity function.
            sim_mat[item_a, item_b] =  sim_mat[item_b, item_a] = correlation(mat_idx=X_idx, mat_data=X_data,  A_start=start_item_A, A_end=end_item_A, B_start=start_item_B, B_end=end_item_B, len=num_users)

# City Block distance.
# The distance function itself.
cdef float cityblock(int[:] mat_idx, float[:] mat_data, int A_start, int A_end, int B_start, int B_end) nogil:
#The cityblock function is a distance function, to make it a simlarity function we are going to use its inverse.
    cdef:
        float distance = 0.0
        int A_iter, B_iter

    # Save the current position we are in.
    A_iter = A_start
    B_iter = B_start

    # If there is no data for either of the items, return 0.
    if A_start == A_end and B_start == B_end:
        return 1.0

    # Now iterate over the two vectors and compute the neccesary parts.
    while((A_iter < A_end) and (B_iter < B_end)): 
        if mat_idx[A_iter] < mat_idx[B_iter]:
            # Collect for the individual vector
            distance = distance + abs(mat_data[A_iter])
            A_iter += 1

        elif mat_idx[A_iter] == mat_idx[B_iter]:
            distance = distance + abs(mat_data[B_iter] - mat_data[A_iter])  
            # Proceed in the vector
            A_iter += 1
            B_iter += 1

        elif mat_idx[A_iter] > mat_idx[B_iter]:
            # Collect for the individual vector
            distance = distance + abs(mat_data[B_iter])
            B_iter += 1

    while(A_iter < A_end):
        distance = distance + abs(mat_data[A_iter])
        A_iter += 1

    while(B_iter < B_end):
        distance = distance + abs(mat_data[B_iter])
        B_iter += 1

    # make the distance to a smilarity
    # so that if dist(a, b) = 0 -> sim(a, b) = 1
    return 1 / (1 + distance)

# Compute the distance function for a matrix.
cpdef void _compute_cityblock_iter(float[:, :] sim_mat, int[:] X_indptr, int[:] X_idx, float[:] X_data, int num_items, int cores):
    cdef:
        int start_item_A, start_item_B, end_item_A, end_item_B, item_a, item_b
                    
    for item_a in prange(num_items, nogil=True, num_threads=cores, schedule='dynamic'):
        for item_b in range(item_a + 1, num_items):

            # Extract the vectors of the columns that represent item_a, item_b
            start_item_A = X_indptr[item_a]
            end_item_A = X_indptr[item_a + 1]
            start_item_B = X_indptr[item_b] 
            end_item_B = X_indptr[item_b + 1]

            # Compute correlation between itemA, itemB based on the similarity function.
            sim_mat[item_a, item_b] =  sim_mat[item_b, item_a] = cityblock(mat_idx=X_idx, mat_data=X_data,  A_start=start_item_A, A_end=end_item_A, B_start=start_item_B, B_end=end_item_B)

# M Distance
cdef float minkowski_distance(int[:] mat_idx, float[:] mat_data, int A_start, int A_end, int B_start, int B_end, int p) nogil:

    cdef:
        float distance = 0.0
        int A_iter, B_iter

    # Save the current position we are in.
    A_iter = A_start
    B_iter = B_start

    # If there is no data for either of the items, return 0.
    if A_start == A_end and B_start == B_end:
        return 1.0

    # Now iterate over the two vectors and compute the neccesary parts.
    while((A_iter < A_end) and (B_iter < B_end)): 
        if mat_idx[A_iter] < mat_idx[B_iter]:
            # Collect for the individual vector
            distance = distance + (abs(mat_data[A_iter]) ** p)
            A_iter += 1
            
        elif mat_idx[A_iter] == mat_idx[B_iter]:
            distance = distance + (abs(mat_data[B_iter] - mat_data[A_iter]) ** p)  
            # Proceed in the vector
            A_iter += 1
            B_iter += 1

        elif mat_idx[A_iter] > mat_idx[B_iter]:
            # Collect for the individual vector
            distance = distance + (abs(mat_data[B_iter]) ** p)
            B_iter += 1

    while(A_iter < A_end):
        distance = distance + (abs(mat_data[A_iter]) ** p)
        A_iter += 1

    while(B_iter < B_end):
        distance = distance + (abs(mat_data[B_iter]) ** p)
        B_iter += 1

    # normalize distance
    """with gil:
        print(distance)"""

    distance = distance ** (1.0 / p)

    """with gil:
        print(distance)"""

    # make the distance to a smilarity
    # so that if dist(a, b) = 0 -> sim(a, b) = 1
    return 1.0 / (1.0 + distance)

cpdef void _minkowski_distance_iter(float[:, :] sim_mat, int[:] X_indptr, int[:] X_idx, float[:] X_data, int num_items, int p, int cores):
    cdef:
        int start_item_A, start_item_B, end_item_A, end_item_B, item_a, item_b
                    
    for item_a in prange(num_items, nogil=True, num_threads=cores, schedule='dynamic'):
        for item_b in range(item_a + 1, num_items):

            # Extract the vectors of the columns that represent item_a, item_b
            start_item_A = X_indptr[item_a]
            end_item_A = X_indptr[item_a + 1]
            start_item_B = X_indptr[item_b] 
            end_item_B = X_indptr[item_b + 1]

            # Compute correlation between itemA, itemB based on the similarity function.
            sim_mat[item_a, item_b] =  sim_mat[item_b, item_a] = minkowski_distance(mat_idx=X_idx, mat_data=X_data,  A_start=start_item_A, A_end=end_item_A, B_start=start_item_B, B_end=end_item_B, p=p)

# Adjusted distances: adjusted_correlation and adjusted cosine.
cdef float adjusted_correlation(int[:] A_idx, int[:] B_idx, float[:] A_data, float[:] B_data, int A_start, int A_end, int B_start, int B_end, float A_mean, float B_mean) nogil:
    cdef:
        float den = 0.0, num_1 = 0.0, num_2 = 0.0
        int A_iter = 0, B_iter = 0
    
    if A_start == A_end or B_start == B_end:
        return 0.0

    A_iter = A_start
    B_iter = B_start
    
    while(A_iter < A_end and B_iter < B_end):
            if A_idx[A_iter] > B_idx[B_iter]:
                B_iter += 1
            elif A_idx[A_iter] == B_idx[B_iter]:
                den += (A_data[A_iter] - A_mean) * (B_data[B_iter] - B_mean)
                num_1 += (A_data[A_iter] - A_mean) ** 2
                num_2 += (B_data[B_iter] - B_mean) ** 2
            
                B_iter += 1
                A_iter += 1
            else:
                A_iter += 1   
    # Avoid zero-devision error if num_1 or num_2 is zero. If den is zero 0.0 is returned anyways.
    if den == 0.0 or (num_1 ** 0.5) == 0.0 or (num_2 ** 0.5) == 0.0:
        return 0.0
    else:
        return den / ((num_1 ** 0.5) * (num_2 ** 0.5) + FLT_EPSILON)

cdef float adjusted_cosine(int[:] A_idx, int[:] B_idx, float[:] A_data, float[:] B_data, int A_start, int A_end, int B_start, int B_end, float[:] means) nogil:
    cdef:
        float den = 0.0, num_1 = 0.0, num_2 = 0.0, A_mean = 0.0, B_mean = 0.0
        int A_iter = 0, B_iter = 0

    if A_start==A_end or B_start==B_end:
        return 0.0

    A_iter = A_start
    B_iter = B_start
    while(A_iter < A_end and B_iter < B_end):
            if A_idx[A_iter] > B_idx[B_iter]:
                B_iter += 1
            elif A_idx[A_iter] == B_idx[B_iter]:
                den += (A_data[A_iter] - means[A_idx[A_iter]]) * (B_data[B_iter] - means[A_idx[A_iter]])
                num_1 += (A_data[A_iter] - means[A_idx[A_iter]]) ** 2
                num_2 += (B_data[B_iter] - means[A_idx[A_iter]]) ** 2
            
                B_iter += 1
                A_iter += 1
            else:
                A_iter += 1

    if den == 0.0 or (num_1 ** 0.5) == 0.0 or (num_2 ** 0.5) == 0.0:
        return 0.0
    else:
        return den / ((num_1 ** 0.5) * (num_2 ** 0.5) + FLT_EPSILON)

cpdef void _compute_sim_iter(float[:, :] sim_mat, int[:] X_indptr, int[:] X_idx, float[:] X_data, float[:] means, int num_items,int num_users, int sim, int cores):
    cdef:
        int start_item_A, start_item_B, end_item_A, end_item_B, item_a, item_b
        float similarity
                    
    for item_a in prange(num_items, nogil=True, num_threads=cores, schedule='dynamic'):
        for item_b in range(item_a + 1, num_items):

            # Extract the vectors of the columns that represent item_a, item_b
            start_item_A = X_indptr[item_a]
            end_item_A = X_indptr[item_a + 1]
            start_item_B = X_indptr[item_b] 
            end_item_B = X_indptr[item_b + 1]

            # Compute similarity between itemA, itemB based on the similarity function.
            if sim == 2:
                # Adjusted correlation
                similarity = adjusted_correlation(A_idx=X_idx, B_idx=X_idx, A_data=X_data, B_data=X_data, A_start=start_item_A, A_end=end_item_A, B_start=start_item_B,
                B_end=end_item_B, A_mean=means[item_a], B_mean=means[item_b])
            
            elif sim == 3:
                # Adjusted cosine 
                similarity = adjusted_cosine(A_idx=X_idx, B_idx=X_idx, A_data=X_data, B_data=X_data, A_start=start_item_A, A_end=end_item_A, 
                B_start=start_item_B, B_end=end_item_B, means=means)

            sim_mat[item_a, item_b] = similarity
            sim_mat[item_b, item_a] = similarity