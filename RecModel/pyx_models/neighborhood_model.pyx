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
from sklearn.metrics import pairwise_distances

# C Imports 
cimport numpy as np
cimport sparse_tools
cimport cd_fast
from cython cimport floating
from base_class cimport RecModel
from libc.stdlib cimport malloc, free

# C Debug imports
from libc.stdio cimport printf

# Helper functions

# Get the qsort function for the internal argsort!
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compare)(const_void *, const_void *)) nogil

# This struct is used to track the elements in argsort.
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
    order = <int*> malloc(sizeof(int) * result_n)
    
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
    
# Dot product between rel_sim and rel_data scaled by the sum of rel_data
cdef float _create_pred(float* rel_sim, float[:] rel_data, int* idx, int idx_len) nogil:
    cdef:
        float scaling_factor = 0.0, dot_product = 0.0
        int i = 0
    for i in range(idx_len):
        scaling_factor += rel_sim[idx[i]]
        dot_product += rel_sim[idx[i]] * rel_data[idx[i]]
    return dot_product / scaling_factor

# Copy the relevant similarities from row item and colums in rel_item from sim to rel_sim
cdef void _cpy(int item, int[:] rel_items, float* rel_sim, float[:, :] sim) nogil:
    cdef:
        int i
    for i in range(len(rel_items)):
        rel_sim[i] = sim[item, rel_items[i]]

# Prediction function that is used within the neighborhood models.
cdef void _predict_neighborhood_iterative(float[:, :] sim, int user, int[:] items, int nb_size, float[:] data, int[:] idx, int[:] indptr, float[:] predictions, int offset) nogil:
    """ Fast, hopefully parallelized implentation of the predict method. Currently only working with item-based neighborhood methods!
    """
    cdef:
        int start_user, end_user, i, actual_nh_size, thread, n_items
        int[:] rel_items
        int * sorted_idx_internal
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
    rel_sim = <float*> malloc(sizeof(float) * len(rel_items))

    # The actual neighborhood size is the minimum of the desired nh size and how many items the user consumed.
    actual_nh_size = min(nb_size, len(rel_items))

    for i in range(len(items)):
        # Extract the relevant similarity, it assumes item-based 
        #rel_sim = sim[item, rel_items] <- is implemented here.
        _cpy(items[i], rel_items, rel_sim, sim)
        
        # Sorted_idx_internal is int* pointing to the ordered ranks of the indices sorted by rel_sim.
        # This is not efficient as all entries of rel_sim are sorted but only the first
        # actual_nh_size are used. 
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

cdef void _predict_neighborhood_parallel(float[:, :] sim, int user, int[:] items, int nb_size, float[:] data, int[:] idx, int[:] indptr, float[:] predictions, int offset, int cores) nogil:
    """ Fast, hopefully parallelized implentation of the predict method. Currently only working with item-based neighborhood methods!
    """
    cdef:
        int start_user, end_user, i, actual_nh_size, thread, n_items
        int[:] rel_items
        int ** sorted_idx_internal
        float** rel_sim
        float[:] rel_data

    # Extract the row that belongs to the user from the sparse matrix.
    start_user = indptr[user]
    end_user = indptr[user + 1]
    n_items = len(items)

    if start_user == end_user:
        for i in range(n_items):
            predictions[offset + i] = 0
        return

    # BE AWARE! these are the original arrays. Use rel_items[:] = ... if you want to copy!
    rel_items = idx[start_user:end_user]
    rel_data = data[start_user:end_user]

    # Alloate rel_sim where the relevant similarity will be saved!
    rel_sim = <float**> malloc(sizeof(float*) * cores)
    sorted_idx_internal = <int**> malloc(sizeof(int*) * cores)

    for i in range(cores):
        rel_sim[i] = <float*> malloc(sizeof(float) * len(rel_items))

    # The actual neighborhood size is the minimum of the desired nh size and how many items the user consumed.
    actual_nh_size = min(nb_size, len(rel_items))

    for i in prange(n_items, nogil=True, num_threads=cores, schedule='dynamic'):
    #for i in range(len(items)):
        with gil:
            thread = threadid()
        
        # Extract the relevant similarity, it assumes item-based 
        # Is a copy here advantageous?
        _cpy(items[i], rel_items, rel_sim[thread], sim)
        
        # This index orders the items by similarity from least to maximum
        # Sorted_idx_internal is int* pointing to the ordered ranks of the indices
        # This is not efficient as all entries of rel_sim are sorted byt only the first
        # actual_nh_size are used. 
        sorted_idx_internal[thread] = _argsort(rel_sim[thread], len(rel_items), result_n=actual_nh_size)
    
        # Put it in a typed memoryview to allow indexing with it
        # Only take the top actual_nh_size values.
        # Would be more efficient only to order the actual_nh_size top items ...       
        # With the top rated items create the prediction with a weighted sum.
        predictions[offset + i] = _create_pred(rel_sim[thread], rel_data, idx=sorted_idx_internal[thread], idx_len=actual_nh_size)

        # Don't forget to free the sorted_idx.
        free(sorted_idx_internal[thread])

    # Don't forget to free rel_sim!
    for i in range(cores):
        free(rel_sim[i])
    free(rel_sim)
    free(sorted_idx_internal)


# These functions compute the similarity functions correlation and adjusted cosine for the Neighborhood extension type.
cdef double correlation(int[:] A_idx, int[:] B_idx, float[:] A_data, float[:] B_data, int A_start, int A_end, int B_start, int B_end, float A_mean, float B_mean) nogil:
    cdef:
        float den = 0.0, num_1 = 0.0, num_2 = 0.0
        int A_iter = 0, B_iter = 0, len_A, len_B
    
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
        return den / ((num_1 ** 0.5) * (num_2 ** 0.5))

cdef double adj_cosine(int[:] A_idx, int[:] B_idx, float[:] A_data, float[:] B_data, int A_start, int A_end, int B_start, int B_end, float[:] means) nogil:
    """
    Means is a bit special. If the similarity between items has to be calcualted this should be the 
    mean of the users, if the item similarity has to be computed this should be the mean of the items.
    """
    cdef:
        float den = 0.0, num_1 = 0.0, num_2 = 0.0, A_mean = 0.0, B_mean = 0.0
        int A_iter = 0, B_iter = 0, len_A, len_B

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
        return den / ((num_1 ** 0.5) * (num_2 ** 0.5))

#  Helpers are over.
# Extension type of the neighborhood models.
cdef class Neighborhood(RecModel):
    cdef:
        int axis, nb_size
        public int num_items, num_users
        public int[:] X_indptr, X_idx
        public float[:] X_data
        public float[:, :] sim
    
    def __init__(self, int axis, int num_items, int num_users, int nb_size):
        self.axis = axis
        self.num_items = num_items
        self.num_users = num_users

        # For item-based cf it does not make sense to use a neighborhood size larger then the number of items - 1.
        self.set_nb_size(nb_size)

    cpdef set_nb_size(self, int nb_size):
        # For item-based cf it does not make sense to use a neighborhood size larger then the number of items - 1.
        if self.axis == 1:
            if self.nb_size > self.num_items - 1:
                self.nb_size = self.num_items - 1
            else:
                self.nb_size = nb_size
        else:
            if self.nb_size > self.num_users - 1:
                self.nb_size = self.users - 1
            else:
                self.nb_size = nb_size
           
    cpdef np.ndarray[np.int32_t, ndim=1] rank(self, np.ndarray[np.int32_t, ndim=1] items, int user, unsigned int topn, unsigned int cores):   
        cdef: 
            float[:] predictions

        predictions = np.empty(len(items), dtype=np.float32)

        topn = min(topn, len(items))

        # Create the predictions
        _predict_neighborhood_iterative(self.sim, user, items,  self.nb_size, self.X_data, self.X_idx, self.X_indptr, predictions, 0)

        # Sort the predictions accordingly!
        return np.flip(items[np.argpartition(predictions, -np.array(list(range(topn, 0, -1)), dtype=np.int32))[(len(predictions) - topn):]])

    cpdef np.ndarray[np.double_t, ndim=1] predict(self, int[:] users, int[:] items):
        cdef:
            int out_len, i, user
            float[:] res

        out_len = len(items)
        res = np.empty(out_len, dtype=np.float32)

        for i in range(len(users)):
            _predict_neighborhood_iterative(self.sim, users[i], items[i:(i+1)],  self.nb_size, self.X_data, self.X_idx, self.X_indptr, res, i)

        return np.array(res)

    cpdef void train(self, object X, int sim, int cores):
        cdef:
            int row, start_item_A, end_item_A, start_item_B, end_item_B, item_a, item_b
            float similarity = 0.0
            float[:] means, X_data
            int[:] X_indptr, X_idx

        # Test similarity as input

        
        # Copy X, and generate one csc and csr version.
        X = X.copy()
        X_csr = X.tocsr()
        X_csr.sort_indices()
        X_csc = X.tocsc()
        X_csc.sort_indices()

        # Store the matrix for later prediction.
        if self.axis == 1:
            # We need to iterate the row fast -> csr format
            self.X_indptr = X_csr.indptr
            self.X_idx = X_csr.indices
            self.X_data = X_csr.data
        else:
            # We need to iterate the colums fast -> csc format
            self.X_indptr = X_csc.indptr
            self.X_idx = X_csc.indices
            self.X_data = X_csc.data
        
        if self.axis == 0:
            # User-based neighborhood
            raise ValueError("User-based neighborhood not implemented yet!")
            
        elif self.axis == 1:
            # Prepare distance matrix for item-based filtering.

            if sim == 3:
            # Pre compute the row means:
                means = X.sum(axis = 1).A1

            elif sim == 2:
            # Pre compute the row means:
                means = X.sum(axis = 0).A1
            
            # Compute the similarity between all items

            if sim == 'cosine':
                # Compute the cosine.

                # These normalizing constants square root of the sum of the squared entries of X_csr. Dividing each columns by the constant
                # will make the l2 norm of the columns 1. Then the cosine is simply the dot product between X_csr and X_csr.
                normalize_const_cols = np.sqrt(X_csr.multiply(X_csr).sum(axis=0).A1)

                # Only the nonzero values have to be normalized as 0 / x will be 0.
                X_csr.data = X_csr.data.copy() / normalize_const_cols[X_csr.indices]
                
                # Return the dot product of the scaled matrix as cosine
                res = X_csr.T.dot(X_csr).todense()
                np.fill_diagonal(res, 0.0)
                self.sim = res

            elif sim == 'jaccard':
                # compute the jaccard distance

                # The cosine is only defined for binary vetors, therefore, overwrite the values of the sparse matrix.
                X_csc.data[:] = 1

                # The sum of the column is simply the number of non-zero entries as each value is 1.
                col_sum = X_csc.getnnz(axis=0)

                # The numerator of the cosine similarity is the inersection of each of the columns of X.
                # With a binary vector the intersection is simply the product between the vectors (only if the entry in both is one, the result will be one)
                xy = X_csc.T * X_csc

                # Key to compute the denomintor fastly is to see that |A or B| = |A| + |B| - |A and B|, then you simply need to combine the column sums using the sparse structure.
                # for rows

                # Combine the row means to account for the structure of the distance matrix of all columns with each other
                xx = np.repeat(col_sum, xy.getnnz(axis=0))
                yy = col_sum[xy.indices]

                # Finally combein the result
                simi = xy.copy()
                simi.data = simi.data / (xx + yy - xy.data + 1e-10)
                simi.data = simi.data.astype(np.float32)
                
                # Save the result as dense matrix and set the diagonal.
                res = simi.todense()

                # Set the diagonal to small value so that the item itself will never be considered.
                np.fill_diagonal(res, 0.0)
                self.sim = res
            elif sim == 'mean_squared':
               # The is simply the scaled, quadratic difference bettween the items
                res = pairwise_distances(X.T, metric='l2', n_jobs=cores, squared=True) 
                res =  1 / (res + 1)
                np.fill_diagonal(res, 0.0)
                self.sim = res
            else:
                # I did not figure a way out to vectorize the computation of the correlation of adjusted cosine. Therefore we need to iterate :(
                # Pre assign similarity.
                self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)

                # To iterate faster store X_csc internal arrays more efficiently.
                X_data = X_csc.data
                X_indptr = X_csc.indptr
                X_idx = X_csc.indices

                for item_a in prange(self.num_items, nogil=True, num_threads=cores, schedule='dynamic'):
                    for item_b in range(item_a + 1, self.num_items):
        
                        # Extract the vectors of the columns that represent item_a, item_b
                        start_item_A = X_indptr[item_a]
                        end_item_A = X_indptr[item_a + 1]
                        start_item_B = X_indptr[item_b] 
                        end_item_B = X_indptr[item_b + 1]

                        # Compare similarity between itemA, itemB based on the similarity function.
                        if sim == 'correlation' :
                            similarity = correlation(A_idx=X_idx, B_idx=X_idx, A_data=X_data, B_data=X_data, A_start=start_item_A, A_end=end_item_A, 
                            B_start=start_item_B, B_end=end_item_B, A_mean=means[item_a], B_mean=means[item_b])
                        
                        elif sim == 'adjusted_cosine':
                            # Adjusted cosine 
                            similarity = adj_cosine(A_idx=X_idx, B_idx=X_idx, A_data=X_data, B_data=X_data, A_start=start_item_A, A_end=end_item_A, 
                            B_start=start_item_B, B_end=end_item_B, means=means)

                        self.sim[item_a, item_b] = similarity
                        self.sim[item_b, item_a] = similarity  
        else:
            raise ValueError(f"Axis can only take value 0, 1 not {self.axis}")
