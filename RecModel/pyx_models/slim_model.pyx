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

cdef class Slim(RecModel):
    cdef public unsigned int num_items, num_users
    cdef public int[:] A_indptr, A_idx, W_indptr, W_idx
    cdef public double[:] A_data, W_data

    def __init__(self, unsigned int num_items, unsigned int num_users):
        self.num_users = num_users       
        self.num_items = num_items

    cpdef np.ndarray[np.int32_t, ndim=1] rank(self, np.ndarray[np.int32_t, ndim=1] items, long user, unsigned int topn, unsigned int cores):           
        cdef: 
            int[:] items_view = items
            np.ndarray[np.int32_t, ndim=1] users = np.full(len(items), user, dtype=np.int32)
            int[:] users_view = users
            np.ndarray[np.double_t, ndim=1] predictions

        predictions = self.predict(users_view, items_view, cores=cores)
        #print(f"predictions: {predictions}, len(predictions): {len(predictions)}")
        #print(-np.array(list(range(topn, 0, -1)), dtype=np.int32))
        #print(np.flip(items[np.argpartition(predictions, np.array(list(range(topn, 0, -1)), dtype=np.int32))[(len(predictions) - topn):]]))
        
        # Ranking optimized for sorting only the first few entries. This is useful if topn << |item| and sorting
        # all ranked items would take significantly longer.
        return np.flip(items[np.argpartition(predictions, -np.array(list(range(topn, 0, -1)), dtype=np.int32))[(len(predictions) - topn):]])

    cpdef void train(self, object X, floating alpha, floating l1_ratio, int max_iter, floating tolerance, int cores, int verbose):
        X = X.copy()
        X_train = X.tocsc()
        X_train.sort_indices
        
        self.W_indptr, self.W_idx, self.W_data = cd_fast.train_Slim(X=X_train, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tolerance, cores=cores, verbose=verbose)
        
        # To predit a csr matrix is needed.        
        X_eval = X.tocsr()
        X_eval.sort_indices()
        self.A_indptr = X_eval.indptr
        self.A_idx = X_eval.indices
        self.A_data = X_eval.data

    cpdef np.ndarray[np.double_t, ndim=1] predict(self, int[:] users, int[:] items, unsigned int cores):
        cdef:
            int elem
            unsigned int start_A
            unsigned int end_A
            unsigned int start_W
            unsigned int end_W
            unsigned int n_items = users.shape[0]

        # Either items of length 1 and users of different length, 
        if (users.shape[0] != items.shape[0]):
            raise ValueError(f"Users and items need to have the same shape, but users.shape[0]={users.shape[0]} and items.shape[0]={items.shape[0]} is not possible!")

        # Allocate output array
        cdef np.ndarray[np.double_t, ndim=1] myarr = np.empty(n_items, dtype=np.float64)
        cdef double[:] output_view = myarr
        
        if cores > 1:
            # Parallel computation use prange
            # Iterate over users, items and make base computation.    
            for elem in prange(n_items, nogil=True, num_threads=cores):
                start_A = self.A_indptr[users[elem]]
                end_A = self.A_indptr[users[elem] + 1]
                start_W = self.W_indptr[items[elem]]
                end_W = self.W_indptr[items[elem] + 1]
                if (end_A - start_A) > 0 and (end_W - start_W):
                    sparse_tools.sparse_mult_vecs(elem, output_view, self.A_idx[start_A:end_A], self.A_data[start_A:end_A], self.W_idx[start_W:end_W], self.W_data[start_W:end_W])
                else:
                    output_view[elem] = 0.0
        else:
             for elem in range(n_items):
                start_A = self.A_indptr[users[elem]]
                end_A = self.A_indptr[users[elem] + 1]
                start_W = self.W_indptr[items[elem]]
                end_W = self.W_indptr[items[elem] + 1]
                if (end_A - start_A) > 0 and (end_W - start_W) > 0:
                    sparse_tools.sparse_mult_vecs(elem, output_view, self.A_idx[start_A:end_A], self.A_data[start_A:end_A], self.W_idx[start_W:end_W], self.W_data[start_W:end_W])
                else:
                    output_view[elem] = 0.0
        return myarr
    @property
    def W(self):
        indptr = np.array(self.W_indptr).copy()
        idx = np.array(self.W_idx).copy()
        data = np.array(self.W_data).copy()

        return scipy.sparse.csc_matrix((data, idx, indptr), dtype=np.float64, shape = (self.num_items, self.num_items))
