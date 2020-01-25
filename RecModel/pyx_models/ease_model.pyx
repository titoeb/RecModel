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
from sparse_tools import _predict_ease
import time

cdef class Ease(RecModel):
    cdef: 
        public unsigned int num_items, num_users
        public int[:] X_indptr, X_idx, 
        public float[:] X_data
        public float[:, :] W

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

        return np.flip(items[np.argpartition(predictions, -np.array(list(range(topn, 0, -1)), dtype=np.int32))[(len(predictions) - topn):]])

    cpdef void train(self, object X, int cores, float alpha, int verbose):
       
        # Make X to csr as this is needed for prediction and advantageous when training.
        X_csr = X.copy().tocsr()

        # Save X_csr internally to predict later on.
        self.X_indptr = X_csr.indptr.astype(np.int32)
        self.X_idx = X_csr.indices.astype(np.int32)
        self.X_data = X_csr.data.astype(np.float32)

        G = np.dot(X_csr.T, X_csr).todense().astype(np.float32)

        np.fill_diagonal(G, G.diagonal() + alpha)
        if verbose > 0:
            start = time.time()

        res = np.linalg.inv(G)
        if verbose > 0:
            print(f"Computing the inverse took {time.time() -start} seconds!")

        res = res / (-res.diagonal())

        np.fill_diagonal(res, 0)

        self.W = res.astype(np.float32)

    cpdef np.ndarray[np.float32_t, ndim=1] predict(self, int[:] users, int[:] items, unsigned int cores):
        return sparse_tools._predict_ease(self.X_indptr, self.X_idx, self.X_data, self.W, users, items)


