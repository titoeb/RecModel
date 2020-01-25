
# Imports
import numpy as np
import scipy.sparse
import ctypes
import os

# Imports from own package. 
from RecModel.py_models.base_model import RecModel
from RecModel.py_models.fast_utils.slim_utils import _predict_slim, train_Slim

class Slim(RecModel):
    def __init__(self, num_items, num_users):
        self.num_users = num_users       
        self.num_items = num_items

    def rank(self, items, users, topn):           
        predictions = self.predict(np.full(len(items), users, dtype=np.int32), items.astype(np.int32))
        return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

    def train(self, X, alpha, l1_ratio, max_iter, tolerance, cores, verbose):
        X = X.copy()
        X_train = X.tocsc()
        X_train.sort_indices
        
        self.W_indptr, self.W_idx, self.W_data = train_Slim(X=X_train, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tolerance, cores=cores, verbose=verbose)
        
        # To predit a csr matrix is needed.        
        X_eval = X.tocsr()
        X_eval.sort_indices()
        self.A_indptr = X_eval.indptr
        self.A_idx = X_eval.indices
        self.A_data = X_eval.data

    def predict(self, users, items):

        # Either items of length 1 and users of different length, 
        if (users.shape[0] != items.shape[0]):
            raise ValueError(f"Users and items need to have the same shape, but users.shape[0]={users.shape[0]} and items.shape[0]={items.shape[0]} is not possible!")

        # Allocate output array
        pred = np.empty(items.shape[0], dtype=np.float64)
        _predict_slim(users, items, pred, self.A_indptr, self.W_indptr, self.A_idx, self.W_idx, self.A_data, self.W_data)
        
        return pred

    @property
    def W(self):
        indptr = np.array(self.W_indptr).copy()
        idx = np.array(self.W_idx).copy()
        data = np.array(self.W_data).copy()

        return scipy.sparse.csc_matrix((data, idx, indptr), dtype=np.float64, shape = (self.num_items, self.num_items))
