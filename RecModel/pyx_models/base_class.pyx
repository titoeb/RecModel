#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
#cython: nonecheck=False

# Imports
import numpy as np
from multiprocessing import Pool
from functools import partial

# C Imports 
cimport numpy as np


cdef class RecModel:
    def __init(self):
        pass

    def train(self):
        pass

    def predict(self, user_item):
        pass

    def rank(self):
        pass

    cdef unsigned int compute_hit(self, int user, int[:] items_selected, unsigned int rand_sampled, int[:] topn, int[:] hits, int max_topn, unsigned int cores):
        cdef:
            unsigned int item, item_num, missing_items, i

        if len(items_selected) == 0:
            return 0
        else:
            rand_items = np.random.randint(0, self.num_items, size=rand_sampled, dtype=np.int32)
           
            #print(f"rand_items: {rand_items}")
            # For each item the user bought.
            for item_num in range(len(items_selected)):

                item = items_selected[item_num]
                # Get the topn items form the random sampled items plus the truly bought item
                candidates = self.rank(user=user, items=np.append(item, rand_items).astype(np.int32), topn=max_topn, cores=1)

                # If the true item was in the topn we have a hit!
                for i in range(len(topn)):
                    if item in candidates[:topn[i]]:
                        hits[i] += 1


    cpdef np.ndarray[np.int32_t, ndim=1] compute_hit_par(self, int thread, int[:] X_idx, int[:] X_indptr, unsigned int rand_sampled, int[:] topn, int max_topn, int n_threads):
        cdef:
            long start_user, end_user, user
            int item_num, item, i
            np.ndarray[np.int32_t, ndim=1] hits
            int[:] items_selected
            np.ndarray[np.int32_t, ndim=1] rand_items
            int[:] candidates
        # Allocate the hits array.
        hits = np.full(len(topn), 0, dtype=np.int32)

        # First compute the number of users for which this thread has to compute the hits.
        start_user = (thread / n_threads) * self.num_users
        end_user = ((thread + 1) / n_threads) * self.num_users
        

        for user in range(start_user, end_user):
            
            rand_items = np.random.randint(0, self.num_items, size=rand_sampled, dtype=np.int32)

            # Get the items the user did consume.
            # Maybe make this a np.array?
            items_selected =  X_idx[X_indptr[user]:X_indptr[user+1]]

            if len(items_selected) > 0:
                # For each item the user bought.
                for item_num in range(len(items_selected)):

                    item = items_selected[item_num]
                    # Get the topn items form the random sampled items plus the truly bought item
                    candidates = self.rank(user=user, items=np.append(item, rand_items).astype(np.int32), topn=max_topn, cores=1)

                    # If the true item was in the topn we have a hit!
                    for i in range(len(topn)):
                        if item in candidates[:topn[i]]:
                            hits[i] += 1
        return hits

    def eval_topn(self, test_mat, unsigned int rand_sampled, int[:] topn, int cores, save):
        cdef:
            unsigned int max_topn = 0, row = 0, num_user = test_mat.shape[0]
            unsigned int low, high, i, core
            double recall = 0.0
            dict recall_dict = {}
            int[:] A_indptr = test_mat.indptr.astype(np.int32)
            int[:] A_idx = test_mat.indices.astype(np.int32)
            np.ndarray[np.int32_t, ndim=1] hits = np.zeros(len(topn), dtype=np.int32)

        test_mat = test_mat.copy()
        
        # Compute the max of top hit
        for i in range(len(topn)):
            max_topn = max(max_topn, topn[i])
     
        if cores == 1:
            # Old, Completely non-parallelized version
            for user in range(num_user):
                # Extract information from sparse matrix.
                low, high = A_indptr[user], A_indptr[user+1]
                self.compute_hit(user, A_idx[low:high],  rand_sampled=rand_sampled, topn=topn, hits=hits, max_topn=max_topn, cores=cores)
        else:
            raise ValueError("Unfortunately, parallelizing eval_topn is not possible at the moment.")

        for i in range(len(hits)):
             recall = hits[i] / len(test_mat.data)
             recall_dict[f"Recall@{topn[i]}"] = recall
        return recall_dict
                
    cpdef eval_prec(self, mat, metric, cores):
        cdef:
            int[:] items_view, users_view
            
        metric = metric.upper()
        mat = mat.copy()
        if metric in ['MSE', 'RMSE', 'MAE']:

            # For these metric the model.predict() method has to be used.
            # For the following elements we have to create a prediction.
            non_zero_elements = mat.nonzero()
            users_view = non_zero_elements[0].astype(np.int32)
            items_view = non_zero_elements[1].astype(np.int32)

            # Get predictions from the model
            predictions = self.predict(users_view, items_view, cores=cores).reshape(1, -1)

            # Apply the corresponding eval metric to get result.
            if metric == 'RMSE':
                return np.sqrt(np.mean(np.square(mat[non_zero_elements] - predictions)))

            elif metric == 'MSE':
                return np.mean(np.square(mat[non_zero_elements] - predictions))

            else:
                return np.mean(np.abs(mat[non_zero_elements] - predictions))

        else:
            raise ValueError("Metric {metric} is not implemented.")
