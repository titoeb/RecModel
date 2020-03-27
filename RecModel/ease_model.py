import numpy as np
import scipy.sparse
import time
import os
from multiprocessing import Pool
from functools import partial
import sharedmem

# Import from RecModels
from RecModel.fast_utils.ease_utils import _predict_ease
from RecModel.base_model import RecModel

def compute_hit(process, n_processes, n_users, rand_sampled, topn, max_topn, W, X_indptr, X_idx, X_data, test_indptr, test_idx):
        sum_hits = np.zeros(topn.shape)

        start = int((process / n_processes) * n_users)
        end = int(((process + 1) / n_processes) * n_users)

        for user in range(start, end):
            
            # Which items did the so user consume?
            user_items = test_idx[test_indptr[user]:test_indptr[user+1]]

            if len(user_items) > 0:
                # Sample sonp.me random items
                rand_items = np.random.randint(0, W.shape[0], size=(rand_sampled + 1)).astype(np.int32)
                users = np.full((rand_items.shape[0] + 1), user, dtype=np.int32)

                rand_pos = np.random.randint(0, rand_sampled- (2 * topn.max()))
            
                for item in user_items:
                    these_items = rand_items
                    these_items[max_topn + rand_pos] = item
                    # Get the topn items form the random sampled items plus the truly bought item
                    predictions = _predict_ease(X_indptr, X_idx, X_data, W, users, these_items)

                    #predictions = np.zeros(these_items.shape)
                    candidates = these_items[np.argpartition(predictions, list(range(-max_topn, 0, 1)))[-max_topn:]][::-1]

                    # If the true item was in the topn we have a hit!
                    for pos in range(len(topn)):
                        if item in candidates[:topn[pos]]:
                            sum_hits[pos] += 1
        return sum_hits

class Ease(RecModel):
    def __init__(self, num_items, num_users):
        self.num_items = num_items
        self.num_users = num_users
        self.W = None

    def rank(self, items, users, topn=None):
        predictions = _predict_ease(self.X_indptr, self.X_idx, self.X_data, self.W, np.full(items.shape[0], users ,dtype=np.int32), items.astype(np.int32))
        return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]
    
    def load_mat(self, path='W_mat.npy', X=None):
        # Try to load mat!
        try:
            self.W = np.load(path).astype(np.float32)
            self.W_shared = sharedmem.empty(self.W.shape, dtype=np.float32)
            self.W_shared[:, :] = self.W[:, :]
            print("The weight matrix was loaded succesfully!")
            if X is not None:
                # Make X to csr as this is needed for prediction and advantageous when training.
                X_csr = X.copy().tocsr()

                # Save X_csr internally to predict later on.
                self.X_indptr = X_csr.indptr.astype(np.int32)
                self.X_idx = X_csr.indices.astype(np.int32)
                self.X_data = X_csr.data.astype(np.float32)

        except FileNotFoundError:
            print("The weight matrix could not be loaded!")

    def save_mat(self, path='W_mat.npy'):
        if self.W is not None:
            np.save(path, self.W)
        else:
            print("Matrix could not be saved, please fit model first!")

    def train(self, X, alpha, verbose, cores):
        os.environ["MKL_NUM_THREADS"]=f"{cores}" 
        os.environ["MKL_DYNAMIC"]="FALSE"

        # Make X to csr as this is needed for prediction and advantageous when training.
        X_csr = X.copy().tocsr()

        # Save X_csr internally to predict later on.
        self.X_indptr = X_csr.indptr.astype(np.int32)
        self.X_idx = X_csr.indices.astype(np.int32)
        self.X_data = X_csr.data.astype(np.float32)

        G = np.dot(X_csr.T, X_csr).todense().astype(np.float32)
        if verbose > 0:
            print("Compute the dot product")

        np.fill_diagonal(G, G.diagonal() + alpha)
        if verbose > 0:
            start = time.time()

        res = np.linalg.inv(G)
        if verbose > 0:
            print(f"Computing the inverse took {time.time() -start} seconds!")

        res = res / (-res.diagonal() + 1e-9)

        np.fill_diagonal(res, 0)
        if verbose > 0:
            print("Starting to create sharedmem")
        self.W = res.astype(np.float32)
        self.W_shared = sharedmem.empty(self.W.shape, dtype=np.float32)
        self.W_shared[:, :] = self.W[:, :]
        if verbose > 0:
            print("Training finished!")

    def predict(self, users,  items):
        return _predict_ease(self.X_indptr, self.X_idx, self.X_data, self.W, users, items)

    def eval_topn(self, test_mat, topn, rand_sampled=1000, cores=1, random_state=1993):
        hits = np.zeros(topn.shape)
        np.random.seed(random_state)

        if cores == 1:
            return super().eval_topn(test_mat=test_mat, topn=topn, rand_sampled =rand_sampled, cores=1, random_state=random_state, dtype='float32')

        elif cores > 1:
            pool = Pool(cores)
            compute_hit_args = partial(compute_hit, n_processes=cores, n_users=test_mat.shape[0], rand_sampled=rand_sampled, topn=topn, max_topn=topn.max(), W=self.W_shared,
                X_indptr=self.X_indptr, X_idx=self.X_idx, X_data=self.X_data, test_indptr=test_mat.indptr, test_idx=test_mat.indices)
            res = pool.map(compute_hit_args, (core for core in range(cores)))
            pool.close()
            pool.join()
            hits = np.stack(res).sum(axis=0)

            #hits += compute_hit(process=0, n_processes=1, n_users=test_mat.shape[0], rand_sampled=rand_sampled, topn=topn, max_topn=topn.max(), W=self.W_shared,
            #    X_indptr=self.X_indptr, X_idx=self.X_idx, X_data=self.X_data, test_indptr=test_mat.indptr, test_idx=test_mat.indices)
     
        # Compute the precision at topn
        recall_dict = {}
        recall = hits / len(test_mat.nonzero()[0])
        precision = recall / topn
        for pos in range(len(topn)):
            recall_dict[f"Recall@{topn[pos]}"] = recall[pos]

        return recall_dict
