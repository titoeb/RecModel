#distutils: extra_compile_args = -fopenmp
#distutils: extra_link_args = -fopenmp
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

# Imports
import numpy as np
import scipy.sparse
from sklearn.metrics import pairwise_distances
import sharedmem
from multiprocessing import Pool
from functools import partial

# Imports from own package. 
from RecModel.base_model import RecModel
from RecModel.fast_utils.neighborhood_utils import _predict_neighborhood_iterative, _predict_neighborhood, _predict_neighborhood_iterative, _predict_neighborhood_iterative_weights_only
from RecModel.fast_utils.neighborhood_utils import _compute_sim_iter, _compute_correlation_iter, _compute_cityblock_iter, _minkowski_distance_iter

#  Helpers are over.
# Extension type of the neighborhood models.
class Neighborhood(RecModel):
    
    def __init__(self, num_items, num_users, nb_size, weights_only=False):
        self.num_items = num_items
        self.num_users = num_users
        self.sim_shared = None
        self.weights_only = weights_only

        # For item-based cf it does not make sense to use a neighborhood size larger then the number of items - 1.
        self.set_nb_size(nb_size)

    def set_nb_size(self, nb_size):
        # For item-based cf it does not make sense to use a neighborhood size larger then the number of items - 1.
        if nb_size > self.num_items - 1:
            self.nb_size = self.num_items - 1
        else:
            self.nb_size = nb_size
           
    def rank(self, items, users, topn, cores=1):   
        predictions = np.empty(len(items), dtype=np.float32)

        # Create the predictions      
        if self.weights_only is False:
            _predict_neighborhood_iterative(self.sim, users, items.astype(np.int32),  self.nb_size, self.X_data, self.X_idx, self.X_indptr, predictions, 0)
        
        elif self.weights_only is True:
             _predict_neighborhood_iterative_weights_only(self.sim, users, items.astype(np.int32),  self.nb_size, self.X_idx, self.X_indptr, predictions, 0)

        else:
            raise ValueError(f"weights only can either be true or false and cannot be {self.eights_only}")

        # Sort the predictions accordingly!
        #return np.flip(items[np.argpartition(predictions, -np.array(list(range(topn, 0, -1))))[(len(predictions) - topn):]])
        return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

    def predict(self, users, items): 
        res = np.zeros(len(items), dtype=np.float32)
        if self.weights_only is False:
            _predict_neighborhood(sim=self.sim, users=users, items=items, nb_size=self.nb_size, data=self.X_data, idx=self.X_idx, indptr=self.X_indptr, predictions=res, weights_only=0)
        elif self.weights_only is True:
            _predict_neighborhood(sim=self.sim, users=users, items=items, nb_size=self.nb_size, data=self.X_data, idx=self.X_idx, indptr=self.X_indptr, predictions=res, weights_only=1)
        else:
            raise ValueError(f"weights only can either be true or false and cannot be {self.weights_only}")

        return res
       
    def train(self, X, similarity_function, cores, p=None):
        # Check the similarity.
        if similarity_function not in ['jaccard','cosine', 'euclidean', 'correlation', 'adjusted_cosine', 'adjusted_correlation', 'cityblock', 'minowski']:
            raise ValueError(f"sin be one of ['jaccard','cosine', 'euclidean', 'correlation', 'adjusted_cosine', 'adjusted_correlation', 'cityblock', 'minowski'] but not {similarity_function}")
        
        # Copy X, and generate one csc and csr version.
        X = X.copy()

        # Store the matrix as csr
        X_csr = X.tocsr()
        X_csr.sort_indices()

        # Store the matrix as csc.
        X_csc = X.tocsc()
        X_csc.sort_indices()

        # Store the matrix for later prediction.
        # We need to iterate the row fast -> csr format
        self.X_indptr = X_csr.indptr.copy()
        self.X_idx = X_csr.indices.copy()
        self.X_data = X_csr.data.copy()

        # Pre-compute means for normalization within the distance measures.
        if similarity_function == 'adjusted_cosine':
        # Pre compute the row means:
            means = X.mean(axis = 1).A1

        elif similarity_function == 'adjusted_correlation':
        # Pre compute the row means:
            means = X.mean(axis = 0).A1
        
        # Compute the similarity matrix.
        if similarity_function == 'cosine':
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

        elif similarity_function == 'jaccard':
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

            # Store similarity as float32
            self.sim = res.astype(np.float32)

        elif similarity_function== 'euclidean':
            # The is simply the scaled, quadratic difference bettween the items
            res = pairwise_distances(X.T, metric='euclidean', n_jobs=cores, squared=True) 
            res =  1 / (res + 1)
            np.fill_diagonal(res, 0.0)
            self.sim = res.astype(np.float32)
        
        elif similarity_function == 'adjusted_cosine':
            self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)
            _compute_sim_iter(sim_mat=self.sim, X_indptr=X_csc.indptr, X_idx=X_csc.indices, X_data=X_csc.data, means=means,
                num_items=self.num_items, num_users=self.num_users, sim=3, cores=cores)

        elif similarity_function == 'adjusted_correlation':
            self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)
            _compute_sim_iter(sim_mat=self.sim, X_indptr=X_csc.indptr, X_idx=X_csc.indices, X_data=X_csc.data, means=means,
                num_items=self.num_items,num_users=self.num_users, sim=2, cores=cores)

        elif similarity_function == 'correlation':
            self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)
            _compute_correlation_iter(sim_mat=self.sim, X_indptr=X_csc.indptr, X_idx=X_csc.indices, X_data=X_csc.data, num_items=self.num_items,
            num_users=self.num_users, cores=cores)

        elif similarity_function == 'cityblock':
            self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)
            _compute_cityblock_iter(sim_mat=self.sim, X_indptr=X_csc.indptr, X_idx=X_csc.indices, X_data=X_csc.data, num_items=self.num_items, cores=cores)

        elif similarity_function == 'minowski':
            if p is None:
                raise ValueError("If you want to use the Minowski distance you need to specify p!")
            self.sim = np.full((X.shape[1], X.shape[1]), 0.0, dtype = np.float32)
            _minkowski_distance_iter(sim_mat=self.sim, X_indptr=X_csc.indptr, X_idx=X_csc.indices, X_data=X_csc.data, num_items=self.num_items, cores=cores, p=p)
        
        else:
            raise AttributeError(f"Similarity function {similarity_function} is not supported yet.")
            
    def share_sim(self):
         # Make the similarity matrix available for parallelized evaluation of topn recall.
        if self.sim_shared is None:
            self.sim_shared = sharedmem.empty(self.sim.shape, dtype=np.float32)
            self.sim_shared[:, :] = self.sim[:, :]

    def eval_topn(self, test_mat, topn, rand_sampled=1000, cores=1, random_state=1993):
        np.random.seed(random_state)

        if cores == 1:
            return super().eval_topn(test_mat=test_mat, topn=topn, rand_sampled=rand_sampled, cores=1, random_state=random_state, dtype='float32')

        elif cores > 1:
            self.share_sim()
            max_topn = topn.max()
            pool = Pool(cores)
            compute_hit_args = partial(compute_hit, rand_sampled=rand_sampled, topn=topn, max_topn=max_topn, sim=self.sim_shared,
                X_indptr=self.X_indptr, X_idx=self.X_idx, X_data=self.X_data, test_indptr=test_mat.indptr, test_idx=test_mat.indices,
                 nb_size=self.nb_size, weights_only=self.weights_only)
            res = pool.map(compute_hit_args, (user for user in range(test_mat.shape[0])))
            hits = np.stack(res).sum(axis=0)
            pool.close()
            pool.join()

        # Compute the precision at topn
        recall_dict = {}
        recall = hits / len(test_mat.nonzero()[0])
        precision = recall / topn
        for pos in range(len(topn)):
            recall_dict[f"Recall@{topn[pos]}"] = recall[pos]

        return recall_dict

def compute_hit(user, rand_sampled, topn, max_topn, sim, X_indptr, X_idx, X_data, test_indptr, test_idx, nb_size, weights_only):
    hits = np.zeros(topn.shape)
    # Which items did the so user consume?
    user_items = test_idx[test_indptr[user]:test_indptr[user+1]]

    if len(user_items) > 0:
        # Sample sonp.me random items 
        rand_items = np.random.randint(0, sim.shape[0], size=(rand_sampled + 1), dtype=np.int32)

        predictions = np.empty(rand_items.shape[0], dtype=np.float32)
        rand_pos = np.random.randint(0, rand_sampled- (2 * max_topn))
    
        for item in user_items:
            these_items = rand_items.copy()
            these_items[max_topn + rand_pos] = item

            # Get the topn items form the random sampled items plus the truly bought item
            if weights_only is False:
                _predict_neighborhood_iterative(sim, user, these_items,  nb_size, X_data, X_idx, X_indptr, predictions, 0)
            elif weights_only is True:
                _predict_neighborhood_iterative_weights_only(sim, user, these_items, nb_size, X_idx, X_indptr, predictions, 0)
            else:
                raise ValueError(f"weights only can either be true or false and cannot be {weights_only}") 

            candidates = these_items[np.argpartition(predictions, list(range(-max_topn, 0, 1)))[-max_topn:]][::-1]

            # If the true item was in the topn we have a hit!
            for pos in range(len(topn)):
                if item in candidates[:topn[pos]]:
                    hits[pos] += 1
    return hits