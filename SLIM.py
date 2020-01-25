from utils import utils
import numpy as np
import scipy.sparse
from sklearn.linear_model import cd_fast
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Calling the Cython on which sklearn is based directly.
@ignore_warnings(category=ConvergenceWarning)
def fast_Elastic_net(X_data, X_idx, X_indptr, y: np.ndarray, l1_reg: float, l2_reg: float,
                     max_iter: int, tolerance: float, rng: np.random.mtrand.RandomState, X_sparse_scaling, coefs,
                     random=False, positive=True):


    # Fit Model
    weights, _, _, _ = cd_fast.sparse_enet_coordinate_descent(w=coefs, alpha=l1_reg, beta=l2_reg,
                                                                                 X_data=X_data,
                                                                                 X_indices=X_idx, X_indptr=X_indptr,
                                                                                 y=y, X_mean=X_sparse_scaling,
                                                                                 max_iter=max_iter, tol=tolerance,
                                                                                 rng=rng, random=random,
                                                                                 positive=positive)

    return scipy.sparse.lil_matrix(weights).reshape(-1, 1)

class SLIM(utils.RecModel):

    def __init__(self, num_items, num_users):
        self.num_items = num_items
        self.num_users = num_users

    def rank(self, items, users, topn=None):
        if topn is None:
            topn = len(items)

        if isinstance(users, list):
            if not type(items) == np.ndarray:
                users = np.array(users)
            return [self.rank(items, user, topn) for user in users]
        else:
            if not type(items) == np.ndarray:
                items = np.array(items)
            predictions = self.predict(users=users, items=items)

            # simple heuristic: If the number of topn is higher than half the number of prediction it may useful to
            # sort the whole vector.
            if len(predictions) * 0.5 > topn:
                # Ranking optimized for sorting only the first few entries. This is useful if topn << |item| and sorting
                # all ranked items would take significantly longer.
                return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

            else:
                # Ranking sorting all item, useful if topn near |items|.
                return items[np.argsort(predictions)[-topn:]][::-1]

    def train(self, utility_mat, random_state, verbose=0, eval_mat=None, l1_reg=0.5, l2_reg=0.01, max_iter=25,
              tolerance=0.001, cores=8, stopping_rounds=3, min_improvement=0.0001, dtype='float64'):
        with utils.MKLThreads(1):

            # Pre_compute some objects needed for model fitting
            if not type(utility_mat) == scipy.sparse.csc.csc_matrix:
                utility_mat_col = scipy.sparse.csc_matrix(utility_mat)
            else:
                utility_mat_col = utility_mat

            if verbose > 0:
                print(f"l1_reg: {l1_reg}, l2_reg: {l2_reg}")
                print(f"{self.num_items} models need to be fitted.")

            # Scaling coefficient:
            X_sparse_scaling = np.asfortranarray(np.zeros(self.num_items, dtype=dtype, order='F'))

            # Pre-allocate the cofficients:
            coefs = np.asfortranarray(np.empty(self.num_items, dtype=dtype, order='F'))

            # Create empty sparse matrix to collect the results.
            # Use type lil_matrix since inserting values is more efficient. Later convert to csc or csr sparse matrix.
            self.W = scipy.sparse.lil_matrix((self.num_items, self.num_items), dtype=dtype)

            # Proprocess the random state.
            random_state = np.random.mtrand.RandomState(random_state)

            # Also save the train_mat for prediction!
            self.A = utility_mat

            if cores == 1:
                # As utilty__mat_col is in csc format, iter_row will iterate its columns.
                for elem in utils.iter_cols(utility_mat_col):

                    # Create target y
                    item, data, idx = elem
                    y = np.zeros(self.num_users, dtype='float64', order='F')
                    y[idx] = data

                    if verbose > 0:
                        if item % 100 == 0:
                            print(f"Currently working on model {item}")

                    self.W[:, item] = fast_Elastic_net(utility_mat_col.data, utility_mat_col.indices,
                                                       utility_mat_col.indptr,
                                                       y=y, l1_reg=l1_reg, l2_reg=l2_reg, max_iter=max_iter,
                                                       tolerance=tolerance,
                                                       rng=random_state, X_sparse_scaling=X_sparse_scaling, coefs=coefs,
                                                       random=False, positive=True)

            # No make the lil_matrix to csr_format matrix for faster computation in prediction.
                self.W = self.W.tocsr()
                #self.W.setdiag(0)
                
            else:
                print(f"Multiple Cores not implemented yet!")

    def predict(self, users, items):
        # Input checks
        if (type(users) == list or type(users) == np.ndarray) and (type(items) == list or type(items) == np.ndarray):
            if len(users) != len(items):
                if not (len(users) == 1 or len(items) == 0):
                    raise ValueError(
                        "users and items need to have the same length or only one user / item has to be provided.")

        return self.A[users, :].multiply(self.W[:, items].T).sum(axis=1).A1

if __name__ == '__main__':
    # Small test cases.
    small = True

    if small == False:
        # Load Data
        train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
        eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")

        # For test purposes very popular items and users that buy frequently.
        user_counts = train_mat.sum(axis=1).A1
        top_users = np.argsort(user_counts)[-int(0.1 * user_counts.shape[0]):][::-1]

        train_mat = train_mat[top_users, :]
        eval_mat = eval_mat[top_users, :]

        item_counts = train_mat.sum(axis=0).A1
        top_items = np.argsort(item_counts)[-int(0.1 * item_counts.shape[0]):][::-1]

        train_mat = train_mat[:, top_items]
        eval_mat = eval_mat[:, top_items]


        # Create object for SLIM model
        test_slim_model = SLIM(num_items=train_mat.shape[1], num_users=train_mat.shape[0])

        # Train Slim model.
        test_slim_model.train(train_mat, random_state=1993, verbose=1, eval_mat=eval_mat, l1_reg=0.0000001, l2_reg=0.0001,
                              max_iter=25, tolerance=0.001, cores=1, stopping_rounds=3, min_improvement=0.0001,
                              dtype='float64')

        # Evaluate SLIM model.
        print(f"MSE was {test_slim_model.eval_prec(eval_mat)}")
        print(f" Top@50: {test_slim_model.eval_topn(eval_mat, train_mat, topn=50, rand_sampled=500, metric='PRECISION', cores=8)}")
    else:
        # Create utlity matrix and weight matrix for training
        test_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_train.npz")
        test_eval_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_validate.npz")

        # Select 10% of the items and users as test.
        np.random.seed(seed=1993)
        n_users, n_items = test_utility_mat.shape
        rel_users = np.random.choice(n_users, size=int(n_users * 0.05), replace=False)
        rel_items = np.random.choice(n_items, size=int(n_items * 0.05), replace=False)
        test_utility_mat = test_utility_mat[rel_users, :][:, rel_items]

        # sort the indicess
        test_utility_mat = test_utility_mat.sorted_indices()
        # print(test_utility_mat.shape)
        # print(len(test_utility_mat.indptr))
        non_zero = test_utility_mat.nonzero()

        test_slim_model = SLIM(num_items=test_utility_mat.shape[1], num_users=test_utility_mat.shape[0])

        # Train Slim model.
        test_slim_model.train(test_utility_mat, random_state=1993, verbose=1, eval_mat=test_eval_utility_mat, l1_reg=0.0000001,
                              l2_reg=0.0001,
                              max_iter=25, tolerance=0.001, cores=1, stopping_rounds=3, min_improvement=0.0001,
                              dtype='float64')

        print('hi')
