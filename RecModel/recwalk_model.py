# Imports
import numpy as np
import scipy.sparse
import ctypes
import os

# Imports from own package. 
from RecModel.base_model import RecModel
from RecModel.fast_utils.slim_utils import _predict_slim, train_Slim

# Helper functions
def fill_empty_row_or_col(mat, fill_value=1.0):
    mat=mat.copy()

    # First fill the empty rows
    empty_rows = (mat.sum(axis=1).A1 == 0)

    # Sample one random item for each of them
    random_items = np.random.randint(0, mat.shape[1], empty_rows.sum())
    mat[empty_rows, random_items] = fill_value

    # If there are some columns remaining zero also fill them!
    empty_cols = (mat.sum(axis=0).A1 == 0)
    if empty_cols.any():
        random_users=np.random.randint(0, mat.shape[0], empty_cols.sum())
        mat[random_users, empty_cols] = fill_value
    return mat

def create_A_g(mat):
    n_users, n_items = mat.shape

    # Built upper half.
    zeros_upper_left = scipy.sparse.csr_matrix((n_users, n_users), dtype=np.float32) 
    upper_half = scipy.sparse.hstack([zeros_upper_left, mat], format='csr')

    # Build lower half.
    zeros_lower_right = scipy.sparse.csr_matrix((n_items, n_items), dtype=np.float32)    
    lower_half = scipy.sparse.hstack([mat.T, zeros_lower_right], format='csr')

    A_g = scipy.sparse.vstack([upper_half, lower_half], format='csr')
    return A_g

def create_H(mat):
    A_g = create_A_g(mat)

    # np.dot(A, 1_n)
    row_sums = A_g.sum(axis=1).A1

    # fill the digonal matrix with the inverse of row sums.
    diag_inv_row_sum = scipy.sparse.diags(diagonals=(1 / row_sums), offsets=0, dtype=np.float32, format='csr')

    return diag_inv_row_sum.dot(A_g)

def create_M_i(W_indptr, W_indices, W_data, n_items):
    """
    M_i is create by making W row stochastic. 
    """
    W = scipy.sparse.csr_matrix((W_data, W_indices, W_indptr), shape=(n_items, n_items), dtype=np.float32)
    W_normalized = W.copy()

    # Compute maximal row sums
    row_sums = W.sum(axis=1).A1
    row_sum_max = row_sums.max()

    # Normalize W by the maximal row sums.
    W_normalized.data /= row_sum_max

    # Create diagonal mat that reintroduces the residuals to make the final Matrix row stochastic.
    diag_mat = scipy.sparse.diags(diagonals = 1 - (row_sums / row_sum_max), offsets=0, dtype=np.float32, format='csr')

    M_i = W_normalized + diag_mat

    return M_i

def create_M(indptr, indices, data, n_users, n_items):
    # Make W row stochastic.
    M_i = create_M_i(indptr, indices, data, n_items)

    # Fill the rest of the matrix with zeros and the upper left with an identity matrix.
    I = scipy.sparse.diags(diagonals=np.full(n_users, 1, dtype=np.float32), offsets=0, dtype=np.float32, format='csr')
    zeros_upper_right = scipy.sparse.csr_matrix((n_users, n_items), dtype=np.float32)    
    zeros_lower_left = scipy.sparse.csr_matrix((n_items, n_users), dtype=np.float32)    

    upper_half = scipy.sparse.hstack([I, zeros_upper_right], format='csr')
    lower_half = scipy.sparse.hstack([zeros_lower_left, M_i], format='csr')

    M = scipy.sparse.vstack([upper_half, lower_half], format='csr')
    return M

# Recwalk model
class Recwalk(RecModel):
    def __init__(self, num_items, num_users, k_steps, eval_method, damping=None, slim_W=None):
        
        if not eval_method in ['k_step', 'PR']:
            raise ValueError(f"eval method needs to be one of ['k_step', 'PR], not {eval_method}")

        self.num_users = num_users       
        self.num_items = num_items
        self.k = k_steps
        self.eval_method = eval_method
        self.P = None

        if eval_method == 'PR':
            if damping is None:
                raise ValueError("If you want to use the power method to rank the items (eval_method = 'PR') please provide the damping factor.")
            self.damping = damping

        if not slim_W is None:
            self.slim_W = slim_W
            self.slim_trained = True
        else:
            self.slim_trained = False

    def rank(self, items, users, topn):     
        # Create the representation of the user:
        user_vec = np.zeros(self.num_items + self.num_users, dtype=np.float32)
        user_vec[users] = 1.0  

        # Choose the prediction method.
        if self.eval_method == 'k_step':

            # prediction via k-step landing probabilities
            for _ in range(self.k):
                user_vec = scipy.sparse.csr_matrix.dot(self.P, user_vec)
            predictions = user_vec[items + self.num_users]
            
        else:
            # Prediction based on the stationary distribution with restarts.
            # make hard copy of user_vec as user_vec is used throughout the computation.
            vec_out = user_vec.copy()
            for _ in range(self.k):
                vec_out = self.damping * scipy.sparse.csr_matrix.dot(self.P, vec_out) + (1-self.damping) * user_vec
                vec_out = vec_out / (np.linalg.norm(vec_out) + 1e-10)
            predictions = vec_out[items + self.num_users]

        # Extract the relevant predictions from the computed vector (take the relevant item nodes)
        
        # Finally sort the prediction and return the relevant items.
        return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

    def predict(self, users, items):
        raise NotImplementedError("Predict is not defined for the RecWalk method.") 

    def train(self, train_mat, phi, alpha, l1_ratio, max_iter, tolerance, cores, verbose):
        """
        Hyper parameter:
        -> phi, weighting paramter to combine the two parts of the P matrix.
        -> n_closest, the number of closest neighbors to be consideren in SLIM.
        -> alpha is the regularization paramters for the slim model
        -> l1_ratio is the second part of the regularization for slim
        """
        # Ensure that the matrix is not changed and that it is a matrix of type csr.
        train_mat = train_mat.tocsr().astype(np.float64)
        train_mat.sort_indices()

        # Slim needs a csc matrix
        train_mat_csc = train_mat.tocsc()
        train_mat_csc.sort_indices()

        n_users, n_items = train_mat.shape

        # Train underlying slim model.
        if self.slim_trained is False:
            # Train slim 
            indptr, indices, data = train_Slim(X=train_mat_csc, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tolerance, cores=cores, verbose=verbose)
        else:
            # Use pre-trained slim weights
            indptr, indices, data = (self.slim_W.indptr, self.slim_W.indices, self.slim_W.data)
        
        # Fill the empty rows / columsn of the train mat with an entry of 1.0 at a random position to ensure stochasticity of the matrix.
        train_mat = fill_empty_row_or_col(train_mat)

        M = create_M(indptr, indices, data, n_users, n_items)
        H = create_H(train_mat)

        # P = alpha * H + (1 - alpha) * M
        H.data *= phi
        M.data *= (1 - phi)

        # We will later on predict np.dot(P.T, phi) instead of np.dot(phi.T, P)
        self.P = (H + M).T

    def load_slim(self, dir='W.npz'):
        try:
            self.slim_W = scipy.sparse.load(dir)
            self.slim_trained = True
        except FileNotFoundError as e:
            Warning(f"The file {dir} could not be loaded.")

    def save_P(self, dir='P.npy'):
        if self.P is not None:
            np.save(dir, self.P)
        else:
            raise ValueError("There is no computed P matrix because the model was not trained yet. Please train it with Recwalker.train().")

    def load_P(self, dir='P.npy'):
        try:
            self.P = np.load(dir)
        except FileNotFoundError as e:
            Warning(f"The file {dir} could not be loaded.")
