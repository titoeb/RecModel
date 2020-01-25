import scipy.sparse
import numpy as np
import time
import RecModel
import SLIM as slim

# old params:
# alpha = 1e-9
# l1_ratio = 0.9

# Params
alpha = 12
#alpha= 0.1d
l1_ratio = 0.12
tol = 0.00001
max_iter = 20

l1_reg = alpha * l1_ratio
l2_reg = (1 - l1_ratio) * alpha

test_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_train.npz")
test_eval_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_validate.npz")


# Sample a small number of products.
np.random.seed(1993)

n_users, n_items = test_utility_mat.shape
rand_items = np.random.choice(np.arange(n_items), size = int(0.1 * n_items), replace = False)
rand_users = np.random.choice(np.arange(n_users), size = int(0.1 * n_users), replace = False)
test_utility_mat = test_utility_mat[rand_users, :][:, rand_items].copy()
test_eval_utility_mat = test_eval_utility_mat[rand_users, :][:, rand_items].copy()

n_users, n_items = test_utility_mat.shape

test_utility_mat.sort_indices()
test_utility_mat = test_utility_mat.astype(np.float64)

test_eval_utility_mat.sort_indices()
test_eval_utility_mat = test_eval_utility_mat.astype(np.float64)

# Create the two class objects
SLIM = RecModel.Slim(num_items=n_items, num_users=n_users)
test_slim_model = slim.SLIM(num_items=n_items, num_users=n_users)


# Train the model
start  = time.time()
SLIM.train(X=test_utility_mat, alpha=l1_reg, l1_ratio=l2_reg, max_iter=max_iter, tolerance=tol, cores=1, verbose=1)
print(f"Execution took {(time.time() - start) / 60} minutes")

test_slim_model.W = SLIM.W
test_slim_model.A = test_utility_mat.tocsc()
"""
indptr = np.array(SLIM.W_indptr).copy()
idx = np.array(SLIM.W_idx).copy()
data = np.array(SLIM.W_data).copy()

W = scipy.sparse.csr_matrix((data, idx, indptr), dtype=np.float64, shape = (n_items, n_items))
test_slim_model.W = W
test_slim_model.A = test_utility_mat

start = time.time()
recall = test_slim_model.eval_topn(test_eval_utility_mat.tocsr(), rand_sampled=1000, topn=np.array([4]), random_state=1993)
print(f"Recall was {recall[0]['recall']} and execution took {time.time() - start} seconds")

print(f"Slim found {np.array(SLIM.W_data)[:100]} non-zero coefficients")

"""
topn = np.array([4, 10, 20, 50], dtype=np.int32)
# Evaluate the model
print("Evaluation with extension type:")
start = time.time()
recall = SLIM.eval_topn(test_eval_utility_mat, rand_sampled=1000, topn=topn, random_state=1993, cores=1)
print(f"Recall was {recall} and execution took {(time.time() - start) / 60} minutes")

print(SLIM.eval_prec(test_eval_utility_mat, 'MSE', 1))

"""

# Extract weight matrix from SLIM object
indptr = SLIM.W_indptr
idx = SLIM.W_idx
data = SLIM.W_data

W = scipy.sparse.csc_matrix((data, idx, indptr), dtype=np.float64, shape = (n_items, n_items))
print(f"W.shape: {W.shape}, A.shape: {test_utility_mat.shape} (n_users, n_items): ({n_users}, {n_items})")



W.setdiag(0)
W.eliminate_zeros()
test_slim_model.W = W
test_slim_model.A = test_utility_mat

start  = time.time()
test_slim_model.train(test_utility_mat, random_state=1993, verbose=1, eval_mat=test_eval_utility_mat, l1_reg=l1_reg,
                             l2_reg=l2_reg, max_iter=max_iter, tolerance=tol, cores=1, stopping_rounds=3, min_improvement=0.0001, dtype='float64')
print(f"Execution took {(time.time() - start) / 60} minutes")
"""
"""
start  = time.time()
test_slim_model.train(test_utility_mat, random_state=1993, verbose=1, eval_mat=test_eval_utility_mat, l1_reg=l1_reg,
                             l2_reg=l2_reg, max_iter=max_iter, tolerance=tol, cores=1, stopping_rounds=3, min_improvement=0.0001, dtype='float64')
print(f"Execution took {(time.time() - start) / 60} minutes")

start = time.time()
recall = test_slim_model.eval_topn(test_eval_utility_mat, rand_sampled=1000, topn=np.array([4]), random_state=1993)
print(f"Recall was {recall[0]['recall']} and execution took {time.time() - start} seconds")
"""


