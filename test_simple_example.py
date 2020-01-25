import scipy.sparse
import numpy as np
import time
import reco_models
import SLIM as slim

# old params:
# alpha = 1e-9
# l1_ratio = 0.9

# Params
alpha = 2e-3
#alpha= 0.1
l1_ratio = 0.9
tol = 0.0001
max_iter = 25

l1_reg = alpha * l1_ratio
l2_reg = (1 - l1_ratio) * alpha

n_users, n_items = 4, 4

np.random.seed(1993)
test_mat = np.random.randint(low=0, high= 2, size = n_users * n_items).astype(np.float64).reshape(n_users, n_items)
test_mat = scipy.sparse.csc_matrix(test_mat, dtype=np.float64)

print(test_mat.todense())

SLIM = reco_models.SLIM(A=test_mat, num_items=n_items, num_users=n_users)
SLIM.train(X=test_mat, alpha=l1_reg, l1_ratio=l2_reg, max_iter=max_iter, tolerance=tol, cores=1, verbose=1)

# Extract weight matrix from SLIM object
indptr = SLIM.W_indptr
idx = SLIM.W_idx
data = SLIM.W_data

W = scipy.sparse.csc_matrix((data, idx, indptr), dtype=np.float32, shape = (n_items, n_items))

print(np.round(np.dot(W.todense(), test_mat.todense()), 2))


"""
# Sample a small number of products.



test_utility_mat.sort_indices()
test_utility_mat = test_utility_mat.astype(np.float64)

test_eval_utility_mat.sort_indices()
test_eval_utility_mat = test_eval_utility_mat.astype(np.float64)


# Create the two class objects
SLIM = reco_models.SLIM(A=test_utility_mat, num_items=n_items, num_users=n_users)
test_slim_model = slim.SLIM(num_items=n_items, num_users=n_users)


# Train the model
start  = time.time()
SLIM.train(X=test_utility_mat, alpha=l1_reg, l1_ratio=l2_reg, max_iter=max_iter, tolerance=tol, cores=1, verbose=1)
print(f"Execution took {(time.time() - start) / 60} minutes")

# Evaluate the model
start = time.time()
recall = SLIM.eval_topn(test_eval_utility_mat, rand_sampled=1000, topn=4, random_state=1993, cores=1)
print(f"Recall was {recall} and execution took {time.time() - start} seconds")


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


