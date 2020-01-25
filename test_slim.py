import scipy.sparse
import numpy as np
import time
import RecModel
import os

# old params:
# alpha = 1e-9
# l1_ratio = 0.9

# Params
l1_reg =  3.6294357280497724
l2_reg = 27.46258427998741
tol = 0.016047184959054224
max_iter = 13

first_n_users = 40000
first_n_items = 7000

test_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_train.npz")
test_eval_utility_mat = scipy.sparse.load_npz("../Reco_Models/data/mat_bin_validate.npz")

test_utility_mat.sort_indices()
test_utility_mat = test_utility_mat.astype(np.float64)

test_eval_utility_mat.sort_indices()
test_eval_utility_mat = test_eval_utility_mat.astype(np.float64)

n_users, n_items = test_utility_mat.shape

# Create the two class objects
slim = RecModel.Slim(num_items=n_items, num_users=n_users)

# Train the model
start  = time.time()
slim.train(X=test_utility_mat, alpha=l1_reg, l1_ratio=l2_reg, max_iter=max_iter, tolerance=tol, cores=4, verbose=1)
print(f"Execution took {(time.time() - start) / 60} minutes")

# Evaluate the model
start = time.time()
recall = slim.eval_topn(test_eval_utility_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=8)
print(f"Recall was {recall} and execution took {time.time() - start} seconds")

# Save W
W = slim.W
scipy.sparse.save_npz('slim_W.npz', W)
