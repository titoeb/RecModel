import RecModel
import numpy as np
import scipy.sparse
import multiprocessing
import time

train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")[:1000, :5000]
eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")[:1000, :5000]
count_mat = scipy.sparse.load_npz("data/mat_count_train.npz")[:1000, :5000]

# params to set
phi = 0.5
alpha = 0.1
l1_ratio=0.5
max_iter = 5 
tolerance=0.1
cores=1 
verbose=1
const = 1e-9

# Define and train model.
rec = RecModel.Recwalk(num_items=train_mat.shape[1], num_users=train_mat.shape[0], k_steps=10, eval_method='k_step')
rec.train(train_mat=train_mat, phi=phi, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tolerance=tolerance, cores=cores, verbose=verbose, const=const)
