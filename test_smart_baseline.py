import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")
train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")

# Static variables
rand_sampled = 1000
cores = 1

test_smart_baseline = RecModel.SmartBaseline(eval_mat.shape[1])

test_smart_baseline.train(train_mat)

perf_all = test_smart_baseline.eval_topn(test_mat=eval_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993)
print(f"The recalls are {perf_all}")

