import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")

# Fill in from the mlflow run

# Static variables
rand_sampled = 1000
cores = 1

test_baseline = RecModel.Baseline(eval_mat.shape[0])

test_baseline.train(train_mat.astype(np.float32))

perf_all = test_baseline.eval_topn(test_mat=eval_mat, rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993)

print(f"The recalls are {perf_all}")


