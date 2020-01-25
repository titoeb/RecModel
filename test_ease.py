import RecModel
import numpy as np
import scipy.sparse
import multiprocessing
import time

train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")[:1000, :5000]
eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")[:1000, :5000]
count_mat = scipy.sparse.load_npz("data/mat_count_train.npz")[:1000, :5000]

# Fill in from the mlflow run
iter = 30
verbose = 0
cores = multiprocessing.cpu_count()
dim = 50
gamma = 100
stopping_rounds = 2
stopping_percentage = 0.01
seed = 1993
rand_sampled=1000


ease = RecModel.Ease(num_items=train_mat.shape[1], num_users=train_mat.shape[0])

ease.load_mat(X=train_mat)
if ease.W is None:
    start = time.time()
    ease.train(train_mat.copy(), alpha=10, verbose=1, cores=5)
    print(f"fitted ease in  {time.time() - start} seconds")

#ease.save_mat()


start = time.time()
for cores in range(6, 0, -1):
    print(f"starting to evaluate ease with {cores} cores")
    start = time.time()
    print(ease.eval_topn(test_mat=eval_mat.copy(), topn=np.array([4, 10, 20, 50], dtype=np.int32), rand_sampled =1000, cores=cores))
    print(f"Execution took {time.time() - start} seconds")

'''
train_mat = scipy.sparse.load_npz("data/mat_count_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_count_validate.npz")

ease = RecModel.Ease(num_items=train_mat.shape[1], num_users=train_mat.shape[0])
start = time.time()

ease.train(train_mat.copy(), alpha=1000, verbose=1, cores=8)
print(f"Training Done! It took {time.time() - start} seconds")

start = time.time()
perf=ease.eval_topn(test_mat=eval_mat.copy(), topn=np.array([4, 10, 20, 50]).astype(np.int32), rand_sampled =1000, cores=1, random_state=1993)
print(f"perf: {perf} and the execution took {time.time() - start}")'''
