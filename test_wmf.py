import RecModel
import numpy as np
import scipy.sparse
import multiprocessing

train_mat = scipy.sparse.load_npz("data/mat_bin_train.npz")
eval_mat = scipy.sparse.load_npz("data/mat_bin_validate.npz")
count_mat = scipy.sparse.load_npz("data/mat_count_train.npz")

# Fill in from the mlflow run
iterations = 30
verbose = 1
cores = multiprocessing.cpu_count()
dim = 200
gamma = 200
stopping_rounds = 2
stopping_percentage = 0.01
seed = 1993

train_mat_save = train_mat.copy()
eval_mat_save = eval_mat.copy()
count_mat_save = count_mat.copy()


# Static variables
rand_sampled = 1000

test_MF = RecModel.WMF(num_items=train_mat.shape[1], num_users=train_mat.shape[0], dim=dim, gamma=gamma, weighted=True, bias=False)
iter_run = test_MF.train(train_mat.copy(), iterations=iterations, verbose=verbose, eval_mat=eval_mat.copy(), cores=8,
                     stopping_rounds=stopping_rounds, count_mat=count_mat.copy())


acc = test_MF.eval_prec(train_mat)
print(f"Accuracy on train is {acc}")

acc = test_MF.eval_prec(eval_mat)
print(f"Accuracy on eval is {acc}")


perf_all = test_MF.eval_topn(eval_mat, train_mat, topn=np.array([4, 10, 20, 50]), rand_sampled=rand_sampled, cores=cores)
print(f"The recalls are {perf_all}")