import cd_fast_own
import numpy as np
import scipy.sparse
import time

# Params
alpha = 1e-2
#alpha= 0.1
l1_ratio = 0.5
tol = 0.1
max_iter = 1

# Make one test case with a matrix that is truly dense but defined as sparse.
X = scipy.sparse.load_npz("../Reco_Models/data/mat_count_train.npz").astype(np.float64)
X = X.tocsc()
start  = time.time()
res = cd_fast_own.train_Slim(X=X, alpha=alpha, beta=l1_ratio, max_iter=max_iter, tol=tol, cores=1, verbose=1)
print(f"Execution took {(time.time() - start) / 60} minutes")
print(res.data)

