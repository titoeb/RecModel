import numpy as np
import scipy.sparse
import time 
import sparse_mult

# What do we need to do?
# Given two sparse Matrix, A in csr_format and W in csc format.
# And two lists / integers users and items
# to get the sum of the elementwise multiplication of the rows users of A with the columns items of W
# To do so their are the following cases:
    # users is list, items is list -> Then A, B do need to have the same shape
    # users is list, items is int -> Then items need to replicated for each user
    # users is int, items is list -> Then users need to be replicated for each item
    # user is int, item is int -> no trouble.



# Let's start with simulating A, W of correct types.
n_users, n_items = 5000, 10000
A = scipy.sparse.random(n_users, n_items, density=0.01, format='csr', dtype=np.float64, random_state=1993)
W = scipy.sparse.random(n_items, n_items, density=0.01, format='csc', dtype=np.float64, random_state=1993)

# Then we will create users and items. Use the standard case, users int and items is np.array.
user = np.random.randint(0, n_users, size=10000)
items = np.random.randint(0, n_items, size=10000)


def predict(users, items, A, W):
    return A[users, :].multiply(W[:, items].T).sum(axis=1).A1


# Now simulate some runs of the multiplications
n_runs = 4000

true_result = predict(users=user, items=items, A=A, W=W)
sparse_result = sparse_mult.predict(user, items, A.indptr, A.indices, A.data, W.indptr, W.indices, W.data, 8)
print(f"results are {np.allclose(true_result, sparse_result)}")

print("Start evaluation!")
start_python = time.time()
for _ in range(n_runs):
    _ = predict(users=user, items=items, A=A, W=W)
end_python = time.time()
avr_time = (end_python - start_python) / n_runs

print(f"average time of python-based computation was {avr_time}")


print("Start evaluation!")
start_cython = time.time()
for _ in range(n_runs):
    _ = sparse_mult.predict(user, items, A.indptr, A.indices, A.data, W.indptr, W.indices, W.data, 8)
end_cython = time.time()
avr_time_cython = (end_cython - start_cython) / n_runs

print(f"average time of cython-based computation was {avr_time_cython}, speedup was {avr_time / avr_time_cython}")
