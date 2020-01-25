import RecModel
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import time

# Create test marix
def create_mats(dim, scale, sparsity, binary=False, seed=1993):
    np.random.seed(seed)

     # Make it artifically sparse
    mask = np.random.binomial(n= 1, p = sparsity, size=dim*dim).astype(np.float32).reshape(dim, dim) 
    if binary is True:
        test_mat = mask
    else:
        # Create dense test_matrix:
        test_mat = np.random.random(dim * dim).astype(np.float32).reshape(dim, dim) * scale
        test_mat = test_mat * mask

    # Convert it to a
    test_mat_sparse = scipy.sparse.csc_matrix(test_mat)
    test_mat_sparse.eliminate_zeros()
    test_mat_sparse.sort_indices()

    return test_mat, test_mat_sparse

# Cosine test
def test_cosine(dim=100, scale=10, sparsity=0.01, verbose=0):
    test_mat, test_mat_sparse = create_mats(dim, scale, sparsity)
    
    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_items=dim, num_users=dim)
    start = time.time()
    test_neighbor.train(test_mat_sparse, 1, cores=1)
    if verbose > 0:
        print(f"Execution took {time.time() - start} seconds.")
    sim_neighbor = np.array(test_neighbor.sim)

    start = time.time()
    true_res = cosine_similarity(test_mat.T)
    np.fill_diagonal(true_res, 0.0)

    if verbose > 0:
        print(f"Execution took {time.time() - start} seconds.")
    if np.allclose(true_res, sim_neighbor):
        if verbose > 0:
            print("All test succesful")
        return True
    else:
        if verbose > 0:
            print('Results are not the same!')
            print(f"Expected result is \n{true_res}\nacutal result was \n{sim_neighbor}")
        return False

# Correlation test and helper functions
def correlation_dense_intern(vec1, vec2):
    # This is a small, not scalable helper to test the real correlation
    a = vec1.copy()
    b = vec2.copy()
    a_b_nonzero = np.logical_and(a != 0, b != 0)

    return np.sum((a[a_b_nonzero] - a.mean()) * (b[a_b_nonzero] - b.mean())) / ((np.sqrt(np.sum((a[a_b_nonzero] - a.mean()) ** 2))) * (np.sqrt(np.sum((b[a_b_nonzero] - b.mean()) ** 2)))  + 1e-10)

def correlation_dense_mat(mat):
    output = np.zeros((mat.shape[1], mat.shape[1]), (np.float32))

    for row_a in range(mat.shape[1]):
        for row_b in range(row_a + 1, mat.shape[1]):
            res = correlation_dense_intern(mat[:,row_a], mat[:, row_b])
            output[row_a, row_b] = res
            output[row_b, row_a] = res
    return output

def test_correlation_performance(verbose=0, cores=1):
    test_mat_sparse = scipy.sparse.load_npz("data/mat_bin_train.npz")
    
    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_users=test_mat_sparse.shape[0], num_items=test_mat_sparse.shape[1])
    start = time.time()
    test_neighbor.train(test_mat_sparse, 2, cores=cores)
    if verbose > 0:
        print(f"Execution took {time.time() - start} seconds")

def test_correlation(dim=100, scale=10, sparsity=0.01, verbose=0, cores=1):
    test_mat, test_mat_sparse = create_mats(dim, scale, sparsity)
    expected_result = correlation_dense_mat(test_mat)
    
    test_neighbor = RecModel.Neighborhood(axis=1,nb_size=10,  num_users=test_mat_sparse.shape[0], num_items=test_mat_sparse.shape[1])
    start = time.time()
    test_neighbor.train(test_mat_sparse, 2, cores=cores)
    if verbose > 0:
        print(f"Execution took {time.time() - start} seconds")
    given_result = np.array(test_neighbor.sim)

    true_result = np.allclose(expected_result, given_result, atol=1e-07)
    if true_result is True:
        if verbose > 0:
            print("Test succesful!")
        return True
    else:
        if verbose > 0:
            print("Results were wrong")
            print(f"True results: \n{expected_result}\n, own_result: \n{given_result}")  
            print(f"summed difference = {np.abs(expected_result - given_result).sum()}")
        return False
    return True
    
# Adjusted_cosine test and helpers

def test_adjusted_correlation_performance(verbose=0, cores=1):
    test_mat_sparse = scipy.sparse.load_npz("data/mat_bin_train.npz")
    
    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_users=test_mat_sparse.shape[0], num_items=test_mat_sparse.shape[1])
    start = time.time()
    test_neighbor.train(test_mat_sparse, 3, cores=cores)
    if verbose > 0:
        print(f"Execution took {time.time() - start} seconds")

def adj_cosine_dense_intern(vec1, vec2, user_mean):
    # This is a small, not scalable helper to test the real correlation
    a = vec1.copy()
    b = vec2.copy()
    a_b_nonzero = np.logical_and(a != 0, b != 0)

    return np.sum((a[a_b_nonzero] - user_mean[a_b_nonzero]) * (b[a_b_nonzero] - user_mean[a_b_nonzero])) / ((np.sqrt(np.sum((a[a_b_nonzero] - user_mean[a_b_nonzero]) ** 2))) * (np.sqrt(np.sum((b[a_b_nonzero] - user_mean[a_b_nonzero]) ** 2)))  + 1e-10)

def adj_cosine_dense_mat(mat):
    output = np.zeros((mat.shape[1], mat.shape[1]), (np.float32))
    user_mean = mat.sum(axis=1)

    for col_a in range(mat.shape[1]):
        for col_b in range(col_a + 1, mat.shape[1]):
            res = adj_cosine_dense_intern(mat[:,col_a], mat[:, col_b], user_mean)
            output[col_a, col_b] = res
            output[col_b, col_a] = res
    return output

def test_adjusted_cosine(dim, scale=10, sparsity=0.01, verbose=0):
    test_mat, test_mat_sparse = create_mats(dim, scale, sparsity)
   
    expected_result = adj_cosine_dense_mat(test_mat)
    
    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_items=test_mat.shape[0], num_users=test_mat.shape[1])
    test_neighbor.train(test_mat_sparse, sim=3, cores=1)
    given_result = np.array(test_neighbor.sim)

    if np.allclose(expected_result, given_result, atol=1e-07) is True:
        if verbose > 0:
            print("Test succesful!")
        return True
    else:
        if verbose > 0:
            print("Results were wrong")
            print(f"True results: \n{expected_result}\n, own_result: \n{given_result}")  
            print(f"summed difference = {np.abs(expected_result - given_result).sum()}")
        return False
    
# MSD test
def msd_test(dim, scale = 10, sparsity=0.01, verbose=0):
    _, sparse_mat = create_mats(dim, scale, sparsity)

    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_items=dim, num_users=dim)
    test_neighbor.train(sparse_mat, 5, cores=8)   
    test = np.array(test_neighbor.sim)

    if verbose > 0:
        print(test)

# Jaccard test
def jaccard_dense_intern(vec1, vec2):
    # This is a small, not scalable helper to test the real correlation
    a = vec1.copy()
    b = vec2.copy()
    return  np.logical_and(a, b).sum()/(np.logical_or(a, b).sum()+ 1e-10)

def jaccard_dense_mat(mat):
    mat = mat.copy()
    output = np.zeros((mat.shape[1], mat.shape[1]), (np.float32))

    if mat.dtype != 'bool':
        mat = mat.astype('bool')

    for row_a in range(mat.shape[1]):
        for row_b in range(row_a + 1, mat.shape[1]):
            res = jaccard_dense_intern(mat[:,row_a], mat[:, row_b])
            output[row_a, row_b] = res
            output[row_b, row_a] = res
    return output
        
def test_jaccard(dim, scale=10, sparsity=0.01, verbose=0):
    test_mat, test_mat_sparse = create_mats(dim, scale, sparsity)
   
    expected_result = jaccard_dense_mat(test_mat)
    
    test_neighbor = RecModel.Neighborhood(axis=1, nb_size=10, num_items=test_mat_sparse.shape[0], num_users=test_mat_sparse.shape[1])
    test_neighbor.train(test_mat_sparse, sim=4, cores=1)
    given_result = np.array(test_neighbor.sim)

    if np.allclose(expected_result, given_result, atol=1e-07) is True:
        if verbose > 0:
            print("Test succesful!")
        return True
    else:
        if verbose > 0:
            print("Results were wrong")
            print(f"True results: \n{expected_result}\n, own_result: \n{given_result}")  
            print(f"summed difference = {np.abs(expected_result - given_result).sum()}")
        return False

def test_model_prediction():
    sim = np.array([
        [0.0, 2.1, 0.0, 0.9],
        [2.1, 0.0, 0.3, 1.0],
        [0.0, 0.3, 0.0, 1.7],
        [0.9, 1.0, 1.7, 0.0]
    ], dtype=np.float32)
    data = np.array([
        [0, 1, 0, 2],
        [1, 0, 0, 1],
        [2, 0, 0, 0],
        [3, 3, 0, 1],
        [4, 2, 2, 0],
        [5, 100, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float32)
    data_sparse = scipy.sparse.csr_matrix(data)

    test_neighbor = RecModel.Neighborhood(axis=1, num_items=data_sparse.shape[0], nb_size = 2,  num_users=data_sparse.shape[1])
    test_neighbor.X_data = data_sparse.data
    test_neighbor.X_idx = data_sparse.indices
    test_neighbor.X_indptr = data_sparse.indptr
    test_neighbor.sim=sim

    # Test with a user that did not consume anything.
    actual_result= test_neighbor.predict(np.array([6], dtype=np.int32), np.array([1], dtype=np.int32)) 
    expected_result = np.array([0.0], dtype=np.float32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")

    pred = test_neighbor.predict(np.array([1], dtype=np.int32), np.array([3], dtype=np.int32)) 
    expected_result = np.array([1.0], dtype=np.float32)
    if pred != 1.0:
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")

    actual_result= test_neighbor.predict(np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)) 
    expected_result = np.array([1.85], dtype=np.float32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")

    
    test_neighbor.set_nb_size(1)
    actual_result = test_neighbor.predict(np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)) 
    expected_result = np.array([2.0], dtype=np.float32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")
    test_neighbor.set_nb_size(2)
    
    actual_result = test_neighbor.predict(np.array([0,  1], dtype=np.int32), np.array([2, 3], dtype=np.int32)) 
    expected_result = np.array([1.85, 1.0], dtype=np.float32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")

    
    test_neighbor.set_nb_size(3)
    actual_result= test_neighbor.predict(np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)) 
    expected_result = np.array([1.85], dtype=np.float32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be {expected_result} but is {actual_result}")
    test_neighbor.set_nb_size(2)

    return True

def test_ranking():
    # Create test data
    sim = np.array([
        [0.0, 2.1, 0.0, 0.9],
        [2.1, 0.0, 0.3, 1.0],
        [0.0, 0.3, 0.0, 1.7],
        [0.9, 1.0, 1.7, 0.0]
    ], dtype=np.float32)
    data = np.array([
        [0, 1, 0, 2],
        [1, 0, 0, 5],
        [2, 0, 0, 0],
        [3, 3, 0, 1],
        [4, 2, 2, 0],
        [5, 100, 0, 0]
    ], dtype=np.float32)
    data_sparse = scipy.sparse.csr_matrix(data)

    test_neighbor = RecModel.Neighborhood(axis=1,num_items=data_sparse.shape[1], num_users=data_sparse.shape[0], nb_size=2)
    test_neighbor.X_data = data_sparse.data
    test_neighbor.X_idx = data_sparse.indices
    test_neighbor.X_indptr = data_sparse.indptr
    test_neighbor.sim=sim

    actual_result= test_neighbor.rank(np.array([1, 2], dtype=np.int32), 1, 2, 1)
    expected_result = np.array([2, 1], dtype=np.int32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be 1.15 but is {actual_result}")

    actual_result= test_neighbor.rank(np.array([1, 2], dtype=np.int32), 1, 2, 2)
    expected_result = np.array([2, 1], dtype=np.int32)
    if not np.allclose(actual_result, expected_result):
        raise ValueError(f"pred should be 1.15 but is {actual_result}")

    return True

def test_ranking_pract():
    train_mat = scipy.sparse.load_npz("data/mat_count_train.npz")
    test_mat = scipy.sparse.load_npz("data/mat_bin_test.npz")

    test_neighbor = RecModel.Neighborhood(axis=1,num_items=train_mat.shape[1], num_users=train_mat.shape[0], nb_size=50)
    start = time.time()
    test_neighbor.train(train_mat.copy(), 'adjusted_cosine', cores=8)
    print(f"Training Done! It took {time.time() - start} seconds")
    
    start = time.time()
    perf=test_neighbor.eval_topn(test_mat=test_mat.copy(), rand_sampled=1000, topn=np.array([4, 10, 20, 50], dtype=np.int32), random_state=1993, cores=7)
    print(f"perf: {perf}\n. The execution took {time.time() - start}")

if __name__ == "__main__":
    """
    if test_cosine(dim=5, sparsity=0.01) is True:
        print("Cosine test was succesfull.")
    else:
        print("Cosine test failed!")

    if  test_correlation(dim=100, sparsity=0.001, verbose=0) is True:
        print("Correlation test was succesfull")
    else:
        print("Correlation test failed")

    if test_adjusted_cosine(dim=5, sparsity=0.01) is True:
        print("Adjusted cosine similarity test was succesfull")
    else:
        print("Adjusted cosine similarity test failed")

    if test_jaccard(dim=100, sparsity=0.01, verbose=0) is True:
        print("Jaccard similarity test was succesfull")
    else:
        print("Jaccard similarity test faW.astype(np.float32)iled")

    #test_correlation_performance(1, 8)
    #test_adjusted_correlation_performance(1, 8)
    
    if test_model_prediction() is True:
        print("Prediction test was succesfull")
    else:
        print("Prediction test failed")
    
    
    if test_ranking() is True:
        print("Ranking test was succesfull")
    else:
        print("Ranking test failed")
    """
    test_ranking_pract()

    

    