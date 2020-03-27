import numpy as np

def test_coverage(cls, Train, topN):
    """Testing the coverage of the algorithm:
        It is assumed cls is a object of classes derived from RecModel and is able to rank items with a rank function.
    """
    item_counts = np.zeros(Train.shape[0], dtype=np.int32)

    for user in range(Train.shape[0]):
        start_usr = Train.indptr[user]
        end_usr = Train.indptr[user+1]

        items_to_rank = np.delete(np.arange(Train.shape[1], dtype=np.int32), Train.indices[start_usr:end_usr])
        ranked_items = cls.rank(users=user, items=items_to_rank, topn=topN).reshape(-1)
        item_counts[ranked_items[:topN]] += 1
    
    return item_counts

def train_test_split_sparse_mat(matrix, train=0.8, seed=1993):
    np.random.seed(seed)
    train_data = matrix.tocoo()
    test_data = matrix.tocoo()

    is_train_data = (np.random.rand(len(matrix.data)) < train)

    train_data.data[~is_train_data] = 0.0
    test_data.data[is_train_data] = 0.0

    train_data = train_data.tocsr()
    test_data = test_data.tocsr()

    train_data.eliminate_zeros()
    test_data.eliminate_zeros()
    
    return [train_data, test_data]