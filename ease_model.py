from utils import utils
import numpy as np
import scipy.sparse
import time

class Ease(utils.RecModel):

    def __init__(self, num_items, num_users):
        self.num_items = num_items
        self.num_users = num_users

    def rank(self, items, users, topn=None):
        if topn is None:
            topn = len(items)

        if isinstance(users, list):
            raise ValueError(f"users need to be an integer!")
        else:
            predictions = self.predict(users=users, items=items)

        return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

    def train(self, X, alpha, verbose, cores):

        # Copy X and make it csr as is optimal for prediction.
        X_csr = X.copy().tocsr()

        # Store X internally for later prediction.
        self.X = X_csr.copy()

        G = np.dot(X_csr.T, X_csr).todense()

        np.fill_diagonal(G, G.diagonal() + alpha)

        res = np.linalg.inv(G)
        res = res / (-res.diagonal())

        np.fill_diagonal(res, 0)

        self.W = res
      
    def predict(self, users, items):
        rel_users = self.X[users, :]
        rel_weights = self.W[:, items]
        return np.dot(rel_users.todense().A1, rel_weights)
