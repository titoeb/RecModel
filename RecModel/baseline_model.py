import numpy as np
import scipy.sparse
import time

# Imports from own package. 
from RecModel.base_model import RecModel

class Baseline(RecModel):
    
    def __init__(self, num_items):
        self.num_items = num_items

    def train(self, X):
        self.top_items = np.argsort(X.sum(axis = 0).A1.astype(np.float32))[::-1]
        
    def rank(self,  items, users, topn):           
        return np.array(self.top_items[:topn])

    def predict(self, users, items):
        print("The Baseline cannot predict, only topn can be used.")