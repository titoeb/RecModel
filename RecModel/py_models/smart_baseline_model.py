import numpy as np
import scipy.sparse
import time

# Imports from own package. 
from RecModel.py_models.base_model import RecModel

class SmartBaseline(RecModel):

    def __init__(self, num_items):
        self.num_items = num_items

    def train(self, X):
        self.item_counts = X.sum(axis = 0).A1.astype(np.float32)
        
    def rank(self, items, users, topn):           
        relevant_item_scores = np.empty(len(items), dtype=np.float32)    
        
        relevant_item_scores = self.item_counts[items]

        return items[np.argsort(relevant_item_scores)[::-1]]

    def predict(self, users, items):
        print("The naive Baseline cannot predict, only topn can be used.")
