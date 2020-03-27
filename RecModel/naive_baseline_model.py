import numpy as np
import scipy.sparse
import time

# Imports from own package. 
from RecModel.base_model import RecModel

class NaiveBaseline(RecModel):
    
    def __init__(self, num_items):
        self.num_items = num_items

    def rank(self, items, users, topn):           

        if topn > self.num_items:
            print(f"Sampling {topn} elements from a vector of length {num_items} without replacement does not make sense!")
        else:
            return np.random.choice(items, size=topn, replace=False)

    def train(self):
        print("This naive Baseline does not need to be trained.")

    def predict(self, users, items, cores):
        print("The naive Baseline cannot predict, only topn can be used.")