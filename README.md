# RecModel
Python and Cython implementation of state-of-the-art collaborative filtering models.

## Details
I implemented the RecModel package in my master thesis, during which I implemented and tested state-of-the art collaborative filtering models with a Python interface. The implemented models are:

* [Neighbor](https://dl.acm.org/doi/10.1145/371920.372071): Item-to-item neighborhood-based collaborative filtering models using the euclidian, minowski, cosine, jaccard, correlation, adjusted cosine and adjusted correlation as similarity functions.

* [SLIM](https://dl.acm.org/doi/10.1109/ICDM.2011.134): Sparse Linear Methods for Top-N Recommender Systems.

* [VAE](https://dl.acm.org/doi/abs/10.1145/3178876.3186150): Variational Autoencoders for Collaborative Filtering.

* [EASE](https://dl.acm.org/doi/abs/10.1145/3308558.3313710): Embarrassingly Shallow Autoencoders for Sparse Data.

* [WMF](https://dl.acm.org/doi/10.1109/ICDM.2008.22): Weighted and non-weighted Matrix factorization, including optional user and item biases.

* [RecWalk](https://dl.acm.org/doi/abs/10.1145/3289600.3291016): Nearly Uncoupled Random Walks for Top-N Recommendation.

## Getting Started
The best way to get started with the package is to look at the [example.ipynb](example.ipynb) Notebook!

## Prerequisites
To run the models and compile the cython code the following packages need be installed:

* numpy
* pandas
* scipy
* torch
* sklearn
* tqdm
* cython
* ctypes
* sharedmem

Additionally, the Cython code needs to be compiled. To do so change to the Models/fast_utils directory:

```
cd RecModel/fast_utils
```

and compile the Cython code with:
```
python setup_models.py build_ext --inplace
```

## Authors

* **Tim Toebrock** - [titoeb](https://github.com/titoeb)




