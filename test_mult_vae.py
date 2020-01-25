import scipy.sparse
import torch.optim
from RecModel import Mult_VAE
from sklearn.preprocessing import normalize
import numpy as np

train_mat = scipy.sparse.load_npz('data/mat_count_train.npz')
eval_mat = scipy.sparse.load_npz('data/mat_bin_validate.npz')

# Create a train test VAE.
# Hyper paramter that are static
batch_size = 3500
max_epochs = 30
verbose = 1
rand_sampled = 1000
model_path='models/Mult_VAE'
max_epochs_without_improvement = 6
n_random_samples = 10

# Hyper paramter for VAE that are optimized
# # General Ones
k = 49
final_beta = 0.6723108968
beta_step = 0.02
weight_decay = 0.00031008910
learning_rate = 0.0072375777624

# NN Encoder for mu
dense_layers_encoder_mu = [704]
batch_norm_encoder_mu = False
dropout_rate_encoder_mu = 	0.007031033502185702
dropout_rate_sparse_encoder_mu = 0.4657489854224814

# NN Encoder for sigma
dense_layers_encoder_sigma = [102]
batch_norm_encoder_sigma = False
dropout_rate_encoder_sigma = 0.38913850933983707
dropout_rate_sparse_encoder_sigma = 0.10546784156218052

# NN Encoder for Decoder
dense_layers_decoder=[597]
batch_norm_decoder = True
dropout_rate_decoder = 	0.47923017547464863

# Create Mult_VAE
test = Mult_VAE(k = k, num_items = train_mat.shape[1], dense_layers_encoder_mu=dense_layers_encoder_mu, dense_layers_encoder_sigma=dense_layers_encoder_sigma, dense_layers_decoder=dense_layers_decoder,
    batch_norm_encoder_mu=batch_norm_encoder_mu, batch_norm_encoder_sigma=batch_norm_encoder_sigma,  batch_norm_decoder=batch_norm_decoder, dropout_rate_decoder=dropout_rate_decoder,
    dropout_rate_encoder_mu=dropout_rate_encoder_mu, dropout_rate_encoder_sigma=dropout_rate_encoder_sigma,  dropout_rate_sparse_encoder_mu=dropout_rate_sparse_encoder_mu,
    dropout_rate_sparse_encoder_sigma=dropout_rate_sparse_encoder_sigma, final_beta=final_beta, beta_step=beta_step, model_path=model_path)

test.set_optimizer(torch.optim.Adam(test.parameters(), lr=learning_rate, weight_decay=weight_decay))
#test.set_writer('logs/')

test.train(X_train=train_mat.copy(), X_validate=eval_mat.copy(), batch_size=batch_size, epochs=max_epochs, verbose=verbose, max_epochs_without_improvement=max_epochs_without_improvement, n_random_samples=n_random_samples)

top_n_on_train = test.eval_topn(test_mat=train_mat.copy(), batch_size=batch_size, topn=np.array([4, 10, 20, 50]), rand_sampled =1000, random_state=None)
print(f"topn on train: {top_n_on_train}")

top_n_on_test = test.eval_topn(test_mat=eval_mat.copy(), batch_size=batch_size, topn=np.array([4, 10, 20, 50]), rand_sampled =1000, random_state=None)
print(f"topn on test: {top_n_on_test}")
