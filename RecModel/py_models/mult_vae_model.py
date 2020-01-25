import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.sparse as sp
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import torchsummary
from datetime import datetime
from sklearn.metrics import confusion_matrix
import scipy
import torch.autograd
from torch.nn.utils import clip_grad_norm_
from RecModel.py_models.base_model import RecModel

def make_coo_to_sparse_tensor(coo_mat):
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def row_sum_dim_1(sparse_tensor, device):
    x = torch.sparse.FloatTensor(
            indices=torch.stack([
                sparse_tensor._indices()[0],
                torch.LongTensor(1).to(device).zero_().expand_as(sparse_tensor._indices()[0]),
            ]),
            values=sparse_tensor._values(),
            size=[sparse_tensor.shape[0], 1]).to(device)

    x.coalesce()
    return x.to_dense().squeeze()

def sparse_dropout(tensor, p):
    nnz = tensor._nnz()
    keep_mask = np.random.binomial(n=1, p=1-p, size=nnz).astype(bool)
    idx = tensor._indices()[:, keep_mask] 
    values = torch.ones_like(idx[0,:], dtype=torch.float)

    mask_tensor = torch.sparse.FloatTensor(idx, values, tensor.size())
    return tensor.mul(mask_tensor)

def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size()).to_dense()

def sparse_dense_mul_sparse_output(s, d):
      i = s._indices()
      v = s._values()
      dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
      return torch.sparse.FloatTensor(i, v * dv, s.size())

def DenseLayers(in_channels, out_channels, batch_norm=False, dropout_rate=None, activation=False):
    individual_layers = []
        
    if not dropout_rate is None:
        individual_layers.append(nn.Dropout(p=dropout_rate))
        
    individual_layers.append(nn.Linear(in_channels, out_channels))

    if activation is not None:
        individual_layers.append(nn.Tanh())
        
    if batch_norm is not None:
        individual_layers.append(nn.BatchNorm1d(num_features=out_channels))

    return nn.Sequential(*individual_layers)
    
class Encoder(nn.Module):
    def __init__(self, k, n_items, dense_layers, batch_norm, activation, dropout_rate, dropout_rate_sparse):
        super().__init__()
        self.k  = k
        self.additional_layer = len(dense_layers)
        self.dropout_rate_sparse = dropout_rate_sparse
        
        if self.additional_layer > 2:
            all_layers = [DenseLayers(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate,
                                                          batch_norm=batch_norm, activation=activation) for (in_channels, out_channels) in zip(dense_layers[0:-1], dense_layers[1:])]
            
            all_layers[0:0] = [DenseLayers(in_channels=n_items, out_channels=dense_layers[0], batch_norm=batch_norm, dropout_rate=None, activation=activation)]
            self.dense_network = nn.Sequential(*all_layers)
        elif self.additional_layer >= 1:
            self.dense_network = DenseLayers(in_channels=n_items, out_channels=dense_layers[0], batch_norm=batch_norm, dropout_rate=None, activation=activation)
        else:
            pass
        self.out_layer = DenseLayers(in_channels=dense_layers[-1], out_channels=k, batch_norm=None, dropout_rate=dropout_rate, activation=None)
        
    def forward(self, X):
        if self.training is True:
            # Apply sparse droput
            X = sparse_dropout(X, p=self.dropout_rate_sparse)

        if self.additional_layer >= 1:
            X = self.dense_network(X)
        return self.out_layer(X)
    
class Decoder(nn.Module):
    def __init__(self,k,  n_items, dense_layers, batch_norm, activation, dropout_rate):
        super().__init__()
        self.n_items = n_items
        self.additional_layer = len(dense_layers)
        if self.additional_layer > 2:
            all_layers = [DenseLayers(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate,
                                                          batch_norm=batch_norm, activation=activation) for (in_channels, out_channels) in zip(dense_layers[0:-1], dense_layers[1:])]
            
            all_layers[0:0] = [DenseLayers(in_channels=k, out_channels=dense_layers[0], batch_norm=None, dropout_rate=dropout_rate, activation=activation)]
            self.dense_network = nn.Sequential(*all_layers)

        elif self.additional_layer >= 1:
            self.dense_network = DenseLayers(in_channels=k, out_channels=dense_layers[0], batch_norm=batch_norm, dropout_rate=dropout_rate, activation=activation)
        else:
            pass
        self.out_layer = DenseLayers(in_channels=dense_layers[-1], out_channels=n_items, batch_norm=None, dropout_rate=dropout_rate, activation=None)
    def forward(self, X):
        if self.additional_layer >= 1:
            X = self.dense_network(X)
        return self.out_layer(X)

class Mult_VAE(nn.Module):
    def __init__(self, device, final_beta, beta_step, k, n_items, dense_layers_encoder_mu, dense_layers_encoder_sigma,  dense_layers_decoder, batch_norm_encoder_mu, batch_norm_encoder_sigma,  batch_norm_decoder,
                 dropout_rate_encoder_mu, dropout_rate_encoder_sigma, dropout_rate_sparse_encoder_mu, dropout_rate_sparse_encoder_sigma, dropout_rate_decoder):
        super().__init__()
        self.k = k
        self.Encoder_mu = Encoder(k=k, n_items=n_items,dense_layers=dense_layers_encoder_mu, batch_norm=batch_norm_encoder_mu, activation=True, dropout_rate=dropout_rate_encoder_mu, dropout_rate_sparse=dropout_rate_sparse_encoder_mu)
        self.Encoder_sigma = Encoder(k=k,n_items=n_items, dense_layers=dense_layers_encoder_sigma, batch_norm=batch_norm_encoder_sigma,
         activation=True, dropout_rate=dropout_rate_encoder_sigma, dropout_rate_sparse=dropout_rate_sparse_encoder_sigma)
        self.Decoder = Decoder(n_items=n_items,k=k, dense_layers=dense_layers_decoder, batch_norm=batch_norm_decoder, activation=True, dropout_rate=dropout_rate_decoder)
        self.device = device
        self.beta=final_beta
        self.beta_step = beta_step
        self.optimizer = None
        self.writer = None

    def set_optimizer(self, optimizer):
        self.optimizer=optimizer

    def set_writer(self, writer_path='logs/'):
        time_stamp = ''.join(str(datetime.now()).split('.')[:-1]).replace(' ', '_').replace(':', '_').replace('-', '_')
        self.writer = SummaryWriter(writer_path + time_stamp)

    def forward(self, X):
        mu = self.Encoder_mu(X)
        log_sigmas = self.Encoder_sigma(X)
        z = Variable(torch.randn(X.shape[0], self.k, requires_grad=False)).to(self.device)
        scaled_z = z.mul(torch.sqrt(torch.exp(log_sigmas))).add(mu)
        pi_not_normalized = self.Decoder(scaled_z)
        return mu, log_sigmas, pi_not_normalized

    def predict(self, users, items=None, topn=None):
        with torch.no_grad():
            self.Encoder_mu.train(False)
            self.Encoder_sigma.train(False)
            self.Decoder.train(False)

            rel_data = self.X_train[users, :]
            rel_data = make_coo_to_sparse_tensor(rel_data.tocoo()).to(self.device)

            z = self.Encoder_mu(rel_data)
            predictions = self.Decoder(z).cpu().numpy()

            if not items is None:
                predictions = predictions[:, items]

            if topn is None:
                return predictions
            else:
                raise ValueError("Ordering the predictions is not implemented yet")
    
    def loss(self, X, mu, log_sigmas, pi_not_normalized):
        """if self.writer is not None:
            self.writer.add_histogram('mu', mu.cpu().detach().numpy(), global_step=self.epoch)
            self.writer.add_histogram('sigmas', torch.exp(log_sigmas.cpu().detach()).numpy(), global_step=self.epoch)
            self.writer.add_histogram('pi', F.softmax(pi_not_normalized.cpu().detach(), dim =1).numpy(), global_step=self.epoch)"""
        
        #log_likelihood = sparse_dense_mul(X, log_pi).sum(dim=1)
        neg_log_likelihood = - torch.sparse.sum(sparse_dense_mul_sparse_output(X, F.log_softmax(pi_not_normalized, dim=1)), dim=1).to_dense().to(self.device).mean()
        
        KL = torch.mean(torch.sum(0.5 * (-log_sigmas + torch.exp(log_sigmas) + mu ** 2 - 1), dim=1))

        if self.writer is not None:
            self.writer.add_scalar('KL_Divergence', KL.cpu().detach().numpy(), global_step=self.epoch)
            self.writer.add_scalar('Negative_log_likelihood', neg_log_likelihood.cpu().detach().numpy(), global_step=self.epoch)
    
        return neg_log_likelihood + self.beta * KL
    
    def forward_pass(self, X):
        self.zero_grad()
        mu, log_sigmas, pi_not_normalized = self.__call__(X)
        loss = self.loss(X=X, mu=mu, log_sigmas=log_sigmas, pi_not_normalized=pi_not_normalized)
        loss.backward()

         #print(f"loss: {loss}")
        """for name, param in self.named_parameters():
            if param.requires_grad:
                self.writer.add_scalar(f"{name}_gradient", param.grad.mean(), global_step=self.epoch)"""
                
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 10)

        self.optimizer.step()
        return loss

    def batch_run(self, X, batch_size):
        avr_loss = 0.0
        self.Encoder_mu.train()
        self.Encoder_sigma.train()
        self.Decoder.train()
        
        for batch_idx in range(0, X.shape[0], batch_size):
            batch_X = make_coo_to_sparse_tensor(X[batch_idx:(batch_idx + batch_size), :].tocoo()).to(self.device)
            cur_loss = self.forward_pass(batch_X)
            avr_loss += cur_loss

        return avr_loss
    
    def train_model(self, X_train, X_validate, batch_size, epochs,  max_epochs_without_improvement, n_random_samples, verbose=0):          
        if verbose > 0:
            print("The following models will be used:")
            print(self.Encoder_mu)
            print(self.Encoder_sigma)
            print(self.Decoder)

        if self.optimizer is None:
            raise ValueError("Please assign first an optimizer to the model with model.optimizer = torch.optim.optimizer")

        # Make sure poor X_train is not hurt in the dirty process of training the VAE.
        X_train = X_train.copy()
        self.X_train = X_train

        last_recall_at_4 = 0.0
        epochs_without_improvement = 0
        
        """if not self.writer is None:
            # Log the model graph with a small example.
            self.writer.add_graph(self, make_coo_to_sparse_tensor(X_train[:min(X_train.shape[0], 10)].tocoo()).to(self.device))"""

        # The training is separated into two steps: Firstly run the pre-training, where it is trained until the final beta is reached without early stopping
        # Then the main training begins including early stopping
        if self.beta_step < self.beta:
            if verbose > 0:
                print("Starting pre-training!")
            for beta in tqdm(np.arange(start=0.0, stop=self.beta, step=self.beta_step), disable = verbose < 1):
                __ = self.batch_run(X=X_train, batch_size=batch_size)
            if verbose > 0:
                print("Beta path ascended now training final model.")

        # Main training run
        for epoch in tqdm(range(epochs), disable = verbose < 1):

            # set epoch for logging!
            self.epoch=epoch

            avr_loss_train = self.batch_run(X=X_train, batch_size=batch_size)

            if not self.writer is None:
                self.writer.add_scalar('loss_train', avr_loss_train, global_step=epoch)

            if verbose > 0:
                print(f" test Performance in epoch {epoch} was {avr_loss_train}.")  
        return epoch

class vae_model(RecModel):
    def __init__(self, final_beta, beta_step, k, num_items, dense_layers_encoder_mu, dense_layers_encoder_sigma, dense_layers_decoder, batch_norm_encoder_mu,batch_norm_encoder_sigma,  batch_norm_decoder,
                 dropout_rate_encoder_mu, dropout_rate_encoder_sigma, dropout_rate_decoder, dropout_rate_sparse_encoder_mu, dropout_rate_sparse_encoder_sigma,  model_path):

        self.num_items = num_items
        self.VAE = Mult_VAE(device=self.get_device(),  final_beta=final_beta,beta_step=beta_step, k=k, n_items=num_items, dense_layers_encoder_mu=dense_layers_encoder_mu, dense_layers_encoder_sigma=dense_layers_encoder_sigma,
            dense_layers_decoder=dense_layers_decoder, batch_norm_encoder_mu=batch_norm_encoder_mu, batch_norm_encoder_sigma=batch_norm_encoder_sigma,  batch_norm_decoder=batch_norm_decoder,
            dropout_rate_encoder_mu=dropout_rate_encoder_mu, dropout_rate_encoder_sigma=dropout_rate_encoder_sigma, dropout_rate_sparse_encoder_sigma=dropout_rate_sparse_encoder_sigma,
            dropout_rate_sparse_encoder_mu=dropout_rate_sparse_encoder_mu, dropout_rate_decoder=dropout_rate_decoder).to(self.get_device())
        self.VAE.model_path = model_path

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu") 
        return device

    def set_optimizer(self, *kwargs):
        self.VAE.set_optimizer(*kwargs)

    def set_writer(self, *kwargs):
        self.VAE.set_writer(*kwargs)

    def parameters(self):
        return self.VAE.parameters()

    def train(self, X_train, X_validate, batch_size, epochs, max_epochs_without_improvement, n_random_samples, verbose=0):
        self.VAE.Encoder_mu.train(True)
        self.VAE.Encoder_sigma.train(True)
        self.VAE.Decoder.train(True)
        return self.VAE.train_model(X_train=X_train, X_validate=X_validate, batch_size=batch_size, epochs=epochs,
         verbose=verbose, max_epochs_without_improvement=max_epochs_without_improvement,
         n_random_samples=n_random_samples)

    def rank(self, items, users, topn=None):
        return self.VAE.predict(users=users, items=items, topn=topn)
    
    def eval_topn(self, test_mat, batch_size, topn=np.array([10]), rand_sampled=1000, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        # if topn is not list make is one.
        if not isinstance(topn, np.ndarray):
            raise ValueError("Topn has to be a np.array")
        
        hits = np.zeros(topn.shape)
        max_topn = topn.max()
        for batch_idx in range(0, test_mat.shape[0], batch_size):
            batch = test_mat[batch_idx:(batch_idx + batch_size), :]

            predictions = self.VAE.predict(users=np.arange(batch_idx, min(batch_idx+batch_size, test_mat.shape[0])), items=None, topn=None)

            for user in range(0, len(batch.indptr) - 1):
                if batch.indptr[user + 1] - batch.indptr[user] > 0:
                    rand_items = np.random.randint(0, self.num_items, size=rand_sampled)
                    items_selected = batch.indices[batch.indptr[user]:batch.indptr[user + 1]]
                    for item in items_selected:
                        items = np.append(item, rand_items)
                        rel_predictions = predictions[user, :][items]
                        candidates = items[np.argpartition(a=rel_predictions, kth=list(range(-max_topn, 0, 1)))][-max_topn:][::-1]
                        for pos in range(len(topn)):
                            if item in candidates[:topn[pos]]:
                                hits[pos] += 1

        # Compute the precision at topn
        recall_dict = {}
        recall = hits / len(test_mat.nonzero()[0])
        for pos in range(len(topn)):
            recall_dict[f"Recall@{topn[pos]}"] = recall[pos]

        return recall_dict