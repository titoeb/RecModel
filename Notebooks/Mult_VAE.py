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

from sklearn.preprocessing import normalize

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
    def __init__(self, k, dense_layers, batch_norm, activation, dropout_rate):
        super().__init__()
        self.k  = k
        self.additional_layer = len(dense_layers)
        if self.additional_layer > 2:
            all_layers = [DenseLayers(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate,
                                                          batch_norm=batch_norm, activation=activation) for (in_channels, out_channels) in zip(dense_layers[1:-1], dense_layers[2:])]
            
            all_layers[0:0] = [DenseLayers(in_channels=dense_layers[0], out_channels=dense_layers[1], batch_norm=batch_norm, dropout_rate=None, activation=activation)]
            self.dense_network = nn.Sequential(*all_layers)
        elif self.additional_layer > 1:
            self.dense_network = DenseLayers(in_channels=dense_layers[0], out_channels=dense_layers[1], batch_norm=batch_norm, dropout_rate=None, activation=activation)
        else:
            pass
        self.out_layer = nn.Linear(dense_layers[-1], k)
        
    def forward(self, X):
        if self.additional_layer > 1:
            X = self.dense_network(X)
        return self.out_layer(X)
    
class Decoder(nn.Module):
    def __init__(self, n_items, dense_layers, batch_norm, activation, dropout_rate):
        super().__init__()
        self.n_items = n_items
        self.additional_layer = len(dense_layers)
        if self.additional_layer > 2:
            all_layers = [DenseLayers(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate,
                                                          batch_norm=batch_norm, activation=activation) for (in_channels, out_channels) in zip(dense_layers[1:-1], dense_layers[2:])]
            
            all_layers[0:0] = [DenseLayers(in_channels=dense_layers[0], out_channels=dense_layers[1], batch_norm=batch_norm, dropout_rate=None, activation=activation)]
            print(all_layers)
            self.dense_network = nn.Sequential(*all_layers)
        elif self.additional_layer > 1:
            self.dense_network = DenseLayers(in_channels=dense_layers[0], out_channels=dense_layers[1], batch_norm=batch_norm, dropout_rate=None, activation=activation)
        else:
            pass
        self.out_layer = nn.Linear(dense_layers[-1], n_items)
        
    def forward(self, X):
        if self.additional_layer > 1:
            X = self.dense_network(X)
        return self.out_layer(X)

class Mult_VAE(nn.Module):
    def __init__(self, device, beta, k, n_items, dense_layers_encoder, dense_layers_decoder, batch_norm_encoder, batch_norm_decoder,
                 dropout_rate_encoder, dropout_rate_decoder, optimizer=None, writer=None):
        super().__init__()
        self.k = k
        self.Encoder_mu = Encoder(k=k, dense_layers=dense_layers_encoder, batch_norm=batch_norm_encoder, activation=True, dropout_rate=dropout_rate_encoder)
        self.Encoder_log_sigmas = Encoder(k=k, dense_layers=dense_layers_encoder, batch_norm=batch_norm_encoder, activation=True, dropout_rate=dropout_rate_encoder)
        self.Decoder = Decoder(n_items=n_items, dense_layers=dense_layers_decoder, batch_norm=batch_norm_decoder, activation=True, dropout_rate=dropout_rate_decoder)
        self.device = device
        self.beta = beta
        self.optimizer = optimizer
        self.writer = writer
    
    def forward(self, X):
        mu = self.Encoder_mu(X)
        log_sigmas = self.Encoder_log_sigmas(X)
        z = Variable(torch.randn(X.shape[0], self.k,  requires_grad=False)).to(self.device)
        pi_not_normalized = self.Decoder(z.mul(torch.sqrt(torch.exp(log_sigmas))).add(mu))
        return mu, log_sigmas, pi_not_normalized

    def predict(self, users, topn=None):
        with torch.no_grad():
            rel_data = self.X_train[users, :]
            rel_data = make_coo_to_sparse_tensor(rel_data.tocoo()).to(self.device)

            z = self.Encoder_mu(rel_data)
            predictions = self.Decoder(z)

            print(predictions)

            if topn is None:
                return predictions
            else:
                return np.argpartition(a=predictions.cpu().numpy(), kth=list(range(-topn, 0, 1)), axis=1)[:, -topn:]
    
    def loss(self, X, mu, log_sigmas, pi_not_normalized): # be careful don't overwrite some pre-defined loss from the super class.
        """self.writer.add_histogram('mu', mu, global_step=self.epoch)
        self.writer.add_histogram('sigmas', sigmas, global_step=self.epoch)
        self.writer.add_histogram('pi', pi, global_step=self.epoch)"""
        
        #log_likelihood = sparse_dense_mul(X, log_pi).sum(dim=1)
        log_likelihood = torch.sparse.sum(sparse_dense_mul_sparse_output(X, F.log_softmax(pi_not_normalized, dim=1)), dim=1).to_dense().to(self.device)
        
        #log_likelihood = torch.mul(X, pi).sum(dim=1)
        #print(f"log_sigmas.mean(): {log_sigmas.mean()}, log_sigmas.max(): {log_sigmas.max()}, log_sigmas.min(): {log_sigmas.min()}")
        KL = torch.sum(0.5 * (-log_sigmas + torch.exp(log_sigmas) + mu ** 2 - 1), dim=1) * self.beta
        
        """print(f"KL: {KL.sum()}, log_likelihood: {log_likelihood.sum()}")
        print(f"mu: {mu.sum()}, log_sigmas: {log_sigmas.sum()}, log_pi: {log_pi.sum()}")
        
        self.writer.add_histogram('log_likelihood', log_likelihood, global_step=self.epoch)
        self.writer.add_histogram('KL_part1', KL_part1, global_step=self.epoch)
        self.writer.add_histogram('KL_part2', KL_part2, global_step=self.epoch)
        self.writer.add_histogram('KL_part3', KL_part3, global_step=self.epoch)
        self.writer.add_histogram('KL', KL, global_step=self.epoch)"""
        
        #KL = 0.5 * (torch.sum(torch.log(sigmas), dim=1) - self.k + torch.sum((1/(sigmas + 1e-10)), dim=1) + torch.sum(torch.mul(sigmas, torch.mul(mu, mu)), dim=1))
        #print(f"log_likleihood: {log_likelihood}, part_1: {KL_part1}, part_2: {KL_part2}, part_3: {KL_part3}, KL: {KL}")
        #print(f"KL.shape: {KL.shape}, log_likelihood.shape: {log_likelihood.shape}")
        return torch.sum(torch.add(KL, log_likelihood))
    
    def forward_pass(self, X):
        self.train(True)
        self.zero_grad()
        mu, log_sigmas, pi_not_normalized = self.__call__(X)
        loss = self.loss(X=X, mu=mu, log_sigmas=log_sigmas, pi_not_normalized=pi_not_normalized)
        loss.backward()

         #print(f"loss: {loss}")
        """for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad.sum()}")"""
                
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 20)

        self.optimizer.step()
        return loss

    def batch_run(self, X, batch_size, verbose=0):
        avr_loss = 0.0
        
        if verbose > 0:
            for batch_idx in tqdm(range(0, X.shape[0], batch_size)):
                batch_X = make_coo_to_sparse_tensor(X[batch_idx:(batch_idx + batch_size), :].tocoo()).to(self.device)
                cur_loss = self.forward_pass(batch_X)
                avr_loss += cur_loss
        else:
            for batch_idx in range(0, X.shape[0], batch_size):
                batch_X = make_coo_to_sparse_tensor(X[batch_idx:(batch_idx + batch_size), :].tocoo()).to(self.device)
                cur_loss = self.forward_pass(batch_X)
                avr_loss += cur_loss
        
        return avr_loss
    
    def train_model(self, X_train,  batch_size, epochs, verbose=0):          
        if self.optimizer is None:
            raise ValueError("Please assign first an optimizer to the model with model.optimizer = torch.optim.optimizer")

        # Make sure poor X_train is not hurt in the dirty process of training the VAE.
        X_train = X_train.copy()
        self.X_train = X_train
        
        if not self.writer is None:
            # Log the model graph with a small example.
            self.writer.add_graph(self, make_coo_to_sparse_tensor(X_train[:min(X_train.shape[0], 10)].tocoo()).to(self.device))

        for epoch in tqdm(range(epochs)):
            self.epoch=epoch # RM LATER!!
            avr_loss_train  = self.batch_run(X=X_train, batch_size=batch_size, verbose=0)

            if not self.writer is None:
                self.writer.add_scalar('loss_train', avr_loss_train, global_step=epoch)
                """for name, params in self.named_parameters():
                    self.writer.add_histogram(name, params, global_step=epoch)"""

            if verbose > 0:
                print(f" test Performance in epoch {epoch} was {avr_loss_train}.")

def give_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu") 
    return device

if __name__ == "__main__":

    train_mat = scipy.sparse.load_npz('../data/mat_bin_train.npz')[:10000, :2000]
    eval_mat = scipy.sparse.load_npz('../data/mat_bin_validate.npz')
    train_mat_normalized = normalize(X=train_mat, norm='l2', axis=1)

    # Get GPU if available else CPU
    device = give_best_device() 
    print(f"device is {device}")
    # Create a train test VAE.
    test = Mult_VAE(device=device, k = 10, n_items = train_mat.shape[1], dense_layers_encoder=[train_mat.shape[1], 1000,100, 100], dense_layers_decoder=[10, 100, 100, 1000], batch_norm_encoder=True,
                batch_norm_decoder=True, dropout_rate_decoder=0.5, dropout_rate_encoder=0.5, beta=0.2).to(device)

    time_stamp = ''.join(str(datetime.now()).split('.')[:-1]).replace(' ', '_').replace(':', '_').replace('-', '_')
    test.writer = SummaryWriter('./logs/' + time_stamp)

    test.optimizer = optim.Adam(test.parameters(), lr=1e-3)
    test.train_model(X_train=train_mat_normalized, batch_size=1000, epochs=5, verbose=1)
    prediction = test.predict(users=np.array([1, 4, 11, 100, 12]), topn=10)
    print(prediction)
