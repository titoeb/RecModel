import os
import numpy as np
import scipy.sparse
import yaml
import mlflow
import time
import utils.config_helpers
import os
import RecModel
import datetime
import hydra
import logging
import hyperopt as hp
from functools import partial
import pickle
import torch
import torch.optim

log = logging.getLogger(__name__)

# Helper functions
def eval_VAE(params, cfg, train_mat_bin, train_mat_count, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    with mlflow.start_run(experiment_id=experiment):

        # Log the config
        utils.config_helpers.log_config(dict(cfg.model))        

        n_users, n_items = train_mat_bin.shape

        # Some simple pre-procesing steps for the params
        params['k'] = max(int(params['k']), 1)
        params['dense_layers_encoder_mu'] = max(int(params['dense_layers_encoder_mu']), 1)
        params['dense_layers_encoder_sigma'] = max(int(params['dense_layers_encoder_sigma']), 1)
        params['dense_layers_decoder'] = max(int(params['dense_layers_decoder']), 1)

        # Log relevant parameters for this run.
        print("Testing the following hyper parmaters!")
        for key, val in dict(params).items():
            mlflow.log_param(key, val)
            print(f"{key}: {val}")
        
        # Select the correct matrix to train.
        if params['mat'] == 'count':
            train_mat = train_mat_count.copy()
        elif params['mat'] == 'bin':
            train_mat = train_mat_bin.copy()
        else:
            raise ValueError(f"mat can only take values 'count' or 'bin' and not {params['mat']}")
        
       # Create model
       # Create Mult_VAE
        vae = RecModel.Mult_VAE(k = int(params['k']),  num_items = train_mat.shape[1],
            dense_layers_encoder_mu=[int(params['dense_layers_encoder_mu'])],
            dense_layers_encoder_sigma= [int(params['dense_layers_encoder_sigma'])],
            dense_layers_decoder= [int(params['dense_layers_decoder'])],
            batch_norm_encoder_mu=params['batch_norm_encoder_mu'],
            batch_norm_encoder_sigma=params['batch_norm_encoder_sigma'],
            batch_norm_decoder=params['batch_norm_decoder'],
            dropout_rate_decoder=params['dropout_rate_decoder'],
            dropout_rate_encoder_mu=params['dropout_rate_encoder_mu'],
            dropout_rate_encoder_sigma=params['dropout_rate_encoder_sigma'],
            dropout_rate_sparse_encoder_mu= params['dropout_rate_sparse_encoder_mu'],
            dropout_rate_sparse_encoder_sigma=params['dropout_rate_sparse_encoder_sigma'],
            final_beta=params['final_beta'], beta_step=params['beta_step'], model_path=cfg.model.model_path)

        # Set optimizer
        vae.set_optimizer(torch.optim.Adam(vae.parameters(), lr=params['learning_rate'],
         weight_decay=params['weight_decay']))

        print(f"start training!")
        start = time.time()
        epochs = vae.train(X_train=train_mat.copy(), X_validate=eval_mat.copy(),
         batch_size=int(cfg.model.batch_size), epochs=int(params['n_epochs']),
        verbose=int(cfg.model.verbose), 
        max_epochs_without_improvement=int(cfg.model.max_epochs_without_improvement),
        n_random_samples=int(cfg.model.n_random_samples))

        # Log run-time
        mlflow.log_metric("Runtime", int(round(time.time() - start, 0)))
        mlflow.log_metric("epochs_training", epochs + 1)

        # Evaluate model
        perf_all = vae.eval_topn(test_mat=eval_mat.copy(),
         batch_size=int(cfg.model.batch_size), topn=np.array(cfg.model.top_n_performances),
          rand_sampled = int(cfg.model.rand_sampled), random_state=None)
        
        # Log the performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])
        
        #We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.
        rel_topn_perf = perf_all[f"Recall@{cfg.model.top_n_performances[0]}"]
  
        log.info(f"Current recallAT{cfg.model.top_n_performances[0]}_of_{cfg.model.rand_sampled} performance was {rel_topn_perf} and model ran for {epochs + 1} epochs")
        loss = -rel_topn_perf
        return {'loss': loss, 'status': hp.STATUS_OK, 'eval_time': time.time()}

def hyper_opt_fmin(space, fun, additional_evals, verbose = 0, trials_path='../trials.p', **kwargs):
    # This is a wrapper around the training process that enables warm starts from file.

    objective = partial(fun, **kwargs)
    
    # Try to recover trials object, else create new one!
    try:
        trials = pickle.load(open(trials_path, "rb"))
        if verbose > 0:
            print(f"Loaded trails from {trials_path}")
    except FileNotFoundError:
        trials = hp.Trials()
        
    # Compute the effect number of new trials that have to be run.
    past_evals = len(trials.losses())
    new_evals = past_evals + additional_evals

    best = hp.fmin(fn = objective, space=space, algo=hp.tpe.suggest,  max_evals = new_evals, trials=trials)
    if verbose > 0:
        print(f"HyperOpt got best loss {trials.best_trial['result']['loss']} with the following hyper paramters: \n{trials.best_trial['misc']['vals']}")
        
    # Store the trials object
    pickle.dump(trials, open(trials_path, "wb"))
    
    return best, trials

# Work around to get the working directory (after release use hydra.utils.get_original_cwd())
from hydra.plugins.common.utils import HydraConfig
def get_original_cwd():
    return HydraConfig().hydra.runtime.cwd

@hydra.main(config_path='configs/config.yaml')
def my_app(cfg):
    # Main 

    # Load mat.
    # Be aware that hydra changes the working directory
    train_mat_bin = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_bin_path))
    train_mat_count = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_count_path))
    n_users, n_items = train_mat_bin.shape
    eval_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.eval_mat_path))

    train_mat_bin = train_mat_bin.astype(np.float64)
    train_mat_count = train_mat_count.astype(np.float64)
    eval_mat = eval_mat.astype(np.float64)
    eval_mat.sort_indices()
    train_mat_bin.sort_indices()
    train_mat_count.sort_indices()

    # Setup HyperOpt
    space = space = {
        # General Hyper paramters
        'k': hp.hp.uniform('k', 0, 300),
        'final_beta': hp.hp.uniform('final_beta', 0, 1),
        'beta_step': hp.hp.uniform('beta_step', 0, 1),
        'weight_decay': hp.hp.uniform('weight_decay', 0, 1),
        'learning_rate': hp.hp.uniform('learning_rate', 0, 0.1),
        'mat': hp.hp.choice('mat', ['count', 'bin']),
        'n_epochs': hp.hp.uniform('n_epochs', 1, 300),
        # Hyper paramters for the encoder for mu
        'dense_layers_encoder_mu': hp.hp.uniform('dense_layers_encoder_mu', 0, 1000),
        'batch_norm_encoder_mu': hp.hp.choice('batch_norm_encoder_mu', [True, False]),
        'dropout_rate_encoder_mu': hp.hp.uniform('dropout_rate_encoder_mu', 0, 1),
        'dropout_rate_sparse_encoder_mu': hp.hp.uniform('dropout_rate_sparse_encoder_mu', 0, 1),
        # Hyper parameter for the encoder for sigma
        'dense_layers_encoder_sigma': hp.hp.uniform('dense_layers_encoder_sigma', 0, 1000),
        'batch_norm_encoder_sigma': hp.hp.choice('batch_norm_encoder_sigma', [True, False]),
        'dropout_rate_encoder_sigma': hp.hp.uniform('dropout_rate_encoder_sigma', 0, 1),
        # Hyper paramter for the decoder
        'dropout_rate_sparse_encoder_sigma': hp.hp.uniform('dropout_rate_sparse_encoder_sigma', 0, 1),
        'dense_layers_decoder': hp.hp.uniform('dense_layers_decoder', 0, 1000),
        'batch_norm_decoder': hp.hp.choice('batch_norm_decoder', [True, False]),
        'dropout_rate_decoder': hp.hp.uniform('dropout_rate_decoder', 0, 1)
        }

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_VAE_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")
    hyper_opt_fmin(space, eval_VAE, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat_count=train_mat_count, train_mat_bin=train_mat_bin, eval_mat=eval_mat, experiment=experiment)
    
    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")
    
    
if __name__ == "__main__":
    my_app()
