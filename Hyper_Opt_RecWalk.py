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
from RecModel import unfold_config

log = logging.getLogger(__name__)

# Helper functions
def eval_recwalk(params, cfg, train_mat_bin, train_mat_count, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    print(experiment)
    with mlflow.start_run(experiment_id=experiment):

        # flatten the config.
        params = unfold_config(params)

        # Log the config in hydra
        utils.config_helpers.log_config(dict(cfg.model))   

        # Log the params in mlflow
        utils.config_helpers.log_config(params) 
        
        n_users, n_items = train_mat_bin.shape
        np.random.seed(seed=cfg.model.seed)

        # Log this run
        log.info(f"Hyper parameter for this run are {params}")

        # Create model and train and evaluate it.
        slim_W = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.slim_W))

        if params['eval_method'] == 'PR':
            recwalk = RecModel.Recwalk(num_items=n_items, num_users=n_users, eval_method=params['eval_method'], k_steps=params['steps'], damping=params['damping'], slim_W=slim_W)
        else:
             recwalk = RecModel.Recwalk(num_items=n_items, num_users=n_users, eval_method=params['eval_method'], k_steps=params['steps'], damping=None, slim_W=slim_W)

        start = time.time()
        if params['train_mat'] == 'count':
            recwalk.train(train_mat_count.copy(), phi=params['phi'], alpha=0.0, l1_ratio=0.0, max_iter=0.0, tolerance=0.0, cores=cfg.model.cores, verbose=cfg.model.verbose)
        else:
            recwalk.train(train_mat_bin.copy(), phi=params['phi'], alpha=0.0, l1_ratio=0.0, max_iter=0.0, tolerance=0.0, cores=cfg.model.cores, verbose=cfg.model.verbose)

        # Log the training time
        mlflow.log_metric("training_time", int(round(time.time() - start, 0)))

        start = time.time()
        perf_all = recwalk.eval_topn(test_mat=eval_mat.copy(), topn=np.array(cfg.model.top_n_performances,
         dtype=np.int32), rand_sampled=int(cfg.model.rand_sampled), cores=int(cfg.model.cores), random_state= int(cfg.model.seed))
        mlflow.log_metric("Topn_evaluation_time", int(round(time.time() - start, 0)))

        # Log the topn performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])

        # We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.
        rel_topn_perf = perf_all[f"Recall@{cfg.model.top_n_performances[0]}"]     
        log.info(f"Current recallAT{cfg.model.top_n_performances[0]}_of_{cfg.model.rand_sampled} performance was {rel_topn_perf}")
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

    # The algorithms rely on the sparse matrices being sorted by idx.
    train_mat_bin = train_mat_bin.astype(np.float32)
    train_mat_count = train_mat_count.astype(np.float32)
    eval_mat = eval_mat.astype(np.float32)
    eval_mat.sort_indices()
    train_mat_bin.sort_indices()
    train_mat_count.sort_indices()

    # Setup HyperOpt
    space = {'eval_method' : hp.hp.choice('eval_method', [
                {'type': 'PR',
                'damping' : hp.hp.uniform('damping', 0, 1),
                'steps': hp.hp.choice('steps_PR', np.arange(1, 20))
                },

                {'type': 'k_step',
                'steps': hp.hp.choice('steps_k_steps', np.arange(1, 20))
                }
            ]),
            'train_mat': hp.hp.choice('train_mat', ['count', 'bin']),
            'phi': hp.hp.uniform('phi', 0, 1)
            }

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_recwalk_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")
    hyper_opt_fmin(space, eval_recwalk, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat_count=train_mat_count, train_mat_bin=train_mat_bin, eval_mat=eval_mat, experiment=experiment)  

    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")

if __name__ == "__main__":
    my_app()
